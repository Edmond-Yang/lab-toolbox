import math
import torch
import numpy as np
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

from enum import Enum
from torch.autograd import Variable
from scipy.signal import butter, filtfilt

EPSILON = 1e-10
BP_LOW=2/3
BP_HIGH=3.0
BP_DELTA=0.1


def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = signal.shape[-1]
        pad_len = int(zero_pad / 2 * L)
        signal = F.pad(signal, (pad_len, pad_len), mode='constant', value=0)

    freqs = torch.fft.fftfreq(signal.shape[-1], 1 / Fs) * 60  # in bpm
    ps = torch.abs(torch.fft.fft(signal, dim=-1)) ** 2
    cutoff = len(freqs) // 2
    return freqs[:cutoff], ps[:, :cutoff]


def predict_heart_rate(signal, Fs=30, min_hr=40., max_hr=180.):
    signal -= torch.mean(signal, dim=-1, keepdim=True)
    freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)

    # Make sure freqs is on the same device as signal
    freqs = freqs.to(signal.device)

    mask = (freqs >= min_hr) & (freqs <= max_hr)
    freqs = freqs[mask]
    ps = ps[:, mask]

    max_ind = torch.argmax(ps, dim=-1)
    max_bpm = torch.zeros(signal.shape[0], device=signal.device)

    for i in range(signal.shape[0]):
        if 0 < max_ind[i] < len(freqs) - 1:
            inds = max_ind[i] + torch.tensor([-1, 0, 1], device=signal.device)
            x = ps[i][inds]
            f = freqs[inds]
            d1 = x[1] - x[0]
            d2 = x[1] - x[2]
            offset = (1 - min(d1, d2) / max(d1, d2)) * (f[1] - f[0])
            if d2 > d1:
                offset *= -1
            max_bpm[i] = f[1] + offset
        elif max_ind[i] == 0:
            max_bpm[i] = freqs[0]
        elif max_ind[i] == len(freqs) - 1:
            max_bpm[i] = freqs[-1]

    return max_bpm

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def compute_complex_absolute_given_k(output, k, N):
    two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
    hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

    k = k.type(torch.FloatTensor).cuda()
    two_pi_n_over_N = two_pi_n_over_N.cuda()
    hanning = hanning.cuda()

    output = output.view(1, -1) * hanning
    output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
    k = k.view(1, -1, 1)
    two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
    complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                       + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

    return complex_absolute

def complex_absolute(output, Fs, bpm_range=None):
    output = output.view(1, -1)

    N = output.size()[1]

    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz

    # only calculate feasible PSD range [0.7,4]Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)

    return (1.0 / complex_absolute.sum()) * complex_absolute


def butter_bandpass_torch(sig: torch.Tensor,
                          lowcut: float,
                          highcut: float,
                          fs: float,
                          order: int = 2) -> torch.Tensor:

    # ========== 1. 頻率向量 ==========
    L = sig.shape[-1]
    freqs = torch.fft.rfftfreq(L, d=1.0 / fs, device=sig.device)

    # 避免除以零
    eps = 1e-12
    f_safe = freqs + eps

    # ========== 2. Butterworth 振幅響應 ==========
    #   |H_HP| = 1 / sqrt(1 + (fc / f)^(2n))
    #   |H_LP| = 1 / sqrt(1 + (f / fc)^(2n))
    hp = 1.0 / torch.sqrt(1.0 + (lowcut / f_safe) ** (2 * order))
    lp = 1.0 / torch.sqrt(1.0 + (f_safe / highcut) ** (2 * order))
    H  = hp * lp
    H[0] = 0.0  # 直流分量強制抑制

    # 將 H reshape 成 (..., 1, F) 以便廣播
    H_shape = [1] * (sig.ndim - 1) + [H.numel()]
    H = H.view(*H_shape)

    # ========== 3. FFT → 相乘 → IFFT ==========
    spec      = torch.fft.rfft(sig, dim=-1)
    spec_filt = spec * H
    filtered  = torch.fft.irfft(spec_filt, n=L, dim=-1)

    return filtered

class NegativePearsonLoss(nn.Module):

    def __init__(self):
        super(NegativePearsonLoss, self).__init__()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        loss = 0.0
        batch_size = preds.shape[0]
        num_elements = preds.shape[1]

        for i in range(batch_size):

            pred = preds[i]
            label = labels[i]

            sum_x = torch.sum(pred)
            sum_y = torch.sum(label)
            sum_xy = torch.sum(pred * label)
            sum_x2 = torch.sum(torch.pow(pred, 2))
            sum_y2 = torch.sum(torch.pow(label, 2))

            # 計算 Pearson 相關係數
            numerator = num_elements * sum_xy - sum_x * sum_y
            denominator = torch.sqrt(
                (num_elements * sum_x2 - torch.pow(sum_x, 2)) *
                (num_elements * sum_y2 - torch.pow(sum_y, 2))
            )

            # 避免除以零的錯誤
            if denominator == 0:
                pearson = torch.tensor(0.0, device=preds.device)  # 或者可以處理為其他值
            else:
                pearson = numerator / denominator

            # 根據 Pearson 相關係數計算損失
            loss += 1 - pearson

        # 計算批次的平均損失
        loss = loss / batch_size
        return loss


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.fps = 30
        self.min_hr = 40.

    def _calculate_loss(self, pred_signal: torch.Tensor, gt_hr: torch.Tensor) -> torch.Tensor:
        bpm_range = torch.arange(40, 181, dtype=torch.float).cuda()
        hr = complex_absolute(pred_signal, self.fps, bpm_range)
        return F.cross_entropy(hr, gt_hr.view((1)).type(torch.long))

    def forward(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        preds = butter_bandpass_torch(preds, 0.6, 4.0, 30)
        gts = butter_bandpass_torch(gts, 0.6, 4.0, 30)

        loss = 0.
        gt_hrs = predict_heart_rate(gts)

        for i in range(preds.shape[0]):
            pred_signal = preds[i]
            gt_hr = gt_hrs[i]

            loss += self._calculate_loss(pred_signal, gt_hr-self.min_hr)

        return loss / preds.shape[0]


class LabelDistributionLoss(nn.Module):

    def __init__(self):
        super(LabelDistributionLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.fps = 30
        self.min_hr = 40.
        self.gaussian_std = 1.0
        self.bpm_range = torch.arange(40, 181, dtype=torch.float).cuda()

    def _kl_loss(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        criterion = torch.nn.KLDivLoss(reduction='none')
        outputs = torch.log(preds)
        loss = criterion(outputs, gts)
        loss = loss.sum()
        return loss

    def _calculate_kl_loss(self, pred_signal: torch.Tensor, gt_hr: torch.Tensor) -> torch.Tensor:
        target_distribution = [normal_sampling(int(gt_hr), i, self.gaussian_std) for i in range(141)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).cuda()

        bpm_range = torch.arange(40, 181, dtype=torch.float).cuda()
        hr = complex_absolute(pred_signal, self.fps, bpm_range)
        fre_distribution = F.softmax(hr.view(-1), dim=0)
        return self._kl_loss(fre_distribution, target_distribution)

    def forward(self, preds, gts):

        preds = butter_bandpass_torch(preds, 0.6, 4.0, 30)
        gts = butter_bandpass_torch(gts, 0.6, 4.0, 30)

        loss = 0.
        gt_hrs = predict_heart_rate(gts)

        for i in range(preds.shape[0]):
            pred_signal = preds[i]
            gt_hr = gt_hrs[i]

            # Calculate KL loss
            loss += self._calculate_kl_loss(pred_signal, gt_hr-self.min_hr)

        return loss / preds.shape[0]


class MeanAbsoluteError(nn.Module):

    def __init__(self):
        super(MeanAbsoluteError, self).__init__()
        self.criterion = nn.L1Loss()
    def forward(self, preds, gts):

        preds = butter_bandpass_torch(preds, 0.6, 4.0, 30)
        gts = butter_bandpass_torch(gts, 0.6, 4.0, 30)

        pred_hr = predict_heart_rate(preds)
        gt_hr = predict_heart_rate(gts)
        return self.criterion(pred_hr, gt_hr)


def select_loss(losses='bsv'):

    criterion_funcs = {
            'bandwidth': IPR_SSL, ## bandwidth loss
            'sparsity': SNR_SSL, ## sparsity loss
            'variance': EMD_SSL, ## variance loss
            'snrharm': SNR_harmonic_SSL, ## sparsity with harmonics (not recommended)
            'normnp': NP_SUPERVISED ## supervised negative pearson loss
    }

    criterions = {}
    if losses == "supervised":
        criterions['supervised'] = criterion_funcs[arg_obj.supervised_loss]
    elif losses == "supervised_priors":
        criterions['supervised'] = criterion_funcs[arg_obj.supervised_loss]
        criterions['bandwidth'] = criterion_funcs[arg_obj.bandwidth_loss]
        criterions['sparsity'] = criterion_funcs[arg_obj.sparsity_loss]
    else:
        if 'b' in losses:
            criterions['bandwidth'] = criterion_funcs['bandwidth']
        if 's' in losses:
            criterions['sparsity'] = criterion_funcs['sparsity']
        if 'v' in losses:
            criterions['variance'] = criterion_funcs['variance']

    return criterions


def _IPR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    use_freqs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    zero_freqs = torch.logical_not(use_freqs)
    use_energy = torch.sum(psd[:,use_freqs], dim=1)
    zero_energy = torch.sum(psd[:,zero_freqs], dim=1)
    denom = use_energy + zero_energy + EPSILON
    ipr_loss = torch.mean(zero_energy / denom)
    return ipr_loss


def IPR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    if speed is None:
        ipr_loss = _IPR_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
    else:
        batch_size = psd.shape[0]
        ipr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            psd_b = psd[b].view(1,-1)
            ipr_losses[b] = _IPR_SSL(freqs, psd_b, low_hz=low_hz_b, high_hz=high_hz_b, device=device)
        ipr_loss = torch.mean(ipr_losses)
    return ipr_loss


def _EMD_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth mover's distance to uniform distribution.
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    if not normalized:
        psd = normalize_psd(psd)
    B,T = psd.shape
    psd = torch.sum(psd, dim=0) / B
    expected = ((1/T)*torch.ones(T)).to(device) #uniform distribution
    emd_loss = torch.mean(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))
    return emd_loss


def EMD_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth movers distance to uniform distribution.
    '''
    if speed is None:
        emd_loss = _EMD_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        B = psd.shape[0]
        expected = torch.zeros_like(freqs).to(device)
        for b in range(B):
            speed_b = speed[b]
            low_hz_b = low_hz * speed_b
            high_hz_b = high_hz * speed_b
            supp_idcs = torch.logical_and(freqs >= low_hz_b, freqs <= high_hz_b)
            uniform = torch.zeros_like(freqs)
            uniform[supp_idcs] = 1 / torch.sum(supp_idcs)
            expected = expected + uniform.to(device)
        lowest_hz = low_hz*torch.min(speed)
        highest_hz = high_hz*torch.max(speed)
        bpassed_freqs, psd = ideal_bandpass(freqs, psd, lowest_hz, highest_hz)
        bpassed_freqs, expected = ideal_bandpass(freqs, expected[None,:], lowest_hz, highest_hz)
        expected = expected[0] / torch.sum(expected[0]) #normalize expected psd
        psd = normalize_psd(psd) # treat all samples equally
        psd = torch.sum(psd, dim=0) / B # normalize batch psd
        emd_loss = torch.mean(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))
    return emd_loss


def _SNR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    signal_freq_idx = torch.argmax(psd, dim=1)
    signal_freq_idx = signal_freq_idx.to(freqs.device)
    signal_freq = freqs[signal_freq_idx].view(-1, 1)
    freqs = freqs.repeat(psd.shape[0],1)
    low_cut = signal_freq - freq_delta
    high_cut = signal_freq + freq_delta
    band_idcs = torch.logical_and(freqs >= low_cut, freqs <= high_cut).to(device)
    signal_band = torch.sum(psd * band_idcs, dim=1)
    noise_band = torch.sum(psd * torch.logical_not(band_idcs), dim=1)
    denom = signal_band + noise_band + EPSILON
    snr_loss = torch.mean(noise_band / denom)
    return snr_loss


def SNR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if speed is None:
        snr_loss = _SNR_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        batch_size = psd.shape[0]
        snr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            snr_losses[b] = _SNR_SSL(freqs, psd[b].view(1,-1), low_hz=low_hz_b, high_hz=high_hz_b, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
        snr_loss = torch.mean(snr_losses)
    return snr_loss


def _SNR_harmonic_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' This sparsity loss incorporates the power in the second harmonic.
        We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
    signal_freq_idx = torch.argmax(psd, dim=1)
    signal_freq = freqs[signal_freq_idx].view(-1,1)
    freqs = freqs.repeat(psd.shape[0],1)
    # First harmonic
    low_cut1 = signal_freq - freq_delta
    high_cut1 = signal_freq + freq_delta
    band_idcs = torch.logical_and(freqs >= low_cut1, freqs <= high_cut1).to(device)
    signal_band = torch.sum(psd * band_idcs, dim=1)
    # Second harmonic
    low_cut2 = 2*signal_freq - freq_delta
    high_cut2 = 2*signal_freq + freq_delta
    harm_idcs = torch.logical_and(freqs >= low_cut2, freqs <= high_cut2).to(device)
    harm_band = torch.sum(psd * harm_idcs, dim=1)
    total_power = torch.sum(psd, dim=1)
    numer = total_power - (signal_band + harm_band)
    denom = total_power + EPSILON
    snr_harm_loss = torch.mean(numer / denom)
    return snr_harm_loss


def SNR_harmonic_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' This sparsity loss incorporates the power in the second harmonic.
        We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if speed is None:
        snr_loss = _SNR_harmonic_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        batch_size = psd.shape[0]
        snr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            snr_losses[b] = _SNR_harmonic_SSL(freqs, psd[b].view(1,-1), low_hz=low_hz_b, high_hz=high_hz_b, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
        snr_loss = torch.mean(snr_losses)
    return snr_loss


def NP_SUPERVISED(x, y, spectral1=None, spectral2=None):
    ''' Same as negative pearson loss, but the result is between 0 and 1.
    '''
    if len(x.shape) < 2:
        x = torch.reshape(x, (1,-1))
    mean_x = torch.mean(x, 1)
    mean_y = torch.mean(y, 1)
    xm = x.sub(mean_x[:, None])
    ym = y.sub(mean_y[:, None])
    r_num = torch.einsum('ij,ij->i', xm, ym)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1) + EPSILON
    r_vals = r_num / r_den
    r_val = torch.mean(r_vals)
    return (1 - r_val)/2


def ideal_bandpass(freqs, psd, low_hz, high_hz):
    freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:,freq_idcs]
    return freqs, psd


def normalize_psd(psd):
    return psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities


def torch_power_spectral_density(x, nfft=5400, fps=90, low_hz=BP_LOW, high_hz=BP_HIGH, return_angle=False, radians=True, normalize=True, bandpass=True):
    centered = x - torch.mean(x, keepdim=True, dim=1)
    rfft_out = fft.rfft(centered, n=nfft, dim=1)
    psd = torch.abs(rfft_out)**2
    N = psd.shape[1]
    freqs = fft.rfftfreq(2*N-1, 1/fps)
    if return_angle:
        angle = torch.angle(rfft_out)
        if not radians:
            angle = torch.rad2deg(angle)
        if bandpass:
            freqs, psd, angle = ideal_bandpass(freqs, psd, low_hz, high_hz, angle=angle)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd, angle
    else:
        if bandpass:
            freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd
