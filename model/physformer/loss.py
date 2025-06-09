import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from torch.autograd import Variable
from scipy.signal import butter, filtfilt


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
