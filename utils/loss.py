import torch
import math
from enum import Enum
import numpy as np
from torch.autograd import Variable
from scipy.signal import butter, filtfilt
import torch.nn.functional as F
import pdb
import torch.nn as nn


import torch
import torch.nn as nn


class NP_Loss(nn.Module):
    """
    計算基於 Pearson 相關係數的損失函數。

    Pearson 相關係數的範圍是 [-1, 1]。
    - 若相關係數 < 0，則損失為 abs(相關係數)。
    - 若相關係數 >= 0，則損失為 1 - 相關係數。

    這個實作計算每個樣本的 Pearson 相關係數，然後取平均作為最終損失。
    """
    def __init__(self):
        """
        初始化 NP_Loss 類別。
        """
        super(NP_Loss, self).__init__()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        計算預測值和標籤之間的 NP 損失。

        Args:
            preds (torch.Tensor): 形狀為 (batch_size, num_elements) 的預測張量。
            labels (torch.Tensor): 形狀為 (batch_size, num_elements) 的標籤張量。

        Returns:
            torch.Tensor: 計算出的平均 NP 損失。
        """
        loss = 0.0
        batch_size = preds.shape[0]
        num_elements = preds.shape[1]

        for i in range(batch_size):
            # 從預測和標籤中提取單個樣本
            pred = preds[i]
            label = labels[i]

            # 計算 Pearson 相關係數所需的各項總和
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

def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = signal.shape[-1]
        pad_len = int(zero_pad / 2 * L)
        signal = F.pad(signal, (pad_len, pad_len), mode='constant', value=0)

    freqs = torch.fft.fftfreq(signal.shape[-1], 1 / Fs) * 60  # in bpm
    ps = torch.abs(torch.fft.fft(signal, dim=-1)) ** 2
    cutoff = len(freqs) // 2
    return freqs[:cutoff], ps[:, :cutoff]


def predict_heart_rate(signal, Fs, min_hr=40., max_hr=180.):
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

def kl_loss(inputs, labels):
    criterion = torch.nn.KLDivLoss(reduction='none')
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum()
    return loss

def butter_bandpass(sig_list, lowcut, highcut, fs, order=2):
    """PyTorch version of butterworth bandpass filter using SciPy backend"""
    device = sig_list.device
    dtype = sig_list.dtype

    # Convert to numpy
    sig_numpy = sig_list.detach().cpu().numpy()

    # Process each signal
    y_list = []
    for sig in sig_numpy:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)

    # Convert back to torch tensor
    filtered = torch.tensor(np.array(y_list), dtype=dtype, device=device)
    return filtered


class HeartRateEvaluator:
    def __init__(self, Fs=30, min_hr=40., max_hr=180., lowcut=0.6, highcut=4.0, filter_order=2, gaussian_std=1):
        self.Fs = Fs
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.gaussian_std = gaussian_std

    def _predict_heart_rates(self, signals):
        """Helper function to predict heart rates for a batch of signals, including filtering."""
        filtered_signals = butter_bandpass(signals, self.lowcut, self.highcut, self.Fs, order=self.filter_order)
        heart_rates = predict_heart_rate(filtered_signals, self.Fs, self.min_hr, self.max_hr)
        return heart_rates

    def calculate_ce_and_ld(self, pred_signal, gt_hr):
        target_distribution = [normal_sampling(int(gt_hr), i, self.gaussian_std) for i in range(141)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).cuda()

        bpm_range = torch.arange(40, 181, dtype=torch.float).cuda()
        ca = complex_absolute(pred_signal, self.Fs, bpm_range)
        fre_distribution = F.softmax(ca.view(-1), dim=0)
        loss_distribution_kl = kl_loss(fre_distribution, target_distribution)

        return loss_distribution_kl, F.cross_entropy(ca, gt_hr.view((1)).type(torch.long))

    def __call__(self, predicted_signal, gt_signal):

        predicted_hr = self._predict_heart_rates(predicted_signal)
        gt_hr = self._predict_heart_rates(gt_signal)

        results = {}

        results['MAE'] = torch.mean(torch.abs(predicted_hr - gt_hr))
        results['MSE'] = torch.mean((predicted_hr - gt_hr) ** 2)
        results['RMSE'] = torch.sqrt(torch.mean((predicted_hr - gt_hr) ** 2))

        ## Cross Entropy Loss (CE) and Label Distribution Loss (LD)
        ce_loss = 0.
        ld_loss = 0.

        for i in range(predicted_hr.size(0)):
            ce, ld =self.calculate_ce_and_ld(predicted_signal[i], gt_hr[i] - self.min_hr)
            ce_loss += ce
            ld_loss += ld

        results['CE'] = ce_loss / predicted_hr.size(0)
        results['LD'] = ld_loss / predicted_hr.size(0)

        return results