import math
import torch
import numpy as np
import torch.nn as nn

from scipy import signal
from scipy import sparse
from einops import rearrange


class POS_WANG(nn.Module):

    def __init__(self,):
        super(POS_WANG, self).__init__()

    def _process_video(self, frames):
        """Calculates the average value of each frame."""
        RGB = []
        for frame in frames:
            summation = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(summation / (frame.shape[0] * frame.shape[1]))
        return np.asarray(RGB)

    # from unsupervised_methods/methods/utils.py
    def detrend(self, input_signal, lambda_value):
        signal_length = input_signal.shape[0]
        # observation matrix
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = sparse.spdiags(diags_data, diags_index,
                    (signal_length - 2), signal_length).toarray()
        filtered_signal = np.dot(
            (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
        return filtered_signal

    def forward(self, rgb):

        rgb = rgb.cpu().numpy()

        fs = 30
        WinSec = 1.6
        rgb = self._process_video(rgb)

        N = rgb.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fs)

        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(rgb[m:n, :], np.mean(rgb[m:n, :], axis=0))
                Cn = np.asmatrix(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])

        BVP = H
        BVP = self.detrend(np.asmatrix(BVP).H, 100)
        BVP = np.asarray(np.transpose(BVP))[0]
        b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
        BVP = signal.filtfilt(b, a, BVP.astype(np.double))
        return BVP