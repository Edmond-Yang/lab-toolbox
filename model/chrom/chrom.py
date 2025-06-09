import math
import torch
import numpy as np
import torch.nn as nn

from scipy import signal

class CHROM_DEHAAN(nn.Module):

    def __init__(self,):
        super(CHROM_DEHAAN, self).__init__()

    def process_video(self, frames):
        "Calculates the average value of each frame."
        RGB = []
        for frame in frames:
            sum = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(sum / (frame.shape[0] * frame.shape[1]))
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

        LPF = 0.7
        HPF = 2.5
        WinSec = 1.6

        FS = 30

        RGB = self.process_video(rgb)
        FN = RGB.shape[0]
        NyquistF = 1 / 2 * FS
        B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')

        WinL = math.ceil(WinSec * FS)
        if (WinL % 2):
            WinL = WinL + 1
        NWin = math.floor((FN - WinL // 2) / (WinL // 2))
        WinS = 0
        WinM = int(WinS + WinL // 2)
        WinE = WinS + WinL
        totallen = (WinL // 2) * (NWin + 1)
        S = np.zeros(totallen)

        for i in range(NWin):
            RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
            RGBNorm = np.zeros((WinE - WinS, 3))
            for temp in range(WinS, WinE):
                RGBNorm[temp - WinS] = np.true_divide(RGB[temp], RGBBase)
            Xs = np.squeeze(3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1])
            Ys = np.squeeze(1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2])
            Xf = signal.filtfilt(B, A, Xs, axis=0)
            Yf = signal.filtfilt(B, A, Ys)

            Alpha = np.std(Xf) / np.std(Yf)
            SWin = Xf - Alpha * Yf
            SWin = np.multiply(SWin, signal.windows.hann(WinL))

            temp = SWin[:int(WinL // 2)]
            S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL // 2)]
            S[WinM:WinE] = SWin[int(WinL // 2):]
            WinS = WinM
            WinM = WinS + WinL // 2
            WinE = WinS + WinL

        BVP = S
        return BVP