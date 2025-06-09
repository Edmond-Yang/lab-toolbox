import torch

from .loss import *
from .physnet import *
from ..templates import *


class Model(ModelTemplate):

    def __init__(self, model_cfg: ModelConfig = ModelConfig()):
        super().__init__(model_cfg)

        channel = 3
        dropout = 0.5

        self.model = PhysNet(
            input_channels=channel, drop_p=dropout
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.total_loss = {'total': 0.0, 'Bandwidth Loss': 0.0, 'Sparsity Loss': 0.0, 'Variance Loss': 0.0}

    def train(self, data):

        self.optimizer.zero_grad()

        rgb = data['augment_rgb'].to(self.device)
        speed = data['speed']

        pred = self.model(rgb)
        pred = self.add_noise_to_constants(pred)
        freqs, psd = torch_power_spectral_density(pred, fps=30, low_hz=0.66666667, high_hz=3.0, normalize=False, bandpass=False)
        loss = self.loss(freqs, psd, speed)

        loss["total"].backward()
        self.optimizer.step()


    def add_noise_to_constants(self, predictions):
        B, T = predictions.shape
        for b in range(B):
            if torch.allclose(predictions[b][0], predictions[b]):  # constant volume
                predictions[b] = torch.rand(T) - 0.5
        return predictions

    def test(self, data):
        rgb = data['rgb'].to(self.device)
        return self.model(rgb)


    def loss(self, freqs, psd, speed):

        criterions_str = 'bsv'
        criterions = select_loss(criterions_str)
        low_hz = 0.66666667
        high_hz = 3.0

        total_loss = 0.0
        losses_dict = {}
        if 'b' in criterions_str:
            bandwidth_loss = criterions['bandwidth'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                     device=self.device)
            total_loss += (1.0 * bandwidth_loss)
            losses_dict['bandwidth'] = bandwidth_loss
            self.total_loss['Bandwidth Loss'] += bandwidth_loss.item()

        if 's' in criterions_str:
            sparsity_loss = criterions['sparsity'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                   device=self.device)
            total_loss += (1.0 * sparsity_loss)
            losses_dict['sparsity'] = sparsity_loss
            self.total_loss['Sparsity Loss'] += sparsity_loss.item()

        if 'v' in criterions_str:
            variance_loss = criterions['variance'](freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                   device=self.device)
            total_loss += (1.0 * variance_loss)
            losses_dict['variance'] = variance_loss
            self.total_loss['Variance Loss'] += variance_loss.item()

        losses_dict['total'] = total_loss
        self.total_loss['total'] += total_loss.item()
        Logger.info(f"Epoch {self.epoch} - Batch {self.batch_idx}\n[Losses] "
                    f"bandwidth: {losses_dict['bandwidth']:.4f}, "
                    f"sparsity: {losses_dict['sparsity']:.4f}, "
                    f"variance: {losses_dict['variance']:.4f}, "
                    f"total: {losses_dict['total']:.4f}"
        )
        return losses_dict



