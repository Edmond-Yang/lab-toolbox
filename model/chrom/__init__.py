import torch

from .chrom import *
from ..templates import *

class Model(ModelTemplate):

    def __init__(self, model_cfg: ModelConfig = ModelConfig()):

        super().__init__(model_cfg)

        self.device = torch.device('cpu')
        self.model = CHROM_DEHAAN()

        if self.mode == "train":
            Logger.warning("This method requires no training.")
            exit()

    def train(self, data):
        raise NotImplementedError("Training is not implemented for this model.")

    def test(self, data):
        rgb = data['rgb'].squeeze().to(self.device).permute(1,2,3,0)
        Logger.detail(f"RGB Shape: {rgb.shape}")
        signals = self.model(rgb)
        signals = [torch.from_numpy(signals.copy())]
        signals = torch.stack(signals)
        Logger.detail(f"Signals shape: {signals.shape}")
        return signals

    def loss(self, preds, gts):
        raise NotImplementedError("Loss is not implemented for this model.")