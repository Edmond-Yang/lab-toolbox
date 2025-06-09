import os
import torch

from utils import *
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ModelMode(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

@dataclass
class ModelConfig:
    mode: ModelMode = ModelMode.TRAIN
    device: torch.device = torch.device('cuda')
    n_batch: int = 12

@dataclass
class ForwardConfig:
    epoch: int = 0
    batch_idx: int = 0
    data: dict = None


class ModelTemplate(ABC):

    def __init__(self, model_cfg: ModelConfig = ModelConfig()):

        self.model = None
        self.model_name = os.environ.get("MODEL_NAME", "default_model").lower()

        self.mode = model_cfg.mode if isinstance(model_cfg.mode, ModelMode) else ModelMode(model_cfg.mode)
        self.device = model_cfg.device
        self.n_batch = model_cfg.n_batch

        self.epoch = None
        self.batch_idx = None

        self.total_loss = {}
        self.weight_dir = pathManager.get_weight_path(args.model, args.train)

        if self.mode not in ModelMode:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.n_batch is None:
            raise ValueError("n_batch must be specified.")

    def postprocess(self):
        if self.batch_idx == self.n_batch:
            self.save_weights()
            epoch_msg = f"Epoch {self.epoch:03d}\n[Losses] Total loss: {self.total_loss['total'] / self.n_batch:.4f}"
            for key, value in self.total_loss.items():
                if key != 'total':
                    epoch_msg += f", {key}: {value / self.n_batch:.4f}"
                self.total_loss[key] = 0.
            print('-' * (len(epoch_msg) - 5))
            Logger.info(epoch_msg)
            print('-' * (len(epoch_msg) - 5))

    def load_weights(self, name=None):
        self.model.load_state_dict(torch.load(self.weight_dir / f'{self.epoch:04d}.pth' if name is None else  self.weight_dir / f"{name}.pth", map_location=self.device))

    def save_weights(self, name=None):
        torch.save(self.model.state_dict(), self.weight_dir / f'{self.epoch:04d}.pth' if name is None else self.weight_dir / f"{name}.pth")

    def __call__(self, data: ForwardConfig):
        self.epoch = data.epoch
        self.batch_idx = data.batch_idx
        if self.mode == ModelMode.TRAIN:
            self.preprocess()
            self.train(data.data)
            return self.postprocess()
        else:
            if self.model_name not in ['pos', 'chrom']:
                self.load_weights()
            return self.test(data.data)

    def preprocess(self):
        pass

    @abstractmethod
    def loss(self, preds, gts):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def test(self, data):
        pass