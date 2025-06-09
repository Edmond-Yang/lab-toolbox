import torch

from .loss import *
from .Physformer import *
from ..templates import *


class Model(ModelTemplate):

    def __init__(self, model_cfg: ModelConfig = ModelConfig()):
        super().__init__(model_cfg)

        self.model_name = 'physforemr_best_0.5'
        self.weight_dir = pathManager.get_weight_path(self.model_name, args.train)

        self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(args.length, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
            num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, in_ch=3
        ).to(self.device)

        self.np = NegativePearsonLoss()
        self.ce = CrossEntropyLoss()
        self.ld = LabelDistributionLoss()
        self.mae = MeanAbsoluteError()

        self.total_loss = {
            "total": 0.0,
            "Negative Pearson Loss": 0.0,
            "Cross Entropy Loss": 0.0,
            "Label Distribution Loss": 0.0,
            "Mean Absolute Error": 0.0,
            "Bandwidth Loss": 0.0,
            "Sparsity Loss": 0.0,
            "Variance Loss": 0.0
        }

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.00005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)


    def train(self, data):

        # Load data
        rgb = data['rgb'].to(self.device)
        gt = data['gt'].to(self.device)

        Logger.detail(f"RGB Shape: {rgb.shape}, GT Shape: {gt.shape}")

        preds, _, _, _ = self.model(rgb, 2.0)

        Logger.detail(f"Pred Shape: {preds.shape}")

        loss = self.loss(preds, gt)
        loss.backward()
        self.optimizer.step()

    def preprocess(self):

        # learning rate
        self.a = 0.4

        if self.epoch >= 25:
            self.b = 5.0
        else:
            self.b = 1.0 * math.pow(5.0, self.epoch / 25.0)

        # announce total loss from an epoch and save weights
        if self.epoch != 0 and self.batch_idx == 0:
            self.scheduler.step()


    def loss(self, preds, gts):

        # Calculate NP Loss
        np_loss = self.np(preds, gts)
        ce_loss = self.ce(preds, gts)
        ld_loss = self.ld(preds, gts)
        mae_loss = self.mae(preds, gts)

        self.total_loss["Negative Pearson Loss"] += np_loss.item()
        self.total_loss["Cross Entropy Loss"] += ce_loss.item()
        self.total_loss["Label Distribution Loss"] += ld_loss.item()
        self.total_loss["Mean Absolute Error"] += mae_loss.item()

        # Calculate bandwidth, sparsity, and variance loss
        B = preds.shape[0]
        speed = torch.stack([torch.tensor(1.0)] * B)

        freq, psd = torch_power_spectral_density(preds, fps=30, low_hz=0.66666667, high_hz=3.0, normalize=False,
                                                 bandpass=False)

        low_hz = 0.66666667
        high_hz = 3.0
        criterions = select_loss('bsv')
        bandwidth_loss = criterions['bandwidth'](freq, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                 device=self.device)
        sparsity_loss = criterions['sparsity'](freq, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                device=self.device)
        variance_loss = criterions['variance'](freq, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                device=self.device)
        self.total_loss["Bandwidth Loss"] += bandwidth_loss.item()
        self.total_loss["Sparsity Loss"] += sparsity_loss.item()
        self.total_loss["Variance Loss"] += variance_loss.item()

        # Calculate total loss
        total_loss = self.a * (np_loss) + self.b * (ce_loss + ld_loss) + bandwidth_loss + sparsity_loss + variance_loss
        self.total_loss["total"] += total_loss

        Logger.detail(
            f"Epoch: {self.epoch} - Batch: {self.batch_idx}\n[Losses] NP Loss: {np_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, LD Loss: {ld_loss.item():.4f}, MAE Loss: {mae_loss.item(): .4f}",
        )
        Logger.detail(
            f"Epoch: {self.epoch} - Batch: {self.batch_idx}\n[Losses] Bandwidth Loss: {bandwidth_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}, Variance Loss: {variance_loss.item(): .4f}",
        )
        Logger.detail(f"Epoch: {self.epoch} - Batch: {self.batch_idx}\n[Total Loss]: {total_loss.item():.4f}")

        return total_loss

    def test(self, data):
        rgb = data['rgb'].to(self.device)
        preds, _, _, _ = self.model(rgb, 2.0)
        Logger.detail(f'preds: {preds.shape}')
        return preds
