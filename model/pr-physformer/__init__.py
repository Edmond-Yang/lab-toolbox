import torch
import torch.functional as F

from .loss import *
from .transform import *
from .Physformer import *
from ..templates import *


class Model(ModelTemplate):

    def __init__(self, model_cfg: ModelConfig = ModelConfig()):
        super().__init__(model_cfg)

        self.teacher = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(args.length, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
            num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, in_ch=3
        ).to(self.device)

        self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(args.length, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
            num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, in_ch=3
        ).to(self.device)

        if self.mode == ModelMode.TRAIN:
            teacher_path =  str(pathManager.get_weight_path('physformer', 'IT_all') / "0085.pth")
            self.teacher.load_state_dict(torch.load(teacher_path))

        self.total_loss = {
            "total": 0.0,
            "Feature Loss": 0.0,
            "Bandwidth Loss": 0.0,
            "Sparsity Loss": 0.0,
            "Variance Loss": 0.0
        }

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0.00005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,
            gamma=0.75
        )

    def test(self, data):

        rgb = data['rgb'].to(self.device)
        _, pred = self.model(rgb, 2.0)
        
        return pred


    def train(self, data):
        # Load data
        rgb = data['rgb'].to(self.device)
        augmented_rgb = data['augment_rgb'].to(self.device)

        Logger.detail(f"RGB Shape: {rgb.shape}, augmented RGB Shape: {augmented_rgb.shape}")

        t_feature, gt = self.teacher(rgb, 2.0)
        s_feature, pred = self.model(augmented_rgb, 2.0)

        Logger.detail(f"t_feature Shape: {t_feature.shape}, s_feature Shape: {s_feature.shape}")
        Logger.detail(f"GT shape: {gt.shape} Pred Shape: {pred.shape}")

        freq, psd = torch_power_spectral_density(pred, fps=30, low_hz=0.66666667, high_hz=3.0, normalize=False,
                                                  bandpass=False)
        loss = self.loss(t_feature, s_feature, data['speed'], freq, psd)
        loss.backward()
        self.optimizer.step()

    def postprocess(self):
        super().postprocess()

        if self.batch_idx == self.n_batch:
            self.scheduler.step()


    def loss(self, tf, sf, speed, freq, psd):

        # feature

        cos_sim = F.cosine_similarity(tf, sf, dim=-1)
        cosine_loss_matrix = 1.0 - cos_sim  # [B, N]
        cosine_loss = cosine_loss_matrix.mean()

        mse_loss = F.mse_loss(tf, sf, reduction='mean')  # scalar
        rmse_loss = torch.sqrt(mse_loss)

        feature_loss = 0.1 * (rmse_loss + 10 * cosine_loss)  # Adjusted weight for feature loss

        # feature_loss = torch.sqrt(F.mse_loss(tf, sf)) + (1 - F.cosine_similarity(tf, sf, dim=1)).mean()
        # feature_loss = feature_loss.mean()  # Usually redundant, but kept for consistency


        # calculate signal
        low_hz = 0.66666667
        high_hz = 3.0
        criterions_str = 'bsv'
        criterions = select_loss(criterions_str)

        ## bandwidth loss
        bandwidth_loss = criterions['bandwidth'](freq, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                 device=self.device)
        ## sparsity loss
        sparsity_loss = criterions['sparsity'](freq, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                    device=self.device)
        ## variance loss
        variance_loss = criterions['variance'](freq, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                    device=self.device)

        # calculate total loss
        total_loss = feature_loss + 1.0 * bandwidth_loss + 1.0 * sparsity_loss + 1.0 * variance_loss

        self.total_loss["total"] += total_loss.item()
        self.total_loss["Feature Loss"] += feature_loss.item()
        self.total_loss["Bandwidth Loss"] += bandwidth_loss.item()
        self.total_loss["Sparsity Loss"] += sparsity_loss.item()
        self.total_loss["Variance Loss"] += variance_loss.item()


        Logger.detail(f"[Losses] Feature Loss: {feature_loss.item():.4f}, Bandwidth Loss: {bandwidth_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}, Variance Loss: {variance_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

        return total_loss


