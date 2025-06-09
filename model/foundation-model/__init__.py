import torch
import torch.nn.functional as F

from .vit import *
from .fsfm import *
from ..templates import *
from einops import rearrange


class Model(ModelTemplate):

    def __init__(self, model_cfg: ModelConfig = ModelConfig()):
        super().__init__(model_cfg)

        self.t = 5
        self.teacher = TeacherModel(model_cfg)
        self.student = StudentModel(model_cfg)

        # NOTE: Loss

        self.total_loss = {"total": 0., 'Spatial Loss': 0. , 'Temporal Loss': 0.}

        self.optimizer = torch.optim.AdamW(self.student.model.parameters(), lr=1e-4, weight_decay=0.00005)


    def train(self, data):

        # Load data
        rgb = data['rgb'].to(self.device)
        augment_rgb = data['augment_rgb'].to(self.device)

        self.B = rgb.shape[0]
        self.T = rgb.shape[2]

        self.N = (rgb.shape[3] // 16) ** 2
        rgb = rearrange(rgb, 'b c t h w -> (b t) c h w')

        Logger.detail(f"RGB Shape: {rgb.shape}, Augmented RGB Shape: {augment_rgb.shape}")

        t_pred = self.teacher(rgb)
        s_pred = self.student(augment_rgb, self.epoch, self.batch_idx)

        Logger.detail(f"Teacher Feature Shape: {t_pred.shape}, Student Feature Shape: {s_pred.shape}")

        loss = self.loss(s_pred, t_pred)
        loss.backward()
        self.optimizer.step()


    def loss(self, preds, gts):

        gt_cls = gts[:, 0:1, :]
        gts = gts[:, 1:, :]

        gts = rearrange(gts, '(b t) n c -> b (n t) c', b=self.B, t=self.T)
        gt_cls = rearrange(gt_cls, '(b t) 1 c -> b t c', b=self.B, t=self.T)


        max_gts = torch.zeros((gts.shape[0], gts.shape[1] // 2 + 1, gts.shape[2])).to(self.device)

        # Average or max the feature dimension per two frames
        max_gts[:, 0, :] = torch.max(gt_cls, dim=1)[0]
        for i in range(gts.shape[1] // 2):
            max_gts[:, i+1, :] = torch.max(gts[:, i * 2:(i + 1) * 2, :], dim=1)[0]

        if max_gts.shape != preds.shape:
            print(f"Max GTS Shape: {max_gts.shape}, Preds Shape: {preds.shape}")
            raise Exception("Shape mismatch")

        # Spatial loss
        s_loss = F.mse_loss(preds, max_gts)

        # Temporal loss
        preds = preds[:, 1:, :]
        max_gts = max_gts[:, 1:, :]
        preds = rearrange(preds, 'b (n t) c -> b n t c', n=self.N)
        gts = rearrange(max_gts, 'b (n t) c -> b n t c', n=self.N)

        diff_preds = preds[:, :, self.t:, :] - preds[:, :, :-self.t, :]
        diff_gts = gts[:, :, self.t:, :] - gts[:, :, :-self.t, :]

        t_loss = F.mse_loss(diff_preds, diff_gts)

        # Total loss
        total_loss = 0.2 * s_loss + t_loss

        self.total_loss['Spatial Loss'] += s_loss.item()
        self.total_loss['Temporal Loss'] += t_loss.item()
        self.total_loss['total'] += total_loss.item()

        Logger.detail(f"[Epoch {self.epoch} Batch {self.batch_idx}] - Spatial Loss: {s_loss:2.4f}, Temporal Loss: {t_loss:2.4f}, Total Loss: {total_loss:2.4f}")

        return total_loss

    def test(self, data):
        raise Exception("Student model does not have test function")

    def load_weights(self, name=None):
        self.student.load_weights(name)

    def save_weights(self, name=None):
        self.student.save_weights(name)


class TeacherModel(ModelTemplate):

    def __init__(self, model_cfg: ModelConfig = ModelConfig()):
        super().__init__(model_cfg)

        self.model = vit_base_patch16_128()

        model_path = pathManager.get_weight_path('fsfm', 'pretrain') / 'checkpoint.pth'
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)['model']
        state_dict = self.model.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint and checkpoint.shape != state_dict[k].shape:
                del checkpoint[k]

        self.interpolate_pos_embed(checkpoint)
        self.model.load_state_dict(checkpoint, strict=False)

        # requires no grad
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

    def interpolate_pos_embed(self, checkpoint_model):

        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.model.patch_embed.num_patches
            num_extra_tokens = self.model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

    def __call__(self, data):
        return self.train(data)


    def train(self, data):
        return self.model(data)

    def test(self, data):
        raise Exception("Teacher model does not have test function")

    def loss(self):
        raise Exception("Teacher model does not have loss function")

    def load_weights(self, name=None):
        pass

    def save_weights(self, name=None):
        pass



class StudentModel(ModelTemplate):
    def __init__(self, model_cfg: ModelConfig = ModelConfig()):
        super().__init__(model_cfg)
        self.model_name = 'FoundationModel1'
        self.model = video_vit_base_patch16_128().to(self.device)

    def __call__(self, data, epoch, batch_idx):
        self.epoch = epoch
        self.batch_idx = batch_idx
        return self.train(data)

    def train(self, data):
        return self.model(data)

    def test(self, data):
        return self.model(data)

    def loss(self):
        raise Exception("Student model does not have loss function")
