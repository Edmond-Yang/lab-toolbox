import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models import register_model
from timm.layers import to_2tuple, trunc_normal_


class TubeletEmbedding(nn.Module):

    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=768, num_frames=30, tubelet_size=2):

        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.num_tubelets = ((img_size // patch_size) ** 2) * (num_frames // tubelet_size)

        self.embed = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size, patch_size),
            stride=(self.tubelet_size, patch_size, patch_size)
        )

    def get_num_tubelets(self):
        return self.num_tubelets

    def forward(self, x):
        x = self.embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop_rate=0., proj_drop_rate=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        # qkv: 3 x B x num_head x N x head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # query, key, value
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = attn @ v

        # attn: B x N x (num_head x head_dim)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(attn)
        return x

class MLP(nn.Module):

    def __init__(self, embed_dim, mlp_ratio, activation_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.activation = activation_layer()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoVisionTransFormer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
                 num_frames=30, tubelet_size=2,):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.embed = TubeletEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size
        )
        self.num_tubelets = self.embed.get_num_tubelets()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tubelets + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):

        # tubelet embedding
        x = self.embed(x)

        # add cls token and positional embedding
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device)

        # attention and mlp
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


@register_model
def video_vit_base_patch16_128(**kwargs):
    return VideoVisionTransFormer(
        img_size=128,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_frames=30,
        tubelet_size=2
    )
