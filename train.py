import os
import torch
import numpy as np

os.environ["MODE"] = 'train'

from utils import *
from model import *
from einops import rearrange


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Logger.detail(f'Train {args.model} on {args.train} dataset')
    Logger.detail(f'Using device: {device}')

    model_cfg = ModelConfig(mode=ModelMode.TRAIN, device=device, n_batch=len(train_loader))
    model = Model(model_cfg)

    for e in range(args.epoch):
        for i, data in enumerate(train_loader):
            model(
                ForwardConfig(
                    epoch=e+1,
                    batch_idx=i+1,
                    data=data
                )
            )

    Logger.info(f'Finished Training {model_name} on {args.train} dataset!')


if __name__ == '__main__':

    Logger.set_prefix(f'Train_{args.train}')

    dataset_cfg = DatasetConfig(
        size=args.size,
        length=args.length,
        sample=None,
        preload=args.preload,
        fixed_sample=False,
        extra_transforms=None,
    )
    loader_cfg = LoaderConfig(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    train_loader = get_loader_from_protocol(
        protocol=args.train,
        config=dataset_cfg,
        loader_config=loader_cfg
    )

    train()
