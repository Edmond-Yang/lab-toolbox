import os
import torch
import numpy as np

os.environ["MODE"] = 'test'

from utils import *
from model import *
from einops import rearrange


def test():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Logger.detail(f'Test {args.model} on {args.test} dataset (Train on {args.train})')
    Logger.detail(f'Using device: {device}')

    model_cfg = ModelConfig(mode=ModelMode.TEST, device=device, n_batch=len(test_loader))
    model = Model(model_cfg)
    metrics = HeartRateEvaluator()

    with torch.no_grad():
        for e in range(args.epoch):

            gt_signals = []
            pred_signals = []

            for i, data in enumerate(test_loader):

                preds = model(
                    ForwardConfig(
                        epoch=e+1,
                        batch_idx=i,
                        data=data
                    )
                )

                gt_signals.append(data['gt'].to(device))
                pred_signals.append(preds)

            gt_signals = torch.cat(gt_signals, dim=0)
            pred_signals = torch.cat(pred_signals, dim=0)

            ## Metrics
            pred_hr, gt_hr, results = metrics(pred_signals, gt_signals)

            Logger.detail(f'pred_hr: {pred_hr}')
            Logger.detail(f'gt_hr: {gt_hr}')

            Logger.info(f'Epoch: {e+1:3d}, MAE: {results["MAE"]:2.4f}, RMSE: {results["RMSE"]:2.4f}, R: {results["R"]:2.4f}')

    Logger.info(f"Completed evaluation of {model_name} (trained on {args.train}) on the {args.test} dataset.")


if __name__ == '__main__':
    Logger.set_prefix(f'Test_{args.test}_Train_{args.train}')

    dataset_cfg = DatasetConfig(
        size=args.size,
        length=args.length,
        sample=None,
        preload=args.preload,
        fixed_sample=True,
        extra_transforms=None,
    )
    loader_cfg = LoaderConfig(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    test_loader = get_loader_from_protocol(
        protocol=args.test,
        config=dataset_cfg,
        loader_config=loader_cfg
    )

    test()
