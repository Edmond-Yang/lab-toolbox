import os
from .transform import *


def apply_augmentation(rgb, nir, idx, length, num_augments=1):

    video_data = dict()
    start, end = idx

    if rgb is not None:
        video_data['rgb'] = rgb[start:end]
    if nir is not None:
        video_data['nir'] = nir[start:end]

    if os.environ.get("MODE", "").lower() == "test":
        return video_data

    return augment(rgb, nir, video_data, idx, length, num_augments)
