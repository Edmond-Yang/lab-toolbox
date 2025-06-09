import os
import cv2
import math
import torch
import random
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from utils.logger import *
from scipy.stats import norm

def augment(rgb, nir, video_data, idx, length, num_augments):

    augment_video_list = []
    augment_speed_list = []

    for i in range(num_augments):
        video = rgb.detach().numpy()

        video, speed = augment_speed(video, idx, length)
        video = augment_horizontal_flip(video)
        video = augment_time_reversal(video)
        video = augment_illumination_noise(video)
        video = augment_gaussian_noise(video)
        video = random_resized_crop(video)

        video = np.clip(video, 0, 255)
        video = torch.from_numpy(video)

        video = video.permute(1, 0, 2, 3)  # [T, C, H, W]

        augment_video_list.append(video.clone())
        augment_speed_list.append(speed)

    video_data['augment_rgb'] = augment_video_list if num_augments > 1 else augment_video_list[0]
    video_data['speed'] = augment_speed_list if num_augments > 1 else augment_speed_list[0]

    return video_data

def resample_clip(video, length):
    video = np.transpose(video, (3,0,1,2)).astype(float)
    video = interpolate_clip(video, length)
    video = np.transpose(video, (1,2,3,0))
    return video


def arrange_channels(imgs, channels):
    d = {'b':0, 'g':1, 'r':2, 'n':3}
    channel_order = [d[c] for c in channels]
    imgs = imgs[:,:,:,channel_order]
    return imgs


def prepare_clip(clip, channels):
    clip = arrange_channels(clip, channels)
    clip = np.transpose(clip, (3, 0, 1, 2)) # [C,T,H,W]
    clip = clip.astype(np.float64)
    return clip

def augment_speed(video, idx, length, channels='rgb', speed_slow=0.6, speed_fast=1.4):
    ''' Interpolates clip to frames_per_clip length given slicing indices, which
        can be floats.
    '''
    vid_len = len(video)
    min_idx = idx[0]
    speed_fast = min(speed_fast, math.floor((vid_len - min_idx) / length * 10) / 10)

    speed = np.random.uniform(speed_slow, speed_fast)
    max_idx = np.round(length * speed + min_idx).astype(int)
    if max_idx > vid_len:
        print(f"max_idx: {max_idx}, vid_len: {vid_len}, speed: {speed}")
        raise Exception("max_idx > vid_len")

    clip = video[min_idx:max_idx]
    clip = prepare_clip(clip, channels)
    interpolated_clip = interpolate_clip(clip, length)
    return interpolated_clip, speed


def interpolate_clip(clip, length):
    '''
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [C,T,H,W]
    '''
    clip = torch.from_numpy(clip[np.newaxis])
    clip = F.interpolate(clip, (length, 64, 64), mode='trilinear', align_corners=True)
    return clip[0].numpy()


def resize_clip(clip, length):
    '''
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [C,T,H,W]
    '''
    T = clip.shape[1]
    clip = torch.from_numpy(np.ascontiguousarray(clip[np.newaxis]))
    clip = F.interpolate(clip, (T, length, length), mode='trilinear', align_corners=False)
    return clip[0].numpy()


def random_resized_crop(clip, crop_scale_lims=[0.5, 1]):
    ''' Randomly crop a subregion of the video and resize it back to original size.
    Arguments:
        clip (np.array): expects [C,T,H,W]
    Returns:
        clip (np.array): same dimensions as input
    '''
    C,T,H,W = clip.shape
    crop_scale = np.random.uniform(crop_scale_lims[0], crop_scale_lims[1])
    crop_length = np.round(crop_scale * H).astype(int)
    crop_start_lim = H - (crop_length)
    x1 = np.random.randint(0, crop_start_lim+1)
    y1 = x1
    x2 = x1 + crop_length
    y2 = y1 + crop_length
    cropped_clip = clip[:,:,y1:y2,x1:x2]
    resized_clip = resize_clip(cropped_clip, H)
    return resized_clip


def augment_gaussian_noise(clip):
    clip = clip + np.random.normal(0, 2, clip.shape)
    return clip


def augment_illumination_noise(clip):
    clip = clip + np.random.normal(0, 10)
    return clip


def augment_time_reversal(clip):
    if np.random.rand() > 0.5:
        clip = np.flip(clip, 1)
    return clip


def augment_horizontal_flip(clip):
    if np.random.rand() > 0.5:
        clip = np.flip(clip, 3)
    return clip