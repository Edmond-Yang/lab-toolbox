# import module
import math
from logging import Logger

import h5py
import json
import torch
import numpy as np
import concurrent.futures

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.path import *
from utils.logger import *
from .augmentation import *

# DataType
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Optional


# Simple config class for dataset and loader
@dataclass
class DatasetConfig:
    """Dataset Parameters"""
    size: int = 128
    length: int = 300
    sample: Optional[int] = None

    preload: bool = False
    fixed_sample: bool = False
    extra_transforms: Optional[List] = None


@dataclass
class LoaderConfig:
    """DataLoader Parameters"""
    batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 16
    pin_memory: bool = False


@dataclass
class SampleInfo:
    video: Optional[Tuple[str]] = None
    segment: int = 0
    segment_start: int = 0
    segment_end: int = 0


# converts the dataset images and corresponding gt to hdf5
class HDF5Saver:

    def __init__(self):

        # default values
        self._size = 128
        self._transforms = transforms.Compose([
            transforms.Resize((self._size, self._size)),
        ])

    def create_hdf5_from_config(self, videos: List[Tuple[str]]) -> Dict[Tuple[str], Path]:

        videos_path = dict()
        datasets = set([vid[0] for vid in videos])

        # Create preload directory for each dataset
        for dataset in datasets:
            pathManager.make_preload_directory(dataset)

        num_workers = 6
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_vid = {
                executor.submit(self._process_video, vid): vid
                for vid in videos
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_vid), desc="Creating HDF5 files", unit="video",
                               total=len(videos)):
                vid, file_path = future.result()
                videos_path[vid] = file_path

        return videos_path

    def _process_video(self, video: Tuple[str]) -> Tuple[Tuple[str], Path]:

        dataset = video[0]
        subfolder = video[1:]
        path = pathManager.get_preload_path(dataset, subfolder)

        if self._check_hdf5_exists(dataset, subfolder):
            return video, path

        rgb_path = pathManager.get_preprocessed_video_path(dataset, subfolder, "RGB")
        nir_path = pathManager.get_preprocessed_video_path(dataset, subfolder, "NIR")
        gt_path = pathManager.get_preprocessed_gt_path(dataset, subfolder)

        self._create_hdf5(dataset, subfolder,
                          {
                              'rgb': rgb_path if rgb_path.exists() else None,
                              'nir': nir_path if nir_path.exists() else None,
                              'gt': gt_path if gt_path.exists() else None
                          })

        return video, path

    def _check_hdf5_exists(self, dataset: str, subfolder: Tuple[str]) -> bool:
        hdf5_file_path = pathManager.get_preload_path(dataset, subfolder)
        if hdf5_file_path.exists():
            try:
                with h5py.File(hdf5_file_path, 'r') as h5f:
                    if "rgb" in h5f or "nir" in h5f or "gt" in h5f:
                        return True
                    else:
                        return False
            except Exception as e:
                return False
        else:
            return False

    def _create_hdf5(self, dataset: str, subfolder: Tuple[str], paths: Dict[str, Optional[Path]]) -> None:

        def __get_data(path: Path, length: int) -> Tuple[Optional[np.array], Optional[int]]:
            if path is None:
                return None, length
            array = self._get_videos_from_dirs(path)
            if array is not None:
                length = min(length, array.shape[0])
            return array, length

        try:

            length = math.inf
            hdf5_file_path = str(pathManager.get_preload_path(dataset, subfolder))

            rgb_array, length = __get_data(paths['rgb'], length)
            nir_array, length = __get_data(paths['nir'], length)

            if paths['gt'] is not None:
                gt_path = paths['gt']
                with open(gt_path, 'r') as f:
                    gt_line = f.readline().strip().split()
                    gt_array = np.array([float(x) for x in gt_line][30:], dtype=np.float64)
                if gt_array is not None:
                    length = min(length, gt_array.shape[0])
            else:
                gt_array = None

            with h5py.File(hdf5_file_path, 'w') as h5f:
                if rgb_array is not None:
                    h5f.create_dataset("rgb", data=rgb_array[:length], compression="gzip")
                if nir_array is not None:
                    h5f.create_dataset("nir", data=nir_array[:length], compression="gzip")
                if gt_array is not None:
                    h5f.create_dataset("gt", data=gt_array[:length], compression="gzip")


        except Exception as e:
            Logger.error(f"Error in creating HDF5 file for {dataset}/{subfolder[0]}: {e}")
            raise e

    def _get_videos_from_dirs(self, dir: Path) -> np.array:

        if not dir.exists():
            raise FileNotFoundError(f"Directory {dir} does not exist.")
        if not dir.is_dir():
            raise NotADirectoryError(f"{dir} is not a directory.")

        # Get all image files in the directory
        images_paths = list(sorted(dir.glob("*.png")))[30:]
        if len(images_paths) == 0:
            Logger.warning(f"No images found in {dir}.")
            return None
        else:
            return np.stack([np.array(self._transforms(Image.open(image).convert("RGB"))) for image in images_paths],
                            axis=0)


# video dataset class
class VideoDataset(Dataset):

    def __init__(self, videos: List[Tuple[str]], config: DatasetConfig):

        self.config = config

        # transforms
        if self.config.extra_transforms is None:
            self.config.extra_transforms = list()

        # HDF5 file path
        # { 'video_id': Path }
        hdf5_handler = HDF5Saver()
        self.video_path = hdf5_handler.create_hdf5_from_config(videos)

        # [ video_id, segment, start, end ]
        self.index_map: List[SampleInfo] = list()
        self._generate_index(videos)

        # samples of each video
        self.total_videos_number: int = len(self.index_map)

        # Preload data
        self.preload_data: Optional[Dict[Tuple[str], Dict[str, torch.Tensor]]] = self._preload_data()

    def _generate_index_for_video(self, video_id: Tuple[str], vid_length: Optional[int]) -> Tuple[
        Tuple[str], List[SampleInfo], bool, int]:

        if_write = vid_length is None

        if vid_length is None:
            min_length = math.inf
            dataset, subfolder = video_id[0], video_id[1:]
            hdf5_file_path = str(pathManager.get_preload_path(dataset, subfolder))

            with h5py.File(hdf5_file_path, 'r') as h5f:
                if "rgb" in h5f:
                    min_length = h5f["rgb"].shape[0]
                elif "nir" in h5f:
                    min_length = h5f["nir"].shape[0]
                elif "gt" in h5f:
                    min_length = h5f["gt"].shape[0]

            vid_length = min_length

        dataset, subfolder = video_id[0], video_id[1:]
        hdf5_file_path = str(pathManager.get_preload_path(dataset, subfolder))

        with h5py.File(hdf5_file_path, 'r') as h5f:
            if "gt" in h5f:
                gt = h5f["gt"][:]
            else:
                gt = None

        n_sample = int(vid_length // self.config.length * .75)
        n_sample = min(n_sample, self.config.sample) if self.config.sample is not None else n_sample

        if n_sample <= 0:
            if vid_length < self.config.length:
                raise ValueError(f"Sample is less than or equal to 0: {n_sample} in video {video_id}")
            else:
                n_sample = 1

        drop_list = list()
        index_map = list()
        sample_length = vid_length // n_sample
        for idx in range(n_sample):
            start_idx = idx * sample_length
            end_idx = start_idx + sample_length - self.config.length - 1

            if gt is not None and self._filter_video_via_signal(gt[start_idx:end_idx]):
                drop_list.append((video_id, idx, start_idx, end_idx))
                continue

            data = SampleInfo(video_id, idx, start_idx, end_idx)
            index_map.append(data)

        return video_id, index_map, if_write, vid_length, drop_list

    def _generate_index(self, videos: List[Tuple[str]]) -> List[Dict[str, Union[Tuple[str], int]]]:

        sample = list()
        length_path = pathManager.get_length_path()

        if_write = False
        videos_length = {}

        if length_path.exists():
            with open(length_path, 'r') as f:
                videos_length = json.load(f)

        num_workers = 16
        drop_list = list()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {
                executor.submit(self._generate_index_for_video, video, videos_length.get('_'.join(video))): video
                for video in videos
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_video), desc="Generating Index", unit="video",
                               total=len(videos)):
                video, index_map, vid_if_write, vid_length, drop = future.result()
                self.index_map.extend(index_map)
                drop_list.extend(drop)
                if_write = if_write or vid_if_write
                if vid_if_write:
                    videos_length['_'.join(video)] = vid_length

        Logger.detail(f"Total videos: {len(videos)}")
        Logger.detail(f"Total samples: {len(self.index_map)}")
        Logger.detail(f"Dropped samples: {len(drop_list)}")

        if if_write:
            with open(length_path, 'w') as f:
                json.dump(videos_length, f)

        return sample

    def _filter_video_via_signal(self, signal) -> bool:

        # TODO: if signal is flat return true, else return false
        diff = np.diff(signal)

        for i in range(0, diff.shape[0]-3):
            if np.abs(diff[i]) < 1e-3 and np.abs(diff[i+1]) < 1e-3 and np.abs(diff[i+2]) < 1e-3:
                return True

        return False

    def _preload_video(self, video: Tuple[str], path: Path) -> torch.Tensor:

        data = dict()
        with h5py.File(path, 'r') as h5f:
            if "rgb" in h5f:
                rgb = torch.from_numpy(h5f["rgb"][:])  # (T,H,W,3) uint8
                rgb = rgb.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W) uint8
                data["rgb"] = rgb
            if "nir" in h5f:
                nir = torch.from_numpy(h5f["nir"][:])
                nir = nir.permute(0, 3, 1, 2).contiguous()
                data["nir"] = nir
            if "gt" in h5f:
                data["gt"] = torch.from_numpy(h5f["gt"][:]).float()

        return video, data

    def _preload_data(self) -> Optional[Dict[Tuple[str], Dict[str, torch.Tensor]]]:
        if self.config.preload:
            data = dict()
            num_workers = 4
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_preload = {
                    executor.submit(self._preload_video, video, path): video
                    for video, path in self.video_path.items()
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_preload), desc="Preload Videos",
                                   unit="video", total=len(self.video_path)):
                    video_name, video_data = future.result()
                    data[video_name] = video_data

            return data
        else:
            return None

    def _resize(self, video: torch.Tensor) -> torch.Tensor:
        if self.config.size != 128:
            video = F.interpolate(video, size=self.config.size,
                                  mode="bilinear", align_corners=False)
        return video

    def __len__(self):
        return self.total_videos_number

    def __getitem__(self, idx):

        def __process_video(segment):
            segment = segment.float().div_(255.)
            segment = self._resize(segment)
            return rearrange(segment, "t c h w -> c t h w")

        try:
            info = self.index_map[idx]

            video = info.video
            path = self.video_path[video]
            start = info.segment_start
            end = info.segment_end

            if self.config.preload:
                clip = self.preload_data[video]
            else:
                _, clip = self._preload_video(video, path)

            rgb = clip.get("rgb")
            nir = clip.get("nir")
            gt = clip.get("gt")

            start_idx = start if self.config.fixed_sample else np.random.randint(start, end)
            end_idx = start_idx + self.config.length

            # augmentation
            video_data = apply_augmentation(rgb, nir, (start_idx, end_idx), self.config.length)

            if self.config.extra_transforms is not None:
                for transform in self.config.extra_transforms:
                    video_data = transform(video_data)

            # transform to float and resize
            for k, v in video_data.items():
                if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                    video_data[k] = __process_video(v)

                if isinstance(v, list) and len(v[0].shape) == 4:
                    for i in range(len(v)):
                        v[i] = __process_video(v[i])

            if gt is not None:
                video_data["gt"] = gt[start_idx:end_idx]

            video_data["id"] = '_'.join(video) + f',{info.segment:03d}'
            video_data["idx"] = f'{start_idx:04d},{end_idx:04d}'

            return video_data
        except Exception as e:
            Logger.error(f"Error in __getitem__ at index {idx}: {e}")
            raise e


def get_loader_from_protocol(protocol: str, config: Optional[DatasetConfig] = None,
                             loader_config: Optional[LoaderConfig] = None, ) -> DataLoader:
    # Support for multiple protocol files

    protocols = protocol.split(',')
    Logger.info(f"Protocol: " + ', '.join(protocols))

    videos = []

    for p in protocols:

        p = p.strip()
        path, if_exist = pathManager.get_protocol_path(p)

        if not if_exist:
            Logger.error(f"Protocol file not found: {p}")
            raise FileNotFoundError(f"Protocol file not found: {p}")

        with open(path, "r") as f:
            lines = f.readlines()

        videos.extend([
            tuple(parts)
            for line in lines if (parts := line.strip().split(","))
        ])

    dataset = VideoDataset(videos=videos, config=config)
    dataloader = DataLoader(dataset, batch_size=loader_config.batch_size, shuffle=loader_config.shuffle,
                            num_workers=loader_config.num_workers)

    return dataloader


if __name__ == '__main__':

    # Example usage
    protocol = "Tokyo_train"
    # protocol = args.train
    # protocol = args.test

    dataset_config = DatasetConfig(
        size=128,
        length=300,
        sample=None,
        preload=True,
        fixed_sample=False,
        extra_transforms=None,
    )

    loader_config = LoaderConfig(
        batch_size=4,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    dataloader = get_loader_from_protocol(protocol, dataset_config, loader_config)

    for i, (rgb, nir, gt, vid) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"ID: {vid}")
        print(f"RGB shape: {rgb.shape}")
        print(f"NIR shape: {nir.shape}")
        print(f"GT shape: {gt.shape}")