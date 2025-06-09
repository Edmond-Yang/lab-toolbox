import os
import argparse

args = argparse.ArgumentParser(description="Preprocess videos and images for face detection and tracking.")
args.add_argument('--raw_dir', type=str, default='/mnt/faceDatasets/original_sequences/actors/',
                    help='Directory containing the raw videos.')
args.add_argument('--preprocessed_dir', type=str, default='/mnt/faceDatasets/',)
args.add_argument('--video_dir', type=str, default='FaceForensics_Actor',
                    help='Directory to save the preprocessed videos.')
args.add_argument('--method', type=str, default='retinaface',)
args = args.parse_args()
video_dir = args.video_dir

os.environ["RAW_DIR"] = args.raw_dir
os.environ["PREPROCESSED_DIR"] = args.preprocessed_dir
os.environ["METHOD"] = args.method


# import necessary libraries
import cv2

from path import *
from tqdm import tqdm
from preprocess import *
from logger import Logger
from create_video import *


def crop_video():

    Logger.detail(f"Preprocessing videos in {video_dir}...")

    # Get the videos from the first folder
    folder = pathManager.get_all_raw_path()[0]
    videos = list(folder.glob('*.mp4'))
    length = len(videos)

    # Create the output directory
    out_dir = pathManager.get_preprocessed_video_path(video_dir, [], 'RGB')
    pathManager.make_directory(out_dir)

    # model
    model = FaceCropper(scale_factor=1.0)

    for video_path in tqdm(videos, total=length, desc="Crop Images From Videos"):

        model.reset()
        v_dir = out_dir / str(video_path.stem)
        pathManager.make_directory(v_dir)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片檔：{video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cropped_images = model(frame, re_detect=True)

            if cropped_images is not None:

                for img in cropped_images:
                    filename = v_dir / f"{frame_idx:05d}.png"
                    cv2.imwrite(str(filename), img)
                    frame_idx += 1

        cap.release()
        create_video_from_images(v_dir)

        # Logger.detail(f"Cropped {frame_idx} images from {video_path}")


if __name__ == '__main__':
    Logger.set_prefix('PreProcessing')
    Logger.detail(f"Preprocessor Version: {PREPROCESSOR_VERSION}")
    Logger.detail(f"Preprocessor Published Date: {PREPROCESSOR_PUBLISHED_DATE}")
    crop_video()


