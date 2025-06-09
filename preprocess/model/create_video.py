import cv2
import os
from pathlib import Path
import re
from tqdm import tqdm


def natural_sort_key(s):
    """使用自然排序方式處理檔名"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def create_video_from_images(folder_path):
    """
    將資料夾中的PNG圖片序列轉換成AVI影片

    參數:
    folder_path: 存放圖片的資料夾路徑
    """
    folder = Path(folder_path) if isinstance(folder_path, str) else folder_path

    # 只取得PNG檔案並按照數字順序排序
    image_files = list(folder.glob('*.png'))
    image_files.sort(key=natural_sort_key)

    if not image_files:
        print(f"在 {folder_path} 中找不到PNG檔案!")
        return

    # 設定輸出影片檔名
    output_path = folder / f"movie.mov"

    # 讀取第一張圖片來獲取尺寸
    frame = cv2.imread(str(image_files[0]))
    height, width, layers = frame.shape

    # 設定影片編碼器和輸出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, 30, (width, height))

    # 利用tqdm顯示進度取代掉print

    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        video.write(frame)

    # 釋放資源
    video.release()


def process_all_subjects(base_path):
    """
    處理所有subject開頭的資料夾

    參數:
    base_path: 主資料夾路徑
    """
    base_folder = Path(base_path)

    # 尋找所有subject開頭的資料夾
    subject_folders = sorted(
        [f for f in base_folder.iterdir() if f.is_dir() and f.name.startswith('subject31')],
        key=natural_sort_key
    )

    if not subject_folders:
        print("找不到subject開頭的資料夾!")
        return

    print(f"找到 {len(subject_folders)} 個subject資料夾")

    # 處理每個資料夾
    for folder in subject_folders:
        create_video_from_images(folder)


if __name__ == "__main__":
    # 設定主資料夾路徑（包含所有subject資料夾的路徑）
    base_folder = "/shared/data/raw/UBFC/"  # 當前資料夾，你可以改成完整路徑

    create_video_from_images('/mnt/disk1/rPPG_datasets/new_processed/RGB_crop/subject12_garage_still_940/')
    create_video_from_images('/mnt/disk1/rPPG_datasets/new_processed/RGB_crop/subject15_garage_still_940/')
# 12星座

