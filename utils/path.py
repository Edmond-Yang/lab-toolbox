# Module imports

import os

from logger import *
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Union, Tuple

# Datasets and corresponding folder names
class DatasetName(Enum):
    MR_NIRP = "MR-NIRP_Car_reprocessed"
    MR_NIRP_indoor = "MR-NIRP_indoor"
    Tokyo = "Tokyo"
    PURE = "pure"
    COHFACE = "COHFACE"
    VIPL = "VIPL"
    UBFC = "UBFC"
    FaceForensics_Youtube = 'FaceForensics-Youtube'
    FaceForensics_Actor = 'FaceForensics-Actor'

load_dotenv()

# PathManager class to manage dataset paths
class PathManager:

    def __init__(self, ):

        for dir in ["RAW_DIR", "PREPROCESSED_DIR", "PREPROCESSED_DIR2", "PRELOAD_DIR", "PROTOCOL_DIR"]:
            if os.environ.get(dir) is None:
                raise Exception(f"Please set the {dir} environment variable in .env file")

        self.raw_dir = Path(os.environ.get("RAW_DIR"))
        self.preprocessed_dir = Path(os.environ.get("PREPROCESSED_DIR"))
        self.preprocessed_dir2 = Path(os.environ.get("PREPROCESSED_DIR2"))
        self.preload_dir = Path(os.environ.get("PRELOAD_DIR"))
        self.protocol_dir = Path(os.environ.get("PROTOCOL_DIR"))

        self.dataset_names = {name.name: name.value for name in DatasetName}

    def get_dataset_dir(self, dataset: str) -> Path:
        return self.dataset_names.get(dataset, dataset)

    def get_folder_recursively(self, root: Path, ) -> Optional[List[Path]]:

        found = list()
        for folder in root.iterdir():
            if folder.is_dir():
                found.extend(self.get_folder_recursively(folder))
            else:
                found.append(root)
                return found
        return found

    # raw dataset path               ex: raw_root / {subfolder}
    def get_all_raw_path(self) -> List[Path]:
        folders = self.get_folder_recursively(self.raw_dir)
        return folders

    # preprocessed dataset path      ex: root / dataset / dataset_name / subject / {subfolder}
    def get_preprocessed_video_path(self, dataset: str, subfolder: Optional[List], modality: str = "RGB") -> Tuple[Path, bool]:
        subfolder = subfolder if subfolder is not None else []
        path = (self.preprocessed_dir / "dataset" / self.get_dataset_dir(dataset) / f"{modality.upper()}_crop").joinpath(*subfolder)
        path2 = (self.preprocessed_dir2 / "dataset" / self.get_dataset_dir(dataset) / f"{modality.upper()}_crop").joinpath(*subfolder)

        return path2 if path2.exists() else path


    def get_preprocessed_gt_path(self, dataset: str, subfolder: List) -> Tuple[Path, bool]:
        path = self.preprocessed_dir / "dataset" / self.get_dataset_dir(dataset) / "GT" / subfolder[-1] / "ground_truth.txt"
        return path

    # preload path                   ex: root / preload / dataset_name / subject
    def get_preload_path(self, dataset: str, subject: List) -> Path:
        return self.preload_dir / "preload" / self.get_dataset_dir(dataset) / f"{subject[-1]}.h5"

    # min length
    def get_length_path(self):
        return self.preload_dir / "preload" / "length.json"

    def get_weight_path(self, model_name: str, dataset: str) -> Path:
        path = self.preload_dir / "weights" / model_name / dataset
        self.make_weight_directory(model_name, dataset)
        return path

    def get_protocol_path(self, protocol_name: str) -> Tuple[Path, bool]:
        path = self.protocol_dir / 'protocol'
        self.make_directory(path)
        return path / f'{protocol_name}.txt', path.exists()

    # make directory
    def make_directory(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)

    def make_preload_directory(self, dataset: str) -> None:
        path = self.preload_dir / "preload" / self.get_dataset_dir(dataset)
        self.make_directory(path)

    def make_weight_directory(self, model_name: str, dataset: str) -> None:
        path = self.preload_dir / "weights" / model_name / dataset
        self.make_directory(path)

    def make_protocol_directory(self) -> None:
        path = self.protocol_dir / 'protocol'
        self.make_directory(path)

    def get_logger_path(self, model_name, train, test) -> Path:
        path = Path('./logger') / model_name
        self.make_directory(path)
        if train is None and test is not None:
            path = path / f"Test-{test}"
            self.make_directory(path)
            return path
        elif train is not None and test is None:
            path = path / f"Train-{train}"
            self.make_directory(path)
            return path
        elif train is not None and test is not None:
            path = path / f"Train-{train}_to_Test-{test}"
            self.make_directory(path)
            return path
        else:
            raise Exception("Invalid logger path")
        pass


pathManager: PathManager = PathManager()