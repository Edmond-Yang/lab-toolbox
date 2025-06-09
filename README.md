# Lab-Toolbox

## Introduction
It's a toolbox for training and testing remote PPG algorithms, including data preprocessing, model training, evaluation, and visualization. It supports various remote PPG methods such as PhysFormer, SINC, POS, and CHROM.

## 📋 Prerequisites

1. Copy `.env-templates` to `.env` and fill in your environment variables before running `train_and_test.sh`, `train.py`, or `test.py`.
2. Create `protocol.yaml` (or duplicate `templates.yaml` and rename it) before using the protocol-related scripts.

---

## Current Version
- **Current Version**: 1.0
- **Last Updated**: 2025-06-09
- **Author**: [Your Name](Edmond Yang)
- **License**: [MIT License](https://opensource.org/license/mit/)
- **GitHub Repository**: [Lab-Toolbox]()

## Version History
- **1.0** [2025-06-09]: Initial release with basic functionality for training and testing remote PPG algorithms.

---

## 📂 Project Structure

```
lab-toolbox/
│
├─ train_and_test.sh
├─ train.py
├─ test.py
├─ tsne.py
├─ .env
├─ .env-templates
│
├─ utils/
│   ├─ dataloader.py
│   ├─ args.py
│   ├─ path.py
│   ├─ augmentation.py
│   ├─ logger.py
│   ├─ metrics.py
│   ├─ loss.py
│   └─ transform/
│       └─ … (method-specific augmentation modules)
│
├─ model/
│   ├─ templates.py
│   └─ (remote PPG algorithms: physformer/ sinc/ pos/ chrom)
│
├─ preprocess/
│   ├─ preprocess.py
│   └─ model/
│       ├─ mtcnn.py
│       ├─ mediapipe.py
│       └─ retinaFaceDetection.py
│
├─ protocol/
│   ├─ protocol.py
│   ├─ protocol.yaml
│   └─ templates.yaml
│
├─ logger/
│   └─ (log files, TensorBoard logs, etc.)
│
└─ visualization/
    └─ (vidualization charts and figures)

```

####  train\_and\_test.sh

One-step shell script to run both training and testing; automatically loads environment variables and calls `train.py` and `test.py`.

####  train.py

Entry point for model training: loads data, initializes the model, runs training loops, and saves the best weights.

#### test.py

Entry point for model evaluation: loads a saved model, computes heart-rate metrics, and exports visualization charts.

####  tsne.py

Utility for t-SNE dimensionality reduction and plotting (currently a work in progress).

####  .env

Environment variable file; must be created by the user to store data paths, API keys, and other sensitive settings.

#### .env-templates

Template for `.env`, demonstrating available variable names and example values.

---

### 📂 utils/

Library of helper modules.

#### dataloader.py

Generates PyTorch DataLoader objects based on splits defined in `protocol.yaml`.

#### args.py

Parses command-line arguments and converts user inputs into configuration objects.

#### path.py

Centralized management of all input/output paths to avoid hardcoding.

#### augmentation.py

Combines various data augmentation methods and applies them dynamically during training.

#### logger.py

Handles logging to files and console output (TODO: improve formatting and features).

#### metrics.py

Calculates heart-rate evaluation metrics such as MAE, RMSE, MAPE, and Pearson correlation.

#### loss.py

Defines template loss functions to facilitate customization.

#### transform/

Contains augmentation modules for different methods (e.g., POS, CHROM).

---

### 📂 model/

Implementations of remote PPG algorithms.

#### templates.py

Abstract base class enforcing implementation of `train`, `forward`, and other common interfaces.

#### physformer/
Yu et al., “PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer,” CVPR 2022.
[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.pdf)   [GitHub](https://github.com/ZitongYu/PhysFormer)

#### sinc/



#### pos/

Wang et al., “Algorithmic Principles of Remote PPG (POS),” IEEE Trans. Biomed. Eng., 2016.

#### chrom/

Haan et al., “Robust Pulse Rate from Chrominance-Based rPPG (CHROM),” IEEE Trans. Biomed. Eng., 2013.

---

### 📂 preprocess/

Data preprocessing scripts.

#### preprocess.py

Preprocesses raw video and labels: frame extraction, normalization, and conversion to `.npy`.

#### mtcnn.py

Face detection and cropping using PyTorch MTCNN.

#### mediapipe.py

Face detection and cropping using MediaPipe.

#### retinaFaceDetection.py

Face detection and cropping using TensorFlow RetinaFace.

---

### 📂 protocol/

Experiment configuration and data split logic.

#### protocol.py

Parses and generates the `protocol.yaml` file to control train/validation/test splits.

#### protocol.yaml

User-defined experiment settings; must be created before running protocol scripts.

#### templates.yaml

Template for `protocol.yaml`, showing all configurable fields and default values.

---

### 📂 logger/

Directory for storing all log files (e.g., `.log`, TensorBoard, `.mat`).

### 📂 visualization/

Directory for output charts and figures (loss curves, Bland-Altman plots, t-SNE scatter plots, etc.).

