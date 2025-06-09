import os
import cv2
import torch
import shutil
import numpy as np

from tqdm import tqdm
from facenet_pytorch import MTCNN

class FaceCropper:
    def __init__(self, scale_factor=1.0, size=200, smooth_alpha=0.15, min_shift=7, max_shift=50):

        # Initialize the model for crop human face
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(device=self.device)

        # Set the scale factor for fitting the human face
        self.size = size
        self.scale_factor = scale_factor

        self.min_shift = min_shift
        self.max_shift = max_shift
        self.smooth_alpha = smooth_alpha

        # Store the last detected bounding box
        self.last_box = None

        # Waiting list for cropping
        self.waiting_list = []

    def detect_faces(self, img, re_detect=True):

        if re_detect or self.last_box is None:

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            boxes_gray, _ = self.model.detect(img_gray)
            boxes_bgr, _ = self.model.detect(img)

            if boxes_bgr is None and boxes_gray is None:
                return

            box = boxes_bgr[0] if boxes_bgr is not None else boxes_gray[0]

            if self.last_box is None:
                self.last_box = box
            else:
                # Calculate center shift to reject outliers
                last_center = np.array([(self.last_box[0] + self.last_box[2]) / 2,
                                        (self.last_box[1] + self.last_box[3]) / 2])
                new_center = np.array([(box[0] + box[2]) / 2,
                                       (box[1] + box[3]) / 2])
                dist = np.linalg.norm(new_center - last_center)

                box_len = int(max(box[2] - box[0], box[3] - box[1]))

                # If movement is reasonable, apply exponential smoothing; otherwise ignore
                if dself.min_shift <= dist <= self.max_shift and box_len > 100:
                    self.last_box = (1 - self.smooth_alpha) * self.last_box + self.smooth_alpha * box
                # else: keep self.last_box unchanged to avoid large jumps


    def crop_face(self, img):

        # Use Last Box
        box_len = int(max(self.last_box[2] - self.last_box[0], self.last_box[3] - self.last_box[1]))
        box_half_len = np.round(box_len / 2 * self.scale_factor).astype('int')
        box_mid_y = int((self.last_box[3] + self.last_box[1]) / 2)
        box_mid_x = int((self.last_box[2] + self.last_box[0]) / 2)
        left_x, right_x = max(box_mid_x - box_half_len, 0), min(box_mid_x + box_half_len, img.shape[1])
        top_y, bottom_y = max(box_mid_y - box_half_len, 0), min(box_mid_y + box_half_len, img.shape[0])

        # Crop the image
        cropped_face = cv2.resize(img[top_y:bottom_y, left_x:right_x], (self.size, self.size))

        return cropped_face

    def __call__(self, img, path, re_detect=True):
        # Detect faces
        self.detect_faces(img, re_detect)

        # has no face
        if self.last_box is None:
            self.waiting_list.append(img)
            return None

        self.waiting_list.append(img)

        # Crop face
        cropped_face_list = [self.crop_face(p) for p in self.waiting_list]
        self.waiting_list.clear()
        return cropped_face_list

    def reset(self):
        if len(self.waiting_list) != 0:
            raise RuntimeError("Waiting list is not empty!")

        self.last_box = None