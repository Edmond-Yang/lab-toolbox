import cv2
import numpy as np
import mediapipe as mp


class FaceCropper:

    def __init__(
        self,
        scale_factor: float = 1.0,
        size: int = 200,
        model_selection: int = 0,
        min_detection_confidence: float = 0.6,
    ):
        """初始化

        Args:
            scale_factor: 放大或縮小偵測框的比例（>1 放大, <1 縮小）。
            size: 輸出圖像邊長（正方形）。
            model_selection: MediaPipe FaceDetection 的 model_selection；0 針對 2m 內、1 針對遠距離 (>5m)。
            min_detection_confidence: 偵測閾值；低於此值的候選框會被濾除。
        """

        self.scale_factor = scale_factor
        self.size = size
        self.last_box: np.ndarray | None = None
        self.waiting_list: list[np.ndarray] = []

        # 初始化 MediaPipe Face Detection
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _media_pipe_bbox_to_xyxy(self, bbox, img_w: int, img_h: int) -> np.ndarray:
        """將相對座標 bbox 轉為絕對 pixel 座標 [x1, y1, x2, y2]."""
        x1 = max(int(bbox.xmin * img_w), 0)
        y1 = max(int(bbox.ymin * img_h), 0)
        x2 = min(int((bbox.xmin + bbox.width) * img_w), img_w)
        y2 = min(int((bbox.ymin + bbox.height) * img_h), img_h)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def detect_faces(self, img: np.ndarray, re_detect: bool = True) -> None:
        """更新 self.last_box。若 re_detect=False 且已有 last_box，則不重新偵測。"""
        if not re_detect and self.last_box is not None:
            return

        img_h, img_w = img.shape[:2]
        results = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return  # 無人臉

        # 取最高信心度的人臉
        best_det = max(results.detections, key=lambda d: d.score[0])
        box = self._media_pipe_bbox_to_xyxy(best_det.location_data.relative_bounding_box, img_w, img_h)

        # 與上一框進行平滑
        if self.last_box is not None:
            diff = np.linalg.norm(box[:2] - self.last_box[:2])
            if diff >= 20:  # 避免劇烈抖動
                self.last_box = 0.7 * self.last_box + 0.3 * box
        else:
            self.last_box = box

    def crop_face(self, img: np.ndarray) -> np.ndarray:
        """依據 self.last_box 將人臉裁切後放大/縮小到目標 size。"""
        x1, y1, x2, y2 = self.last_box.astype(int)
        box_len = int(max(x2 - x1, y2 - y1))
        half = int(box_len / 2 * self.scale_factor)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        img_h, img_w = img.shape[:2]
        left = max(cx - half, 0)
        right = min(cx + half, img_w)
        top = max(cy - half, 0)
        bottom = min(cy + half, img_h)

        face = img[top:bottom, left:right]
        face = cv2.resize(face, (self.size, self.size))
        return face

    def __call__(self, img: np.ndarray, path, re_detect: bool = True):
        """向類別實例傳入影像，回傳裁切後的人臉列表或 None。"""
        self.detect_faces(img, re_detect)

        # 若當前沒有偵測到人臉則加入等待佇列
        if self.last_box is None:
            self.waiting_list.append(img)
            return None

        # 將等待佇列與當前影像一起裁切
        self.waiting_list.append(img)
        faces = [self.crop_face(frame) for frame in self.waiting_list]
        self.waiting_list.clear()
        return faces

    def reset(self):
        if self.waiting_list:
            raise RuntimeError("Waiting list is not empty!")
        self.last_box = None


# ---------------------------- Demo Usage ----------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cropper = FaceCropperMP(scale_factor=1.3, size=224, model_selection=0, min_detection_confidence=0.5)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        faces = cropper(frame, re_detect=True)
        if faces:
            cv2.imshow("Cropped Face", faces[-1])
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()