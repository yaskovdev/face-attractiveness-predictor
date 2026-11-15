"""Face detection and alignment shared by training (SCUT-FBP5500) and inference (arbitrary photos).

Using the very same detector and alignment for both is essential: otherwise the model sees differently cropped faces at
inference time than it was trained on.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = Path(__file__).resolve().parent.parent / "pretrained" / "face_detection_yunet_2023mar.onnx"

ALIGNED_SIZE = 256

# Canonical positions of the right eye, left eye, nose tip, right and left mouth corners (the order YuNet uses) on the
# well-known 112x112 ArcFace template. They are rescaled so that the crop also includes the forehead, hair and chin.
_ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
_FACE_SCALE = 0.55
_FACE_CENTER_Y = 0.55


def _template(size):
    face_center = _ARCFACE_TEMPLATE.mean(axis=0)
    return (_ARCFACE_TEMPLATE - face_center) * (size / 112.0) * _FACE_SCALE + np.array([size / 2.0, size * _FACE_CENTER_Y])


class FaceAligner:
    def __init__(self, size=ALIGNED_SIZE, score_threshold=0.6):
        if not YUNET_PATH.exists():
            YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        self._detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold, 0.3, 5000)
        self._size = size
        self._template = _template(size).astype(np.float32)

    def detect(self, image):
        """Returns the 5 landmarks of the largest face in a BGR image, or None if there is no face."""
        height, width = image.shape[:2]
        # Very large photos slow the detector down without improving accuracy.
        scale = min(1.0, 1280.0 / max(height, width))
        resized = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image
        self._detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = self._detector.detect(resized)
        if faces is None or len(faces) == 0:
            return None
        largest = max(faces, key=lambda face: face[2] * face[3])
        return largest[4:14].reshape(5, 2) / scale

    def align(self, image):
        """Returns a size x size BGR crop of the largest face, rotated and scaled to the canonical pose, or None."""
        landmarks = self.detect(image)
        if landmarks is None:
            return None
        matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), self._template, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(image, matrix, (self._size, self._size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def read_image(path):
    """cv2.imread does not support non-ASCII paths on Windows, hence imdecode."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
