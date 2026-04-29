from pathlib import Path

import threading
from typing import Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1


class FaceService:
    def __init__(self, yunet_model_path: str, input_size: Tuple[int, int] = (320, 320)):
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #  MODEL CHECK (ADDED)
        ensure_model_exists(yunet_model_path)

        # LOWER THRESHOLD (IMPORTANT)
        self.detector = cv2.FaceDetectorYN.create(
            yunet_model_path,
            "",
            input_size,
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    #  FACE DETECTION (ROBUST)
    def detect_faces(self, frame: np.ndarray):
        if frame is None:
            print(" Frame is None")
            return []

        h, w = frame.shape[:2]
        with self.lock:
         print(" Frame size:", w, h)

        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)

        #  FALLBACK
        if faces is None:
            print(" No face detected, trying resize...")

            resized = cv2.resize(frame, (320, 320))
            self.detector.setInputSize((320, 320))

            _, faces = self.detector.detect(resized)

            if faces is not None:
                scale_x = w / 320
                scale_y = h / 320

                faces[:, 0] *= scale_x
                faces[:, 1] *= scale_y
                faces[:, 2] *= scale_x
                faces[:, 3] *= scale_y

        if faces is None:
            print(" Still no face detected")
            return []

        print(" Faces detected:", len(faces))
        return faces

    #  CROP FACE
    def get_face_crop(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        x, y, bw, bh = face_row[:4].astype(int)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + bw)
        y2 = min(frame.shape[0], y + bh)

        if x2 <= x1 or y2 <= y1:
            return np.array([])

        return frame[y1:y2, x1:x2]

    #  EMBEDDING
    def embedding_from_crop(self, frame: np.ndarray, face_box: np.ndarray) -> np.ndarray:
        x, y, w, h = face_box[:4].astype(int)

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            return np.array([])

        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0

        tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.embedder(tensor).cpu().numpy()[0]

        norm = np.linalg.norm(emb)
        if norm == 0:
            return np.array([])

        return emb / norm


#  COSINE SIMILARITY
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


#  MODEL CHECK
def ensure_model_exists(model_path: str) -> None:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"YUNet model not found at: {model_path}\n"
            "Download face_detection_yunet_2023mar.onnx and pass correct path."
        )



 # VERY IMPORTANT PART (ADD THIS)

#  AUTO PATH (no hardcode issue)
MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "ml_models/face_detection_yunet_2023mar.onnx"
)

#  CREATE GLOBAL INSTANCE
face_service = FaceService(MODEL_PATH)