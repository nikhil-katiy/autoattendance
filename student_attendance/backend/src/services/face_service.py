import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
import threading

class FaceService:
    def __init__(self, model_path):
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(model_path).exists():
            raise FileNotFoundError("Model not found")

        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), 0.5, 0.3, 5000
        )

        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def detect_faces(self, frame):
        if frame is None:
            return []

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        return faces

    def embedding_from_crop(self, frame, face_box):
        x, y, w, h = map(int, face_box[:4])
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            return None

        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0

        tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.embedder(tensor).cpu().numpy()[0]

        norm = np.linalg.norm(emb)
        if norm == 0:
            return None

        return emb / norm


MODEL_PATH = str(Path(__file__).resolve().parents[2] / "ml_models/face_detection_yunet_2023mar.onnx")
face_service = FaceService(MODEL_PATH)