
import base64
import numpy as np
import cv2
import os
import threading


from src.services.face_service import FaceService
from src.db.database import fetch_all_embeddings, save_attendance
from src.schemas.face_schema import EnrollSchema, ImageSchema
from src.services.attendance_service import mark_attendance, get_current_lecture
from src.db.database import get_conn
from src.schemas.face_schema import EnrollSchema
from fastapi import APIRouter

router = APIRouter()

#  MODEL PATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "face_detection_yunet_2023mar.onnx")

print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

face_service = FaceService(MODEL_PATH)


#  MATCH FUNCTION (FIXED - OUTSIDE)

def match_face(query_emb):
    rows = fetch_all_embeddings()

    if not rows:
        return None, "Unknown", 0

    grouped = {}

    for sid, name, angle, emb in rows:
        grouped.setdefault((sid, name), []).append(emb)

    best_score = -1
    best_id = None
    best_name = "Unknown"

    query_emb = query_emb / np.linalg.norm(query_emb)

    for (sid, name), embs in grouped.items():
        avg_emb = np.mean(embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)

        score = np.dot(query_emb, avg_emb)

        if score > best_score:
            best_score = score
            best_id = sid
            best_name = name

    if best_score > 0.6:
        return best_id, best_name, float(best_score)

    return None, "Unknown", float(best_score)

#  ADD HERE (enroll ke upar)

def is_already_registered(new_emb):
    rows = fetch_all_embeddings()

    for sid, name, angle, emb in rows:
        score = np.dot(new_emb, emb) / (
            np.linalg.norm(new_emb) * np.linalg.norm(emb)
        )

        print(" Checking:", name, "Score:", score)

        if score > 0.75:
            return True

    return False

@router.post("/enroll")
def enroll(data: EnrollSchema):
    try:
        print("\n========== ENROLL START ==========")

        student_id = data.face_id
        full_name = f"{data.first_name} {data.last_name}"

        if not data.image:
            return {"message": "No image provided"}

        #  BASE64 FIX
        img_data = data.image
        if "," in img_data:
            img_data = img_data.split(",")[1]

        try:
            img_bytes = base64.b64decode(img_data)
        except Exception as e:
            print(" BASE64 ERROR:", e)
            return {"message": "Invalid image format"}

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            print(" Frame decode failed")
            return {"message": "Invalid image"}

        print(" Frame shape:", frame.shape)

        #  FACE DETECT
        faces = face_service.detect_faces(frame)

        if faces is None or len(faces) == 0:
            print(" No face detected")
            return {"message": "No face detected"}

        print(" Faces found:", len(faces))

        #  DUPLICATE ID CHECK
        existing = fetch_all_embeddings()
        already_id = any(sid == student_id for sid, _, _, _ in existing)

        if already_id:
            return {"message": "ID already exists"}

        #   ONLY ONE FACE (FIXED)
        f = faces[0]

        x, y, w, h = map(int, f[:4])

        h_img, w_img = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return {"message": "Face crop failed"}

        #  EMBEDDING
        emb = face_service.embedding_from_crop(frame, f)

        if emb is None or emb.size == 0:
            return {"message": "Bad face"}

        #  FACE DUPLICATE CHECK
        if is_already_registered(emb):
            return {"message": "Face already registered"}

        emb_bytes = emb.astype(np.float32).tobytes()

        #  INSERT
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO embeddings 
            (student_id, name, angle, embedding, image,
             first_name, last_name, mobile, email, gender, role)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            student_id,
            full_name,
            "front",
            emb_bytes,
            img_bytes,
            data.first_name,
            data.last_name,
            data.mobile,
            data.email,
            data.gender,
            data.role
        ))

        conn.commit()
        conn.close()

        print(" ENROLL SUCCESS")

        return {
            "message": "Face enrolled successfully",
            "role": data.role
        }

    except Exception as e:
        print(" ENROLL ERROR:", e)
        return {"message": "Server error"}

#  UPDATE STUDENT
@router.put("/update-student")
def update_student(data: EnrollSchema):
    try:
        conn = get_conn()
        cur = conn.cursor()

        #  decode image
        img_data = data.image
        if "," in img_data:
            img_data = img_data.split(",")[1]

        img_bytes = base64.b64decode(img_data)

        #  re-embedding
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = face_service.detect_faces(frame)

        if faces is None or len(faces) == 0:
            return {"message": "No face detected"}

        emb = face_service.embedding_from_crop(frame, faces[0])
        emb_bytes = emb.astype(np.float32).tobytes()

        full_name = f"{data.first_name} {data.last_name}"

        #  UPDATE
        cur.execute("""
            UPDATE embeddings
            SET 
                name=%s,
                first_name=%s,
                last_name=%s,
                mobile=%s,
                email=%s,
                gender=%s,
                role=%s,
                image=%s,
                embedding=%s
            WHERE student_id=%s
        """, (
            full_name,
            data.first_name,
            data.last_name,
            data.mobile,
            data.email,
            data.gender,
            data.role,
            img_bytes,
            emb_bytes,
            data.face_id
        ))

        conn.commit()
        conn.close()

        return {"message": "Student updated successfully"}

    except Exception as e:
        print("UPDATE ERROR:", e)
        return {"message": "Update failed"}

#  RECOGNIZE
@router.post("/recognize")
def recognize(data: ImageSchema):

    try:
        print("\n========== NEW REQUEST ==========")

        img_data = data.image

        if "," in img_data:
            img_data = img_data.split(",")[1]

        img_bytes = base64.b64decode(img_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Frame decode failed")
            return {"faces": []}

        frame = cv2.resize(frame, (640, 480))

        if cv2.Laplacian(frame, cv2.CV_64F).var() < 50:
            print("Blurry frame skipped")
            return {"faces": []}

        faces = face_service.detect_faces(frame)

        if faces is None or len(faces) == 0:
            print("No faces detected")
            return {"faces": []}

        face = faces[0]

        emb = face_service.embedding_from_crop(frame, face)

        if emb is None or emb.size == 0:
            print("Invalid embedding")
            return {"faces": []}

        x, y, w, h = map(int, face[:4])

        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            return {"faces": []}

        success, buffer = cv2.imencode(".jpg", face_img)

        if not success:
            return {"faces": []}

        capture_bytes = buffer.tobytes()

        img_base64 = base64.b64encode(capture_bytes).decode("utf-8")

        #  MATCH FIRST (IMPORTANT)
        student_id, name, score = match_face(emb)

        print("MATCH RESULT:", student_id, name, score)

        #  SAFE CHECK
        if student_id is not None and score > 0.65:

            print("MATCH FOUND:", student_id)

            save_attendance(student_id, name, capture_bytes)

        else:
            name = "Unknown"
            student_id = None

        return {
            "faces": [{
                "student_id": student_id,
                "name": name,
                "score": float(round(score, 3)),
                "box": [x, y, w, h],
                "image": img_base64
            }]
        }

    except Exception as e:
        print("RECOGNIZE ERROR:", e)
        return {"faces": []}