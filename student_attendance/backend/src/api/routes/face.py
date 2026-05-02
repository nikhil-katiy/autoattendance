
import base64
import numpy as np
import cv2
import os
import threading

from torch import cosine_similarity
from src.db.database import get_students
from src.services.face_service import FaceService
from src.db.database import fetch_all_embeddings, save_attendance
from src.schemas.face_schema import EnrollSchema, ImageSchema
from src.services.attendance_service import mark_attendance, get_current_lecture
from src.db.database import get_conn
from src.schemas.face_schema import EnrollSchema
from fastapi import APIRouter
from collections import defaultdict

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

def is_good_face(face, landmarks):

    # 🔥 blur (loose karo)
    blur = cv2.Laplacian(face, cv2.CV_64F).var()
    print("BLUR:", blur)

    if blur < 30:   # ❌ 60 → ✅ 30
        return False

    # 🔥 size check
    h, w = face.shape[:2]
    if w < 80 or h < 80:   # ❌ 100 → ✅ 80
        return False

    # # 🔥 eye alignment (loose karo)
    # left_eye = (landmarks[5], landmarks[6])
    # right_eye = (landmarks[7], landmarks[8])

    # eye_diff = abs(left_eye[1] - right_eye[1])
    # print("EYE DIFF:", eye_diff)

    # if eye_diff > 35:   # ❌ 20 → ✅ 35
    #     return False

    return True

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0

    return float(np.dot(a, b) / denom)


# 🔥 GLOBAL MEMORY (top of file)
# captured_counts = {}

@router.post("/enroll")
def enroll(data: EnrollSchema):
    try:
        print("\n========== ENROLL ==========")

        student_id = data.face_id
        full_name = f"{data.first_name} {data.last_name}"

        if not data.image or len(data.image) < 100:
            return {"status": "RED", "message": "No image"}
        
        if not data.face_id or not data.first_name or not data.mobile:
            return {"status": "RED", "message": "Fill all required fields"}

        # decode
        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "RED", "message": "Invalid image"}

        faces = face_service.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return {"status": "RED", "message": "No face"}

        x, y, w, h = map(int, faces[0][:4])
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            return {"status": "RED", "message": "Bad face"}

        # embedding
        h, w = face_crop.shape[:2]
        emb = face_service.embedding_from_crop(face_crop, [0, 0, w, h])
        if emb is None or len(emb) == 0:
            return {"status": "RED", "message": "Embedding failed"}

        emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return {"status": "RED", "message": "Bad embedding"}

        emb = emb / norm
        emb_bytes = emb.tobytes()
        
        rows = fetch_all_embeddings()

        for sid, name, angle, db_emb in rows:
         sim = cosine_similarity(emb, db_emb)

         if sim > 0.75 and sid != student_id:
          return {
            "status": "RED",
            "message": "Face already registered with another ID"
        }

        conn = get_conn()
        cur = conn.cursor()

        # 🔥 DB COUNT CHECK
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE student_id=%s", (student_id,))
        count = cur.fetchone()[0]

        if count >= 5:
            conn.close()
            return {"status": "DONE", "message": "Face Added Successfully"}

        # insert
        cur.execute("""
            INSERT INTO embeddings 
            (student_id, name, angle, embedding, image,
             first_name, last_name, mobile, email, gender, role)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            student_id, full_name, "auto", emb_bytes, img_bytes,
            data.first_name, data.last_name, data.mobile,
            data.email, data.gender, data.role
        ))

        conn.commit()
        conn.close()

        count += 1
        print("COUNT:", count)

        return {"status": "GREEN", "count": count}

    except Exception as e:
        print("ENROLL ERROR:", e)
        return {"status": "RED", "message": "Server error"}

@router.put("/update-student")
def update_student(data: EnrollSchema):
    try:
        conn = get_conn()
        cur = conn.cursor()

        full_name = f"{data.first_name} {data.last_name}"

        # 🔥 WITHOUT IMAGE UPDATE
        if not data.image:
            cur.execute("""
                UPDATE embeddings
                SET 
                    name=%s,
                    first_name=%s,
                    last_name=%s,
                    mobile=%s,
                    email=%s,
                    gender=%s,
                    role=%s
                WHERE student_id=%s
            """, (
                full_name,
                data.first_name,
                data.last_name,
                data.mobile,
                data.email,
                data.gender,
                data.role,
                data.face_id
            ))

        else:
            # 🔥 WITH IMAGE UPDATE
            img_data = data.image.split(",")[1]
            img_bytes = base64.b64decode(img_data)

            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            faces = face_service.detect_faces(frame)

            if faces is None or len(faces) == 0:
                return {"success": False, "message": "No face"}

            f = faces[0]
            x, y, w, h = map(int, f[:4])
            face_crop = frame[y:y+h, x:x+w]

            h, w = face_crop.shape[:2]
            fake_box = [0, 0, w, h]

            emb = face_service.embedding_from_crop(face_crop, fake_box)

            emb = np.array(emb, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)

            emb_bytes = emb.tobytes()

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

        return {"success": True}

    except Exception as e:
        print("UPDATE ERROR:", e)
        return {"success": False}

#  RECOGNIZE
@router.post("/recognize")
def recognize(data: ImageSchema):
    try:
        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = face_service.detect_faces(frame)

        if faces is None or len(faces) == 0:
            return {"faces": []}

        f = faces[0]
        x, y, w, h = map(int, f[:4])
        face_crop = frame[y:y+h, x:x+w]

        h, w = face_crop.shape[:2]
        emb = face_service.embedding_from_crop(face_crop, [0,0,w,h])

        if emb is None or len(emb) == 0:
            return {"faces": []}

        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        # 🔥 FETCH DB
        rows = fetch_all_embeddings()

        if not rows:
            return {"faces": []}

        # 🔥 GROUP BY STUDENT
        grouped = defaultdict(list)

        for sid, name, angle, db_emb in rows:
            grouped[(sid, name)].append(db_emb)

        best_score = -1
        best_match = None

        # 🔥 COMPARE
        for (sid, name), emb_list in grouped.items():
            scores = []

            for db_emb in emb_list:
                sim = cosine_similarity(emb, db_emb)
                scores.append(sim)

            avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_match = (sid, name)

        print("BEST SCORE:", best_score)

        # 🔥 THRESHOLD (IMPORTANT)
        if best_score < 0.45:
            return {"faces": []}

        return {
            "faces": [{
                "student_id": best_match[0],
                "name": best_match[1],
                "score": float(best_score),
                "box": [int(x), int(y), int(w), int(h)]
            }]
        }

    except Exception as e:
        print("RECOGNIZE ERROR:", e)
        return {"faces": []}