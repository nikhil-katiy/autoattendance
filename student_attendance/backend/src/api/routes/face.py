import base64
import numpy as np
import cv2
import os
import threading
import time

from pydantic import BaseModel
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
from collections import deque
from fastapi import BackgroundTasks
from src.services.email_service import send_attendance_email
from src.services.attendance_service import already_marked_today, mark_attendance
from src.services.attendance_service import save_attendance
from src.services.telegram_service import send_telegram_message
from fastapi import Depends
from src.utils.dependencies import get_current_user

router = APIRouter()

#  MODEL PATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "face_detection_yunet_2023mar.onnx")

face_service = FaceService(MODEL_PATH)

# GLOBALS
angle_buffer = deque(maxlen=5)
required_angles = {"front", "left", "right", "up", "down"}
angle_store = {}  # {student_id: set(angles)}
angle_store_baseline = {}  
enroll_queue = {}  # {student_id: [{"angle": angle, "embedding": emb_bytes, "image": img_bytes, "data": data}, ...]}
last_capture_time = 0

def get_face_angle(landmarks, student_id):
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dx = right_eye[0] - left_eye[0]

    if dx > 30:
        return "right"
    elif dx < -30:
        return "left"
    else:
        return "front"

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

    #  blur (loose karo)
    blur = cv2.Laplacian(face, cv2.CV_64F).var()
    print("BLUR:", blur)

    if blur < 30:   #  60 →  30
        return False

    #  size check
    h, w = face.shape[:2]
    if w < 80 or h < 80:   #  100 →  80
        return False
    return True

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def get_face_angle(landmarks, student_id):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2

    dx = right_eye[0] - left_eye[0]
    dx_nose = nose[0] - eye_center_x

    face_width = abs(dx) + 1
    dx_ratio = dx_nose / (face_width * 0.5)

    mouth_y = (left_mouth[1] + right_mouth[1]) / 2
    face_height = abs(mouth_y - eye_center_y) + 1

    dy_ratio = (nose[1] - eye_center_y) / face_height

    # baseline
    if student_id not in angle_store_baseline:
        angle_store_baseline[student_id] = dy_ratio

    base = angle_store_baseline[student_id]
    delta = dy_ratio - base

    #  FRONT (tight)
    if abs(delta) < 0.025 and abs(dx_ratio) < 0.08:
        return "front"

    #  AXIS DECISION (vertical ko edge)
    if abs(delta) > abs(dx_ratio) * 0.8:
        if delta < -0.06:   #  UP strict 
         return "up"
        elif delta > 0.03:  #  DOWN easy
         return "down"
    else:
        if dx_ratio > 0.18:
            return "right"
        elif dx_ratio < -0.18:
            return "left"

    return "front"

def get_stable_angle(new_angle):
    angle_buffer.append(new_angle)

    # initial frames → raw use karo
    if len(angle_buffer) < 3:
        return new_angle

    return max(set(angle_buffer), key=angle_buffer.count)

# ENROLL
@router.post("/enroll")
def enroll(
    data: EnrollSchema,
    user = Depends(get_current_user)
   ):
    try:
        student_id = data.face_id
        full_name = f"{data.first_name} {data.last_name}"

        # VALIDATION
        if not data.image or len(data.image) < 100:
            return {"status": "RED", "message": "No image"}

        if not data.face_id or not data.first_name or not data.mobile:
            return {"status": "RED", "message": "Fill all required fields"}

        # IMAGE DECODE
        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "RED", "message": "Invalid image"}

        # FACE DETECT
        faces = face_service.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return {"status": "RED", "message": "No face"}

        face = faces[0]
        x, y, w, h = map(int, face[:4])
        landmarks = np.array(face[4:14]).reshape((5, 2))

        # ANGLE DETECT
        current_angle = get_face_angle(landmarks, student_id)

        # INIT STORE
        if student_id not in angle_store:
            angle_store[student_id] = []

        if student_id not in enroll_queue:
            enroll_queue[student_id] = []

        # DUPLICATE ANGLE BLOCK
        if current_angle in [i["angle"] for i in enroll_queue[student_id]]:
            remaining = list(required_angles - set([i["angle"] for i in enroll_queue[student_id]]))
            return {
                "status": "WAIT",
                "message": f"{current_angle} already done",
                "next": remaining
            }

        # FACE CROP
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            return {"status": "RED", "message": "Bad face"}

        # EMBEDDING
        h, w = face_crop.shape[:2]
        emb = face_service.embedding_from_crop(face_crop, [0, 0, w, h])

        if emb is None or len(emb) == 0:
            return {"status": "RED", "message": "Embedding failed"}

        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        emb_bytes = emb.tobytes()
        
        # DUPLICATE FACE CHECK (DB)
        rows = fetch_all_embeddings()
        for sid, name, angle, db_emb in rows:
            sim = cosine_similarity(emb, db_emb)
            if sim > 0.75 and sid != student_id:
                return {
                    "status": "RED",
                    "message": "Face already registered"
                }

        #  STORE IN QUEUE (NOT DB)
        enroll_queue[student_id].append({
            "angle": current_angle,
            "embedding": emb_bytes,
            "image": img_bytes,
            "data": data
        })

        angle_store[student_id].append(current_angle)

        #  IF 5 ANGLES COMPLETE → SAVE DB
        if len(enroll_queue[student_id]) == 5:

            conn = get_conn()
            cur = conn.cursor()

            for item in enroll_queue[student_id]:
                d = item["data"]

                cur.execute("""
                    INSERT INTO embeddings 
                    (student_id, name, angle, embedding, image,
                     first_name, last_name, mobile, email, gender, role)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    student_id,
                    f"{d.first_name} {d.last_name}",
                    item["angle"],
                    item["embedding"],
                    item["image"],
                    d.first_name,
                    d.last_name,
                    d.mobile,
                    d.email,
                    d.gender,
                    d.role
                ))

            conn.commit()
            conn.close()

            # clear queue
            del enroll_queue[student_id]
            angle_store.pop(student_id, None)

            return {
                "status": "DONE",
                "message": "Enrollment Successful"
            }

        # NORMAL RESPONSE
        return {
            "status": "CAPTURED",
            "angle": current_angle,
            "count": len(enroll_queue[student_id]),
            "remaining": list(required_angles - set([i["angle"] for i in enroll_queue[student_id]]))
        }

    except Exception as e:
        print("ENROLL ERROR:", e)
        return {"status": "RED", "message": "Server error"}
   

class RemoveAngleSchema(BaseModel):
    student_id: str
    angle: str
    
@router.post("/remove-angle")
def remove_angle(data: RemoveAngleSchema, user = Depends(get_current_user)):
    student_id = data.student_id
    angle = data.angle

    if student_id in enroll_queue:
        enroll_queue[student_id] = [
            i for i in enroll_queue[student_id]
            if i["angle"] != angle
        ]

    # angle_store भी update करो
    if student_id in angle_store:
        angle_store[student_id] = [
            a for a in angle_store[student_id]
            if a != angle
        ]

    return {"status": "REMOVED"}

@router.put("/update-student")
def update_student(
    data: EnrollSchema,
    user=Depends(get_current_user)
):
    try:

        student_id = data.face_id

        full_name = (
            f"{data.first_name} "
            f"{data.last_name}"
        )

        # VALIDATION
        if not data.image:
            return {
                "status": "RED",
                "message": "No image"
            }

        # IMAGE DECODE
        img_data = data.image.split(",")[1]

        img_bytes = base64.b64decode(
            img_data
        )

        np_arr = np.frombuffer(
            img_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return {
                "status": "RED",
                "message": "Invalid image"
            }

        # FACE DETECT
        faces = face_service.detect_faces(
            frame
        )

        if faces is None or len(faces) == 0:
            return {
                "status": "RED",
                "message": "No face"
            }

        face = faces[0]

        x, y, w, h = map(
            int,
            face[:4]
        )

        landmarks = np.array(
            face[4:14]
        ).reshape((5, 2))

        current_angle = get_face_angle(
            landmarks,
            student_id
        )

        # INIT STORE
        if student_id not in enroll_queue:
            enroll_queue[student_id] = []

        # DUPLICATE ANGLE
        if current_angle in [
            i["angle"]
            for i in enroll_queue[student_id]
        ]:

            remaining = list(
                required_angles - set([
                    i["angle"]
                    for i in enroll_queue[student_id]
                ])
            )

            return {
                "status": "WAIT",
                "message":
                    f"{current_angle} already done",
                "next": remaining
            }

        # FACE CROP
        face_crop = frame[
            y:y+h,
            x:x+w
        ]

        h, w = face_crop.shape[:2]

        emb = (
            face_service.embedding_from_crop(
                face_crop,
                [0, 0, w, h]
            )
        )

        emb = np.array(
            emb,
            dtype=np.float32
        )

        emb = emb / np.linalg.norm(emb)

        emb_bytes = emb.tobytes()

        # STORE TEMP
        enroll_queue[student_id].append({
            "angle": current_angle,
            "embedding": emb_bytes,
            "image": img_bytes,
            "data": data
        })

        # DONE
        if len(enroll_queue[student_id]) == 5:

            conn = get_conn()

            cur = conn.cursor()

            # DELETE OLD
            cur.execute("""
                DELETE FROM embeddings
                WHERE student_id=%s
            """, (student_id,))

            # INSERT NEW
            for item in enroll_queue[
                student_id
            ]:

                d = item["data"]

                cur.execute("""
                    INSERT INTO embeddings
                    (
                        student_id,
                        name,
                        angle,
                        embedding,
                        image,
                        first_name,
                        last_name,
                        mobile,
                        email,
                        gender,
                        role
                    )
                    VALUES (
                        %s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s
                    )
                """, (
                    student_id,
                    full_name,
                    item["angle"],
                    item["embedding"],
                    item["image"],
                    d.first_name,
                    d.last_name,
                    d.mobile,
                    d.email,
                    d.gender,
                    d.role
                ))

            conn.commit()

            conn.close()

            del enroll_queue[student_id]

            return {
                "status": "DONE",
                "message":
                    "Profile Updated Successfully"
            }

        return {
            "status": "CAPTURED",
            "angle": current_angle,
            "count": len(
                enroll_queue[student_id]
            ),
            "remaining": list(
                required_angles - set([
                    i["angle"]
                    for i in enroll_queue[student_id]
                ])
            )
        }

    except Exception as e:

        print("UPDATE ERROR:", e)

        return {
            "status": "RED",
            "message": "Server error"
        }

#  RECOGNIZE
@router.post("/recognize")
def recognize(data: ImageSchema, background_tasks: BackgroundTasks):
    try:
        print("\n========== RECOGNIZE ==========")

        #  DECODE IMAGE
        if not data.image or len(data.image) < 100:
            return {
                "status": "ERROR",
                "message": "Invalid image",
                "faces": []
            }

        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {
                "status": "ERROR",
                "message": "Frame decode failed",
                "faces": []
            }

        #  LIGHT CHECK
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if np.mean(gray) < 50:
            return {
                "status": "WAIT",
                "message": "Low light",
                "faces": []
            }

        #  DETECT FACE
        faces = face_service.detect_faces(frame)

        if faces is None or len(faces) == 0:
            return {
                "status": "WAIT",
                "message": "No face detected",
                "faces": []
            }

        f = faces[0]
        x, y, w, h = map(int, f[:4])

        # face size check
        if w < 80 or h < 80:
            return {
                "status": "WAIT",
                "message": "Move closer",
                "faces": []
            }

        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size == 0:
            return {
                "status": "WAIT",
                "message": "Bad face crop",
                "faces": []
            }

        #  EMBEDDING
        h_, w_ = face_crop.shape[:2]

        emb = face_service.embedding_from_crop(
            face_crop,
            [0, 0, w_, h_]
        )

        if emb is None or len(emb) == 0:
            return {
                "status": "WAIT",
                "message": "Embedding failed",
                "faces": []
            }

        emb = np.array(emb, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        #  FETCH DB
        rows = fetch_all_embeddings()

        if not rows:
            return {
                "status": "ERROR",
                "message": "No data in DB",
                "faces": []
            }

        #  GROUP BY STUDENT
        from collections import defaultdict

        grouped = defaultdict(list)

        for sid, name, angle, db_emb in rows:

            # only front angle
            if angle != "front":
                continue

            grouped[(sid, name)].append(db_emb)

        if not grouped:
            return {
                "status": "ERROR",
                "message": "No front embeddings found",
                "faces": []
            }

        #  BEST MATCH SEARCH
        best_score = -1
        best_match = None

        for (sid, name), emb_list in grouped.items():

            avg_emb = np.mean(emb_list, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)

            sim = cosine_similarity(emb, avg_emb)

            print(f"[CHECK] {sid} ({name}) -> {sim:.3f}")

            if sim > best_score:
                best_score = sim
                best_match = (sid, name)

        print("BEST MATCH:", best_match)
        print("BEST SCORE:", best_score)

        #  THRESHOLD
        THRESHOLD = 0.85

        if best_score < THRESHOLD:
            return {
                "status": "UNKNOWN",
                "message": "Face not recognized",
                "score": float(best_score),
                "faces": []
            }

        #  SUCCESS
        student_id = best_match[0]
        student_name = best_match[1]

        #  GET EMAIL
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "SELECT email FROM embeddings WHERE student_id=%s LIMIT 1",
            (student_id,)
        )

        row = cur.fetchone()

        student_email = row[0] if row else None

        conn.close()

        #  SAVE ATTENDANCE
        result = save_attendance(
            student_id,
            student_name,
            img_bytes
        )

        #  SUCCESS CASE
        if result["success"]:

            # send email
            if student_email:
                background_tasks.add_task(
                    send_attendance_email,
                    student_email,
                    student_name
                )

            return {
                "status": "SUCCESS",
                "message": "Attendance marked",
                "faces": [{
                    "student_id": student_id,
                    "name": student_name,
                    "score": float(best_score),
                    "box": [int(x), int(y), int(w), int(h)]
                }]
            }

        #  ALREADY MARKED
        else:
            return {
                "status": "SKIPPED",
                "message": "Attendance already marked today",
                "faces": [{
                    "student_id": student_id,
                    "name": student_name,
                    "score": float(best_score),
                    "box": [int(x), int(y), int(w), int(h)]
                }]
            }

    except Exception as e:
        print("RECOGNIZE ERROR:", e)

        return {
            "status": "ERROR",
            "message": "Server error",
            "faces": []
        }
    
@router.get("/enroll-images/{student_id}")
def get_enroll_images(student_id: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT image, angle FROM embeddings WHERE student_id=%s",
        (student_id,)
    )

    rows = cur.fetchall()

    import base64

    result = []
    for img, angle in rows:
        base64_img = base64.b64encode(img).decode("utf-8")

        result.append({
            "angle": angle,
            "image": f"data:image/jpeg;base64,{base64_img}"
        })

    return result