from fastapi import APIRouter
from src.db.database import fetch_students, get_conn
import base64
from datetime import datetime
from src.db.database import get_conn
from pydantic import BaseModel
from src.schemas.attendance import AttendanceIn


    
router = APIRouter()   

router = APIRouter(prefix="/attendance")


# ===============================
# 🔵 STUDENT ATTENDANCE
# ===============================
@router.get("/student")
def student_attendance():
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT 
                a.student_id,
                a.name,
                a.date,
                a.status,
                a.enroll_image,
                a.capture_image
            FROM attendance a
            JOIN (
                SELECT DISTINCT student_id, role FROM embeddings
            ) e ON a.student_id = e.student_id
            WHERE e.role = 'student'
            ORDER BY a.date DESC
        """)

        data = cur.fetchall()

        # 🔥 BASE64 FIX
        for row in data:
            if row.get("enroll_image"):
                row["enroll_image"] = base64.b64encode(row["enroll_image"]).decode("utf-8")
            else:
                row["enroll_image"] = None

            if row.get("capture_image"):
                row["capture_image"] = base64.b64encode(row["capture_image"]).decode("utf-8")
            else:
                row["capture_image"] = None

        conn.close()
        return data if data else []

    except Exception as e:
        print("STUDENT ERROR:", e)
        return []


# ===============================
# 🟢 TEACHER ATTENDANCE
# ===============================
@router.get("/teacher")
def teacher_attendance():
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT 
                a.student_id,
                a.name,
                a.date,
                a.status,
                a.enroll_image,
                a.capture_image
            FROM attendance a
            JOIN (
                SELECT DISTINCT student_id, role FROM embeddings
            ) e ON a.student_id = e.student_id
            WHERE e.role = 'teacher'
            ORDER BY a.date DESC
        """)

        data = cur.fetchall()

        # 🔥 BASE64 FIX
        for row in data:
            if row.get("enroll_image"):
                row["enroll_image"] = base64.b64encode(row["enroll_image"]).decode("utf-8")
            else:
                row["enroll_image"] = None

            if row.get("capture_image"):
                row["capture_image"] = base64.b64encode(row["capture_image"]).decode("utf-8")
            else:
                row["capture_image"] = None

        conn.close()
        return data if data else []

    except Exception as e:
        print("TEACHER ERROR:", e)
        return []

@router.post("/mark-attendance")
def mark_attendance(data: AttendanceIn):
    try:
        student_id = data.student_id
        name = data.name

        # 🔥 SAFE IMAGE HANDLE
        capture_bytes = None
        if data.capture_image and "," in data.capture_image:
            img_data = data.capture_image.split(",")[1]
            capture_bytes = base64.b64decode(img_data)

        conn = get_conn()
        cur = conn.cursor()

        now = datetime.now()

        # duplicate check
        cur.execute(
            "SELECT * FROM attendance WHERE student_id=%s AND DATE(date)=%s",
            (student_id, now.date())
        )

        if cur.fetchone():
            conn.close()
            return {"success": True, "message": "Already marked"}

        # enroll image
        cur.execute(
            "SELECT image FROM embeddings WHERE student_id=%s LIMIT 1",
            (student_id,)
        )
        row = cur.fetchone()
        enroll_image = row[0] if row else None

        # insert
        cur.execute("""
            INSERT INTO attendance 
            (student_id, name, date, status, enroll_image, capture_image)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (
            student_id,
            name,
            now,
            "present",
            enroll_image,
            capture_bytes
        ))

        conn.commit()
        conn.close()

        print("✅ ATTENDANCE SAVED:", student_id)

        return {"success": True}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"success": False, "error": str(e)}  # 🔥 IMPORTANT

# ===============================
# 👤 ALL STUDENTS
# ===============================
@router.get("/students")
def get_students():
    return fetch_students()