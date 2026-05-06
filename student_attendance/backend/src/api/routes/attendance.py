from fastapi import APIRouter
from src.db.database import fetch_students, get_conn
import base64
from datetime import datetime
from src.db.database import get_conn
from pydantic import BaseModel
from src.schemas.attendance import AttendanceIn
from src.services.attendance_service import save_attendance, get_current_lecture, already_marked_today, mark_attendance
from src.services.telegram_service import send_telegram_message
  
router = APIRouter()   
router = APIRouter(prefix="/attendance")

# STUDENT ATTENDANCE
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

        #  BASE64 FIX
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

# TEACHER ATTENDANCE
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

        #  BASE64 FIX
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

    capture_bytes = None
    if data.capture_image and "," in data.capture_image:
        img_data = data.capture_image.split(",")[1]
        capture_bytes = base64.b64decode(img_data)

    return save_attendance(
        data.student_id,
        data.name,
        capture_bytes
    )
    
@router.get("/working-days")
def working_days():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(DISTINCT DATE(date))
        FROM attendance
    """)

    total = cur.fetchone()[0]

    conn.close()

    return {
        "working_days": total
    }
    
@router.get("/attendance-percentage/{student_id}")
def attendance_percentage(student_id: str):

    conn = get_conn()
    cur = conn.cursor()

    # present count
    cur.execute("""
        SELECT COUNT(*)
        FROM attendance
        WHERE student_id=%s
    """, (student_id,))

    present_days = cur.fetchone()[0]

    # total working days
    cur.execute("""
        SELECT COUNT(DISTINCT DATE(date))
        FROM attendance
    """)

    total_days = cur.fetchone()[0]

    conn.close()

    percentage = 0

    if total_days > 0:
        percentage = (present_days / total_days) * 100

    return {
        "student_id": student_id,
        "present_days": present_days,
        "total_days": total_days,
        "attendance_percentage": round(percentage, 2)
    }
    
@router.get("/attendance-stats/{student_id}")
def attendance_stats(student_id: str):

    conn = get_conn()
    cur = conn.cursor()

    # present
    cur.execute("""
        SELECT COUNT(*)
        FROM attendance
        WHERE student_id=%s
    """, (student_id,))

    present = cur.fetchone()[0]

    # total days
    cur.execute("""
        SELECT COUNT(DISTINCT DATE(date))
        FROM attendance
    """)

    total = cur.fetchone()[0]

    absent = total - present

    conn.close()

    return {
        "student_id": student_id,
        "present": present,
        "absent": absent,
        "total": total
    }
    
@router.get("/attendance-percentage-all")
def attendance_percentage_all():

    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    # ✅ FIXED WORKING DAYS
    TOTAL_WORKING_DAYS = 30

    # student present counts
    cur.execute("""
        SELECT 
            student_id,
            name,
            COUNT(*) AS present_days
        FROM attendance
        GROUP BY student_id, name
    """)

    rows = cur.fetchall()

    result = []

    for row in rows:

        present_days = row["present_days"]

        # absent calculation
        absent_days = TOTAL_WORKING_DAYS - present_days

        # percentage calculation
        percentage = (
            present_days / TOTAL_WORKING_DAYS
        ) * 100

        result.append({
            "student_id": row["student_id"],
            "name": row["name"],
            "present_days": present_days,
            "absent_days": absent_days,
            "total_working_days": TOTAL_WORKING_DAYS,
            "percentage": round(percentage, 2)
        })

    conn.close()

    return result




from src.services.telegram_service import send_telegram_message


@router.post("/send-telegram-report")
def send_telegram_report():

    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    # =========================
    # ALL ENROLLED STUDENTS
    # =========================
    cur.execute("""
        SELECT DISTINCT student_id, name
        FROM embeddings
    """)

    all_students = cur.fetchall()

    # =========================
    # TODAY PRESENT STUDENTS
    # =========================
    cur.execute("""
        SELECT DISTINCT student_id, name
        FROM attendance
        WHERE DATE(date)=CURDATE()
        AND status='present'
    """)

    present_students = cur.fetchall()

    # =========================
    # CREATE PRESENT ID SET
    # =========================
    present_ids = set()

    for student in present_students:
        present_ids.add(student["student_id"])

    # =========================
    # FIND ABSENT STUDENTS
    # =========================
    absent_students = []

    for student in all_students:

        if student["student_id"] not in present_ids:
            absent_students.append(student)

    # =========================
    # COUNTS
    # =========================
    total_enrolled = len(all_students)

    present_count = len(present_students)

    absent_count = len(absent_students)

    # =========================
    # FORMAT MESSAGE
    # =========================
    message = "📢 TODAY'S ATTENDANCE REPORT\n\n"

    # =========================
    # PRESENT LIST
    # =========================
    message += "✅ PRESENT STUDENTS:\n\n"

    if present_students:

        for student in present_students:

            message += (
                f"ID: {student['student_id']} "
                f"- {student['name']}\n"
            )

    else:
        message += "No present students\n"

    # =========================
    # ABSENT LIST
    # =========================
    message += "\n❌ ABSENT STUDENTS:\n\n"

    if absent_students:

        for student in absent_students:

            message += (
                f"ID: {student['student_id']} "
                f"- {student['name']}\n"
            )

    else:
        message += "No absent students\n"

    # =========================
    # SUMMARY
    # =========================
    message += "\n📊 SUMMARY:\n\n"

    message += f"👥 Total Enrolled: {total_enrolled}\n"

    message += f"✅ Present: {present_count}\n"

    message += f"❌ Absent: {absent_count}\n"

    # =========================
    # SEND TELEGRAM MESSAGE
    # =========================
    send_telegram_message(message)

    conn.close()

    return {
        "success": True,
        "total_enrolled": total_enrolled,
        "present": present_count,
        "absent": absent_count
    }

# ALL STUDENTS
@router.get("/students")
def get_students():
    return fetch_students()