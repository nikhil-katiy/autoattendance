from fastapi import APIRouter
from src.db.database import fetch_students, get_conn

router = APIRouter(prefix="/attendance")

@router.get("/student")
def student_attendance():
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
SELECT 
    a.student_id,
    e.name,
    a.date,
    a.status,
    a.enroll_image,
    a.capture_image
FROM attendance a
JOIN embeddings e ON a.student_id = e.student_id
WHERE a.role = 'student'
ORDER BY a.date DESC
""")

        data = cur.fetchall()

        print(" DATA FROM DB:", data)

        import base64

        #  VERY IMPORTANT (BEFORE RETURN)
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
        print(" ERROR:", e)
        return []


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
        WHERE a.role = 'teacher'
        ORDER BY a.date DESC
        """)

        data = cur.fetchall()

        import base64

        for row in data:
            if row.get("enroll_image"):
                row["enroll_image"] = base64.b64encode(row["enroll_image"]).decode("utf-8")

            if row.get("capture_image"):
                row["capture_image"] = base64.b64encode(row["capture_image"]).decode("utf-8")

        conn.close()

        print(" TEACHER DATA:", data)

        return data if data else []

    except Exception as e:
        print(" TEACHER ERROR:", e)
        return []

@router.get("/students")
def get_students():
    return fetch_students()