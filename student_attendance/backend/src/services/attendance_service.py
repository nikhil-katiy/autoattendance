from src.db.database import get_conn
from datetime import date, datetime
from src.db.database import get_conn
from datetime import datetime
import base64

def get_current_lecture():
    conn = get_conn()
    c = conn.cursor()

    now = datetime.now().strftime("%H:%M")

    c.execute("SELECT * FROM lectures")
    lectures = c.fetchall()

    for lec in lectures:
        if lec[2] <= now <= lec[3]:
            return {"id": lec[0]}

    return None

def already_marked_today(student_id):
    conn = get_conn()
    cursor = conn.cursor()

    query = """
    SELECT 1 FROM attendance 
    WHERE student_id = %s AND DATE(date) = CURDATE()
    """

    cursor.execute(query, (student_id,))
    result = cursor.fetchone()

    conn.close()
    return result is not None

def mark_attendance(student_id):
    conn = get_conn()
    cursor = conn.cursor()

    query = """
    INSERT INTO attendance (student_id, date, status)
    VALUES (%s, NOW(), 'present')
    """

    cursor.execute(query, (student_id,))
    conn.commit()
    conn.close()
    
def save_attendance(student_id, name, capture_image=None):
    try:
        conn = get_conn()
        cur = conn.cursor()
        now = datetime.now()

        # duplicate check (1 day rule)
        cur.execute(
            "SELECT * FROM attendance WHERE student_id=%s AND DATE(date)=%s",
            (student_id, now.date())
        )

        if cur.fetchone():
            conn.close()
            return {"success": False, "message": "Already marked"}

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
            capture_image
        ))

        conn.commit()
        conn.close()

        return {"success": True}

    except Exception as e:
        print("ERROR:", e)
        return {"success": False, "error": str(e)}