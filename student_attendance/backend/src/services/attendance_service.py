from src.db.database import get_conn
from datetime import date, datetime

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

def mark_attendance(student_id, lecture_id, img_bytes):
    conn = get_conn()
    c = conn.cursor()

    from datetime import date
    today = date.today()

    #  DUPLICATE CHECK
    c.execute("""
        SELECT * FROM lecture_attendance
        WHERE student_id=%s AND lecture_id=%s AND date=%s
    """, (student_id, lecture_id, today))

    if not c.fetchone():
        c.execute("""
            INSERT INTO lecture_attendance 
            (student_id, lecture_id, date, status, capture_image)
            VALUES (%s, %s, %s, %s, %s)
        """, (student_id, lecture_id, today, "present", img_bytes))

        conn.commit()

    conn.close()