from fastapi import APIRouter
from src.db.database import get_conn
import base64

router = APIRouter()

@router.get("/students/")
def get_students():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            student_id,
            MAX(name),
            MAX(image),
            MAX(mobile),
            MAX(email),
            MAX(gender),
            MAX(role),
            MAX(first_name),
            MAX(last_name)
        FROM embeddings
        GROUP BY student_id
    """)

    rows = cur.fetchall()
    conn.close()

    result = []

    for r in rows:
        img = None
        if r[2]:
            img = base64.b64encode(r[2]).decode("utf-8")

        result.append({
            "student_id": r[0],
            "name": r[1],
            "image": img,
            "mobile": r[3],
            "email": r[4],
            "gender": r[5],
            "role": r[6],
            "first_name": r[7],
            "last_name": r[8]
        })

    return result

# delete student
@router.delete("/delete-student/{student_id}")
def delete_student(student_id: str):
    try:
        conn = get_conn()
        cur = conn.cursor()

        #  embeddings से delete
        cur.execute("DELETE FROM embeddings WHERE student_id=%s", (student_id,))

        #  attendance से delete
        cur.execute("DELETE FROM attendance WHERE student_id=%s", (student_id,))

        conn.commit()
        conn.close()

        return {"message": "Student deleted successfully"}

    except Exception as e:
        print("DELETE ERROR:", e)
        return {"message": "Server error"}