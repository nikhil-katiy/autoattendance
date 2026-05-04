import mysql.connector
from datetime import datetime
import numpy as np
import base64

# DB CONNECTION
def get_conn():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="attendance_db"
    )

# INIT DB (FIXED TABLES)
def init_db():
    conn = get_conn()
    cur = conn.cursor()
    
    # EMBEDDINGS (add image column)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        student_id VARCHAR(50),
        name VARCHAR(100),
        angle VARCHAR(20),
        embedding LONGBLOB,
        image LONGBLOB
    )
    """)

        # attendance
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(50),
    status VARCHAR(20) DEFAULT 'present',
    name VARCHAR(100),
    session_id INT,
    date DATE,
    status VARCHAR(20) DEFAULT 'present',
    enroll_image LONGBLOB,
    capture_image LONGBLOB,

    UNIQUE KEY unique_attendance (student_id, session_id, date),

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
)
        """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject VARCHAR(100) NOT NULL,
            session_name VARCHAR(100),
            start_time TIME,
            end_time TIME,
            start_date DATE,
            end_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # session_students
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_students (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id INT,
            student_id VARCHAR(50),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
        """)

    

    conn.commit()
    conn.close()

# INSERT EMBEDDING
def insert_embedding(student_id, name, angle, embedding, image_bytes):
    conn = get_conn()
    cur = conn.cursor()

    #  FIX: ensure numpy
    embedding = np.array(embedding)

    norm = np.linalg.norm(embedding)
    if norm != 0:
        embedding = embedding / norm

    emb_bytes = embedding.tobytes()

    cur.execute("""
        INSERT INTO embeddings (student_id, name, angle, embedding, image)
        VALUES (%s, %s, %s, %s, %s)
    """, (student_id, name, angle, emb_bytes, image_bytes))

    conn.commit()
    conn.close()

# FETCH EMBEDDINGS
def fetch_all_embeddings():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT student_id, name, angle, embedding FROM embeddings")
    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        if r[3] is None:
            continue
        emb = np.frombuffer(r[3], dtype=np.float32)
        result.append((r[0], r[1], r[2], emb))

    return result

def get_students():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT student_id, name, first_name, last_name, mobile, email, gender, role
        FROM embeddings
    """)

    rows = cur.fetchall()
    conn.close()

    return rows



# SAVE ATTENDANCE
def save_attendance(student_id, name, capture_bytes):
    from datetime import datetime

    conn = get_conn()
    cur = conn.cursor()

    now = datetime.now()

    print(" TRY SAVE:", student_id, name)

    #  STRONG DUPLICATE CHECK (MySQL side)
    cur.execute("""
    SELECT id FROM attendance
    WHERE student_id=%s AND DATE(date)=CURDATE()
""", (student_id,))

    if cur.fetchone():
        print(" Already marked today")
        conn.close()
        return

    #  GET ENROLL IMAGE + ROLE (SAFE)
    cur.execute("""
        SELECT image, role FROM embeddings
        WHERE student_id=%s
        LIMIT 1
    """, (student_id,))
    
    row = cur.fetchone()

    if not row:
        print(" No embedding found")
        conn.close()
        return

    enroll_img = row[0]
    role = row[1] if row[1] else "student"   # fallback

    #  INSERT
    cur.execute("""
        INSERT INTO attendance 
        (student_id, name, date, status, enroll_image, capture_image, role)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        student_id,
        name,
        now,
        "present",
        enroll_img,
        capture_bytes,
        role
    ))

    conn.commit()
    conn.close()

    print(" Attendance saved")



# FETCH ATTENDANCE ( FIXED OUTSIDE)

def fetch_attendance():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            a.student_id,
            a.name,
            a.date,
            a.status,
            e.image,           -- enroll image
            a.capture_image    -- capture image
        FROM attendance a
        LEFT JOIN embeddings e 
        ON a.student_id = e.student_id
        ORDER BY a.date DESC
    """)

    rows = cur.fetchall()
    conn.close()

    result = []

    for r in rows:
        enroll_img = None
        capture_img = None

        if r[4]:
            enroll_img = base64.b64encode(r[4]).decode("utf-8")

        if r[5]:
            capture_img = base64.b64encode(r[5]).decode("utf-8")

        result.append({
            "student_id": r[0],
            "name": r[1],
            "date": str(r[2]),
            "status": r[3] if r[3] else "present",
            "enroll_image": enroll_img,
            "capture_image": capture_img
        })

    return result
def fetch_students():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT student_id, name, image
        FROM embeddings
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
            "image": img
        })

    return result