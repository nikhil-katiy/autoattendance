from apscheduler.schedulers.background import BackgroundScheduler
from src.db.database import get_conn
from datetime import date

def mark_absent():
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT id FROM lectures")
    lectures = c.fetchall()

    for lec in lectures:
        # mark absent logic (simplified)
        pass

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(mark_absent, 'interval', minutes=5)
    scheduler.start()