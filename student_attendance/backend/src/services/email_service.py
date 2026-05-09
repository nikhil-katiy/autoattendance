import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from src.db.database import get_conn

EMAIL = "nikhilkatiyar90055@gmail.com"
PASSWORD = "gwbd arfi mjhh lqlz"  #  APP PASSWORD (NOT REAL PASSWORD)

def send_attendance_email(to_email, name):
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        subject = "Attendance Marked Successfully"
        body = f"""
        Hello {name},

        Your attendance has been marked successfully.

        Time: {current_time}

        Thank you.
      """

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL
        msg["To"] = to_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL, PASSWORD)
        server.send_message(msg)
        server.quit()

    except Exception as e:
        print("Email Error:", e)
        
