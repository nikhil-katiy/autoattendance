from pydantic import BaseModel

class AttendanceOut(BaseModel):
    student_id: str
    name: str
    date: str
    capture_image: str   
    enroll_image: str
    status: str
    action: str
    
    