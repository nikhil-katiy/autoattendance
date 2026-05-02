from pydantic import BaseModel


class AttendanceIn(BaseModel):
     student_id: str
     name: str
     capture_image: str | None = None   # 🔥 ADD THIS 
class AttendanceOut(BaseModel):
    student_id: str
    name: str
    date: str
    capture_image: str   
    enroll_image: str
    status: str
    action: str
    
    
    
     