from pydantic import BaseModel

from typing import Optional

class EnrollSchema(BaseModel):
    face_id: str
    first_name: str
    last_name: str
    mobile: str
    email: str
    gender: str
    role: str
    image: str   # base64

#  RECOGNIZE
class ImageSchema(BaseModel):
    image: str