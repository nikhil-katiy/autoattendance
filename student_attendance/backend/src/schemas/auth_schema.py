from pydantic import BaseModel, Field

#  REGISTER
class RegisterSchema(BaseModel):

    full_name: str

    username: str

    email: str

    mobile: str

    gender: str

    password: str

    confirm_password: str

#  LOGIN
class LoginSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=3)