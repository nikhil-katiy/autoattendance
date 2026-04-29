from pydantic import BaseModel, field_validator
from typing import List
from datetime import date, time

class SessionCreate(BaseModel):
    subject: str
    session_name: str
    start_time: time
    end_time: time
    start_date: date
    end_date: date
    students: List[str]

    @field_validator("end_time")
    def check_time(cls, v, values):
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("End time must be greater than start time")
        return v

    @field_validator("end_date")
    def check_date(cls, v, values):
        if "start_date" in values and v < values["start_date"]:
            raise ValueError("End date must be after start date")
        return v