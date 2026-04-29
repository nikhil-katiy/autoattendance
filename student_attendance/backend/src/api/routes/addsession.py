from fastapi import APIRouter

router = APIRouter()

@router.post("/add-session")
def add_session():
    return {"message": "Session added"}