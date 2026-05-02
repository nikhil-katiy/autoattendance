from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#  routers import
from src.api.routes import face, attendance, auth, students
from src.db.database import init_db
from src.services.scheduler import start_scheduler
from src.api.routes.addsession import router as addsession_router
from src.api.routes import attendance




#  app create FIRST
app = FastAPI()

#  init
init_db()
start_scheduler()

#  CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  routers (AFTER app)
app.include_router(face.router)
app.include_router(attendance.router)
app.include_router(auth.router)
app.include_router(students.router)
app.include_router(addsession_router) 
  

@app.get("/")
def root():
    return {"message": "API Running Successfully"}