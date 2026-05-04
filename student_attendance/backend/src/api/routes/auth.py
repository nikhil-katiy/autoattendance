from fastapi import Depends, HTTPException
from fastapi import APIRouter
from src.db.database import get_conn
from src.utils.jwt import create_token
from fastapi import APIRouter, HTTPException
from src.schemas.auth_schema import RegisterSchema, LoginSchema
from src.utils.hash import hash_password, verify_password
from src.utils.hash import verify_password
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

router = APIRouter()

#  REGISTER
@router.post("/register")
def register(user: RegisterSchema):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE username=%s", (user.username,))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="User already registered")

    #  HASH PASSWORD
    hashed_password = hash_password(user.password)

    cur.execute(
        "INSERT INTO users (username, password) VALUES (%s, %s)",
        (user.username, hashed_password)
    )

    conn.commit()
    conn.close()

    return {"message": "User Registered Successfully"}

# LOGIN
@router.post("/login")
def login(user: LoginSchema):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "SELECT id, username, password FROM users WHERE username=%s",
            (user.username,)
        )

        db_user = cur.fetchone()

        if not db_user:
            raise HTTPException(status_code=404, detail="User not registered")

        #  VERIFY HASH PASSWORD
        if not verify_password(user.password, db_user[2]):
            raise HTTPException(status_code=401, detail="Wrong password")

        #  CREATE TOKEN
        token = create_token({
            "user_id": db_user[0],
            "username": db_user[1]
        })

        return {
            "message": "Login success",
            "token": token
        }

    except HTTPException:
        raise

    except Exception as e:
        print("LOGIN ERROR:", e)
        raise HTTPException(status_code=500, detail="Server error")

    finally:
        if conn:
            conn.close()
            
            

security = HTTPBearer()

SECRET_KEY = "your_secret_key_123"
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# @router.get("/Dashboard")
# def dashboard(user=Depends(verify_token)):
#     return {
#         "message": "Welcome to dashboard",
#         "user": user
#     }