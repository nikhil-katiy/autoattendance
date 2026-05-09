from fastapi import Depends, HTTPException
from fastapi import APIRouter
from src.db.database import get_conn

from fastapi import APIRouter, HTTPException
from src.schemas.auth_schema import RegisterSchema, LoginSchema

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from src.utils.auth import (
    hash_password,
    validate_password
)

router = APIRouter()

#  REGISTER
@router.post("/register")
def register(data: RegisterSchema):

    # PASSWORD MATCH
    if data.password != data.confirm_password:

        return {
            "status": "ERROR",
            "message": "Passwords do not match"
        }

    # PASSWORD VALIDATION
    error = validate_password(data.password)

    if error:

        return {
            "status": "ERROR",
            "message": error
        }

    conn = get_conn()

    cur = conn.cursor()

    # CHECK EMAIL
    cur.execute(
        "SELECT id FROM users WHERE email=%s",
        (data.email,)
    )

    if cur.fetchone():

        return {
            "status": "ERROR",
            "message": "Email already exists"
        }

    # HASH PASSWORD
    hashed_password = hash_password(
        data.password
    )

    # INSERT USER
    cur.execute(
        """
        INSERT INTO users
        (
            full_name,
            username,
            email,
            mobile,
            gender,
            role,
            password
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            data.full_name,
            data.username,
            data.email,
            data.mobile,
            data.gender,
            "admin",
            hashed_password
        )
    )

    conn.commit()

    conn.close()

    return {
        "status": "SUCCESS",
        "message": "Registration Successful"
    }

# LOGIN
from src.utils.auth import (
    verify_password,
    create_access_token
)

@router.post("/login")
def login(data: LoginSchema):

    conn = get_conn()

    cur = conn.cursor()

    # USERNAME OR EMAIL LOGIN
    cur.execute(
        """
        SELECT *
        FROM users
        WHERE username=%s OR email=%s
        """,
        (data.username, data.username)
    )

    user = cur.fetchone()

    conn.close()

    if not user:

        return {
            "status": "ERROR",
            "message": "Invalid credentials"
        }

    stored_password = user[7]

    # VERIFY PASSWORD
    if not verify_password(
        data.password,
        stored_password
    ):

        return {
            "status": "ERROR",
            "message": "Invalid credentials"
        }

    # CREATE TOKEN
    token = create_access_token({

        "sub": user[1],

        "role": user[6]
    })

    return {

        "status": "SUCCESS",
        "access_token": token,

        "token_type": "bearer",
        "message": "Login Successful"
    }