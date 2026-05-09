import re

from jose import jwt, JWTError

from datetime import datetime, timedelta

from passlib.context import CryptContext


SECRET_KEY = "supersecretkey"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# =========================
# PASSWORD VALIDATION
# =========================
def validate_password(password):

    if len(password) < 8:
        return "Password must be at least 8 characters"

    if not re.search(r"[A-Z]", password):
        return "Password must contain uppercase letter"

    if not re.search(r"[a-z]", password):
        return "Password must contain lowercase letter"

    if not re.search(r"\d", password):
        return "Password must contain number"

    if not re.search(r"[!@#$%^&*]", password):
        return "Password must contain special character"

    return None


# =========================
# HASH PASSWORD
# =========================
def hash_password(password: str):

    return pwd_context.hash(password)


# =========================
# VERIFY PASSWORD
# =========================
def verify_password(
    plain_password,
    hashed_password
):

    return pwd_context.verify(
        plain_password,
        hashed_password
    )


# =========================
# CREATE TOKEN
# =========================
def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return encoded_jwt


# =========================
# VERIFY TOKEN
# =========================
def verify_token(token: str):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        return payload

    except JWTError:

        return None