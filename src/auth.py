"""JWT helpers (kept free of BentoML so they can be unit-tested on their own)."""
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "30"))

# Demo users. Override with the API_USERS env var, e.g. "alice:pw1,bob:pw2".
USERS = dict(
    pair.split(":", 1)
    for pair in os.getenv("API_USERS", "admin:admin123,user:user123").split(",")
    if ":" in pair
)


class AuthError(Exception):
    """Raised when a token is missing, malformed, invalid or expired."""


def authenticate_user(username: str, password: str) -> bool:
    expected = USERS.get(username)
    return expected is not None and hmac.compare_digest(expected, password)


def create_access_token(username: str, expires_delta: timedelta | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=TOKEN_EXPIRE_MINUTES))
    payload = {"sub": username, "iat": now, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid token") from exc


def token_from_header(authorization: str | None) -> str:
    if not authorization:
        raise AuthError("Missing authentication token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")
    return token.strip()
