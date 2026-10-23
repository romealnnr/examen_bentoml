"""Secure BentoML prediction API.

Endpoints (all POST, JSON):
    /login    {"username": ..., "password": ...}       -> {"access_token": ..., ...}
    /predict  Authorization: Bearer <token>
              {"gre_score": 320, "toefl_score": 110, "university_rating": 4,
               "sop": 4.5, "lor": 4.0, "cgpa": 9.1, "research": 1}
              -> {"chance_of_admit": 0.83, "user": "admin"}

Run locally:  bentoml serve src.service:AdmissionService --port 3000
"""
from typing import Annotated

import bentoml
import pandas as pd
from pydantic import Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.auth import (
    AuthError,
    TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    decode_token,
    token_from_header,
)

FEATURES = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
]
PROTECTED_PATHS = {"/predict"}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Rejects requests to protected paths without a valid Bearer token."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PROTECTED_PATHS:
            try:
                token = token_from_header(request.headers.get("Authorization"))
                payload = decode_token(token)
            except AuthError as exc:
                return JSONResponse(
                    status_code=401,
                    content={"detail": str(exc)},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            request.state.user = payload["sub"]
        return await call_next(request)


@bentoml.service(name="admission_prediction_service", traffic={"timeout": 30})
class AdmissionService:
    bento_model = bentoml.models.BentoModel("admission_lr:latest")

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.bento_model)

    @bentoml.api(route="/login")
    def login(
        self,
        username: Annotated[str, Field(min_length=1)],
        password: Annotated[str, Field(min_length=1)],
        ctx: bentoml.Context,
    ) -> dict:
        if not authenticate_user(username, password):
            ctx.response.status_code = 401
            return {"detail": "Incorrect username or password"}
        return {
            "access_token": create_access_token(username),
            "token_type": "bearer",
            "expires_in": TOKEN_EXPIRE_MINUTES * 60,
        }

    @bentoml.api(route="/predict")
    def predict(
        self,
        gre_score: Annotated[float, Field(ge=0, le=340)],
        toefl_score: Annotated[float, Field(ge=0, le=120)],
        university_rating: Annotated[float, Field(ge=1, le=5)],
        sop: Annotated[float, Field(ge=1, le=5)],
        lor: Annotated[float, Field(ge=1, le=5)],
        cgpa: Annotated[float, Field(ge=0, le=10)],
        research: Annotated[int, Field(ge=0, le=1)],
        ctx: bentoml.Context,
    ) -> dict:
        row = pd.DataFrame(
            [[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]],
            columns=FEATURES,
        )
        prediction = float(self.model.predict(row)[0])
        # Linear regression is unbounded: clip to a valid probability.
        prediction = min(max(prediction, 0.0), 1.0)
        user = getattr(ctx.request.state, "user", None)
        return {"chance_of_admit": round(prediction, 4), "user": user}

    @bentoml.api(route="/model_info")
    def model_info(self) -> dict:
        """Public endpoint describing the served model and its test metrics."""
        return {
            "model_tag": str(self.bento_model.tag),
            "features": FEATURES,
            "metrics": dict(self.bento_model.info.metadata),
        }


AdmissionService.add_asgi_middleware(JWTAuthMiddleware)
