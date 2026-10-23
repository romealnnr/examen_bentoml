"""Tests for the admission prediction API (self-contained: they only talk to
the running container over HTTP and use PyJWT to craft test tokens).

Start the service first:
    docker run --rm -d -p 3000:3000 --name admission_api admission_prediction_service:latest
Then:
    pytest -v

Environment overrides:
    API_URL         default http://localhost:3000
    JWT_SECRET_KEY  must match the container's secret (default below)
"""
import os
import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:3000")
LOGIN_URL = f"{API_URL}/login"
PREDICT_URL = f"{API_URL}/predict"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = "HS256"
TIMEOUT = 10

VALID_USER = {"username": "admin", "password": "admin123"}
VALID_INPUT = {
    "gre_score": 324,
    "toefl_score": 107,
    "university_rating": 4,
    "sop": 4.0,
    "lor": 4.5,
    "cgpa": 8.87,
    "research": 1,
}


def make_token(sub="admin", minutes=30, key=JWT_SECRET_KEY):
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(payload, key, algorithm=JWT_ALGORITHM)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def predict(headers=None, payload=VALID_INPUT):
    return requests.post(PREDICT_URL, json=payload, headers=headers or {}, timeout=TIMEOUT)


@pytest.fixture(scope="session", autouse=True)
def api_ready():
    """Wait up to 60 s for the service, otherwise fail with a clear message."""
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if requests.get(f"{API_URL}/readyz", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    pytest.fail(f"API not reachable at {API_URL}: start the container first (see README)")


@pytest.fixture(scope="session")
def token():
    r = requests.post(LOGIN_URL, json=VALID_USER, timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


# ------------------------------------------------------------- 1. JWT auth
class TestJWTAuthentication:
    def test_missing_token_fails(self):
        r = predict()
        assert r.status_code == 401
        assert "missing" in r.json()["detail"].lower()

    def test_malformed_header_fails(self, token):
        r = predict({"Authorization": token})  # no "Bearer " scheme
        assert r.status_code == 401

    def test_invalid_token_fails(self):
        assert predict(bearer("not.a.valid.jwt")).status_code == 401

    def test_token_signed_with_wrong_key_fails(self):
        forged = make_token(key="attacker-secret-key-that-is-long-enough")
        assert predict(bearer(forged)).status_code == 401

    def test_expired_token_fails(self):
        r = predict(bearer(make_token(minutes=-5)))
        assert r.status_code == 401
        assert "expired" in r.json()["detail"].lower()

    def test_valid_token_succeeds(self, token):
        assert predict(bearer(token)).status_code == 200


# ------------------------------------------------------------ 2. Login API
class TestLoginAPI:
    def test_valid_credentials_return_valid_jwt(self):
        r = requests.post(LOGIN_URL, json=VALID_USER, timeout=TIMEOUT)
        assert r.status_code == 200
        body = r.json()
        assert body["token_type"] == "bearer"
        payload = jwt.decode(body["access_token"], JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == VALID_USER["username"]
        assert payload["exp"] > time.time()

    @pytest.mark.parametrize(
        "creds",
        [
            {"username": "admin", "password": "wrong-password"},
            {"username": "unknown", "password": "admin123"},
        ],
    )
    def test_invalid_credentials_return_401(self, creds):
        r = requests.post(LOGIN_URL, json=creds, timeout=TIMEOUT)
        assert r.status_code == 401
        assert "access_token" not in r.json()


# ------------------------------------------------------- 3. Prediction API
class TestPredictionAPI:
    def test_missing_token_returns_401(self):
        assert predict().status_code == 401

    def test_invalid_token_returns_401(self):
        assert predict(bearer("abc")).status_code == 401

    def test_valid_input_returns_prediction(self, token):
        r = predict(bearer(token))
        assert r.status_code == 200
        pred = r.json()["chance_of_admit"]
        assert isinstance(pred, float)
        assert 0.0 <= pred <= 1.0

    def test_stronger_profile_gets_higher_chance(self, token):
        weak = {**VALID_INPUT, "gre_score": 295, "toefl_score": 95, "cgpa": 7.0, "research": 0}
        strong = {**VALID_INPUT, "gre_score": 338, "toefl_score": 119, "cgpa": 9.8}
        p_weak = predict(bearer(token), weak).json()["chance_of_admit"]
        p_strong = predict(bearer(token), strong).json()["chance_of_admit"]
        assert p_strong > p_weak

    @pytest.mark.parametrize(
        "bad_input",
        [
            {**VALID_INPUT, "gre_score": 400},  # out of range
            {**VALID_INPUT, "cgpa": "very good"},  # wrong type
            {k: v for k, v in VALID_INPUT.items() if k != "toefl_score"},  # missing field
            {**VALID_INPUT, "research": 2},  # must be 0 or 1
        ],
        ids=["out_of_range", "wrong_type", "missing_field", "bad_research"],
    )
    def test_invalid_input_returns_error(self, token, bad_input):
        r = predict(bearer(token), bad_input)
        assert r.status_code in (400, 422)
