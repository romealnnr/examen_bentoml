# Examen BentoML — Admission prediction API

A linear regression (StandardScaler + LinearRegression) predicting a student's
chance of admission, served by BentoML behind JWT authentication and packaged
as a Docker image.

| Endpoint      | Method | Auth         | Body                                                                                   |
|---------------|--------|--------------|----------------------------------------------------------------------------------------|
| `/login`      | POST   | none         | `{"username": "admin", "password": "admin123"}`                                        |
| `/predict`    | POST   | Bearer token | `{"gre_score": 324, "toefl_score": 107, "university_rating": 4, "sop": 4.0, "lor": 4.5, "cgpa": 8.87, "research": 1}` |
| `/model_info` | POST   | none         | `{}`                                                                                    |

Demo accounts: `admin / admin123`, `user / user123`. Tokens expire after 30 minutes.

---

## 1. Reviewer: load, run and test the service

Run these commands **in order** from the folder containing this README
(Docker and Python 3.10+ required; port 3000 must be free).

```bash
# 1. Load the Docker image
docker load -i admission_prediction_service.tar

# 2. Run the service in the background on port 3000
docker run --rm -d -p 3000:3000 --name admission_api admission_prediction_service:latest

# 3. Create and activate a virtual environment for the tests
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 4. Install the dependencies
pip install -r requirements.txt

# 5. Run the tests (they wait up to 60 s for the service to be ready)
pytest -v

# 6. Stop the service when finished
docker stop admission_api
```

Optional manual check (with the container running):

```bash
TOKEN=$(curl -s -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"gre_score": 324, "toefl_score": 107, "university_rating": 4, "sop": 4.0, "lor": 4.5, "cgpa": 8.87, "research": 1}'
```

The tests cover: missing / malformed / invalid / wrongly signed / expired tokens
(401), a valid token (200), login with correct (JWT returned) and incorrect
credentials (401), a valid prediction in [0, 1], and invalid inputs
(out of range, wrong type, missing field) rejected with 400/422.

---

## 2. Author: rebuild everything from scratch

Run from the project root (the full repository, which contains `src/` and `bentofile.yaml`).

```bash
# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the dataset into data/raw
curl -o data/raw/admission.csv https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv

# 3. Clean + split -> data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
python src/prepare_data.py

# 4. Train, print R2 / RMSE / MAE, save to the BentoML Model Store
python src/train_model.py

# 5. Verify the model is registered
bentoml models list
bentoml models get admission_lr:latest

# 6. Serve locally and test (in a second terminal: pytest -v)
bentoml serve src.service:AdmissionService --port 3000

# 7. Build the bento and check it
bentoml build
bentoml list

# 8. Containerize and export the image archive
bentoml containerize admission_prediction_service:latest -t admission_prediction_service:latest
docker save admission_prediction_service:latest -o admission_prediction_service.tar

# 9. Create the submission archive (only the 4 required items)
tar -cf examen_bentoml_submission.tar admission_prediction_service.tar requirements.txt README.md tests/test_api.py
```

Configuration (environment variables, read by the container at start-up):
`JWT_SECRET_KEY` (default `change-me-in-production`), `TOKEN_EXPIRE_MINUTES`
(default 30), `API_USERS` (default `admin:admin123,user:user123`).
If you change `JWT_SECRET_KEY` with `docker run -e ...`, export the same value
before running `pytest`.
