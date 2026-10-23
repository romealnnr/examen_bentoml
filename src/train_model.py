"""Train a linear regression on the processed data, evaluate it and save it
to the BentoML Model Store.

Usage (from the project root):
    python src/train_model.py
    bentoml models list          # verify the model is registered
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_NAME = "admission_lr"
MIN_R2 = 0.70  # minimum test R2 required before saving the model


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")


def build_model() -> Pipeline:
    # Scaling lives inside the pipeline, so the API can send raw values
    # and the exact same normalisation is applied at inference time.
    return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
    }


def main() -> None:
    X_train, X_test = load_split("X_train"), load_split("X_test")
    y_train = load_split("y_train").squeeze("columns")
    y_test = load_split("y_test").squeeze("columns")

    model = build_model()
    model.fit(X_train, y_train)

    train_metrics = evaluate(model, X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test)
    print("Performance  |    R2   |  RMSE  |  MAE")
    for label, m in (("train", train_metrics), ("test", test_metrics)):
        print(f"{label:<12} | {m['r2']:.4f}  | {m['rmse']:.4f} | {m['mae']:.4f}")

    coefs = model.named_steps["regressor"].coef_
    print("\nStandardised coefficients:")
    for feature, coef in sorted(zip(X_train.columns, coefs), key=lambda t: -abs(t[1])):
        print(f"  {feature:<18} {coef:+.4f}")

    if test_metrics["r2"] < MIN_R2:
        raise SystemExit(
            f"Test R2 {test_metrics['r2']:.3f} < {MIN_R2}: model not saved."
        )

    import bentoml  # imported here so training/evaluation works without it

    saved = bentoml.sklearn.save_model(
        MODEL_NAME,
        model,
        signatures={"predict": {"batchable": False}},
        metadata={f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
        custom_objects={"feature_names": list(X_train.columns)},
    )
    print(f"\nModel saved to the BentoML Model Store: {saved.tag}")


if __name__ == "__main__":
    main()
