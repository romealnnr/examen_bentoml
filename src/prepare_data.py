"""Load the raw admissions dataset, clean it and split it into train/test sets.

Usage (from the project root):
    python src/prepare_data.py

Outputs (data/processed/): X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "admission.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "chance_of_admit"
FEATURES = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
]

# Valid ranges for each column, used to drop impossible rows.
RANGES = {
    "gre_score": (0, 340),
    "toefl_score": (0, 120),
    "university_rating": (1, 5),
    "sop": (1, 5),
    "lor": (1, 5),
    "cgpa": (0, 10),
    "research": (0, 1),
    "chance_of_admit": (0, 1),
}

TEST_SIZE = 0.2
RANDOM_STATE = 42


def clean_column_name(name: str) -> str:
    """'Chance of Admit ' -> 'chance_of_admit', 'LOR ' -> 'lor'."""
    return "_".join(name.strip().lower().replace(".", "").split())


def load_and_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [clean_column_name(c) for c in df.columns]

    # 'Serial No.' is just a row identifier: no predictive value.
    df = df.drop(columns=[c for c in df.columns if c.startswith("serial")])

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    n_before = len(df)
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().drop_duplicates()
    for col, (low, high) in RANGES.items():
        df = df[df[col].between(low, high)]
    df["research"] = df["research"].astype(int)

    print(f"Rows kept after cleaning: {len(df)}/{n_before}")
    return df.reset_index(drop=True)


def main() -> None:
    df = load_and_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print(f"Train: {X_train.shape}, Test: {X_test.shape} -> saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
