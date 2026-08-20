"""Loads and encodes the three benchmark datasets into a common shape.

Each loader returns (X, y, protected_feature, categorical_columns, display_name):
  X: DataFrame, fully numeric (label-encoded categoricals), median-imputed.
  y: Series of {0, 1}, 1 = adverse outcome (default / delinquency / denial).
  protected_feature: column name usable for C4 bias-detection analysis, or None.
  categorical_columns: names of columns that were label-encoded (treated as
    categorical by the explainer neighborhood construction).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Home Credit's application_train.csv is not redistributed in this repo (Kaggle
# competition data usage terms restrict redistribution). Download it from
# https://www.kaggle.com/c/home-credit-default-risk/data, then either place it
# at research/data/application_train.csv or point HOME_CREDIT_TRAIN_CSV at it.
HOME_CREDIT_TRAIN_CSV = Path(
    os.environ.get("HOME_CREDIT_TRAIN_CSV", DATA_DIR / "application_train.csv")
)


def _label_encode(series: pd.Series) -> pd.Series:
    codes, _ = pd.factorize(series, sort=True)
    return pd.Series(codes, index=series.index, dtype=float).replace(-1, np.nan)


def load_home_credit(n_rows: int | None = None) -> tuple[pd.DataFrame, pd.Series, str, list[str], str]:
    if not HOME_CREDIT_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Home Credit training data not found at {HOME_CREDIT_TRAIN_CSV}. "
            "Download application_train.csv from "
            "https://www.kaggle.com/c/home-credit-default-risk/data and place it "
            "there, or set the HOME_CREDIT_TRAIN_CSV environment variable."
        )
    df = pd.read_csv(HOME_CREDIT_TRAIN_CSV, nrows=n_rows)
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])

    missing_frac = X.isna().mean()
    keep = missing_frac[missing_frac <= 0.80].index.tolist()
    X = X[keep]

    categorical_columns = [
        c for c in X.columns
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c])
    ]
    for col in categorical_columns:
        X[col] = _label_encode(X[col])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    protected_feature = "CODE_GENDER"
    return X, y, protected_feature, categorical_columns, "Home Credit Default Risk"


def load_hmeq() -> tuple[pd.DataFrame, pd.Series, str | None, list[str], str]:
    path = DATA_DIR / "hmeq.csv"
    df = pd.read_csv(path)
    y = df["BAD"].astype(int)
    X = df.drop(columns=["BAD"])

    categorical_columns = ["REASON", "JOB"]
    for col in categorical_columns:
        X[col] = _label_encode(X[col])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # HMEQ has no demographic columns at all -- there is nothing to bias-test.
    return X, y, None, categorical_columns, "HMEQ Home Equity Loan"


def _parse_dti(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    text = text.replace("%", "")
    if text.startswith("<"):
        try:
            return float(text[1:]) - 5.0
        except ValueError:
            return np.nan
    if text.startswith(">"):
        try:
            return float(text[1:]) + 5.0
        except ValueError:
            return np.nan
    if "-" in text:
        parts = [p.replace("<", "").strip() for p in text.split("-")]
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return (lo + hi) / 2.0
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def _parse_age_bin(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in ("8888", "9999"):
        return np.nan
    if "-" in text:
        try:
            lo, hi = text.split("-")
            return (float(lo) + float(hi)) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_hmda_vt() -> tuple[pd.DataFrame, pd.Series, str, list[str], str]:
    path = DATA_DIR / "hmda_vt_sample.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df[df["action_taken"].isin([1, 3])].copy()
    y = (df["action_taken"] == 3).astype(int)  # 1 = denied (adverse)

    # interest_rate / rate_spread / total_loan_costs are excluded: they are
    # null-by-construction for denied applications (no loan closed), so they
    # are a disguised label proxy rather than a genuine pre-decision feature.
    numeric_raw_cols = [
        "loan_amount",
        "loan_to_value_ratio",
        "loan_term",
        "property_value",
        "income",
        "total_units",
    ]
    frame = {}
    for col in numeric_raw_cols:
        frame[col] = pd.to_numeric(df[col], errors="coerce")

    frame["debt_to_income_ratio"] = df["debt_to_income_ratio"].apply(_parse_dti)
    frame["applicant_age"] = df["applicant_age"].apply(_parse_age_bin)

    categorical_source_cols = [
        "derived_sex",
        "derived_race",
        "derived_ethnicity",
        "loan_purpose",
        "occupancy_type",
        "derived_dwelling_category",
        "loan_type",
        "lien_status",
    ]
    categorical_columns = []
    for col in categorical_source_cols:
        if col in df.columns:
            frame[col] = _label_encode(df[col].astype(str))
            categorical_columns.append(col)

    X = pd.DataFrame(frame, index=df.index)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    protected_feature = "derived_sex"
    return X, y, protected_feature, categorical_columns, "HMDA Vermont 2023 (approval/denial)"


LOADERS = {
    "home_credit": load_home_credit,
    "hmeq": load_hmeq,
    "hmda_vt": load_hmda_vt,
}


if __name__ == "__main__":
    for key, loader in LOADERS.items():
        X, y, protected, cats, name = loader()
        print(f"{key:12s} {name:35s} rows={len(X):>7} cols={X.shape[1]:>3} "
              f"positive_rate={y.mean():.4f} protected={protected} n_cat={len(cats)} "
              f"nulls_after_impute={int(X.isna().sum().sum())}")
