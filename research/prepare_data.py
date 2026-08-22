"""Loads and encodes the three benchmark datasets into a common shape.

Each loader returns (X, y, protected_feature, categorical_columns, display_name):
  X: DataFrame, fully numeric (label-encoded categoricals), NaN PRESERVED.
  y: Series of {0, 1}, 1 = adverse outcome (default / delinquency / denial).
  protected_feature: column name usable for C4 bias-detection analysis, or None.
  categorical_columns: names of columns that were label-encoded (treated as
    categorical by the explainer neighborhood construction).

X is intentionally NOT median-imputed here. The paper's own training code
feeds raw, NaN-preserving data to LightGBM (which handles missing values
natively via learned split-direction, per-split) and only ever imputes a
separate copy for LIME, which cannot accept NaN -- and critically, computes
that imputation's fill values from the TRAINING split only, applying them
unchanged to validation/test, exactly like train_models.py's split does
downstream. A prior version of this loader median-imputed X globally, across
the full dataset, before any split existed -- meaning LightGBM was trained on
a materially different (and leakier) dataset than the paper's own code
produces. shap.TreeExplainer and SHAPPlusExplainer both handle NaN natively
(the latter via its own internal per-feature median fallback in
shap_plus/explainer.py), so neither needs a filled copy; only LIME does, and
that fill now happens downstream in benchmark_xai.py, train-only, right
before LIME is invoked.
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

    # Matches the paper's own comment verbatim: "Drop only 100%-missing
    # columns (not the common 40%-threshold)." A prior version of this loader
    # used an 80% threshold instead -- currently zero-impact on this exact
    # CSV (no column exceeds 80% missing), but not what the paper's code
    # actually says, so corrected for fidelity regardless.
    missing_frac = X.isna().mean()
    keep = missing_frac[missing_frac < 1.0].index.tolist()
    X = X[keep]

    categorical_columns = [
        c for c in X.columns
        if X[c].dtype == object or pd.api.types.is_string_dtype(X[c])
    ]
    for col in categorical_columns:
        X[col] = _label_encode(X[col])

    X = X.apply(pd.to_numeric, errors="coerce")

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


def _load_hmda(filename: str, display_name: str) -> tuple[pd.DataFrame, pd.Series, str, list[str], str]:
    path = DATA_DIR / filename
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

    protected_feature = "derived_sex"
    return X, y, protected_feature, categorical_columns, display_name


def load_hmda_vt() -> tuple[pd.DataFrame, pd.Series, str, list[str], str]:
    return _load_hmda("hmda_vt_sample.csv", "HMDA Vermont 2023 (approval/denial)")


def load_hmda_nh() -> tuple[pd.DataFrame, pd.Series, str, list[str], str]:
    return _load_hmda("hmda_nh_sample.csv", "HMDA New Hampshire 2023 (approval/denial)")


# Datasets used to develop and tune SHAP PLUS: any hyperparameter search may
# look at metrics computed on these (subject to the tune/report split within
# each -- see research/select_hyperparameters.py).
LOADERS = {
    "home_credit": load_home_credit,
    "hmeq": load_hmeq,
    "hmda_vt": load_hmda_vt,
}

# Held out completely from hyperparameter selection. This dataset is loaded
# and scored exactly once, after hyperparameters are already frozen, purely
# to check that a configuration chosen on the datasets above generalizes to a
# state (different regional economics, different lender mix, unseen at
# tuning time) it has never influenced in any way.
HOLDOUT_LOADERS = {
    "hmda_nh": load_hmda_nh,
}

ALL_LOADERS = {**LOADERS, **HOLDOUT_LOADERS}


if __name__ == "__main__":
    for key, loader in ALL_LOADERS.items():
        tag = "holdout" if key in HOLDOUT_LOADERS else "dev"
        X, y, protected, cats, name = loader()
        print(f"{key:12s} [{tag:7s}] {name:40s} rows={len(X):>7} cols={X.shape[1]:>3} "
              f"positive_rate={y.mean():.4f} protected={protected} n_cat={len(cats)} "
              f"raw_nulls={int(X.isna().sum().sum())}")
