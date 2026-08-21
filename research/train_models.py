"""Trains a LightGBM classifier per dataset with a stratified 64/16/20 split,
mirroring the conference paper's protocol, and saves model + split indices."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from prepare_data import ALL_LOADERS

OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)


def split_64_16_20(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.36, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=20 / 36, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_one(key: str) -> dict:
    X, y, protected, categorical_columns, name = ALL_LOADERS[key]()
    X_train, X_val, X_test, y_train, y_val, y_test = split_64_16_20(X, y)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    proba_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    pred_label = (proba_test >= 0.5).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, proba_test)),
        "precision": float(precision_score(y_test, pred_label, zero_division=0)),
        "recall": float(recall_score(y_test, pred_label, zero_division=0)),
        "f1": float(f1_score(y_test, pred_label, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "best_iteration": int(booster.best_iteration),
        "positive_rate": float(y.mean()),
    }

    dataset_dir = OUT_DIR / key
    dataset_dir.mkdir(exist_ok=True)
    booster.save_model(str(dataset_dir / "model.txt"))
    X_train.to_pickle(dataset_dir / "X_train.pkl")
    X_test.to_pickle(dataset_dir / "X_test.pkl")
    y_train.to_pickle(dataset_dir / "y_train.pkl")
    y_test.to_pickle(dataset_dir / "y_test.pkl")
    meta = {
        "name": name,
        "protected_feature": protected,
        "categorical_columns": categorical_columns,
        "feature_names": list(X.columns),
        "metrics": metrics,
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    all_meta = {}
    for key in ALL_LOADERS:
        print(f"Training LightGBM on {key} ...")
        meta = train_one(key)
        all_meta[key] = meta
        m = meta["metrics"]
        print(
            f"  {meta['name']:35s} AUC={m['auc_roc']:.3f} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"(train={m['n_train']}, test={m['n_test']})"
        )
    (OUT_DIR / "all_meta.json").write_text(json.dumps(all_meta, indent=2))
