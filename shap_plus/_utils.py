"""Internal deterministic numerical and serialization helpers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


def ensure_frame(
    values: pd.DataFrame | pd.Series | np.ndarray | Sequence[float],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        missing = [name for name in feature_names if name not in values.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing[:10]}")
        return values.loc[:, list(feature_names)].copy()
    if isinstance(values, pd.Series):
        return pd.DataFrame([values.loc[list(feature_names)].to_dict()])
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(feature_names):
        raise ValueError(
            f"Expected shape (n, {len(feature_names)}); received {array.shape}."
        )
    return pd.DataFrame(array, columns=list(feature_names))


def prediction_vector(
    predict_fn: Callable[[pd.DataFrame], Any],
    frame: pd.DataFrame,
    positive_class: int,
) -> np.ndarray:
    raw = np.asarray(predict_fn(frame))
    if raw.ndim == 1:
        return raw.astype(float)
    if raw.ndim == 2:
        if raw.shape[1] == 1:
            return raw[:, 0].astype(float)
        if not 0 <= positive_class < raw.shape[1]:
            raise ValueError(
                f"positive_class={positive_class} is invalid for prediction shape {raw.shape}."
            )
        return raw[:, positive_class].astype(float)
    raise ValueError(f"Prediction function returned unsupported shape {raw.shape}.")


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(weights.sum())
    if denom <= 0:
        return 0.0
    mean = float(np.sum(weights * y_true) / denom)
    ss_res = float(np.sum(weights * (y_true - y_pred) ** 2))
    ss_tot = float(np.sum(weights * (y_true - mean) ** 2))
    if ss_tot <= 1e-15:
        return 1.0 if ss_res <= 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def structural_complexity(conditions: Sequence[str], top_k_reference: int = 10) -> float:
    names = [str(item) for item in conditions]
    count = len(names)
    conditional = sum(
        1 for name in names if any(operator in name for operator in (">", "<", "="))
    )
    mean_length = float(np.mean([len(name) for name in names])) if names else 0.0
    return float(
        0.4 * (count / top_k_reference)
        + 0.4 * (conditional / top_k_reference)
        + 0.2 * (mean_length / 20.0)
    )


def stable_json_hash(payload: Any, prefix: str = "") -> str:
    normalized = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def model_fingerprint(model: Any) -> str:
    for accessor in ("model_to_string", "get_params"):
        attribute = getattr(model, accessor, None)
        if callable(attribute):
            try:
                return stable_json_hash(attribute(), prefix="sha256:")
            except Exception:
                pass
    return stable_json_hash(repr(model), prefix="sha256:")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return default if not math.isfinite(result) else result
    except (TypeError, ValueError):
        return default


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)

