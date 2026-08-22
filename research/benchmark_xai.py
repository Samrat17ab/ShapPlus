"""Empirical CSF benchmark: real SHAP vs real LIME vs SHAP PLUS hybrid.

This is the actual validation the conference paper's own "future work" section
calls for: run genuine SHAP (shap.TreeExplainer) and genuine LIME
(lime.lime_tabular) -- not just SHAP PLUS's internal claims -- against a real
LIME baseline on held-out data from three datasets, and score all three
methods with the paper's own CSF formulas (Table II / Table IV).

Every number below is measured, not asserted. C3b (human comprehensibility)
is intentionally NOT scored here -- it requires a human study, and assigning
it a number without one would repeat the exact flaw this benchmark exists to
fix.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from shap_plus import SHAPPlusExplainer
from shap_plus._utils import jaccard, structural_complexity
from shap_plus.evaluation import (
    _score_bias_gap,
    _score_complexity,
    _score_coverage,
    _score_disparity,
    _score_fidelity,
    _score_jaccard,
)

ARTIFACTS = Path(__file__).parent / "artifacts"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TOP_K = 10
LIME_NUM_SAMPLES = 1000
RIDGE_ALPHA_LIME = None  # use lime defaults

IMMUTABLE = {
    "home_credit": {"CODE_GENDER", "DAYS_BIRTH"},
    "hmeq": set(),
    "hmda_vt": {"derived_sex", "derived_race", "derived_ethnicity", "applicant_age"},
    "hmda_nh": {"derived_sex", "derived_race", "derived_ethnicity", "applicant_age"},
}
ACTIONABLE = {
    "home_credit": {"AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"},
    "hmeq": {"LOAN", "DEBTINC", "CLNO", "DELINQ"},
    "hmda_vt": {"loan_amount", "debt_to_income_ratio"},
    "hmda_nh": {"loan_amount", "debt_to_income_ratio"},
}


def load_artifacts(key: str):
    dataset_dir = ARTIFACTS / key
    meta = json.loads((dataset_dir / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(dataset_dir / "model.txt"))
    X_train = pd.read_pickle(dataset_dir / "X_train.pkl")
    X_test = pd.read_pickle(dataset_dir / "X_test.pkl")
    y_test = pd.read_pickle(dataset_dir / "y_test.pkl")
    return meta, booster, X_train, X_test, y_test


def make_predict_proba(booster: lgb.Booster):
    def fn(X):
        p = booster.predict(np.asarray(X, dtype=float))
        return np.column_stack([1.0 - p, p])
    return fn


# ---------------------------------------------------------------------------
# Plain SHAP baseline
# ---------------------------------------------------------------------------

def shap_top10_conditions(feature_names, shap_row, k=TOP_K):
    order = np.argsort(-np.abs(shap_row))[:k]
    # Bare feature-name labels, no inequality operators: this is what raw
    # SHAP actually surfaces (a scalar attribution per feature), not a rule.
    return [str(feature_names[i]) for i in order], order


def run_plain_shap(booster, feature_names, sample_frame, protected_feature):
    explainer = shap.TreeExplainer(booster)
    out = explainer(sample_frame)
    values = np.asarray(out.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]

    coverages, complexities = [], []
    protected_idx = feature_names.index(protected_feature) if protected_feature else None
    protected_attr = []
    for row in values:
        total = float(np.abs(row).sum())
        conditions, order = shap_top10_conditions(feature_names, row)
        top_abs = float(np.abs(row[order]).sum())
        coverages.append(1.0 if total <= 1e-15 else top_abs / total)
        complexities.append(structural_complexity(conditions))
        if protected_idx is not None:
            protected_attr.append(float(row[protected_idx]))

    return {
        "coverage_mean": float(np.mean(coverages)),
        "complexity_mean": float(np.mean(complexities)),
        "coverage_values": coverages,
        "complexity_values": complexities,
        "raw_values": values,
        "protected_attribution": protected_attr,
        "presence_rate": 1.0 if protected_idx is not None else None,
    }


def shap_consistency(booster, sample_frame):
    """SHAP is a deterministic algorithm; verify (not assume) J == 1.0."""
    explainer = shap.TreeExplainer(booster)
    run_a = np.asarray(explainer(sample_frame).values, dtype=float)
    run_b = np.asarray(explainer(sample_frame).values, dtype=float)
    scores = []
    for a, b in zip(run_a, run_b):
        set_a = set(np.argsort(-np.abs(a))[:TOP_K].tolist())
        set_b = set(np.argsort(-np.abs(b))[:TOP_K].tolist())
        scores.append(jaccard(set_a, set_b))
    exact_match = bool(np.allclose(run_a, run_b))
    return float(np.mean(scores)), exact_match


# ---------------------------------------------------------------------------
# Plain LIME baseline
# ---------------------------------------------------------------------------

def run_plain_lime(booster, X_train, feature_names, categorical_columns, sample_frame, protected_feature, seed=None):
    cat_idx = [feature_names.index(c) for c in categorical_columns]
    predict_fn = make_predict_proba(booster)
    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        categorical_features=cat_idx,
        class_names=["favourable", "adverse"],
        discretize_continuous=True,
        mode="classification",
        random_state=seed,
    )
    fidelities, complexities = [], []
    protected_attr = []
    present_count = 0
    protected_idx = feature_names.index(protected_feature) if protected_feature is not None else None
    for _, row in sample_frame.iterrows():
        exp = lime_explainer.explain_instance(
            row.values, predict_fn, num_features=TOP_K, num_samples=LIME_NUM_SAMPLES
        )
        conditions = [cond for cond, _ in exp.as_list()]
        fidelities.append(float(exp.score))
        complexities.append(structural_complexity(conditions))
        if protected_idx is not None:
            local_exp = dict(exp.local_exp[1])  # {feature_index: weight}
            if protected_idx in local_exp:
                protected_attr.append(float(local_exp[protected_idx]))
                present_count += 1
            else:
                protected_attr.append(float("nan"))
    presence_rate = (
        present_count / len(sample_frame) if protected_feature is not None else None
    )
    return {
        "fidelity_mean": float(np.mean(fidelities)),
        "complexity_mean": float(np.mean(complexities)),
        "fidelity_values": fidelities,
        "complexity_values": complexities,
        "protected_attribution": protected_attr,
        "presence_rate": presence_rate,
    }


def lime_consistency(booster, X_train, feature_names, categorical_columns, sample_frame, fixed_seed):
    cat_idx = [feature_names.index(c) for c in categorical_columns]
    predict_fn = make_predict_proba(booster)

    def build_explainer(seed):
        return LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            categorical_features=cat_idx,
            class_names=["favourable", "adverse"],
            discretize_continuous=True,
            mode="classification",
            random_state=seed,
        )

    seed_a = 42 if fixed_seed else None
    seed_b = 42 if fixed_seed else None
    exp_a = build_explainer(seed_a)
    exp_b = build_explainer(seed_b)
    scores = []
    exact_flags = []
    for _, row in sample_frame.iterrows():
        e1 = exp_a.explain_instance(row.values, predict_fn, num_features=TOP_K, num_samples=LIME_NUM_SAMPLES)
        e2 = exp_b.explain_instance(row.values, predict_fn, num_features=TOP_K, num_samples=LIME_NUM_SAMPLES)
        set1 = {idx for idx, _ in e1.local_exp[1]}
        set2 = {idx for idx, _ in e2.local_exp[1]}
        scores.append(jaccard(set1, set2))
        w1 = dict(e1.local_exp[1])
        w2 = dict(e2.local_exp[1])
        exact_flags.append(
            set1 == set2 and all(abs(w1[i] - w2[i]) < 1e-9 for i in set1)
        )
    return float(np.mean(scores)), bool(np.all(exact_flags))


# ---------------------------------------------------------------------------
# SHAP PLUS hybrid
# ---------------------------------------------------------------------------

DEFAULT_SHAP_PLUS_HYPERPARAMS = {
    "max_rule_terms": 5,
    "neighborhood_size": 512,
    "fidelity_threshold": 0.75,
    # min_leaf_weight_fraction, quantile_grid_size, objective_weights left at
    # the SHAPPlusExplainer class defaults unless overridden by a caller that
    # has actually selected them via tune_hyperparameters.py.
}


def build_shap_plus(
    booster, X_train, feature_names, categorical_columns, protected_feature, dataset_key,
    hyperparams: dict | None = None,
):
    params = {**DEFAULT_SHAP_PLUS_HYPERPARAMS, **(hyperparams or {})}
    return SHAPPlusExplainer(
        booster,
        X_train,
        feature_names=feature_names,
        positive_class=1,
        positive_class_is_adverse=True,
        decision_threshold=0.5,
        top_k=min(TOP_K, len(feature_names)),
        immutable_features=IMMUTABLE.get(dataset_key, set()),
        actionable_features=ACTIONABLE.get(dataset_key, set()),
        **params,
        model_version=f"{dataset_key}-lightgbm-v1",
    )


def run_shap_plus(explainer, sample_frame, protected_feature, feature_names):
    explanations = explainer.explain(sample_frame, include_recourse=False)
    coverages, fidelities, complexities = [], [], []
    visible_attr, visible_present = [], 0
    full_attr = []
    fallback_flags = []
    for exp in explanations:
        coverages.append(exp.coverage)
        fidelities.append(exp.fidelity)
        complexities.append(structural_complexity([t.condition for t in exp.terms]))
        fallback_flags.append(exp.fallback_used)
        if protected_feature is not None:
            full_attr.append(exp.full_shap_values[protected_feature])
            visible_term = next((t for t in exp.terms if t.feature == protected_feature), None)
            if visible_term is not None:
                visible_attr.append(visible_term.shap_value)
                visible_present += 1
            else:
                visible_attr.append(float("nan"))
    n = len(explanations)
    return {
        "coverage_mean": float(np.mean(coverages)),
        "fidelity_mean": float(np.mean(fidelities)),
        "complexity_mean": float(np.mean(complexities)),
        "coverage_values": coverages,
        "fidelity_values": fidelities,
        "complexity_values": complexities,
        "fallback_rate": float(np.mean(fallback_flags)),
        "protected_attribution_full": full_attr,
        "protected_attribution_visible": visible_attr,
        "presence_rate_full": 1.0 if protected_feature is not None else None,
        "presence_rate_visible": (
            visible_present / n if protected_feature is not None else None
        ),
    }


def shap_plus_consistency(explainer, sample_frame):
    run_a = explainer.explain(sample_frame, include_recourse=False)
    run_b = explainer.explain(sample_frame, include_recourse=False)
    scores = []
    exact_flags = []
    for a, b in zip(run_a, run_b):
        scores.append(jaccard(set(a.features), set(b.features)))
        exact_flags.append(a.local_rule == b.local_rule and a.features == b.features)
    return float(np.mean(scores)), bool(np.all(exact_flags))


# ---------------------------------------------------------------------------
# Bias gap helper (two-group)
# ---------------------------------------------------------------------------

def bias_gap_two_group(values, group_labels):
    """Signed-mean attribution gap across groups. NaN entries (feature not
    surfaced in that instance's explanation) are excluded from that group's
    mean, matching the paper's own treatment of LIME's partial coverage."""
    values = np.asarray(values, dtype=float)
    groups = pd.Series(group_labels).reset_index(drop=True).to_numpy()
    means = {}
    counts = {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        group_values = values[mask]
        valid = group_values[~np.isnan(group_values)]
        counts[str(g)] = int(len(valid))
        if len(valid) == 0:
            continue
        means[str(g)] = float(np.mean(valid))
    if len(means) < 2:
        return None, means, counts
    vals = list(means.values())
    return float(max(vals) - min(vals)), means, counts


# ---------------------------------------------------------------------------
# C6 structural checklist (fixed rubric, not per-instance)
# ---------------------------------------------------------------------------

C6_CHECKLIST = {
    "shap": {"direction": True, "magnitude": True, "actionable": False, "traceability": True},
    "lime": {"direction": True, "magnitude": True, "actionable": True, "traceability": False},
    "shap_plus": {"direction": True, "magnitude": True, "actionable": True, "traceability": True},
}


def c6_score(method: str) -> float:
    checklist = C6_CHECKLIST[method]
    return 1.0 + sum(1.0 for v in checklist.values() if v)


def c7_score(exact_reproducible_no_state: bool, exact_reproducible_with_state: bool) -> float:
    if exact_reproducible_no_state:
        return 5.0
    if exact_reproducible_with_state:
        return 2.0
    return 1.0


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_dataset(
    key: str,
    n_explain: int,
    n_consistency: int,
    n_consistency_runs: int,
    seed: int = 42,
    sample_idx: np.ndarray | None = None,
    consistency_idx: np.ndarray | None = None,
    hyperparams: dict | None = None,
) -> dict:
    """
    If sample_idx/consistency_idx are given, those exact positional indices
    into X_test are used (e.g. a report-only pool that a hyperparameter
    search never looked at). Otherwise falls back to random sampling, for
    standalone/exploratory use.
    """
    print(f"\n=== {key} ===")
    t0 = time.time()
    meta, booster, X_train, X_test, y_test = load_artifacts(key)
    feature_names = meta["feature_names"]
    categorical_columns = meta["categorical_columns"]
    protected_feature = meta["protected_feature"]

    rng = np.random.default_rng(seed)
    if sample_idx is None:
        idx = rng.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)
    else:
        idx = np.asarray(sample_idx)
    sample_frame = X_test.iloc[idx].reset_index(drop=True)

    if consistency_idx is None:
        idx_c = rng.choice(len(X_test), size=min(n_consistency, len(X_test)), replace=False)
    else:
        idx_c = np.asarray(consistency_idx)
    consistency_frame = X_test.iloc[idx_c].reset_index(drop=True)

    result = {"dataset": key, "name": meta["name"], "model_metrics": meta["metrics"]}

    # ---- Plain SHAP ----
    print("  SHAP ...", end=" ", flush=True)
    shap_res = run_plain_shap(booster, feature_names, sample_frame, protected_feature)
    shap_cons, shap_exact = shap_consistency(booster, consistency_frame)
    result["shap"] = {
        "coverage": shap_res["coverage_mean"],
        "complexity": shap_res["complexity_mean"],
        "coverage_values": shap_res["coverage_values"],
        "complexity_values": shap_res["complexity_values"],
        "consistency": shap_cons,
        "exact_reproducible": shap_exact,
        "presence_rate": shap_res["presence_rate"],
        "protected_attribution": shap_res["protected_attribution"],
    }
    print(f"coverage={shap_res['coverage_mean']:.3f} consistency={shap_cons:.3f} ({time.time()-t0:.1f}s)")

    # ---- Plain LIME ----
    print("  LIME (stochastic) ...", end=" ", flush=True)
    t1 = time.time()
    lime_res = run_plain_lime(
        booster, X_train, feature_names, categorical_columns, sample_frame, protected_feature, seed=None
    )
    lime_cons_stoch, lime_exact_stoch = lime_consistency(
        booster, X_train, feature_names, categorical_columns, consistency_frame, fixed_seed=False
    )
    print(f"fidelity={lime_res['fidelity_mean']:.3f} consistency={lime_cons_stoch:.3f} ({time.time()-t1:.1f}s)")
    print("  LIME (fixed-seed) ...", end=" ", flush=True)
    t2 = time.time()
    lime_cons_fixed, lime_exact_fixed = lime_consistency(
        booster, X_train, feature_names, categorical_columns, consistency_frame, fixed_seed=True
    )
    print(f"consistency={lime_cons_fixed:.3f} exact={lime_exact_fixed} ({time.time()-t2:.1f}s)")
    result["lime"] = {
        "fidelity": lime_res["fidelity_mean"],
        "complexity": lime_res["complexity_mean"],
        "fidelity_values": lime_res["fidelity_values"],
        "complexity_values": lime_res["complexity_values"],
        "consistency_stochastic": lime_cons_stoch,
        "consistency_fixed_seed": lime_cons_fixed,
        "exact_reproducible_fixed_seed": lime_exact_fixed,
        "presence_rate": lime_res["presence_rate"],
        "protected_attribution": lime_res["protected_attribution"],
    }

    # ---- SHAP PLUS ----
    print("  SHAP PLUS ...", end=" ", flush=True)
    t3 = time.time()
    sp_explainer = build_shap_plus(
        booster, X_train, feature_names, categorical_columns, protected_feature, key,
        hyperparams=hyperparams,
    )
    sp_res = run_shap_plus(sp_explainer, sample_frame, protected_feature, feature_names)
    sp_cons, sp_exact = shap_plus_consistency(sp_explainer, consistency_frame)
    result["shap_plus"] = {
        "coverage": sp_res["coverage_mean"],
        "fidelity": sp_res["fidelity_mean"],
        "complexity": sp_res["complexity_mean"],
        "coverage_values": sp_res["coverage_values"],
        "fidelity_values": sp_res["fidelity_values"],
        "complexity_values": sp_res["complexity_values"],
        "fallback_rate": sp_res["fallback_rate"],
        "consistency": sp_cons,
        "exact_reproducible": sp_exact,
        "presence_rate_full": sp_res["presence_rate_full"],
        "presence_rate_visible": sp_res["presence_rate_visible"],
        "protected_attribution_full": sp_res["protected_attribution_full"],
        "protected_attribution_visible": sp_res["protected_attribution_visible"],
    }
    print(
        f"coverage={sp_res['coverage_mean']:.3f} fidelity={sp_res['fidelity_mean']:.3f} "
        f"consistency={sp_cons:.3f} fallback_rate={sp_res['fallback_rate']:.3f} ({time.time()-t3:.1f}s)"
    )

    # ---- C4 bias detection (only if protected feature exists) ----
    if protected_feature is not None:
        group_labels_full = X_test.iloc[idx][protected_feature].to_numpy()

        gap_shap, means_shap, n_shap = bias_gap_two_group(
            shap_res["protected_attribution"], group_labels_full
        )
        gap_lime, means_lime, n_lime = bias_gap_two_group(
            lime_res["protected_attribution"], group_labels_full
        )
        gap_sp_full, means_sp_full, n_sp_full = bias_gap_two_group(
            sp_res["protected_attribution_full"], group_labels_full
        )
        gap_sp_visible, means_sp_visible, n_sp_visible = bias_gap_two_group(
            sp_res["protected_attribution_visible"], group_labels_full
        )

        result["c4_bias"] = {
            "shap_gap": gap_shap,
            "shap_group_means": means_shap,
            "shap_group_n": n_shap,
            "shap_presence_rate": 1.0,
            "lime_gap": gap_lime,
            "lime_group_means": means_lime,
            "lime_group_n": n_lime,
            "lime_presence_rate": lime_res["presence_rate"],
            "shap_plus_full_gap": gap_sp_full,
            "shap_plus_full_group_means": means_sp_full,
            "shap_plus_full_presence_rate": sp_res["presence_rate_full"],
            "shap_plus_visible_gap": gap_sp_visible,
            "shap_plus_visible_group_means": means_sp_visible,
            "shap_plus_visible_presence_rate": sp_res["presence_rate_visible"],
        }
        print(
            f"  C4 bias gap ({protected_feature}): SHAP={gap_shap} LIME={gap_lime} "
            f"SHAP_PLUS(full)={gap_sp_full} SHAP_PLUS(visible)={gap_sp_visible}"
        )

    # ---- C5 fairness transparency (a model property, ties across methods) ----
    if protected_feature is not None:
        model_predictions = np.asarray(booster.predict(sample_frame))
        favourable = (
            model_predictions < sp_explainer.decision_threshold
            if sp_explainer.positive_class_is_adverse
            else model_predictions >= sp_explainer.decision_threshold
        )
        _, approval_rates, _ = bias_gap_two_group(
            np.where(favourable, 1.0, 0.0), group_labels_full
        )
        if len(approval_rates) >= 2:
            result["c5_disparity"] = float(max(approval_rates.values()) - min(approval_rates.values()))
            result["c5_approval_rates"] = approval_rates
            print(f"  C5 approval-rate disparity ({protected_feature}): {result['c5_disparity']:.4f}")

    result["protected_feature"] = protected_feature
    result["sample_idx"] = idx.tolist()
    result["consistency_idx"] = idx_c.tolist()
    result["hyperparams"] = hyperparams or dict(DEFAULT_SHAP_PLUS_HYPERPARAMS)
    print(f"  dataset total: {time.time()-t0:.1f}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-explain", type=int, default=150)
    parser.add_argument("--n-consistency", type=int, default=25)
    parser.add_argument("--n-consistency-runs", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=["home_credit", "hmeq", "hmda_vt"])
    args = parser.parse_args()

    all_results = {}
    for key in args.datasets:
        all_results[key] = run_dataset(
            key, args.n_explain, args.n_consistency, args.n_consistency_runs
        )

    out_path = RESULTS_DIR / "benchmark_raw.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved raw results to {out_path}")
