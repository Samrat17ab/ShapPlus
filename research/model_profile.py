"""Standalone empirical profile of SHAP PLUS -- no comparison to SHAP or
LIME, no discretized 1-5 CSF tiers. Just what the model actually does,
per dataset, in full distributional detail (mean/median/std/min/max, not
just a single averaged number), using the same frozen hyperparameters
selected by tune_hyperparameters.py.

This exists because the CSF comparison table looks suspiciously uniform
across datasets for real, explainable reasons (C2/C4/C7 are mathematically
guaranteed ties since SHAP PLUS is exactly deterministic and shares SHAP's
attribution vector; C6 is a method-level property, not a per-dataset
measurement) -- but folding everything into discretized tiers and a
three-way comparison obscures the actual continuous, per-instance variation
underneath. This reports that variation directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_xai import build_shap_plus, load_artifacts
from prepare_data import ALL_LOADERS
from shap_plus._utils import jaccard, structural_complexity

RESULTS_DIR = Path(__file__).parent / "results"
N_INSTANCES = 300
N_CONSISTENCY = 40
SEED = 99  # independent of tune (20260821) and report (42) seeds


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def profile_dataset(key: str) -> dict:
    with open(RESULTS_DIR / "selected_hyperparameters.json") as f:
        hyperparams = json.load(f)["selected_hyperparams"]

    meta, booster, X_train, X_test, y_test = load_artifacts(key)
    feature_names = meta["feature_names"]
    protected_feature = meta["protected_feature"]

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_test), size=min(N_INSTANCES, len(X_test)), replace=False)
    sample = X_test.iloc[idx].reset_index(drop=True)
    group_labels = X_test.iloc[idx][protected_feature].to_numpy() if protected_feature else None

    explainer = build_shap_plus(
        booster, X_train, feature_names, meta["categorical_columns"],
        protected_feature, key, hyperparams=hyperparams,
    )
    explanations = explainer.explain(sample, include_recourse=bool(explainer.actionable_features))

    fidelity, complexity, coverage, sign_consistency, fallback, n_terms, predictions = (
        [], [], [], [], [], [], []
    )
    full_attr, visible_attr = [], []
    recourse_achieved, recourse_cost = [], []
    for exp in explanations:
        fidelity.append(exp.fidelity)
        complexity.append(structural_complexity([t.condition for t in exp.terms]))
        coverage.append(exp.coverage)
        sign_consistency.append(exp.sign_consistency)
        fallback.append(bool(exp.fallback_used))
        n_terms.append(len(exp.terms))
        predictions.append(exp.prediction)
        if protected_feature:
            full_attr.append(exp.full_shap_values[protected_feature])
            visible_term = next((t for t in exp.terms if t.feature == protected_feature), None)
            visible_attr.append(visible_term.shap_value if visible_term is not None else float("nan"))
        if exp.recourse is not None and not exp.recourse.achieved and exp.recourse.steps:
            pass  # already-favourable rows excluded below
        if exp.recourse is not None:
            recourse_achieved.append(bool(exp.recourse.achieved))
            if exp.recourse.steps:
                recourse_cost.append(sum(s.normalized_cost for s in exp.recourse.steps))

    # Consistency: repeated explain() calls on the same instances, real Jaccard.
    cons_idx = rng.choice(len(X_test), size=min(N_CONSISTENCY, len(X_test)), replace=False)
    cons_sample = X_test.iloc[cons_idx].reset_index(drop=True)
    run_a = explainer.explain(cons_sample, include_recourse=False)
    run_b = explainer.explain(cons_sample, include_recourse=False)
    jaccards = [jaccard(set(a.features), set(b.features)) for a, b in zip(run_a, run_b)]
    exact_matches = [a.local_rule == b.local_rule for a, b in zip(run_a, run_b)]

    result = {
        "dataset": key,
        "name": meta["name"],
        "n_instances": len(sample),
        "hyperparams": hyperparams,
        "fidelity": summarize(fidelity),
        "complexity": summarize(complexity),
        "coverage": summarize(coverage),
        "sign_consistency": summarize(sign_consistency),
        "n_terms": summarize(n_terms),
        "prediction": summarize(predictions),
        "fallback_rate": float(np.mean(fallback)),
        "consistency_jaccard": summarize(jaccards),
        "consistency_exact_match_rate": float(np.mean(exact_matches)),
    }
    if protected_feature:
        full_arr = np.asarray(full_attr)
        vis_arr = np.asarray(visible_attr)
        vis_present = ~np.isnan(vis_arr)
        groups = pd.Series(group_labels)
        full_group_means = {str(g): float(full_arr[(groups == g).to_numpy()].mean()) for g in groups.unique()}
        result["protected_feature"] = protected_feature
        result["audit_vector_presence_rate"] = 1.0
        result["visible_rule_presence_rate"] = float(np.mean(vis_present))
        result["audit_vector_group_means"] = full_group_means
        result["audit_vector_gap"] = float(max(full_group_means.values()) - min(full_group_means.values()))

        # C5 (Fairness Transparency): approval-rate disparity across the
        # protected feature's groups, from the model's own predictions --
        # a property of the LightGBM model, not of the XAI method, exactly
        # as the paper's own C5 caveat says. Computed here from real
        # decisions, not skipped.
        preds_arr = np.asarray(predictions)
        favourable = (
            preds_arr < explainer.decision_threshold
            if explainer.positive_class_is_adverse
            else preds_arr >= explainer.decision_threshold
        )
        approval_rates = {
            str(g): float(favourable[(groups == g).to_numpy()].mean())
            for g in groups.unique()
            if (groups == g).sum() > 0
        }
        result["approval_rates"] = approval_rates
        result["approval_rate_disparity"] = float(max(approval_rates.values()) - min(approval_rates.values()))
    if recourse_achieved:
        result["recourse_achieved_rate"] = float(np.mean(recourse_achieved))
        if recourse_cost:
            result["recourse_normalized_cost"] = summarize(recourse_cost)
    return result


def main() -> None:
    all_results = {}
    for key in ALL_LOADERS:
        print(f"\n=== {key} ===")
        result = profile_dataset(key)
        all_results[key] = result
        f = result["fidelity"]
        c = result["complexity"]
        print(
            f"  fidelity R^2   mean={f['mean']:.3f} median={f['median']:.3f} std={f['std']:.3f} "
            f"[{f['min']:.3f}, {f['max']:.3f}]  (n={f['n']})"
        )
        c_ = result["complexity"]
        print(
            f"  complexity SC  mean={c_['mean']:.3f} median={c_['median']:.3f} std={c_['std']:.3f} "
            f"[{c_['min']:.3f}, {c_['max']:.3f}]"
        )
        cov = result["coverage"]
        print(f"  coverage       mean={cov['mean']:.3f} median={cov['median']:.3f} std={cov['std']:.3f}")
        nt = result["n_terms"]
        print(f"  rule length    mean={nt['mean']:.2f} terms  [{nt['min']:.0f}, {nt['max']:.0f}]")
        print(f"  fallback rate  {result['fallback_rate']:.1%}")
        sc = result["sign_consistency"]
        print(f"  sign consist.  mean={sc['mean']:.3f} std={sc['std']:.3f}")
        cj = result["consistency_jaccard"]
        print(
            f"  consistency J  mean={cj['mean']:.4f} std={cj['std']:.4f}  "
            f"exact_match_rate={result['consistency_exact_match_rate']:.1%}"
        )
        if "protected_feature" in result:
            print(
                f"  bias audit gap on {result['protected_feature']}: {result['audit_vector_gap']:.4f} "
                f"(100% presence)  visible-rule presence: {result['visible_rule_presence_rate']:.1%}"
            )
        if "recourse_achieved_rate" in result:
            print(f"  recourse achieved: {result['recourse_achieved_rate']:.1%} of non-favourable instances")

    out_path = RESULTS_DIR / "model_profile.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
