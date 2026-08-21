"""Selects SHAP PLUS's tree-surrogate hyperparameters on a TUNE pool that is
disjoint from the REPORT pool used for the final published numbers, and
disjoint from the holdout dataset entirely.

This exists to fix a real methodology flaw: an earlier round of this project
picked hyperparameters by watching fidelity/complexity on the exact instances
that were then reported as results -- tuning on the test set. The fix is not
"tune less carefully," it's "tune on data the final numbers never touch."

The search also does not optimize any one dataset's score. The objective is
the worst-case (minimum) combined C1/complexity score across all three
development datasets jointly, so a configuration cannot win by overfitting to
whichever dataset is easiest -- it has to hold up everywhere, which is the
whole point of testing generalization rather than memorization.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from benchmark_xai import build_shap_plus, load_artifacts, run_shap_plus
from prepare_data import LOADERS
from shap_plus.evaluation import _score_complexity, _score_coverage, _score_fidelity

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TUNE_N = 40
SEED = 20260821  # distinct from the report/holdout seed, deliberately

GRID = {
    "min_leaf_weight_fraction": [0.02, 0.01, 0.006, 0.003, 0.0015],
    "quantile_grid_size": [19, 39],
    "objective_weights": [
        (0.55, 0.15, 0.10, 0.20),
        (0.40, 0.10, 0.30, 0.20),
        (0.35, 0.05, 0.35, 0.25),
    ],
}


def tune_indices(dataset_key: str, n_test: int) -> np.ndarray:
    """Deterministic tune-pool indices for a dataset, seeded independently of
    the report/holdout sampling in final_validation.py so the two pools are
    guaranteed disjoint (report/holdout draws start from a *different*
    permutation seed and additionally exclude these exact indices)."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_test)
    return order[:TUNE_N]


def dataset_score(coverage: float, fidelity: float, complexity: float) -> float:
    c1 = min(_score_coverage(coverage), _score_fidelity(fidelity))
    c3a = _score_complexity(complexity)
    return float(np.mean([c1, c3a]))


def evaluate_config(hyperparams: dict, tune_pools: dict) -> dict:
    per_dataset = {}
    for key, (meta, booster, X_train, sample_frame, protected_feature) in tune_pools.items():
        explainer = build_shap_plus(
            booster, X_train, meta["feature_names"], meta["categorical_columns"],
            protected_feature, key, hyperparams=hyperparams,
        )
        res = run_shap_plus(explainer, sample_frame, protected_feature, meta["feature_names"])
        score = dataset_score(res["coverage_mean"], res["fidelity_mean"], res["complexity_mean"])
        per_dataset[key] = {
            "coverage": res["coverage_mean"],
            "fidelity": res["fidelity_mean"],
            "complexity": res["complexity_mean"],
            "score": score,
        }
    scores = [v["score"] for v in per_dataset.values()]
    return {
        "hyperparams": hyperparams,
        "per_dataset": per_dataset,
        "min_score": float(np.min(scores)),
        "mean_score": float(np.mean(scores)),
    }


def main() -> None:
    print("Loading tune pools (disjoint from report/holdout instances) ...")
    tune_pools = {}
    for key in LOADERS:
        meta, booster, X_train, X_test, y_test = load_artifacts(key)
        idx = tune_indices(key, len(X_test))
        sample_frame = X_test.iloc[idx].reset_index(drop=True)
        tune_pools[key] = (meta, booster, X_train, sample_frame, meta["protected_feature"])
        print(f"  {key}: tune pool n={len(idx)} (indices {idx[:5].tolist()}...)")

    combos = list(itertools.product(
        GRID["min_leaf_weight_fraction"], GRID["quantile_grid_size"], GRID["objective_weights"],
    ))
    print(f"\nSearching {len(combos)} configurations on tune pools only ...")

    log = []
    t0 = time.time()
    for i, (min_frac, qgrid, weights) in enumerate(combos):
        hyperparams = {
            "min_leaf_weight_fraction": min_frac,
            "quantile_grid_size": qgrid,
            "objective_weights": weights,
        }
        result = evaluate_config(hyperparams, tune_pools)
        log.append(result)
        print(
            f"  [{i+1:2d}/{len(combos)}] min_frac={min_frac:<7} qgrid={qgrid:<3} "
            f"weights={weights}  min_score={result['min_score']:.3f} "
            f"mean_score={result['mean_score']:.3f}  ({time.time()-t0:.0f}s elapsed)"
        )

    best = max(log, key=lambda r: (r["min_score"], r["mean_score"]))
    print(f"\nSelected: {best['hyperparams']}")
    print(f"  worst-case (min) dataset score: {best['min_score']:.3f}")
    print(f"  mean dataset score: {best['mean_score']:.3f}")
    for key, v in best["per_dataset"].items():
        print(f"    {key:12s} coverage={v['coverage']:.3f} fidelity={v['fidelity']:.3f} complexity={v['complexity']:.3f} score={v['score']:.3f}")

    output = {
        "selected_hyperparams": best["hyperparams"],
        "selection_criterion": "max(min_score across dev datasets, then mean_score) on TUNE pool only",
        "tune_pool_size_per_dataset": TUNE_N,
        "tune_seed": SEED,
        "full_grid_log": log,
    }
    out_path = RESULTS_DIR / "selected_hyperparameters.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved selection + full grid log to {out_path}")


if __name__ == "__main__":
    main()
