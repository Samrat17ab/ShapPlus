"""Final SHAP vs LIME vs SHAP PLUS benchmark using hyperparameters frozen by
tune_hyperparameters.py. Report-pool instances for the three development
datasets are guaranteed disjoint from the tune pool; the New Hampshire HMDA
dataset is untouched by hyperparameter selection in any way -- loaded and
scored exactly once here, with the same frozen configuration, as a genuine
blind generalization check rather than a second bite at the same data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark_xai import load_artifacts, run_dataset
from prepare_data import HOLDOUT_LOADERS, LOADERS
from tune_hyperparameters import tune_indices

RESULTS_DIR = Path(__file__).parent / "results"

REPORT_N = 150
CONSISTENCY_N = 25
REPORT_SEED = 42


def report_and_consistency_indices(dataset_key: str, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Report/consistency indices, explicitly excluding whatever
    tune_hyperparameters.py used, so hyperparameter selection and final
    reporting never share a single instance."""
    tune_idx = set(tune_indices(dataset_key, n_test).tolist())
    rng = np.random.default_rng(REPORT_SEED)
    order = rng.permutation(n_test)
    available = np.array([i for i in order if i not in tune_idx])
    report_idx = available[:REPORT_N]
    consistency_idx = available[REPORT_N:REPORT_N + CONSISTENCY_N]
    return report_idx, consistency_idx


def main() -> None:
    selected = json.loads((RESULTS_DIR / "selected_hyperparameters.json").read_text())
    hyperparams = selected["selected_hyperparams"]
    print(f"Using frozen hyperparameters (selected on tune pool, never touching report/holdout): {hyperparams}\n")

    all_results = {}

    for key in LOADERS:
        meta, booster, X_train, X_test, y_test = load_artifacts(key)
        report_idx, consistency_idx = report_and_consistency_indices(key, len(X_test))
        overlap = set(report_idx.tolist()) & set(tune_indices(key, len(X_test)).tolist())
        assert not overlap, f"{key}: report pool overlaps tune pool at {overlap}"
        result = run_dataset(
            key, REPORT_N, CONSISTENCY_N, 2,
            sample_idx=report_idx, consistency_idx=consistency_idx,
            hyperparams=hyperparams,
        )
        result["split"] = "report (disjoint from tune pool; see tune_hyperparameters.py)"
        all_results[key] = result

    for key in HOLDOUT_LOADERS:
        meta, booster, X_train, X_test, y_test = load_artifacts(key)
        rng = np.random.default_rng(REPORT_SEED)
        idx_all = rng.permutation(len(X_test))
        sample_idx = idx_all[:REPORT_N]
        consistency_idx = idx_all[REPORT_N:REPORT_N + CONSISTENCY_N]
        result = run_dataset(
            key, REPORT_N, CONSISTENCY_N, 2,
            sample_idx=sample_idx, consistency_idx=consistency_idx,
            hyperparams=hyperparams,
        )
        result["split"] = "holdout -- never loaded during hyperparameter selection"
        all_results[key] = result

    out_path = RESULTS_DIR / "final_benchmark_raw.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved final benchmark to {out_path}")


if __name__ == "__main__":
    main()
