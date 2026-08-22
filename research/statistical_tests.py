"""Statistical rigor for the final benchmark: paired significance tests and
bootstrap confidence intervals on fidelity/complexity (SHAP PLUS vs LIME, on
the exact same instances -- these are paired measurements, not independent
samples, so a paired test is the correct one), plus a Dirichlet
weight-sensitivity Monte Carlo check on the CSF composite ranking, mirroring
the conference paper's own robustness analysis (its Monte Carlo weight
sensitivity over 3,000 Dirichlet draws).

Run after final_validation.py has produced research/results/final_benchmark_raw.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from score_csf import score_all

RESULTS_DIR = Path(__file__).parent / "results"
N_BOOTSTRAP = 5000
N_DIRICHLET = 5000
RNG_SEED = 7


def paired_test(a: list[float], b: list[float], label: str) -> dict:
    """Wilcoxon signed-rank test + bootstrap 95% CI for mean(a) - mean(b),
    on paired per-instance values (same instances, two methods)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    n = len(diff)
    if n < 2 or np.allclose(diff, 0.0):
        return {"label": label, "n": int(n), "note": "insufficient variation for a test"}

    try:
        wilcoxon_stat, p_value = stats.wilcoxon(a, b)
    except ValueError as exc:
        wilcoxon_stat, p_value = float("nan"), float("nan")

    rng = np.random.default_rng(RNG_SEED)
    boot_means = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[sample_idx].mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    cohens_d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 1e-12 else float("nan")

    return {
        "label": label,
        "n": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_statistic": float(wilcoxon_stat),
        "p_value": float(p_value),
        "cohens_d_paired": cohens_d,
        "significant_at_0.05": bool(p_value < 0.05) if np.isfinite(p_value) else None,
    }


def dirichlet_robustness(scored_entry: dict, n_draws: int = N_DIRICHLET) -> dict:
    """What fraction of random criterion-weightings still rank SHAP PLUS >=
    SHAP overall? Mirrors the conference paper's own Monte Carlo weight
    sensitivity check (3,000 Dirichlet(1) draws over the CSF criteria).

    Run twice: once over all eight Table II criteria (matching the paper's
    own exact criterion set, including the two self-assessed ones, C3b and
    C6), and once excluding just those two self-assessed criteria. An
    earlier version of this function used a fabricated C6 that silently
    inflated the "SHAP PLUS is robust to SHAP" reading; reporting both
    numbers now makes clear how much of any robustness finding comes from
    measured data versus author judgment."""
    all_criteria = [c for c in ("C1", "C2", "C3a", "C3b", "C4", "C5", "C6", "C7") if scored_entry["shap"].get(c) is not None]
    quant_criteria = [c for c in all_criteria if c not in ("C3b", "C6")]

    def sweep(criteria: list[str]) -> dict:
        shap_scores = np.array([scored_entry["shap"][c] for c in criteria])
        sp_scores = np.array([scored_entry["shap_plus"][c] for c in criteria])
        rng = np.random.default_rng(RNG_SEED)
        weights = rng.dirichlet(np.ones(len(criteria)), size=n_draws)
        shap_weighted = weights @ shap_scores
        sp_weighted = weights @ sp_scores
        ties_or_wins = sp_weighted >= shap_weighted
        return {
            "criteria_used": criteria,
            "n_draws": n_draws,
            "shap_plus_ties_or_beats_shap_fraction": float(np.mean(ties_or_wins)),
            "shap_plus_strictly_beats_shap_fraction": float(np.mean(sp_weighted > shap_weighted)),
        }

    return {
        "including_qualitative": sweep(all_criteria),
        "quantitative_only": sweep(quant_criteria),
    }


def main() -> None:
    raw = json.loads((RESULTS_DIR / "final_benchmark_raw.json").read_text())
    scored = score_all(raw)

    all_stats = {}
    for key, entry in raw.items():
        print(f"\n=== {entry['name']} ({entry.get('split', 'unknown split')}) ===")
        fidelity_test = paired_test(
            entry["shap_plus"]["fidelity_values"], entry["lime"]["fidelity_values"],
            "SHAP PLUS fidelity vs LIME fidelity (higher is better for both)",
        )
        complexity_test = paired_test(
            entry["shap_plus"]["complexity_values"], entry["lime"]["complexity_values"],
            "SHAP PLUS complexity vs LIME complexity (lower is better for both)",
        )
        robustness = dirichlet_robustness(scored[key])

        for test in (fidelity_test, complexity_test):
            if "note" in test:
                print(f"  {test['label']}: {test['note']}")
                continue
            sig = "significant" if test["significant_at_0.05"] else "not significant"
            print(
                f"  {test['label']}\n"
                f"    mean diff = {test['mean_diff']:+.3f}  (95% CI [{test['ci95_low']:+.3f}, {test['ci95_high']:+.3f}])  "
                f"Wilcoxon p={test['p_value']:.2e} ({sig})  Cohen's d={test['cohens_d_paired']:.2f}"
            )
        incl = robustness["including_qualitative"]
        quant = robustness["quantitative_only"]
        print(
            f"  Weight-sensitivity (all 8 criteria, incl. self-assessed C3b/C6): SHAP PLUS ties-or-beats SHAP in "
            f"{incl['shap_plus_ties_or_beats_shap_fraction']:.1%} of {incl['n_draws']} random weightings "
            f"({incl['shap_plus_strictly_beats_shap_fraction']:.1%} strictly beats)"
        )
        print(
            f"  Weight-sensitivity (quantitative only, C3b/C6 excluded): SHAP PLUS ties-or-beats SHAP in "
            f"{quant['shap_plus_ties_or_beats_shap_fraction']:.1%} of {quant['n_draws']} random weightings "
            f"({quant['shap_plus_strictly_beats_shap_fraction']:.1%} strictly beats)"
        )

        all_stats[key] = {
            "fidelity_test": fidelity_test,
            "complexity_test": complexity_test,
            "weight_sensitivity": robustness,
        }

    out_path = RESULTS_DIR / "final_statistical_tests.json"
    out_path.write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"\nSaved statistical tests to {out_path}")


if __name__ == "__main__":
    main()
