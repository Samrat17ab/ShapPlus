"""Scores SHAP PLUS's own model_profile.py results against the conference
paper's exact CSF formulas and Table IV thresholds -- all eight Table II
criteria (C1, C2, C3a, C3b, C4, C5, C6, C7), same scoring code as
score_csf.py, same source-of-truth constants -- but reports only SHAP
PLUS, no SHAP/LIME comparison columns. "Don't compare with anything, just
test our model" meant drop the three-way comparison table, not drop any of
the paper's own criteria.
"""

from __future__ import annotations

import json
from pathlib import Path

from score_csf import c3b_score, c6_score, c7_score
from shap_plus.evaluation import (
    _score_bias_gap,
    _score_complexity,
    _score_coverage,
    _score_disparity,
    _score_fidelity,
    _score_jaccard,
)

RESULTS_DIR = Path(__file__).parent / "results"


def score_dataset(profile: dict) -> dict:
    c1_coverage = _score_coverage(profile["coverage"]["mean"])
    c1_fidelity = _score_fidelity(profile["fidelity"]["mean"])
    c1 = min(c1_coverage, c1_fidelity)

    c2 = _score_jaccard(profile["consistency_jaccard"]["mean"])
    c3a = _score_complexity(profile["complexity"]["mean"])
    c3b = c3b_score("shap_plus")
    c4 = _score_bias_gap(profile["audit_vector_gap"]) if "audit_vector_gap" in profile else None
    c5 = _score_disparity(profile["approval_rate_disparity"]) if "approval_rate_disparity" in profile else None
    c6 = c6_score("shap_plus")
    c7 = c7_score(profile["consistency_exact_match_rate"] >= 0.999, None)

    scored = {
        "C1": c1, "C1_coverage_only": c1_coverage, "C1_fidelity_only": c1_fidelity,
        "C2": c2, "C3a": c3a, "C3b": c3b, "C4": c4, "C5": c5, "C6": c6, "C7": c7,
    }

    all_8_criteria = ("C1", "C2", "C3a", "C3b", "C4", "C5", "C6", "C7")
    quantitative_criteria = ("C1", "C2", "C3a", "C4", "C5", "C7")
    all_8 = [v for k, v in scored.items() if k in all_8_criteria and v is not None]
    quant = [v for k, v in scored.items() if k in quantitative_criteria and v is not None]
    scored["overall_all_8"] = sum(all_8) / len(all_8)
    scored["overall_quantitative"] = sum(quant) / len(quant)
    return scored


def main() -> None:
    profiles = json.loads((RESULTS_DIR / "model_profile.json").read_text())
    all_scored = {}
    for key, profile in profiles.items():
        scored = score_dataset(profile)
        all_scored[key] = scored
        print(f"\n=== {profile['name']} (SHAP PLUS only, n={profile['n_instances']}) ===")
        print(f"  C1  Coverage/Fidelity      {scored['C1']:.2f}  (coverage-only {scored['C1_coverage_only']:.2f}, fidelity-only {scored['C1_fidelity_only']:.2f}, conservative min applied)")
        print(f"  C2  Consistency            {scored['C2']:.2f}")
        print(f"  C3a Structural complexity  {scored['C3a']:.2f}")
        print(f"  C3b Comprehensibility      {scored['C3b']:.2f}  [self-assessed, see c3b_score() docstring]")
        c4_str = f"{scored['C4']:.2f}" if scored["C4"] is not None else "n/a (no protected attribute)"
        print(f"  C4  Bias detection         {c4_str}")
        c5_str = f"{scored['C5']:.2f}" if scored["C5"] is not None else "n/a (no protected attribute)"
        print(f"  C5  Fairness transparency  {c5_str}  (model property, not XAI-method-specific)")
        print(f"  C6  Human oversight        {scored['C6']:.2f}  [self-assessed, see c6_score() docstring]")
        print(f"  C7  Auditability           {scored['C7']:.2f}")
        print(f"  Overall -- all 8 criteria (paper formula)   {scored['overall_all_8']:.2f}")
        print(f"  Overall -- quantitative only (C3b/C6 out)   {scored['overall_quantitative']:.2f}")

    out_path = RESULTS_DIR / "model_profile_csf_scored.json"
    out_path.write_text(json.dumps(all_scored, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
