"""Scores SHAP PLUS's own model_profile.py results against the conference
paper's exact CSF formulas and Table IV thresholds -- same scoring code as
score_csf.py, same source-of-truth constants -- but reports only SHAP PLUS,
no SHAP/LIME comparison columns. This exists because "don't compare with
anything, just test our model" was about dropping the three-way comparison
table, not about dropping the paper's own scoring formulas.
"""

from __future__ import annotations

import json
from pathlib import Path

from score_csf import c6_score, c7_score
from shap_plus.evaluation import (
    _score_bias_gap,
    _score_complexity,
    _score_coverage,
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
    c4 = _score_bias_gap(profile["audit_vector_gap"]) if "audit_vector_gap" in profile else None
    c6 = c6_score("shap_plus")
    c7 = c7_score(profile["consistency_exact_match_rate"] >= 0.999, None)

    scored = {"C1": c1, "C1_audit_only": c1_coverage, "C1_fidelity_only": c1_fidelity,
              "C2": c2, "C3a": c3a, "C4": c4, "C6": c6, "C7": c7}

    quant = [v for k, v in scored.items() if k in ("C1", "C2", "C3a", "C4", "C7") and v is not None]
    scored["overall_quantitative"] = sum(quant) / len(quant)
    incl = quant + [c6]
    scored["overall_with_checklist"] = sum(incl) / len(incl)
    return scored


def main() -> None:
    profiles = json.loads((RESULTS_DIR / "model_profile.json").read_text())
    all_scored = {}
    for key, profile in profiles.items():
        scored = score_dataset(profile)
        all_scored[key] = scored
        print(f"\n=== {profile['name']} (SHAP PLUS only, n={profile['n_instances']}) ===")
        print(f"  C1  Coverage/Fidelity      {scored['C1']:.2f}  (coverage-only {scored['C1_audit_only']:.2f}, fidelity-only {scored['C1_fidelity_only']:.2f}, conservative min applied)")
        print(f"  C2  Consistency            {scored['C2']:.2f}")
        print(f"  C3a Structural complexity  {scored['C3a']:.2f}")
        print(f"  C4  Bias detection         {scored['C4']:.2f}" if scored["C4"] is not None else "  C4  Bias detection         n/a (no protected attribute in dataset)")
        print(f"  C7  Auditability           {scored['C7']:.2f}")
        print(f"  Overall (quantitative, C1/C2/C3a/C4/C7)  {scored['overall_quantitative']:.2f}   <-- primary figure")
        print(f"  C6  Human Oversight (paper-grounded, qualitative)  {scored['C6']:.2f}")
        print(f"  Overall (incl. C6)         {scored['overall_with_checklist']:.2f}")

    out_path = RESULTS_DIR / "model_profile_csf_scored.json"
    out_path.write_text(json.dumps(all_scored, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
