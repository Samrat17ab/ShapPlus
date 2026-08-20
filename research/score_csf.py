"""Turns the raw benchmark measurements into the paper's 1-5 CSF scale
(Table IV thresholds) and prints a Table-V-style comparison for all three
methods across all three datasets.

C3b (human comprehensibility) is deliberately left unscored: it requires a
human study (exactly what both source documents flag as missing), and this
script's whole purpose is to stop treating unmeasured qualities as measured
ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from shap_plus.evaluation import (
    _score_bias_gap,
    _score_complexity,
    _score_coverage,
    _score_disparity,
    _score_fidelity,
    _score_jaccard,
)

RESULTS_DIR = Path(__file__).parent / "results"

C6_CHECKLIST = {
    "shap": {"direction": True, "magnitude": True, "actionable": False, "traceability": True},
    "lime": {"direction": True, "magnitude": True, "actionable": True, "traceability": False},
    "shap_plus": {"direction": True, "magnitude": True, "actionable": True, "traceability": True},
}


def c6_score(method: str) -> float:
    return 1.0 + sum(1.0 for v in C6_CHECKLIST[method].values() if v)


def c7_score(exact_no_state: bool, exact_with_state: bool | None) -> float:
    if exact_no_state:
        return 5.0
    if exact_with_state:
        return 2.0
    return 1.0


def score_dataset(entry: dict) -> dict:
    shap_r = entry["shap"]
    lime_r = entry["lime"]
    sp_r = entry["shap_plus"]

    c1_shap = _score_coverage(shap_r["coverage"])
    c1_lime = _score_fidelity(lime_r["fidelity"])
    c1_sp = min(_score_coverage(sp_r["coverage"]), _score_fidelity(sp_r["fidelity"]))

    c2_shap = _score_jaccard(shap_r["consistency"])
    c2_lime = _score_jaccard(lime_r["consistency_stochastic"])
    c2_sp = _score_jaccard(sp_r["consistency"])

    c3a_shap = _score_complexity(shap_r["complexity"])
    c3a_lime = _score_complexity(lime_r["complexity"])
    c3a_sp = _score_complexity(sp_r["complexity"])

    c6_shap, c6_lime, c6_sp = c6_score("shap"), c6_score("lime"), c6_score("shap_plus")

    c7_shap = c7_score(shap_r["exact_reproducible"], None)
    c7_lime = c7_score(False, lime_r["exact_reproducible_fixed_seed"])
    c7_sp = c7_score(sp_r["exact_reproducible"], None)

    scored = {
        "shap": {"C1": c1_shap, "C2": c2_shap, "C3a": c3a_shap, "C6": c6_shap, "C7": c7_shap},
        "lime": {"C1": c1_lime, "C2": c2_lime, "C3a": c3a_lime, "C6": c6_lime, "C7": c7_lime},
        "shap_plus": {"C1": c1_sp, "C2": c2_sp, "C3a": c3a_sp, "C6": c6_sp, "C7": c7_sp},
    }

    c4 = entry.get("c4_bias")
    if c4:
        for method, gap_key in (("shap", "shap_gap"), ("lime", "lime_gap")):
            gap = c4.get(gap_key)
            scored[method]["C4"] = _score_bias_gap(gap) if gap is not None else None
        gap_full = c4.get("shap_plus_full_gap")
        scored["shap_plus"]["C4"] = _score_bias_gap(gap_full) if gap_full is not None else None
        scored["_c4_raw"] = c4

    for method in ("shap", "lime", "shap_plus"):
        values = [v for k, v in scored[method].items() if isinstance(v, (int, float))]
        scored[method]["overall_measured"] = sum(values) / len(values) if values else None

    return scored


def main() -> None:
    raw = json.loads((RESULTS_DIR / "benchmark_raw.json").read_text())
    all_scored = {}
    for key, entry in raw.items():
        scored = score_dataset(entry)
        all_scored[key] = scored
        print(f"\n=== {entry['name']} ===")
        header = f"{'Criterion':8s} {'SHAP':>8s} {'LIME':>8s} {'SHAP_PLUS':>10s}"
        print(header)
        for crit in ("C1", "C2", "C3a", "C4", "C6", "C7"):
            row = []
            for method in ("shap", "lime", "shap_plus"):
                v = scored[method].get(crit)
                row.append("  n/a " if v is None else f"{v:6.2f}")
            print(f"{crit:8s} {row[0]:>8s} {row[1]:>8s} {row[2]:>10s}")
        print(
            f"{'Overall':8s} "
            f"{scored['shap']['overall_measured']:8.2f} "
            f"{scored['lime']['overall_measured']:8.2f} "
            f"{scored['shap_plus']['overall_measured']:10.2f}"
        )
        print(
            f"  (raw) coverage/fidelity: SHAP={entry['shap']['coverage']:.3f} cov | "
            f"LIME={entry['lime']['fidelity']:.3f} R2 | "
            f"SHAP_PLUS={entry['shap_plus']['coverage']:.3f} cov / {entry['shap_plus']['fidelity']:.3f} R2 "
            f"(fallback {entry['shap_plus']['fallback_rate']:.0%})"
        )
        if "c4_bias" in entry:
            b = entry["c4_bias"]
            print(
                f"  (raw) bias gap on {entry['protected_feature']}: SHAP={b['shap_gap']:.4f} "
                f"LIME={b['lime_gap']:.4f} (presence {b['lime_presence_rate']:.0%}) "
                f"SHAP_PLUS full-audit={b['shap_plus_full_gap']:.4f} (presence 100%) "
                f"SHAP_PLUS visible-rule={b['shap_plus_visible_gap']}"
                f" (presence {b['shap_plus_visible_presence_rate']:.0%})"
            )
        print(
            "  NOTE: C3b (human comprehensibility) and C5 (portfolio approval-rate "
            "disparity, a model property not an XAI-method property) are intentionally "
            "excluded from 'Overall' above -- C3b has no automated proxy and requires "
            "a human study; including a guessed number for it would repeat the exact "
            "flaw this benchmark exists to fix."
        )

    out_path = RESULTS_DIR / "csf_scored.json"
    out_path.write_text(json.dumps(all_scored, indent=2, default=str))
    print(f"\nSaved scored CSF results to {out_path}")


if __name__ == "__main__":
    main()
