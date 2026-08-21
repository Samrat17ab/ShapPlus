"""Compliance Scoring Framework evaluation helpers for SHAP PLUS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ._utils import jaccard, prediction_vector
from .explainer import SHAPPlusExplainer


@dataclass(frozen=True)
class CSFReport:
    n_explanations: int
    c1_coverage_mean: float
    c1_fidelity_mean: float
    c2_jaccard_mean: float
    c3a_complexity_mean: float
    c4_bias_gap: float | None
    c4_group_means: dict[str, float]
    c5_disparity: float | None
    c5_outcome_rates: dict[str, float]
    fallback_rate: float
    sign_consistency_mean: float
    csf_scores: dict[str, float]
    diagnostic_scores: dict[str, float]
    overall_score: float
    qualitative_scores_are_provisional: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_csf(
    explainer: SHAPPlusExplainer,
    sample: pd.DataFrame,
    *,
    protected_feature: str | None = None,
    n_runs: int = 10,
) -> CSFReport:
    """
    Evaluate SHAP PLUS using the same operational criteria as the paper.

    Repeated runs intentionally reuse the same model, input, and configuration.
    A deterministic implementation should therefore obtain Jaccard = 1.0.
    """
    repeated_top_sets: list[list[set[str]]] = []
    first_run = None
    for _ in range(int(n_runs)):
        explanations = explainer.explain(sample, include_recourse=False)
        if first_run is None:
            first_run = explanations
        repeated_top_sets.append([set(item.features) for item in explanations])
    assert first_run is not None

    jaccard_scores = [
        jaccard(repeated_top_sets[left][row], repeated_top_sets[right][row])
        for left, right in combinations(range(int(n_runs)), 2)
        for row in range(len(sample))
    ]
    group_means: dict[str, float] = {}
    outcome_rates: dict[str, float] = {}
    bias_gap = None
    disparity = None
    if protected_feature is not None:
        if protected_feature not in sample.columns:
            raise ValueError(f"{protected_feature!r} is not present in the sample.")
        groups = sample[protected_feature]
        predictions = prediction_vector(
            explainer._predict_fn,  # Package-level evaluator intentionally uses configured function.
            sample.loc[:, explainer.feature_names],
            explainer.positive_class,
        )
        for group in sorted(groups.dropna().unique(), key=str):
            mask = groups == group
            attributions = [
                item.full_shap_values[protected_feature]
                for item, selected in zip(first_run, mask)
                if selected
            ]
            group_means[str(group)] = float(np.mean(attributions))
            favourable = (
                predictions[mask.to_numpy()] < explainer.decision_threshold
                if explainer.positive_class_is_adverse
                else predictions[mask.to_numpy()] >= explainer.decision_threshold
            )
            outcome_rates[str(group)] = float(np.mean(favourable))
        if len(group_means) >= 2:
            bias_gap = float(max(group_means.values()) - min(group_means.values()))
            disparity = float(max(outcome_rates.values()) - min(outcome_rates.values()))

    coverage = float(np.mean([item.coverage for item in first_run]))
    fidelity = float(np.mean([item.fidelity for item in first_run]))
    consistency = float(np.mean(jaccard_scores)) if jaccard_scores else 1.0
    complexity = float(np.mean([item.structural_complexity for item in first_run]))
    fallback_rate = float(np.mean([item.fallback_used for item in first_run]))
    sign_consistency = float(np.mean([item.sign_consistency for item in first_run]))
    coverage_score = _score_coverage(coverage)
    fidelity_score = _score_fidelity(fidelity)
    scores = {
        # The original CSF uses coverage for SHAP and fidelity for LIME. SHAP
        # PLUS produces both, so the conservative (lower) score prevents the
        # hybrid from passing C1 by excelling on only one side.
        "C1_Hybrid_Coverage_Fidelity": min(coverage_score, fidelity_score),
        "C2_Consistency": _score_jaccard(consistency),
        "C3a_Complexity": _score_complexity(complexity),
        "C3b_Comprehension": 4.0,
        "C4_Bias": _score_bias_gap(bias_gap) if bias_gap is not None else 0.0,
        "C5_Fairness": _score_disparity(disparity) if disparity is not None else 0.0,
        "C6_Oversight": 5.0,
        "C7_Auditability": 5.0,
    }
    diagnostics = {
        "C1_Coverage": coverage_score,
        "C1_Fidelity": fidelity_score,
    }
    scored = [value for value in scores.values() if value > 0]
    return CSFReport(
        n_explanations=len(first_run),
        c1_coverage_mean=coverage,
        c1_fidelity_mean=fidelity,
        c2_jaccard_mean=consistency,
        c3a_complexity_mean=complexity,
        c4_bias_gap=bias_gap,
        c4_group_means=group_means,
        c5_disparity=disparity,
        c5_outcome_rates=outcome_rates,
        fallback_rate=fallback_rate,
        sign_consistency_mean=sign_consistency,
        csf_scores=scores,
        diagnostic_scores=diagnostics,
        overall_score=float(np.mean(scored)),
    )


def _score_coverage(value: float) -> float:
    return 5.0 if value >= 0.90 else 4.0 if value >= 0.80 else 3.0 if value >= 0.70 else 2.0 if value >= 0.60 else 1.0


def _score_fidelity(value: float) -> float:
    # Table IV, "Fidelity R^2 (LIME)" column: <0.55 -> 1 (None/non-compliant).
    return 5.0 if value >= 0.85 else 4.0 if value >= 0.75 else 3.0 if value >= 0.65 else 2.0 if value >= 0.55 else 1.0


def _score_jaccard(value: float) -> float:
    return 5.0 if value >= 0.95 else 4.0 if value >= 0.85 else 3.0 if value >= 0.75 else 2.0 if value >= 0.60 else 1.0


def _score_complexity(value: float) -> float:
    return 5.0 if value <= 0.40 else 4.0 if value <= 0.55 else 3.0 if value <= 0.70 else 2.0 if value <= 0.85 else 1.0


def _score_bias_gap(value: float) -> float:
    value = abs(value)
    return 5.0 if value >= 0.05 else 4.0 if value >= 0.03 else 3.0 if value >= 0.010 else 2.0 if value >= 0.005 else 1.0


def _score_disparity(value: float) -> float:
    # The paper defines no separate C5 threshold table; C5 (approval-rate
    # disparity) is the same kind of gap metric as C4's bias gap, so it uses
    # Table IV's "Bias Gap Delta" column directly, all five tiers included.
    return _score_bias_gap(value)
