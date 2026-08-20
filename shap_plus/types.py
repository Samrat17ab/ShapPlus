"""Serializable result types returned by SHAP PLUS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExplanationTerm:
    feature: str
    value: Any
    condition: str
    shap_value: float
    surrogate_coefficient: float
    direction: str
    rank: int

    def as_pair(self) -> tuple[str, float]:
        """LIME-compatible ``(condition, weight)`` representation."""
        return self.condition, self.shap_value


@dataclass(frozen=True)
class RecourseStep:
    feature: str
    from_value: float
    to_value: float
    direction: str
    normalized_cost: float


@dataclass(frozen=True)
class RecoursePlan:
    achieved: bool
    original_score: float
    resulting_score: float
    target_threshold: float
    steps: tuple[RecourseStep, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class AuditRecord:
    record_id: str
    created_at_utc: str
    package_version: str
    model_version: str
    model_fingerprint: str
    input_hash: str
    prediction: float
    decision: str
    threshold: float
    expected_value: float
    attribution_space: str
    full_signed_attributions: dict[str, float]
    selected_features: tuple[str, ...]
    local_rule: str
    fidelity: float
    stability: float
    sign_consistency: float
    coverage: float
    structural_complexity: float
    explanation_objective: dict[str, float]
    fallback_used: bool
    recourse: RecoursePlan | None
    configuration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SHAPPlusExplanation:
    prediction: float
    decision: str
    expected_value: float
    terms: tuple[ExplanationTerm, ...]
    full_shap_values: dict[str, float]
    local_rule: str
    fidelity: float
    stability: float
    sign_consistency: float
    coverage: float
    structural_complexity: float
    explanation_objective: dict[str, float]
    fallback_used: bool
    recourse: RecoursePlan | None
    audit: AuditRecord
    surrogate_intercept: float

    @property
    def score(self) -> float:
        """LIME-compatible alias for local surrogate fidelity."""
        return self.fidelity

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(term.feature for term in self.terms)

    def as_list(self) -> list[tuple[str, float]]:
        """Return condition-style terms in the same shape as ``LIME.as_list``."""
        return [term.as_pair() for term in self.terms]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["audit"] = self.audit.to_dict()
        return data
