"""Public API for SHAP PLUS."""

from .evaluation import CSFReport, evaluate_csf
from .explainer import SHAPPlusExplainer
from .types import (
    AuditRecord,
    ExplanationTerm,
    RecoursePlan,
    RecourseStep,
    SHAPPlusExplanation,
)

__all__ = [
    "AuditRecord",
    "CSFReport",
    "ExplanationTerm",
    "RecoursePlan",
    "RecourseStep",
    "SHAPPlusExplainer",
    "SHAPPlusExplanation",
    "evaluate_csf",
]

__version__ = "0.1.0"

