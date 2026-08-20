import json

import numpy as np
import pandas as pd

from shap_plus import SHAPPlusExplainer, evaluate_csf


class LinearRiskModel:
    weights = np.array([0.9, -0.7, 0.4, 0.2])

    def predict(self, frame):
        values = np.asarray(frame, dtype=float)
        logits = values @ self.weights
        return 1.0 / (1.0 + np.exp(-logits))

    def get_params(self):
        return {"weights": self.weights.tolist()}


def attribution_fn(frame):
    values = np.asarray(frame, dtype=float)
    reference = np.array([0.0, 0.0, 0.0, 0.0])
    return (values - reference) * LinearRiskModel.weights, 0.0


def make_explainer():
    background = pd.DataFrame(
        {
            "CODE_GENDER": [0, 1] * 40,
            "INCOME": np.linspace(-2, 2, 80),
            "DEBT": np.linspace(-1.5, 1.5, 80),
            "RESERVES": np.linspace(-1, 1, 80),
        }
    )
    return SHAPPlusExplainer(
        LinearRiskModel(),
        background,
        attribution_fn=attribution_fn,
        top_k=4,
        max_rule_terms=3,
        neighborhood_size=64,
        fidelity_threshold=0.5,
        immutable_features={"CODE_GENDER"},
        actionable_features={"INCOME", "DEBT", "RESERVES"},
        model_version="test-linear-v1",
    )


def test_explanation_is_deterministic_and_auditable():
    explainer = make_explainer()
    row = pd.Series(
        {"CODE_GENDER": 1.0, "INCOME": -0.5, "DEBT": 1.2, "RESERVES": -0.4}
    )
    first = explainer.explain_instance(row)
    second = explainer.explain_instance(row)

    assert first.features == second.features
    assert first.local_rule == second.local_rule
    assert first.audit.record_id == second.audit.record_id
    assert first.stability == 1.0
    assert first.sign_consistency == 1.0
    assert len(first.full_shap_values) == 4
    assert "CODE_GENDER" not in {
        step.feature for step in (first.recourse.steps if first.recourse else ())
    }
    json.dumps(first.audit.to_dict())


def test_lime_compatible_surface_and_csf_report():
    explainer = make_explainer()
    sample = pd.DataFrame(
        [
            {"CODE_GENDER": 0.0, "INCOME": -0.8, "DEBT": 1.0, "RESERVES": -0.5},
            {"CODE_GENDER": 1.0, "INCOME": 0.7, "DEBT": -0.5, "RESERVES": 0.4},
            {"CODE_GENDER": 0.0, "INCOME": -0.2, "DEBT": 0.3, "RESERVES": -0.1},
            {"CODE_GENDER": 1.0, "INCOME": 0.4, "DEBT": 0.1, "RESERVES": 0.2},
        ]
    )
    explanation = explainer.explain_instance(sample.iloc[0], num_features=2)
    # SHAP PLUS treats num_features as a ceiling; its explicit complexity
    # objective may select a still smaller faithful rule.
    assert 1 <= len(explanation.as_list()) <= 2
    assert isinstance(explanation.score, float)

    report = evaluate_csf(
        explainer, sample, protected_feature="CODE_GENDER", n_runs=3
    )
    assert report.c2_jaccard_mean == 1.0
    assert report.sign_consistency_mean == 1.0
    assert report.c4_bias_gap is not None
    assert report.c5_disparity is not None
