import numpy as np
import pandas as pd
import pytest

from shap_plus import SHAPPlusExplainer

lgb = pytest.importorskip("lightgbm")
pytest.importorskip("shap")


def test_genuine_tree_shap_with_lightgbm_booster():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "CODE_GENDER": rng.integers(0, 2, 220),
            "AMT_INCOME_TOTAL": rng.normal(150_000, 35_000, 220),
            "AMT_CREDIT": rng.normal(500_000, 120_000, 220),
            "AMT_ANNUITY": rng.normal(26_000, 6_000, 220),
            "EXT_SOURCE_2": rng.uniform(0, 1, 220),
        }
    )
    logit = (
        1.2 * (frame["AMT_CREDIT"] / 500_000)
        - 1.5 * (frame["AMT_INCOME_TOTAL"] / 150_000)
        + 0.8 * frame["CODE_GENDER"]
        - 2.0 * frame["EXT_SOURCE_2"]
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(len(frame)) < probability).astype(int)
    dataset = lgb.Dataset(frame.iloc[:180], label=target[:180])
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 42,
            "num_threads": 2,
        },
        dataset,
        num_boost_round=20,
    )
    explainer = SHAPPlusExplainer(
        model,
        frame.iloc[:180],
        top_k=5,
        max_rule_terms=4,
        neighborhood_size=64,
        fidelity_threshold=0.5,
        immutable_features={"CODE_GENDER"},
        actionable_features={"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"},
        model_version="test-lightgbm",
    )

    first = explainer.explain_instance(frame.iloc[200])
    second = explainer.explain_instance(frame.iloc[200])

    assert len(first.full_shap_values) == len(frame.columns)
    assert first.features == second.features
    assert first.local_rule == second.local_rule
    assert first.audit.record_id == second.audit.record_id
    assert first.sign_consistency == 1.0
    assert first.stability == 1.0
