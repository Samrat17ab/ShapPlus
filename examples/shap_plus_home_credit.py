#!/usr/bin/env python3
"""Drop-in SHAP PLUS section for the existing Home Credit evaluation script."""

import json

import pandas as pd

from shap_plus import SHAPPlusExplainer, evaluate_csf

# This example assumes your existing script has already created:
#   model, X_train_filled, X_sample_raw, feature_names

IMMUTABLE = {
    "CODE_GENDER",
    "DAYS_BIRTH",
    "NAME_FAMILY_STATUS",
}

# These are illustrative research constraints, not lending advice. Adjust them
# with underwriting, legal, and domain experts before evaluating recourse.
ACTIONABLE = {
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
}


def build_shap_plus(model, X_train_filled, feature_names):
    return SHAPPlusExplainer(
        model,
        X_train_filled,
        feature_names=feature_names,
        positive_class=1,                 # Home Credit TARGET=1 means default
        positive_class_is_adverse=True,
        decision_threshold=0.5,
        top_k=10,
        max_rule_terms=5,
        neighborhood_size=512,
        fidelity_threshold=0.75,
        immutable_features=IMMUTABLE,
        actionable_features=ACTIONABLE,
        feature_bounds={
            "AMT_CREDIT": (0, None),
            "AMT_ANNUITY": (0, None),
            "AMT_GOODS_PRICE": (0, None),
            "AMT_INCOME_TOTAL": (0, None),
        },
        model_version="home-credit-lightgbm-v1",
        random_state=42,
    )


def run_shap_plus(model, X_train_filled, X_sample_raw, feature_names):
    explainer = build_shap_plus(model, X_train_filled, feature_names)

    # LIME-compatible single-instance interface:
    one = explainer.explain_instance(X_sample_raw.iloc[0], num_features=10)
    print("SHAP PLUS rule:", one.local_rule)
    print("Local fidelity:", one.score)
    print("Condition/weight terms:", one.as_list())
    print("Recourse:", one.recourse)

    # CSF evaluation on the same shared sample used in your paper:
    report = evaluate_csf(
        explainer,
        X_sample_raw,
        protected_feature="CODE_GENDER",
        n_runs=10,
    )
    with open("shap_plus_results.json", "w", encoding="utf-8") as stream:
        json.dump(report.to_dict(), stream, indent=2)

    # Complete per-decision Article 13/14/26-oriented audit package:
    with open("shap_plus_audit_example.json", "w", encoding="utf-8") as stream:
        json.dump(one.audit.to_dict(), stream, indent=2)
    return report


if __name__ == "__main__":
    raise SystemExit(
        "Import run_shap_plus into your training script after STEP 3, then call "
        "run_shap_plus(model, X_train_filled, X_sample_raw, feature_names)."
    )

