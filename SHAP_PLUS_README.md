# SHAP PLUS

SHAP PLUS is a mortgage and credit-specific hybrid XAI package designed from the architecture proposed in *Feasibility and Design of a Mortgage-Specific Hybrid XAI Model Beyond SHAP and LIME*.

It is an explanation model placed on top of a trained predictor. It does **not** replace LightGBM and it does **not** average SHAP and LIME outputs.

## Architecture

1. **TreeSHAP audit backbone** - stores the complete signed attribution vector for every feature.
2. **Stable feature selection** - combines local absolute SHAP importance with background-level importance.
3. **Deterministic local surrogate** - uses a fixed low-discrepancy neighborhood rather than random LIME perturbations.
4. **SHAP-sign constraints** - surrogate directions must agree with the signed SHAP evidence.
5. **Fidelity gate and fallback** - low-fidelity surrogates fall back to direct TreeSHAP condition rendering.
6. **Rule-format rendering** - returns short condition-style explanations through `as_list()` and `local_rule`.
7. **Counterfactual recourse** - searches only explicitly declared actionable features; immutable features are locked.
8. **Compliance log** - records the input hash, model fingerprint/version, score, threshold, full attribution vector, rule, fidelity, stability, sign consistency, coverage, complexity, recourse, and configuration.

The implemented objective corresponds to the feasibility paper:

`L_explain = alpha*L_fidelity + beta*L_stability + gamma*L_complexity + delta*L_sign_consistency`

Stability is enforced by deterministic construction; complexity by `top_k` and `max_rule_terms`; sign consistency by constrained coefficients; fidelity by weighted local R-squared and a configurable fallback threshold.

SHAP PLUS evaluates rule lengths from one term through `max_rule_terms` and
selects the deterministic minimum of the four-part objective. The default
normalized weights are `(0.55, 0.15, 0.10, 0.20)` for fidelity, stability,
complexity, and sign consistency respectively; they are recorded in every
audit artifact and can be changed for documented ablation studies.

For CSF C1, SHAP PLUS reports both attribution coverage and surrogate fidelity.
Its aggregate C1 score conservatively uses the lower of the two scores, so the
hybrid cannot pass by performing well on only the SHAP side or only the
human-readable surrogate side.

## Installation

From this repository:

```bash
python -m pip install -e ".[research,test]"
```

For the minimum package:

```bash
python -m pip install -e .
```

## Integration with your existing script

After your current STEP 3 has created `model`, `X_train_filled`, `X_sample_raw`, and `feature_names`:

```python
from examples.shap_plus_home_credit import run_shap_plus

report = run_shap_plus(
    model,
    X_train_filled,
    X_sample_raw,
    feature_names,
)
print(report.to_dict())
```

Or use the package directly:

```python
from shap_plus import SHAPPlusExplainer

explainer = SHAPPlusExplainer(
    model,
    X_train_filled,
    feature_names=feature_names,
    positive_class=1,
    positive_class_is_adverse=True,
    top_k=10,
    max_rule_terms=5,
    fidelity_threshold=0.75,
    immutable_features={"CODE_GENDER", "DAYS_BIRTH"},
    actionable_features={"AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"},
    model_version="home-credit-lightgbm-v1",
)

explanation = explainer.explain_instance(X_sample_raw.iloc[0])
print(explanation.local_rule)
print(explanation.as_list())       # compatible shape with LIME's as_list()
print(explanation.score)           # local weighted R-squared
print(explanation.recourse)
print(explanation.audit.to_dict())
```

## Research cautions

- `application_test.csv` has no `TARGET`; it cannot train the LightGBM model or calculate AUC, precision, recall, F1, and label-dependent validation metrics.
- Fit encoders, imputers, quantiles, and the explainer background on the training set only.
- Protected attributes may be logged for authorized fairness analysis, but must never be proposed as recourse.
- The included recourse search is a constrained research baseline. Domain feasibility and causal validity need expert review.
- A CSF score is an evaluation result, not a declaration of legal compliance.
- Validate SHAP PLUS against SHAP, LIME, Anchors, a dedicated recourse method, and an interpretable-by-design model on Home Credit, HMEQ, and HMDA before making a superiority claim.
