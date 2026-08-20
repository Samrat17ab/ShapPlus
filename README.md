# Mortgage XAI Studio

A working research demonstrator for the mortgage-specific hybrid XAI architecture described in *Feasibility and Design of a Mortgage-Specific Hybrid XAI Model Beyond SHAP and LIME*.

The application implements the paper's layered product contract:

1. a deterministic mortgage approval scoring core for interactive demonstration;
2. a complete signed contribution vector as the evidence backbone;
3. a sparse, sign-consistent local rule rendered in underwriting language;
4. counterfactual recourse restricted to mutable features; and
5. an exportable compliance record containing the decision, threshold, model version, input hash, evidence, rule, quality metrics, and recourse.

## Run

```bash
npm install
npm run dev
```

Use `npm test` to create a production build and verify the rendered product contract.

## Research boundary

The included browser model is a transparent deterministic demonstrator, not a trained production credit model. A journal or regulated deployment must replace the scoring core with a validated LightGBM or XGBoost model and its genuine TreeSHAP vector, then complete calibration, multi-dataset evaluation, fairness analysis, baseline comparisons, human studies, legal review, and operational monitoring.
