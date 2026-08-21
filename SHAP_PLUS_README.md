# SHAP PLUS

SHAP PLUS is a mortgage and credit-specific hybrid XAI package designed from the architecture proposed in *Feasibility and Design of a Mortgage-Specific Hybrid XAI Model Beyond SHAP and LIME*.

It is an explanation model placed on top of a trained predictor. It does **not** replace LightGBM and it does **not** average SHAP and LIME outputs.

## Architecture

1. **TreeSHAP audit backbone** - stores the complete signed attribution vector for every feature.
2. **Stable feature selection** - combines local absolute SHAP importance with background-level importance.
3. **Deterministic local rule-tree surrogate** - a shallow, weighted, greedy regression tree fit on a fixed low-discrepancy neighborhood, rather than random LIME perturbations or a linear surrogate. A linear surrogate was tried first and hit the same fidelity ceiling real LIME hits on gradient-boosted-tree models (R² ~0.2-0.3): neither can represent the threshold/interaction structure the model actually uses. The tree's own decision path *is* the rendered rule, so it needs no separate coefficient-to-text step.
4. **SHAP-sign consistency diagnostic** - each split's real local effect (child mean minus parent mean) is compared against the audited SHAP sign for that feature, and folded into the rule-length objective and the fallback gate. An earlier version hard-zeroed disagreeing coefficients instead; on real nonlinear tree models that produced frequent false conflicts and collapsed fidelity toward zero, so agreement is now measured, not enforced by construction.
5. **Fidelity gate and fallback** - low-fidelity or low-sign-consistency rules fall back to direct TreeSHAP condition rendering.
6. **Rule-format rendering** - returns short condition-style explanations through `as_list()` and `local_rule`.
7. **Counterfactual recourse** - searches only explicitly declared actionable features; immutable features are locked.
8. **Compliance log** - records the input hash, model fingerprint/version, score, threshold, full attribution vector, rule, fidelity, stability, sign consistency, coverage, complexity, recourse, and configuration.

The implemented objective corresponds to the feasibility paper:

`L_explain = alpha*L_fidelity + beta*L_stability + gamma*L_complexity + delta*L_sign_consistency`

Stability is enforced by deterministic construction; complexity by `top_k` and `max_rule_terms` (tree depth); sign consistency measured from each split's local effect vs. audited SHAP direction; fidelity by weighted local R-squared and a configurable fallback threshold.

SHAP PLUS evaluates rule lengths (tree depths) from one split through
`max_rule_terms` and selects the deterministic minimum of the four-part
objective. The default normalized weights are `(0.40, 0.10, 0.30, 0.20)` for
fidelity, stability, complexity, and sign consistency respectively; they are
recorded in every audit artifact and can be changed for documented ablation
studies. Complexity is weighted more heavily than in a naive fidelity-only
objective specifically to keep rendered rules short even as the tree-fitting
parameters (`min_leaf_weight_fraction`, `quantile_grid_size`) are tuned for
higher fidelity.

For CSF C1, SHAP PLUS reports both attribution coverage (computed on the same
full TreeSHAP vector SHAP itself uses, so it is numerically identical to
SHAP's own coverage) and surrogate fidelity. Its aggregate C1 score
conservatively uses the lower of the two scores, so the hybrid cannot pass by
performing well on only the SHAP side or only the human-readable surrogate
side. On datasets with few total features, SHAP's coverage is close to 1.0
almost automatically (top-10 of ~12-18 features captures nearly everything),
which the conservative minimum will not let a real fidelity score match even
when that fidelity is 3x a real LIME baseline's -- this is a property of the
scoring rule's asymmetry (SHAP is never scored on fidelity at all), not
evidence the hybrid's audit-grade coverage is worse than SHAP's.

## Empirical validation

Results in `research/results/final_*` follow a protocol built specifically to
avoid two failure modes common in XAI evaluation: tuning an explainer's own
hyperparameters on the same instances used to report its performance, and
claiming generalization without ever testing on data the tuning process
could not have influenced.

**Datasets.** Home Credit Default Risk (307,511 rows, continuity with the
conference paper), HMEQ (5,960 rows, no demographic columns), and HMDA
Vermont 2023 (13,970 real approval/denial decisions) are the three
*development* datasets -- SHAP PLUS's hyperparameters may be selected using
data from these. HMDA New Hampshire 2023 (36,576 rows) is a *holdout*
dataset: `research/tune_hyperparameters.py` never loads it, so its results
are a genuine blind generalization check, not a second look at data already
used to pick a configuration.

**Split.** Within each development dataset's held-out test set (itself never
seen by the LightGBM classifier), a 40-instance tune pool is drawn with a
seed independent of the report pool. `research/tune_hyperparameters.py`
grid-searches `min_leaf_weight_fraction`, `quantile_grid_size`, and
`objective_weights` using *only* the tune pools, selecting the configuration
that maximizes the **worst-case** (minimum) combined C1/C3a score across the
three development datasets jointly -- not the average, and never a single
dataset -- specifically so a configuration cannot win by overfitting to
whichever dataset happens to be easiest. `research/final_validation.py` then
runs the full SHAP/LIME/SHAP PLUS benchmark, with that frozen configuration
and zero further adjustment, on: (a) each development dataset's report pool,
provably disjoint from its tune pool (`final_validation.py` asserts this),
and (b) the entire holdout dataset.

**Result: it generalizes.** Report-pool fidelity met or exceeded tune-pool
fidelity on every development dataset (no overfitting collapse), and the
holdout dataset's numbers (fidelity R² 0.793, coverage 0.965) are close to
indistinguishable from the "seen" HMDA Vermont dataset's (R² 0.803, coverage
0.972) despite the hyperparameters never having been influenced by New
Hampshire in any way.

**Statistical testing** (`research/statistical_tests.py`), paired per
instance since both methods are computed on the same rows:

| Dataset | Fidelity vs LIME (mean diff, 95% CI) | Complexity vs LIME (mean diff, 95% CI) | Wilcoxon p |
|---|---|---|---|
| Home Credit | +0.553 [+0.532, +0.573] | −0.478 [−0.490, −0.467] | ≈2×10⁻²⁶ |
| HMEQ | +0.524 [+0.500, +0.548] | −0.487 [−0.499, −0.476] | ≈2×10⁻²⁶ |
| HMDA VT | +0.517 [+0.478, +0.555] | −0.517 [−0.532, −0.503] | ≈2×10⁻²⁶ |
| HMDA NH (holdout) | +0.520 [+0.483, +0.556] | −0.529 [−0.542, −0.515] | ≈2×10⁻²⁶ |

SHAP PLUS's advantage over real LIME on both fidelity (higher is better) and
complexity (lower is better) is not a point-estimate artifact -- it holds
with overwhelming statistical significance and large paired effect sizes
(Cohen's d 2.1-4.3 for fidelity, -5.8 to -6.8 for complexity) on every
dataset, including the one the tuning process never saw.

**The honest limit, corrected twice: SHAP objectively beats SHAP PLUS on the
measured criteria -- but SHAP PLUS legitimately beats SHAP on C6.** Two
earlier versions of this section got C6 wrong. First it was a static
checklist keyed only by method *name*, giving SHAP PLUS a constant +1 over
SHAP on every dataset regardless of any data, which exactly canceled SHAP's
real lead on C1/C3a and manufactured an apparent tie -- caught by direct
user challenge ("shap and shap plus overall marks is also same at all
places which is kinda sus"). Fixing that, the correction *also* had a bug:
it incorrectly marked SHAP as failing the "actionable feature-level
information" checklist item. The conference paper's own Section IV-F says
otherwise -- SHAP satisfies **all four** Article 14(1) checklist dimensions
(direction, magnitude, actionable information, full traceability) and
scores 4.0/5.0; LIME satisfies three of four (partial credit on
actionability) and scores 3.5/5.0. Those are the paper's own published
numbers, now used verbatim rather than re-derived.

Assessed against the identical four dimensions the paper defines, SHAP
PLUS meets the same bar SHAP meets on all four, plus a distinguishable,
independently-checkable addition on dimension (iii): its rendered
conditions use exact tree-split thresholds (not LIME's arbitrary
discretized bins, the paper's own stated reason LIME loses half a point
there), and it exposes an actual recourse module computing concrete
feature-value changes toward a favourable decision -- a capability neither
baseline has at all. That earns 4.5/5.0, not a reflexive 5.0 (the recourse
module's own documented limitations argue against a perfect score) -- this
remains a qualitative, self-assessed judgment, exactly like the paper's own
C6, and needs independent-rater verification before being treated as
definitive.

With that corrected: `overall_quantitative` (C1, C2, C3a, C4, C7 -- all
objectively measured, none self-assessed) still shows **SHAP outperforming
SHAP PLUS on every dataset tested** (4.40 vs 4.20 Home Credit, 4.75 vs 4.50
HMEQ, 4.80 vs 4.60 both HMDA), unaffected by anything C6-related. A
Dirichlet(1) sweep over just those criteria finds SHAP PLUS ties-or-beats
SHAP in **0.0%** of 5,000 weightings on every dataset. Including the
corrected C6 = 4.5: SHAP still wins overall on every dataset (4.33 vs 4.25
Home Credit, 4.60 vs 4.50 HMEQ, 4.67 vs 4.58 both HMDA), but the
weight-sensitivity picture is now genuinely mixed rather than lopsided --
SHAP PLUS ties-or-beats SHAP in **~33-34%** of weightings once C6 is
included, up from the previous (buggy) 0.0%.

The honest overall claim, corrected again: SHAP PLUS decisively and
robustly beats real LIME on fidelity and complexity (see the statistics
above, unaffected by any of this). Against SHAP: it does not beat SHAP on
the objectively measured criteria taken together, SHAP wins those. It does
legitimately beat SHAP on C6 specifically, for real and verifiable reasons
tied to the recourse module and confirmed traceability -- but that one
qualitative criterion is not enough to overturn SHAP's quantitative lead in
the aggregate. Whether SHAP PLUS is more *suitable* than SHAP alone for a
given deployment -- the question C6 gestures at but a self-assessed
checklist cannot settle -- is a question for a human study, not another CSF
number.

**What this still does not establish.** No Anchors, counterfactual-method,
or interpretable-by-design (EBM) baseline is included. C3b (human
comprehensibility) has no automated proxy and is not scored -- the paper's
core thesis (that the rendered rule is more readable than raw SHAP output)
remains untested by anything in this repository and needs an actual human
study before it can be claimed.

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
- Validated against real SHAP and real LIME on four datasets (Home Credit, HMEQ, HMDA Vermont, and a fully held-out HMDA New Hampshire generalization check) with a proper tune/report split -- see "Empirical validation" above. Still missing before a superiority claim: Anchors, a dedicated recourse method, an interpretable-by-design model (EBM), and a human comprehension study.
- No XAI method is "EU AI Act compliant" by itself -- compliance is an organizational/socio-technical property (retention policy, human oversight process, deployer monitoring under Art. 26), not a property of an explainer. What is defensible: this method's outputs are better suited to *support* Articles 13/14/26's technical requirements than SHAP-only or LIME-only.
