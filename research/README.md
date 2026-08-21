# Empirical validation pipeline

Run in this order. Each step's outputs are consumed by the next.

```bash
# 1. Fetch/prepare all four datasets into a common numeric shape.
#    Home Credit's application_train.csv is not redistributed here (Kaggle
#    terms) -- download it and set HOME_CREDIT_TRAIN_CSV, or place it at
#    research/data/application_train.csv.
python prepare_data.py

# 2. Train a LightGBM classifier per dataset (stratified 64/16/20 split).
python train_models.py

# 3. Grid-search SHAP PLUS's tree-surrogate hyperparameters on a 40-instance
#    TUNE pool per development dataset (home_credit, hmeq, hmda_vt) only.
#    Selects the configuration maximizing worst-case (not average, and never
#    single-dataset) performance, so nothing gets overfit to one dataset.
python tune_hyperparameters.py

# 4. Run the full SHAP / real LIME / SHAP PLUS benchmark with the frozen
#    configuration on: each dev dataset's REPORT pool (disjoint from its
#    tune pool -- asserted in code, not just assumed), and the entire
#    hmda_nh HOLDOUT dataset, which step 3 never loaded.
python final_validation.py

# 5. Score everything against the conference paper's CSF Table IV thresholds.
python score_csf.py final

# 6. Paired significance tests (Wilcoxon signed-rank + bootstrap CI) on
#    fidelity/complexity vs LIME, and a Dirichlet weight-sensitivity check
#    on the SHAP PLUS vs SHAP overall-score ranking.
python statistical_tests.py
```

## Why the split matters

An earlier round of this project selected SHAP PLUS's hyperparameters by
watching fidelity/complexity on the exact instances that were then reported
as results -- tuning on the test set, a real methodology error. Steps 3-4
fix that: hyperparameters are chosen using only a tune pool that step 4's
report pool and the holdout dataset never touch, and `final_validation.py`
asserts the report pool has zero overlap with the tune pool rather than
assuming it.

The holdout dataset (`hmda_nh`, HMDA New Hampshire 2023) exists specifically
to answer "does this generalize, or did it just memorize the development
datasets?" Its results were produced in a single run with the already-frozen
configuration, no adjustment afterward.

## What's in `results/`

- `benchmark_raw.json` / `csf_scored.json` -- the **preliminary** run from
  before the tune/report split existed. Kept for the methodological record
  (the before/after is itself informative), not as a current result.
- `selected_hyperparameters.json` -- the full grid search log and the
  selected configuration, with per-dataset tune-pool scores.
- `final_benchmark_raw.json` / `final_csf_scored.json` -- the current,
  authoritative numbers: report pools + holdout, frozen hyperparameters.
- `final_statistical_tests.json` -- paired tests, bootstrap CIs, and the
  Dirichlet weight-sensitivity sweep.

Always cite `final_*` in the paper, not the unqualified `benchmark_raw.json`.
