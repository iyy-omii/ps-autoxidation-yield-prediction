# Code for: Uncertainty-Aware Yield Prediction for Data-Limited Polystyrene Autoxidation Using Heteroscedastic Ensemble Learning

This repository contains the code used to produce the main quantitative results, tables, and
figures reported in the manuscript. It implements the proposed NGBoost + heteroscedastic-MLP
ensemble framework, its evaluation, prediction-interval analysis, SHAP-based interpretation,
the baseline comparisons, and the training-data-size ablation.

Run each script with `python <script>.py`.

## What each script reproduces

| Script | Reproduces |
|---|---|
| `01_ensemble_training_pipeline.py` | Table 1 (`Ensemble`, `NGBoost`, `MLP` rows); trained-model checkpoint used by scripts 02, 03, 05; 95% prediction interval construction (Eq. 12-14) |
| `02_prediction_interval_analysis.py` | Table 2 (model-disagreement variance term, Eq. 9; recomputed 95% PI; PICP/MPIW) |
| `03_shap_interpretability.py` | Figures 3-5 (SHAP global importance, summary plot, dependence plots), for both test folds |
| `04_training_ratio_ablation.py` | Figure 6 / Table S2 (MAE/RMSE vs. train:test split ratio) |
| `05_baseline_models.py` | Table 1 (`Linear Regression`, `Ridge`, `Lasso`, `Random Forest`, `GPR`, `MC-Dropout BNN`, `Deep Ensemble` rows) |

Run `01` first: `02`, `03`, and `05` all consume the `reported_run/` split it writes.
`04` is independent (re-run it once per `TRAIN_RATIO` value).

## Fixed evaluation split

`01_ensemble_training_pipeline.py` (and everything downstream of it) uses a fixed,
hardcoded group-aware 50:50 two-fold split — the specific split described in Methods 2.1,
recorded as two lists of row indices (not reaction-condition values). Scripts `02`, `03`, and
`05` all reconstruct the same split from `01`'s output rather than generating their own, so
every script is evaluated on identical folds.

## Environment

- Python 3.12, PyTorch 2.11.0, scikit-learn 1.8.0 (as reported in Methods, Section 2)
- See `requirements.txt` for the full dependency list.
- Each script sets a fixed random seed for model training, so re-running it is internally
  reproducible.

```bash
pip install -r requirements.txt
```

## Data availability

This repository does not include the data file itself
(`Polystyrene_autoxidation.xlsx`, 137 samples across 77 reaction conditions).

Expected input format (one row per experimental sample):

| Column | Description |
|---|---|
| PS amount (g) | Polystyrene mass |
| benzoic acid amount (g) | Solvent mass |
| Mn(OAc)2 (wt%) | Catalyst loading |
| NaBr (wt%) | Catalyst loading |
| reaction time (h) | Reaction duration |
| temperature (°C) | Reaction temperature |
| Yield (%) | Target variable |

(Full variable definitions are given in Table S1 of the manuscript.)

To run any script end-to-end, place a copy of the dataset with matching column names in the
same directory as the script.

## Citation

If you use this code, please cite the manuscript (citation details to be added upon
publication).
