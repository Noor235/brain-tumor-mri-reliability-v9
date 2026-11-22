# NeuroScope V9 — Final research warning dashboard

## Install into your existing Brain_Tumor_Reproduction_V9 folder

**Do not unzip this over a V7/V8 project.** Keep the checkpoint, dataset, and pre-existing V9 results in the original project; this ZIP adds only the dashboard, Step 23 analysis, and the Step 23 results. It does not overwrite past experiments.

1. Open VS Code in `Brain_Tumor_Reproduction_V9` and open its Terminal.
2. Paste these two lines:

```bash
unzip -o ~/Downloads/V9_Final_Research_System.zip -d .
bash RUN_V9_FINAL_MAC.sh
```

3. The local Streamlit interface opens at `http://127.0.0.1:8501`.

If your Python environment does not yet have Streamlit, the launcher will attempt to install it. If installation fails, activate your original V9 virtual environment and run `python3 -m pip install streamlit`, then rerun the launcher.

## Modes

- **Random held-out test MRI:** Select a real MRI from the verified V9 1,374-image test split and display its saved measurements and known dataset class. This is an *existing recorded evaluation*, not independent model validation. Optionally rerun the inference and Grad-CAM on that MRI.
- **New MRI upload/camera photo:** Compute the original V9 model prediction plus all five reliability measurements live, with 20 MC Dropout passes and 20 MC Grad-CAM passes. A camera photograph might be very different from the images the model was trained on; do not use clinically.

The five measurements are: calibrated confidence, predictive entropy, 20%-deletion faithfulness advantage over random patches, mean Grad-CAM perturbation cosine stability, and MC Grad-CAM explanation-shape uncertainty. Uses corrected `features[-4]`.

### How warning messages work

- If a trustworthy independent reference class is **provided**, the application can tell whether the model prediction matches that label. That is a known mismatch, not an inferred diagnosis.
- If the reference label is **unknown**, the application cannot tell whether it is wrong. It shows research warning indicators and never claims correctness, clinical trustworthiness, or medical safety.
- For educational explanation, warning categories use **heuristics**, not clinical decision thresholds (calibrated confidence 90%, entropy .05, deletion advantage >0, stability .95, explanation uncertainty relative to the old test-cohort 75th percentile). These were *not independently validated*.
- The optional learned error-warning ranking scores are experimental and **not calibrated probabilities**. Training on the previously inspected V9 test cohort makes them unsuitable for deployment or an independent performance claim.

## Saved Step 23 comparison

The supplied `results/23_reliability_warning/` contains the analysis computed from the four ZIPs uploaded on November 19, 2026, comparing five metrics against simpler approaches. It includes:

- `comparison_summary.json` — five-fold **StratifiedGroupKFold** internal cross-validation, grouped by detected near-duplicate group (not patient ID)
- `oof_error_warning_scores.csv` — 1,374 out-of-fold error-warning scores with labels
- `warning_model.json` — fitted *research demonstration* warning models and reference metadata
- `reference_quantiles.json` — historic distribution for descriptive flags
- `cross_fitted_roc.png` — actual comparison figure

To reproduce the Step 23 output from the original saved results in a new directory:

```bash
python3 -m src.step23_warning_validation --out-name 23_warning_reproduction
```

**No need to rerun Step 22 or train EfficientNet-B4.** The package was validated against a real V9 MRI. The saved checkpoint predictions, calibrated probability, faithfulness, and stability agreed closely with the original stored results when using the same per-image seed.

## Scientific conclusion

For ranking incorrect predictions in this *previously examined* V9 test cohort, direct calibrated confidence ROC-AUC was 0.959 and MC predictive entropy alone was 0.957. The internal cross-fitted logistic models obtained 0.939 (confidence only), 0.945 (confidence + MC), and 0.911 (all five). Adding three explanation metrics **did not improve classification-error detection** in this tested integration. They remain valuable for measuring distinct aspects of explanations, but cannot verify a correct or clinically trustworthy diagnosis.

**Caution:** only 29 classifier errors; the test set was used in prior exploratory model/layer decisions, so these are not fresh independent validation results. External evaluation remains necessary for generalization claims. No data show this system is medically safe or validated for patient care.
