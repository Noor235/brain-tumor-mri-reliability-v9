# Step 28 — V9 confidence-calibration audit

This is the **confidence** measurement (second of the five reliability measurements), *not* the MC Dropout uncertainty study. It uses only V9's existing Step 12 calibration outputs, originally generated from the real EfficientNet-B4 checkpoint. **It never retrains, refits temperature, or reruns your model.**

## Run on your Mac in VS Code

Open VS Code's integrated Terminal, then paste exactly:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
unzip -o ~/Downloads/V9_Step28_Confidence_Calibration_Audit.zip -d .
python3 -m src.step28_confidence_calibration_audit
```

The test outputs appear in `results/28_confidence_calibration_audit/`. Figures from your uploaded data are *already included* in this ZIP, so you can inspect them immediately after unzipping even before running the audit.

The script expects the previously created files:

- `results/12_calibration/calibration_summary.json`
- `results/12_calibration/per_image_test_calibration.csv`

An optional four-image MRI gallery is generated if your original files are under `data/brain-tumor-mri/`. If those image files are absent, the **numerical audit still completes**.

## Which files to open

1. `01_reliability_diagram.png` — observed accuracy versus average confidence; marker sizes reflect the number of images in each of 15 equally spaced confidence bins.
2. `02_ece_nll_brier.png` — calibration error, negative log-likelihood, Brier score before versus after temperature scaling.
3. `03_wrong_predictions_confidence.png` — confidence levels of all 29 wrong predictions, before and after.
4. `04_real_confidently_wrong_mris.png` — four actual V9 MRI examples where the predicted class remains wrong.
5. `all_29_wrong_predictions_before_after.csv` — inspect all 29 wrong images individually.
6. `STEP28_FINDINGS.md` and `step28_calibration_audit_summary.json` — auditable numerical results.

## Results verified from your original saved Step 12 files

- Validation-fitted temperature **T = 2.2844227044** (not fitted to the test labels).
- Test set **1,374 images, 1,345 correct, 29 wrong**.
- Test accuracy **97.889% before and after** (unchanged).
- Expected calibration error (ECE, **15 equal-width bins**) **0.01456988 → 0.00614586**.
- Negative log-likelihood **0.10394750 → 0.06561645**.
- Four-class summed Brier score **0.03249569 → 0.03017656**.
- Wrong predictions with predicted confidence >=95% **13 → 5**.
- Two wrong predictions remain above **99% confidence even after scaling**.

**Interpretation:** Temperature scaling improves the quality of the confidence probabilities *on average*. It does **not** detect or correct every wrong class prediction. ECE depends on binning and can be dominated by dense confidence bins. This is retrospective research on the existing V9 test set, not clinical evidence.

## Stop here for now

Use this as your confidence-calibration figure set for Chapter 4. After inspecting the four example MRI images, the next measurement to audit is **explanation faithfulness** (using the corrected `features[-4]` Grad-CAM results). Do not rerun model training or Step 22.
