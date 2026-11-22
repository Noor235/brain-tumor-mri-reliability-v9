# V9 Selective-Uncertainty Upgrade — Quick Start

This is an add-on to the **original** `Brain_Tumor_Reproduction_V9` project. It bundles the earlier V9 final research dashboard and its Step 23 records, plus the new Step 24 **validation-thresholded MC Dropout referral policy** and analysis.

## Mac installation

Put `V9_Step24_Uncertainty_Referral_Upgrade.zip` in Downloads. In Terminal:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
unzip -o ~/Downloads/V9_Step24_Uncertainty_Referral_Upgrade.zip -d .
bash RUN_V9_FINAL_MAC.sh
```

Open **http://127.0.0.1:8501** if the browser does not open automatically.

It uses the already-trained V9 checkpoint, original results and data directories. Do not unzip into a separate new project: it needs the original V9 `src/` and `results/` files.

### What changed?

- An added section **Step 24 — Uncertainty referral recommendation** appears for saved test MRI selections and uploaded MRI images.
- A **validation-fitted** MC predictive entropy threshold (0.00148676) determines whether to recommend expert review.
- All five reliability metrics remain separate, with their own interpretations.
- The original model and V9 source/results are not retrained or overwritten by the Step 24 analysis; the existing dashboard script itself is replaced by the upgraded version.
- Exported MRI reports include `step24_uncertainty_referral`.
- The upgrade includes its own results and figures: no need to rerun the 1,374-image analysis.

### Reproduce Step 24 results (OPTIONAL)

From the original project root, if you want to check the results without touching packaged files:

```bash
python3 -m src.step24_selective_uncertainty --out-dir results/24_uncertainty_selective_check
```

This reads the existing Step 21 validation MC and Step 09 test MC CSVs and Step 08 deterministic predictions. It does **not** run your neural network or any image-processing experiment.

**Research use only**: no test above verifies medical diagnosis, and even images that receive no uncertainty warning may be incorrectly classified. See `V9_STEP24_RESEARCH_RESULTS.md` for methodology and limitations.
