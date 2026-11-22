# V9 Step 32 — Explanation uncertainty (HiResCAM vs corrected Grad-CAM)

## Outcome

The **same 200 validation MRI images** used for Steps 30 and 31 were analyzed. Each image received **20 identical classifier-dropout draws for both methods**, with the deterministic V9 predicted class fixed as the explanation target. The selected layer was `model.features[-4]`.

| Metric (per MRI, then average) | Grad-CAM | HiResCAM |
|---|---:|---:|
| Mean pixelwise normalized heatmap standard deviation (lower = more consistent) | 0.015606 | **0.008110** |
| Cosine similarity of each MC map to the 20-map mean (higher = more consistent) | 0.995391 | **0.996609** |
| Top-20% heatmap IoU to the mean map (higher = more consistent) | 0.901975 | **0.911134** |

Paired bootstrap 95% intervals for HiResCAM minus Grad-CAM are included in `results/32_hirescam_explanation_uncertainty/summary.json` and `STEP32_FINDINGS.md`. There were 197 correct and 3 incorrect predictions among the selected validation images; there were no entirely empty heatmaps.

## Install the package into your existing V9 folder

Download `V9_Step32_Explanation_Uncertainty.zip` into Downloads, open the **V9 project folder** in VS Code, and use its terminal:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
unzip -o ~/Downloads/V9_Step32_Explanation_Uncertainty.zip -d .
```

This installs `src/step32_hirescam_explanation_uncertainty.py` and the **already completed** validation results under `results/32_hirescam_explanation_uncertainty/`.

**You do not need to rerun the 200-image experiment.** To verify that the saved results and code match your checkpoint and split and regenerate the summary, use:

```bash
python3 -m src.step32_hirescam_explanation_uncertainty --resume
```

If you intentionally want a fresh experimental run, use a **different output name** rather than overwriting the existing results:

```bash
python3 -m src.step32_hirescam_explanation_uncertainty --out-name 32_fresh_validation
```

## Files to inspect

- `results/32_hirescam_explanation_uncertainty/explanation_uncertainty_comparison.png` — comparative chart
- `results/32_hirescam_explanation_uncertainty/examples/` — four examples from real MRIs with mean and variability heatmaps
- `results/32_hirescam_explanation_uncertainty/uncertainty_trials.csv` — both methods for all 200 images (400 rows)
- `results/32_hirescam_explanation_uncertainty/summary.json` — all comparison statistics and paired bootstrap intervals
- `results/32_hirescam_explanation_uncertainty/STEP32_FINDINGS.md` — concise Chapter 4 evidence

## Important scientific limitations

- **Lower attribution variability is not proof of correct tumour localization, correct classification, or clinical validity.** HiResCAM may be more concentrated by construction.
- MC Dropout here is applied only at V9's classifier dropout layer (p = 0.4). It does not sample all sources of uncertainty.
- Each heatmap is max-normalized separately to focus on *shape variation*, so these figures cannot be interpreted as absolute or calibrated pixelwise uncertainty.
- The 200-image experiment is validation data. The full 1,374-image Step 22 results are still **Grad-CAM results**, not HiResCAM results. Do **not** copy/rename the old results as though they apply to HiResCAM.
- The 200-image validation sample contained only three misclassifications, too few to reliably measure whether these explanation-uncertainty metrics detect errors.
- The fast implementation uses the unchanged V9 backbone and classifier head. Because classifier dropout happens **after** pooled features, the 20 stochastic gradient passes can be evaluated using the same deterministic feature activations and independent classifier masks. This was checked against a direct derivative and matched exactly.

**Recommended next step:** Freeze HiResCAM as the explanation candidate, and only then calculate/reconcile the *full-test* HiResCAM reliability results, keeping the original Grad-CAM results as baseline, before writing Chapter 4.
