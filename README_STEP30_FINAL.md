# V9 — Revised Step 30: Final Faithfulness Validation

This **replaces the previous 12-image Step 30 pilot**. It does NOT alter the trained V9 model, the V9 dataset, or existing Step 22 results. You do **not** need to install or run `step30_faithfulness_methods.py`. The new `src/step30_final_faithfulness.py` is self-contained and reuses your existing V9 Python modules.

## One command in VS Code / Mac Terminal

Unzip `V9_Revised_Step30_Final_Faithfulness.zip` into your original V9 root folder. Then:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
python3 -m src.step30_final_faithfulness --out-name 30_final_local_reproduction
```

This *reproduces* the bundled 200-image run but will refuse to overwrite an existing output directory. If you unzip the full package (including results), use a different output name, e.g. `python3 -m src.step30_final_faithfulness --out-name 30_final_local_reproduction`, or read the included results directly.

The method selects **200 V9 validation MRIs**, 50 from each class, with seed 42, and compares:

- Corrected Grad-CAM (`features[-4]`), **HiResCAM**, **Grad-CAM++**.
- Identical 20% patch-area **blur deletion** and **blur insertion** tests, with high/low/random regions (same as Step 13).
- A separate 20% pixel-area **ROAD-style noisy-linear-imputation removal** check, with high/low/random regions. This is a ROAD-inspired **target-logit-margin** adaptation, not a claimed exact reproduction of the package's `ROADCombined` score.
- Image-paired differences from Grad-CAM with bootstrap 95% intervals, plus per-class summaries. Correct/incorrect breakdown is generated but might contain too few validation mistakes for useful inference.

Progress is saved after each MRI. If your Mac crashes or the process stops, **resume**:

```bash
python3 -m src.step30_final_faithfulness --out-name 30_final_local_reproduction --resume
```

The final outputs are:

```
results/30_final_faithfulness_validation/
  progress.json
  trials.csv
  per_image_paired.csv
  summary.json
  STEP30_FINDINGS.md
  method_comparison.png
  actual_mri_cam_gallery.png
```

For a fast 4-image technical check **instead of** the full 200:

```bash
python3 -m src.step30_final_faithfulness --n-per-class 1 --out-name 30_final_smoke_check
```

## How to make the final decision

Compare the **same-image high-minus-random score** for each method in all three tests. Higher means the highlighted regions influenced the *original predicted class margin* more under that perturbation. Use paired differences and confidence intervals; consider all four classes. **Do not select a winner merely because one metric improves marginally** or because a heatmap looks more tumour-shaped.

- If results disagree across tests or classes: **retain the corrected V9 Grad-CAM** as the baseline.
- If an alternative consistently improves important faithfulness outcomes on validation: it is a candidate for replacing Grad-CAM. Because earlier test results were already inspected, we should treat any new test evaluation as confirmatory only with caution; external independent data would be stronger.

## Reproducibility and research limitations

- Uses the previously trained checkpoint and pre-existing group-aware split manifest, verified against the training experiment fingerprint.
- No patient identifiers exist in the source dataset, so patient-level independence **is not established**.
- Same normalized 224×224 grayscale inputs and fixed original predicted class across all perturbations.
- Perturbation experiments create synthetic modified MRI inputs; neither metric establishes true tumour location or clinical validity.
- The noisy imputation scheme is an independently implemented **ROAD-style** check using 8-connected weighted neighbors and Gaussian noise (`0.01` in normalized pixel units). It is not identical to default official `pytorch-grad-cam` ROADCombined aggregation and should be described accurately in Chapter 3/4.
- Report both favorable and unfavorable findings and avoid claiming generalization beyond this dataset.

## Scientific sources

- Draelos & Carin, *Use HiResCAM Instead of Grad-CAM for Faithful Explanations of Convolutional Neural Networks*: https://arxiv.org/abs/2011.08891
- Chattopadhay et al., *Grad-CAM++*: https://arxiv.org/abs/1710.11063
- Rong et al., *A Consistent and Efficient Evaluation Strategy for Attribution Methods* (ICML 2022): https://proceedings.mlr.press/v162/rong22a.html
- Reference ROAD implementation and MIT source attribution: https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/metrics/road.py

This script requires the dependencies already used by V9 plus **SciPy**. If unavailable, run `python3 -m pip install scipy` in the same environment. No new neural-network training is required.
