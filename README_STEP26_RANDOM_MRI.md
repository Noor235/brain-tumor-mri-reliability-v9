# V9 — Random MRI MC Dropout experiment (VS Code / Mac)

This adds **one new script only** to your existing V9 project. Your Step 25 uncertainty lab must already be installed (you have run it successfully).

## Run a different random image each time

Open your existing V9 project in Visual Studio Code and use the terminal:

```bash
python3 -m src.step26_random_mri_uncertainty
```

This randomly selects one entry from your **original V9 held-out split manifest** (`results/04_split/split_manifest.csv`). It does NOT use the original Kaggle `Testing` directory as a shortcut, because the V9 split was reassigned.

It then loads `results/07_training/best_v9.pt`, runs deterministic inference plus **20 real MC Dropout passes**, and shows the true label, prediction, confidence, predictive entropy, mutual information, and an uncertainty referral warning. No retraining.

Plots and data are saved in a fresh subfolder of `results/26_random_mri_uncertainty/`, including `random_v9_mri_20_mc_passes.png`, `20_MC_passes.csv`, and `result.json`.

## Repeat exactly the same random image

```bash
python3 -m src.step26_random_mri_uncertainty --seed 20261119
```

`--seed` determines which image is selected; `--mc-seed` sets dropout sampling separately. Use the same settings to reproduce the result.

## Scientific interpretation

A high entropy value indicates uncertainty in the mean probability distribution. MC Dropout mutual information measures variability of the probabilities across stochastic passes. Neither score independently establishes whether the predicted label is correct.

The **Refer for Review** threshold comes from the earlier V9 Step 24 validation analysis; it is a retrospective research threshold, not a clinical rule. Ground truth is obtained from the V9 dataset manifest and displayed *after* inference; it is never given to the model.
