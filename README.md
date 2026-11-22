# Uncertainty-Aware Explainable AI for Brain Tumor MRI Classification

### V9 Research Implementation

**MSc Artificial Intelligence Dissertation**  
The British University in Dubai | November 2026

**Researcher:** Noor Mekhled Hawari

## Overview

This repository contains the V9 implementation of a research framework for evaluating prediction and explanation reliability in brain tumour MRI classification.

The project extends beyond classification accuracy by investigating whether confident predictions and visual explanations can be considered reliable.

The classification model uses EfficientNet-B4 with channel and spatial attention mechanisms.

## Research Objectives

The framework evaluates five complementary reliability dimensions:

1. Calibrated prediction confidence
2. Predictive uncertainty
3. Explanation faithfulness
4. Explanation stability
5. Explanation uncertainty

These measurements are combined into individual-image reliability profiles while remaining separate, rather than being reduced to one overall trustworthiness score.

## Dataset

The V9 experiment uses a four-class brain tumour MRI dataset:

- Glioma
- Meningioma
- Pituitary tumour
- No tumour

Original dataset: 7,200 MRI images.

After integrity checking and cleaning:
- 322 duplicate images removed
- 6 contradictory-label images removed
- 6,872 images retained

Dataset partitions:
- Training: 4,399
- Validation: 1,099
- Testing: 1,374

Dataset source:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The MRI dataset is not redistributed in this repository.

## Classification Results

The final V9 model achieved:

| Metric | Result |
|---|---|
| Test accuracy | 97.89% |
| Correct predictions | 1,345 |
| Incorrect predictions | 29 |
| Macro-F1 | 0.9789 |

## Predictive Uncertainty and Calibration

Predictive uncertainty was estimated using Monte Carlo Dropout with 20 stochastic forward passes.

Temperature scaling was fitted using validation data.

V9 test Expected Calibration Error (ECE):
- Before calibration: 0.01457
- After calibration: 0.00615

Predictive entropy achieved an ROC-AUC of 0.955 for classification-error detection.

A validation-derived uncertainty threshold identified 25 of 29 test errors, leaving four errors undetected.

## Explainable AI

The research evaluates three attribution methods:

- Grad-CAM
- Grad-CAM++
- HiResCAM

A structured comparison was conducted on 200 validation MRI images.

HiResCAM was selected for the final comprehensive V9 explanation reliability analysis.

Explanation quality was investigated through deletion and insertion faithfulness testing, controlled perturbation stability analysis, and stochastic attribution variability.

## Main Research Finding

High classification accuracy, calibrated confidence, low predictive uncertainty, and apparently favourable explanation measurements do not guarantee an individually correct prediction.

The research therefore supports evaluating multiple complementary reliability dimensions rather than relying only on classification accuracy or visual heatmaps.

## Project Structure

- `src/` — experimental pipeline and research evaluation scripts
- `brainlens_v9.py` — interactive research application
- `v9_final_reliability_app.py` — V9 reliability demonstration interface
- `requirements.txt` — Python dependencies
- `REPRODUCING.md` — original step-by-step experimental instructions
- `RUN_*.sh` — execution scripts
- `V9_Uncertainty_Lab.ipynb` — uncertainty analysis notebook
- `README_STEP*.md` — supporting experimental documentation

Large datasets, trained checkpoints and bulk experimental outputs are excluded from the initial repository.

## Reproducibility

See [REPRODUCING.md](REPRODUCING.md) for the recorded experimental pipeline.

The repository documents the implementation and experimental procedures. Exact reproduction may require the original dataset version, partition manifests, model checkpoints and compatible software dependencies.

## Research Limitations

- Complete patient independence could not be verified for the V9 dataset.
- The experiments relied on publicly available MRI images.
- Independent hospital-based clinical validation was not conducted.
- Explanation reliability metrics do not independently establish anatomical or clinical correctness.

## Disclaimer

This project is intended exclusively for academic research and experimental evaluation.

It is not a clinically validated medical diagnostic system and must not be used to make medical decisions.

## Author

Noor Mekhled Hawari  
MSc Artificial Intelligence  
The British University in Dubai  
2026
