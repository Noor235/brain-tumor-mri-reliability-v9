# BrainLens MRI — V9 Research Website

A minimalist local website with a **white + baby-blue brain theme**, two ways to select MRIs, and five separate reliability interpretations.

## What is inside

- `brainlens_v9.py` — the Streamlit website.
- `src/v9_web_hirescam.py` — live inference and all five reliability calculations using your **final V9 HiResCAM** layer (`features[-4]`).
- `RUN_BRAINLENS_MAC.sh` — Mac launcher.

This is designed as an **add-on to your ORIGINAL V9 project**, not a standalone trained model. It does not contain a second checkpoint and does not modify the original model, training code, original MRI images or test results.

## One-time setup (Mac / VS Code terminal)

1. Download `BrainLens_V9_White_Blue_Website.zip` into `~/Downloads`.
2. Open your V9 project in Visual Studio Code and run:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
unzip -o ~/Downloads/BrainLens_V9_White_Blue_Website.zip -d .
bash RUN_BRAINLENS_MAC.sh
```

The website opens at `http://localhost:8501`. If no browser opens, visit the URL manually.

If Streamlit needs installation, the launcher attempts `python3 -m pip install streamlit`. Ensure you run it in the same Python environment as your original V9 research scripts. If your Python environment forbids global package installation, activate your venv first or install Streamlit in your active environment.

## Requirements from the original V9 folder

- `results/07_training/best_v9.pt`
- `results/12_calibration/calibration_summary.json`
- `results/12_calibration/per_image_test_calibration.csv`
- `results/33_hirescam_full_test/earlier/per_image_reliability_profile.csv` (1,374 cases)
- `data/brain-tumor-mri/` (original image files for random selection)
- Existing `src` dependencies from V9, including `src/step33_hirescam_full_test.py` and `src/step22_full_test_reliability.py`
- Python dependencies used by V9 (torch, torchvision, numpy, opencv-python, scipy, scikit-learn, pandas, Pillow, etc.), plus `streamlit`.

## Two website modes

**Random MRI:** Chooses a genuinely random image from the verified V9 1,374-image TEST cohort (or optionally only from correctly/incorrectly classified subsets), shows the original image and its previously computed **Step 33 five-metric results**, and calculates a fresh single HiResCAM heatmap for display. The displayed five metrics are saved experimental measurements, not freshly recalculated. Known dataset label is available.

**Upload MRI:** Takes a newly uploaded image and runs the actual V9 checkpoint; computes temperature-scaled confidence, **20 MC dropout passes**, HiResCAM faithfulness under deletion/insertion versus random 20% patch masks, stability under noise/brightness/shift, and normalized HiResCAM explanation uncertainty over 20 stochastic explanation maps. The upload never needs to be saved to disk. An optional independently known reference label may be supplied.

### Interpretation

Each measurement gets its own short text, **not a composite trust score**. An uncertainty referral is based on the previously chosen validation entropy threshold `0.00148676`; it is a research-level warning, not clinical approval or verification. Faithfulness, stability and explanation uncertainty use cohort-based comparisons only as descriptive context; no clinical thresholds were derived for those measurements.

A model can report 99% confidence and be wrong. A stable, faithful heatmap does **not** prove tumour localization or correct diagnosis.

## Troubleshooting

If the website says Step 33 results are missing, extract your previously completed `V9_Step33_Final_Results.zip` into the original V9 directory using `unzip -o ~/Downloads/V9_Step33_Final_Results.zip -d .`.

If the checkpoint fails to load, verify the current `python3` uses the same PyTorch/torchvision environment as your tested V9 scripts.

If MRI files are missing, check the `KAGGLE_V9_SOURCE` environment variable or the original project's `src/config.py`. Random selection requires the original dataset; upload mode still needs the saved Step 33 reference profiles.

**Research use only.** No clinical, patient-independent, or external-hospital validation is implied. The 1,374-case V9 test set has been used repeatedly for exploratory experiments.
