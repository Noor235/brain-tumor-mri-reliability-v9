# Brain Tumor Reproduction V9 — Newer Kaggle (four-class, nominal 7,200 MRIs)

A clean, separate adaptation of the V8 experiment with Steps 00–20.
No V8 results, datasets, or checkpoint are copied. Each V9 stage writes only to V9/results.

**Source folder:** defaults to sibling
`Brain_Tumor_Reproduction_V2/data/brain-tumor-mri`, which contains:

```
Training/glioma/  Training/meningioma/  Training/notumor/  Training/pituitary/
Testing/glioma/   Testing/meningioma/   Testing/notumor/   Testing/pituitary/
```

If different, set `KAGGLE_V9_SOURCE` to the absolute folder containing `Training/` and `Testing/`.
Never point V9 at `brain-tumor-mri-2021` (the V8 source).

## Easy start on Mac (from this new V9 project directory)

Reuse the already working V8 Python environment without modifying V8:
```bash
../Brain_Tumor_Reproduction_V8/.venv/bin/python -m src.step00_environment
```
If your V8 folder isn't a sibling, set up a V9 environment:
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m src.step00_environment
```

## One by one, in order

Replace `PYTHON` below with either `../Brain_Tumor_Reproduction_V8/.venv/bin/python`
or `./.venv/bin/python`, and run each command separately, inspecting the output before the next.

```bash
PYTHON -m src.step00_environment
PYTHON -m src.step01_inventory
PYTHON -m src.step02_exact_duplicates
PYTHON -m src.step03_near_duplicates
PYTHON -m src.step04_make_splits
PYTHON -m src.step05_verify
PYTHON -m src.step06_preview
PYTHON -m src.step07_train
PYTHON -m src.step08_evaluate
PYTHON -m src.step09_mc_uncertainty
PYTHON -m src.step10_gradcam
PYTHON -m src.step10_diagnose_gradcam
PYTHON -m src.step10b_background_dependence
PYTHON -m src.step11_write_report
PYTHON -m src.step12_calibration
PYTHON -m src.step12b_real_mri_examples
PYTHON -m src.step13_faithfulness --split val --n-per-class 8
PYTHON -m src.step13b_real_mri_explainer
PYTHON -m src.step14_stability --split val --n-per-class 8
PYTHON -m src.step14b_real_mri_stability_example
PYTHON -m src.step13_faithfulness --split test --n-per-class 4
PYTHON -m src.step14_stability --split test --n-per-class 4
PYTHON -m src.step15_explanation_uncertainty --split test --layer final --n-per-class 4 --passes 20
PYTHON -m src.step16_integrate_reliability --split test --layer final
PYTHON -m src.step17_thesis_report
PYTHON -m src.step18_final_audit --check-only
PYTHON -m src.step19_make_v9_figures
PYTHON -m src.step20_more_v9_thesis_figures
```

**Steps 07 and 08** are intentionally protected against overwriting saved training and test outputs.
**Step 18** with `--check-only` runs checks but does not generate missing work.
If desired, without `--check-only`, Step 18 can run missing post-hoc stages 12–17 (not training).

### V9 research precautions
- A data version with 7,200 images is presumed from previous V2 experiments: Step 01 verifies class counts, expected 1,800 per class. STOP and check if not matched.
- Near-duplicate detection is heuristic; no verified patient IDs. The new V9 splits will not reproduce the original predefined evaluation test; comparisons to published results require a clearly described protocol.
- V9 and V8 datasets may overlap. Neither is guaranteed independent external validation of the other. Cross-dataset audits come before any combined V11 study.
- Dataset files read-only. Result files are separate. No claim of clinical validation.
- Step 13–16 test subsets are exploratory pilots. Calibration and checkpoint choices must use validation only.
