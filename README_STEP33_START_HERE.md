# V9 — Step 33: Final HiResCAM full-test reliability analysis

**Purpose:** Recompute the explanation-related measurements (faithfulness, stability, explanation uncertainty) for the **entire existing 1,374-MRI V9 held-out test split** with **HiResCAM at `features[-4]`**. Combine with the previously saved deterministic predictions, MC Dropout uncertainty, and validation-fitted temperature-scaled confidence. No retraining, no new dataset splits, no changes to previous results.

### Installation in your existing V9 project on Mac

Download `V9_Step33_HiResCAM_Full_Test.zip` and put it in Downloads, then run:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
unzip -o ~/Downloads/V9_Step33_HiResCAM_Full_Test.zip -d .
```

**First: test two MRIs** (creates a resumable output directory):

```bash
python3 -m src.step33_hirescam_full_test --max-images 2
```

If the script reports `PARTIAL: 2/1374`, the smoke test has worked. **Then run all remaining MRIs:**

```bash
caffeinate -i python3 -m src.step33_hirescam_full_test
```

The `caffeinate -i` command helps prevent idle sleep while the process runs. Leave Terminal open. The checkpoint and dataset must remain in their original locations. The new analysis is computationally intensive and runs on your Mac; the ZIP does *not* contain 1,374 newly calculated results.

**If interrupted:** rerun the **same full command**. It uses SQLite progress and skips completed MRIs. Do not delete `results/33_hirescam_full_test/progress.sqlite3` while the run is in progress.

### Output locations

- `results/33_hirescam_full_test/earlier/per_image_reliability_profile.csv` — one row per completed MRI, joining all five measurements.
- `results/33_hirescam_full_test/earlier/faithfulness_trials.csv` — HiResCAM deletion/insertion trials at 10%, 20%, and 30% masked area.
- `results/33_hirescam_full_test/earlier/stability_trials.csv` — noise/brightness/translation perturbations.
- `results/33_hirescam_full_test/earlier/integrated_summary.json` — aggregated per-metric statistics and exploratory error discrimination.
- `results/33_hirescam_full_test/example_maps/earlier/` — selected heatmap panels.
- `results/33_hirescam_full_test/figures/earlier/` — nine visualizations.
- If present, the original corrected Grad-CAM full-test profile from `results/gradcam_features4_full_test/earlier/` is used for an automatic **paired 1,374-image comparison**, saved as `paired_cam_comparison.csv`, `comparison_vs_corrected_gradcam.json` and `.png`.

### Important consistency details

- HiResCAM's formula is `ReLU(sum_channels(activation * spatial_gradient))`, as in Steps 30–32. It uses the selected predicted class as a *fixed* target, not the true class.
- Uses the *exact original Step 22 full-test evaluation protocol* (patchwise blur deletion/insertion, 3 small perturbations, 20 classifier-dropout explanations) to permit a **paired** comparison against the old corrected Grad-CAM full-cohort run. This protocol is not identical to the 200-image Step 30 validation pixel-mask/ROAD-style protocol. Do not compare their absolute scores as if the protocols were identical.
- The output subfolder still uses `earlier` for compatibility with Step 22, but `integrated_summary.json` explicitly identifies **HiResCAM / features[-4]**.
- Results are exploratory because the V9 test set has been inspected in earlier experiments; there are no patient IDs or externally validated clinical thresholds. Improved explanation consistency or deletion scores **do not establish the heatmap localizes tumours or confirms diagnosis**.
- Use old temperature scaling from Step 12 (HTS Step 29 was **not** selected); original predictive entropy from Step 09.

### Expected final check

The successful run prints:

```
COMPLETE: 1374/1374 HiResCAM test images.
Predictions: 1345 correct / 29 incorrect.
```

If the values differ, STOP and investigate data/checkpoint consistency. The run automatically verifies dataset fingerprints, per-file labels, source checkpoint and saved predictions, and refuses mismatched source results.
