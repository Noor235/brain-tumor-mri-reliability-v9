V9 Grad-CAM corner-highlight correction
=====================================
Only TWO existing source modules are changed:
  src/step10_gradcam.py   -> Grad-CAM target features[-4], linear interpolation; separate output dir
  src/research_v9.py     -> 'earlier' Grad-CAM target now features[-4] (not [-2])

Apply from your Brain_Tumor_Reproduction_V9 project root:
  cp src/step10_gradcam.py src/step10_gradcam.py.before_corner_fix
  cp src/research_v9.py src/research_v9.py.before_corner_fix
  unzip -o ~/Downloads/V9_GRADCAM_CORNER_FIX.zip -d .

Regenerate 12 representative Grad-CAM images, original checkpoint unchanged:
  python3 -m src.step10_gradcam

See images in:
  results/10_gradcam_features_minus4/

Optional: recompute full-test explanation metrics for the corrected intermediate layer only:
  caffeinate -i python3 -m src.step22_full_test_reliability --layers earlier --out-name gradcam_features4_full_test

Results of past experiments are not overwritten.
V9 actual diagnostic on three no-tumor test images found mean upper-right corner heatmap mass
0.647 for final block versus 0.017 for the features[-4] intermediate block.
This is a new Grad-CAM target layer, not input masking, retraining, or cosmetic heatmap clipping.
