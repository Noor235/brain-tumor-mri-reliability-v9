V9 Step 04 mixed-label safeguard — READ FIRST

Purpose:
Two near-similarity groups contain conflicting glioma/meningioma labels. These are not independently confirmed diagnoses. Conservatively exclude EVERY representative within those two mixed-label groups from all V9 splits, without touching the source MRI images.

Install: inside the extracted archive, open src/. Copy FOUR files into the existing V9 project's src/ folder, replacing files with the SAME names:
  step04_make_splits.py  (revised Step 04)
  verify.py              (supports excluded groups in Steps 05–20)
  step11_write_report.py (records exclusion for thesis)
  step17_thesis_report.py (records exclusion for thesis)

Do NOT replace config.py or copy the entire older V9 project again. Keep the V9 project open in VS Code. Back up your old files if you wish.

Run ONLY Step 04 for now, from Brain_Tumor_Reproduction_V9:
  ../Brain_Tumor_Reproduction_V8/.venv/bin/python -m src.step04_make_splits

Step 04 creates:
  results/04_split/split_manifest.csv
  results/04_split/excluded_mixed_label_groups.csv
  results/04_split/split_summary.json

It refuses to overwrite an existing split (safety feature). If you already ran Step 04, DO NOT delete its split or run training: send the exact output for review first.

After you send Step 04 output, the next step is:
  ../Brain_Tumor_Reproduction_V8/.venv/bin/python -m src.step05_verify

These changes were tested on a synthetic 140-image dataset, including an intentionally mixed-label group; results must be rechecked with your actual dataset.
