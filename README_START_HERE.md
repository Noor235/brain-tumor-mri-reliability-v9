# V9 Step 27 — Inspect every one of the 29 incorrect MRI predictions

This is a **Visual Studio Code uncertainty-only experiment**. It does not alter NeuroScope, retrain your V9 classifier, or replace any earlier thesis results.

## First run — audit all 29 errors using your EXISTING 20-pass MC Dropout results

Unzip `V9_Step27_Test_All_29_Errors.zip` in the **root of your original V9 project** (the one with the existing `src/`, `data/`, and `results/` folders). In your VS Code Terminal:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
unzip -o ~/Downloads/V9_Step27_Test_All_29_Errors.zip -d .
python3 -m src.step27_audit_29_errors
```

This verifies the original Step 08 and 09 CSVs, recalculates the entropy threshold exclusively from the Step 21 validation data, and examines **every one of the 29 incorrect deterministic predictions**.

Results are saved to `results/27_audit_29_errors/saved_29_errors/`:

- `29_wrong_mri_audit.csv` — true/predicted label, entropy, mutual information, confidence, referral for each wrong MRI
- `29_wrong_mri_gallery.png` — the original **29 MRI images** annotated with the error and decision
- `29_wrong_mri_uncertainty_threshold.png` — all 29 error entropies compared with validation threshold
- `4_missed_wrong_mris.png` — the four cases that the original entropy warning missed
- `FINDINGS.md`, `summary.json` — exportable results

**Expected original saved result:** 25/29 errors referred, 4/29 errors missed, 122 correctly classified cases also referred, using entropy threshold 0.00148676. The results are based on actual prior MC Dropout inference (20 passes per MRI), not synthetic probabilities.

## Optional second run — new LIVE 20-pass MC Dropout on the 29 images

You can also test whether the findings persist when generating **new stochastic dropout samples** from the actual saved V9 checkpoint:

```bash
python3 -m src.step27_audit_29_errors --live
```

This needs the earlier Step 25 package installed (because it imports `src.step25_uncertainty_visual_demo`). You've already used that code in Steps 25/26. It runs **29 × 20 = 580 MC forward passes**, outputs to `results/27_audit_29_errors/live_29_errors/`, and leaves the original saved results untouched.

The exact count referred can vary slightly between runs because MC Dropout is stochastic. **Don't replace original Chapter 4 statistics with the new count unless you explicitly report it as a separate repeat experiment.** The threshold remains the original validation-derived cutoff; the code does not adapt it to test errors.

## What this tells you (and does not)

- Correctness is determined only from the known held-out dataset labels.
- The warning uses **MC predictive entropy**, not confidence alone.
- A wrong prediction that is confidently and consistently wrong can be missed.
- Referral does not prove that a label is wrong; the system also flags correct cases.
- This is a retrospective dataset experiment, not clinical validation.
