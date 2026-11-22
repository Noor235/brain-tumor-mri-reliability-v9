# V9 Step 34 — Test the four missed incorrect MRI predictions (five measurements)

This uses **your original V9 checkpoint**, final **HiResCAM at `features[-4]`**, and exactly the existing Step 33 protocol on the four specific MRI examples. No training, model changes, or full-test rerun.

The four known error cases are `Testing/glioma/Te-gl_159.jpg`, `Testing/glioma/Te-gl_58.jpg`, `Training/meningioma/Tr-me_289.jpg`, and `Training/meningioma/Tr-me_964.jpg`. The latter two images are in the **V9 test split** despite residing in a source folder named Training.

## Install into your original V9 project

Download the ZIP to `~/Downloads`, open VS Code Terminal and run:

```bash
cd ~/Desktop/Thesis/Brain_Tumor_Reproduction_V9
unzip -o ~/Downloads/V9_Step34_Four_Missed_Five_Metrics.zip -d .
```

## Run the full experiment (four images only)

```bash
caffeinate -i python3 -m src.step34_four_missed_full_metrics
```

The script prints all five measurements for each image and writes results to a new timestamped folder under `results/34_four_missed_five_metrics/`. Open `four_real_mris_five_metrics.png` and the individual HiResCAM heatmaps in `example_maps/earlier/`. It also creates CSVs and a Markdown report comparing the **fresh** numbers with the existing saved Step 33 numbers.

**Or view existing values immediately without any model inference:**

```bash
python3 -m src.step34_four_missed_full_metrics --saved-only
```

The full run reads saved Step 09/Step 12 results for verification but **recomputes** prediction confidence from the checkpoint with the validation-fitted temperature, predictive entropy from 20 fresh classifier-dropout passes, and HiResCAM faithfulness, stability, and explanation uncertainty using the original Step 33 protocol. The separately fitted MC entropy from Step 09 is also displayed as **saved** evidence. Because MC dropout is stochastic, fresh entropy may differ slightly. Mutual information is shown only for the saved results, since Step 33's 20-pass CAM routine doesn't calculate it.

The reference threshold for MC referral was selected using validation data: predictive entropy > 0.00148676. The full-test HiResCAM mean spatial standard deviation ~0.008284 is a **descriptive reference**, not a validated warning threshold for explanation uncertainty. Do not interpret positive faithfulness or stable heatmaps as proof of correct diagnosis.

### Technical dependencies

Runs from the same project that successfully ran Step 33. Requires `src/step22_full_test_reliability.py`, `src/step33_hirescam_full_test.py`, results of Steps 08/09/12/33, the original MRI dataset and V9 checkpoint. This ZIP adds only the new script and this README; it does not overwrite your existing scripts or results.
