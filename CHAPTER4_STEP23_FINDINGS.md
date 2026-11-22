# V9 — Exploratory integrated reliability warning experiment (Step 23)

## Research question

Can calibrated confidence, MC Dropout predictive entropy, Grad-CAM faithfulness, explanation stability, and explanation uncertainty improve identification of misclassified brain-tumour MRIs beyond confidence or confidence plus prediction uncertainty?

## Study context

- Original EfficientNet-B4 with channel and spatial attention; unchanged original checkpoint.
- Four labels: glioma, meningioma, notumor, pituitary.
- Previous verified held-out **V9 test** cohort: 1,374 MRIs; 1,345 predictions match Kaggle labels; 29 do not (97.89% accuracy).
- Corrected explanation layer: `model.features[-4]`, previously selected after investigating corner activation.
- Five input features: validation-temperature-scaled deterministic confidence; 20-pass MC predictive entropy; high-minus-random 20%-patch-deletion target-logit margin; mean cosine similarity under small MRI perturbations; mean standardized MC Grad-CAM shape variation.
- We tested three **logistic, class-balanced, internally cross-fitted** error-warning models with fixed regularization and the same stratified five-fold splits. Detected near-duplicate image groups were assigned to the same cross-validation fold. Patient IDs unavailable.
- Direct confidence (`1 − calibrated confidence`) and MC entropy were also evaluated **without fitting a warning model**.
- Feature scaling was fitted only on each training fold. Each row received an out-of-fold score from a model that did not train on that row. No classifier retraining or new Grad-CAM calculations were performed.

## Actual internal comparison

| Warning approach | ROC-AUC | Average precision (error class) |
|---|---:|---:|
| Direct calibrated confidence (no warning-model fitting) | **0.9594** | 0.5247 |
| MC predictive entropy alone | 0.9573 | **0.5554** |
| Learned calibrated confidence alone | 0.9389 | 0.5535 |
| Learned confidence + MC predictive entropy | 0.9454 | 0.3982 |
| Learned **all five** metrics | **0.9108** | 0.3754 |

Average precision should be interpreted relative to classification error prevalence (29/1374 = 2.11%), and does not equal positive predictive value at any particular threshold.

- Difference in ROC-AUC, all-five vs *direct* calibrated confidence: **−0.0486**, stratified paired bootstrap 95% interval **[−0.1079, −0.0096]**.
- Difference in ROC-AUC, all-five vs learned confidence + MC: **−0.0346**, bootstrap 95% interval **[−0.0864, −0.0028]**.

These are exploratory resampling intervals, not independent-study confidence intervals. The model and layer were already inspected on these test cases. The number of misclassifications was limited (29).

## Interpretation for Chapter 4

The proposed five-dimensional reliability assessment was implemented and computed for every test MRI. However, the experiment **did not support the hypothesis that a learned combination of all five measurements improves identification of incorrect tumour classifications.** Calibrated confidence and predictive entropy were stronger individual error-warning indicators. Explanation faithfulness, stability, and explanation uncertainty address properties of a model's rationale and should not be treated as interchangeable with classification correctness. Some incorrect classifications can have convincing, stable and influential explanations.

The resulting NeuroScope V9 prototype therefore presents each dimension separately, uses careful research warning language, and distinguishes a known mismatch (requires independently supplied dataset label) from an estimated reliability concern (when the label is unknown).

## Do NOT claim

- Do not report the learned five-metric score as a validated chance of error, a trustworthiness percentage, or a medical safety score.
- Do not claim the five-metric system outperforms confidence alone; these results show the opposite for the tested combination.
- Do not claim that the corrected Grad-CAM always locates tumour tissue or that 97.89% on this split proves clinical generalization.
- Do not describe previously examined test-set retrospective analysis as a new independent validation cohort.

## Follow-up if scientifically required

An independently sourced and appropriately labelled MRI cohort with patient-independent separation would be necessary for stronger prospective error-warning evaluation; however, that is **not required to operate this research demonstration** or to report the present exploratory findings with the limitations above.
