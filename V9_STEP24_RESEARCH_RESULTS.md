# V9 Step 24 — Validation-thresholded MC Dropout selective referral

## Question
Can predictive uncertainty identify MRI classifications requiring review, reducing errors among the cases that the classifier retains? This is **selective prediction**, not an improvement in the base classifier's accuracy and not a clinical guarantee.

## Relevant published research
1. Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*. NeurIPS 30. Establishes a rejection option for already-trained deep models: https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
2. *A hybrid scale attention network for uncertainty aware brain tumor MRI classification* (2026), *Discover Artificial Intelligence*. MC Dropout (20 passes) achieved misclassification detection ROC-AUC 0.9598, and selective prediction improved accuracy on retained predictions in its own test cohort. https://link.springer.com/article/10.1007/s44163-026-02386-7
3. Li, J., Arefan, D. & Wu, S. (2025 conference; available in PMC October 2026). *Uncertainty Quantification for Deep Learning-based Medical Imaging Classification Model Evaluation and Individualized Risk Estimation*. Uses risk–coverage analysis to evaluate uncertainty in medical imaging. https://pmc.ncbi.nlm.nih.gov/articles/PMC13678565/

## Exact method in *our* V9
- **Unchanged V9 EfficientNet-B4 checkpoint.** No classifier training, no Grad-CAM re-run.
- **Uncertainty estimator:** predictive entropy of 20 Monte Carlo dropout probability vectors (the original V9 Step 09 / Step 21 method).
- **Threshold selection:** 90th percentile of entropy on the 1,099 validation-split MRIs (`results/21_validation_mc/uncertainty.csv`). The threshold is **0.0014867600**, in the same entropy units as the stored CSV. The threshold was not fitted to test labels.
- **Test:** 1,374 V9-held-out images (`results/09_uncertainty/uncertainty.csv`) paired by exact file path with the *deterministic* predictions in `results/08_test/per_image_predictions.csv` (1,345 correct, 29 wrong). This matches the main class prediction displayed in the interface.
- **Action:** if MC entropy > threshold, display **Refer for independent review**. Otherwise display **No uncertainty-based referral; correctness not verified**.
- All 5 existing reliability measurements (calibrated confidence, predictive uncertainty, Grad-CAM faithfulness, explanation stability, and explanation uncertainty) are displayed **separately**, not collapsed into an unvalidated trust score.

## Primary retrospective result (90% target validation coverage)

| Outcome | Result |
|---|---:|
| Total test images | 1,374 |
| Baseline accuracy with all predictions | 97.89% (1,345/1,374) |
| Actual coverage at validation-derived threshold | 89.30% |
| Retained / not referred | 1,227 |
| Referred for review | 147 |
| **Accuracy among retained predictions** | **99.67% (1,223/1,227)** |
| Errors referred | **25 of 29 (86.21% error capture)** |
| Errors *not* referred | **4** |
| Correctly predicted cases unnecessarily referred | **122** |
| Proportion of referred cases that were errors | 17.01% (25/147) |
| Approx. Wilson 95% CI for 25/29 error capture | 69.4–94.5% |
| Approx. Wilson 95% CI for retained accuracy | 99.16–99.87% |

The Wilson intervals reflect sampling uncertainty in proportions, but do not account for near-duplicate dependence or prior exploratory decisions.

**Important:** MC Dropout's own argmax differed from the deterministic prediction on **one** test MRI, yielding **28** MC errors instead of the deterministic **29**. The Step 24 main results explicitly evaluate deterministic prediction correctness so they align with the dashboard's predicted class.

## Baseline uncertainty score comparison (same validation-targeted coverage)

| Score used for referral | ROC-AUC for detecting deterministic errors | # Referred | Errors referred |
|---|---:|---:|---:|
| MC predictive entropy | 0.9573 | 147 | 25/29 |
| MC mutual information | 0.9569 | 144 | 25/29 |
| MC mean confidence (low values) | 0.9568 | 146 | 25/29 |

MC predictive entropy did **not** show a meaningful error-detection advantage over MC mean confidence in this dataset. Do not claim the more complex uncertainty estimator beats confidence on this evidence. These are descriptive, retrospective results.

## Why this is useful, and limitations

This implementation turns *measured* MC Dropout uncertainty into a transparent review workflow. Retaining fewer cases concentrates errors in those marked for review. The original classifier still has exactly the same 97.89% unfiltered accuracy. It is **incorrect** to describe 99.67% as its new overall accuracy.

The existing V9 test set was previously inspected for other research decisions, including Grad-CAM selection and Step 23. Although the **numerical cutoff** is validation-derived, this retrospective result is not a fully independent confirmation of the overall research approach. There are only 29 wrong predictions, unknown patient identity, no radiologist-validated ground truth, and no external hospital validation. The threshold controls approximately the *fraction* referred, not the chance of being wrong for an individual MRI. Future independent testing is required to make stronger claims.

## Chapter 4 phrasing

> Using a validation-derived 90th-percentile threshold on MC Dropout predictive entropy, selective referral retained 1,227 of 1,374 V9 test images (89.30% coverage). Among retained predictions, 1,223 were correct (99.67% selective accuracy), compared with 97.89% unfiltered accuracy. The approach referred 25 of 29 deterministic misclassifications (86.21% error capture), but also referred 122 correctly classified cases and missed four errors. MC predictive entropy, mutual information, and MC mean confidence displayed similar discrimination, so the improvement reflects uncertainty-based selection rather than superiority of a particular uncertainty measure. Results remain exploratory given test-set reuse and the small number of errors.

## Figures and raw outputs

- `results/24_uncertainty_selective/risk_coverage.png`: full empirical risk–coverage curves.
- `results/24_uncertainty_selective/selective_accuracy.png`: selective vs. baseline accuracy, with coverage.
- `results/24_uncertainty_selective/summary.json`: full metrics and secondary sensitivity checks at validation target coverages 80%, 85%, 95%.
- `results/24_uncertainty_selective/per_image_uncertainty_referral.csv`: **all 1,374 individual referral decisions** and main prediction labels.
- `results/24_uncertainty_selective/thresholds.json`: validation threshold used for live UI.
