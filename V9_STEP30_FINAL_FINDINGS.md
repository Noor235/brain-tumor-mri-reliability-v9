# V9 Step 30 — Final revised explanation faithfulness validation

## Question
Can another CAM method produce more faithful V9 explanations than the corrected `features[-4]` Grad-CAM, when using the same pre-trained, frozen V9 EfficientNet-B4 and same target class?

## Design
- 200 **validation** MRIs: 50 glioma, 50 meningioma, 50 no-tumour, 50 pituitary, seed 42.
- 197 were classified correctly; **3 were classified incorrectly** (too few to infer differences on incorrect classifications).
- Three methods: corrected Grad-CAM, HiResCAM, Grad-CAM++.
- Three paired diagnostic tests: blur-deletion and blur-insertion (20% 16×16 patch area, existing V9 Step13 logic), plus **ROAD-style 20%-pixel deletion using noisy linear imputation** (separate method and baseline). All evaluate the original predicted class's logit margin, and compare high-saliency removal/reinsertion against matched random region control.
- All procedures used the identical fixed checkpoint, `features[-4]` layer, image preprocessing, MRI, and predicted-class target. No retraining.

## Actual results

| Mean high-minus-random target-margin advantage | Corrected Grad-CAM | HiResCAM | Grad-CAM++ |
|---|---:|---:|---:|
| Blur deletion | +5.7662 | **+7.0024** | +2.1088 |
| Blur insertion | +5.2752 | **+6.9755** | +1.1771 |
| ROAD-style imputed deletion | +9.8248 | **+11.0498** | +5.3648 |

The *same-image paired differences* (HiResCAM minus corrected Grad-CAM; mean with bootstrap 95% CI, 1,000 draws) were:

| Test | Mean paired improvement | Bootstrap 95% CI |
|---|---:|---|
| Blur deletion | +1.2362 | [+0.5985, +1.9151] |
| Blur insertion | +1.7003 | [+1.1776, +2.2641] |
| ROAD-style imputed deletion | +1.2250 | [+0.5305, +1.9537] |

Positive high-vs-random fractions (Grad-CAM → HiResCAM): deletion 87.0% → **94.0%**; insertion 83.5% → **92.5%**; ROAD-style removal 97.5% → **99.5%**. These are **not** percentages of correctly localized tumours.

## Nuance by MRI class
HiResCAM's average paired improvement versus corrected Grad-CAM:

| Class | Blur deletion | Blur insertion | ROAD-style deletion |
|---|---:|---:|---:|
| Glioma | +1.056 | +1.445 | +0.665 |
| Meningioma | +2.353 | +2.959 | +1.358 |
| No tumour | +0.493 | **−0.843** | +0.712 |
| Pituitary | +1.043 | +3.240 | +2.164 |

The no-tumour blur-insertion disadvantage is an important limitation and shows the method isn't uniformly superior.

## Interpretation / decision
**HiResCAM is the stronger validation candidate on these three predefined faithfulness diagnostics.** Its paired mean advantages were positive with bootstrap intervals above zero for all three. Grad-CAM++ performed substantially worse than both on these tests.

This is a valid **validation-set** improvement in the specified intervention protocols, **not** proof of better tumour localization or clinical reliability. The model may rely on anatomy outside a tumour. Heatmaps must not be described as lesion segmentation.

We should **retain all original Step 22 and Grad-CAM outputs** until deciding whether to change the entire final explanation pipeline. If adopting HiResCAM in the final five-metric framework, explanation faithfulness, stability and explanation uncertainty must be recomputed with that method; earlier Grad-CAM-derived measurements must not be relabeled as HiResCAM. An independent external test set would be needed for stronger generalization claims. The original V9 test set has already been explored, so avoid claims of pristine independent method selection on that test set.

## Scientific limitations
- Small incorrect-validation subgroup (3 images); results principally describe model behaviour on correct classifications.
- Blurring, patching and ROAD-style imputations are synthetic interventions, not clinically realistic MRI modifications.
- The implemented ROAD-style score uses noisy-linear imputation and the V9 target-logit-margin diagnostic, not the official `ROADCombined` metric; do not mislabel it.
- The theoretical HiResCAM guarantee does not automatically apply to the V9 EfficientNet-B4 + attention model at the selected intermediate layer.
- All conclusions are conditional on this dataset and model, and do not validate tumour localization.

## References
- Draelos & Carin, 2021, HiResCAM: https://arxiv.org/abs/2011.08891
- Chattopadhay et al., Grad-CAM++: https://arxiv.org/abs/1710.11063
- Rong et al., 2022, ROAD: https://proceedings.mlr.press/v162/rong22a.html
