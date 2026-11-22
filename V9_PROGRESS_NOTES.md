# V9 — BRAIN TUMOR MRI RESEARCH PROGRESS

## CURRENT STATUS
Completed: Steps 00–06
Next: Step 07 — Model Training
IMPORTANT: Training has NOT started yet.

## 1. RESEARCH GOAL

Reproduce the EfficientNet-B4 dual-attention brain tumor
classification model on the second Kaggle MRI dataset.

Then evaluate:
- Classification performance
- Prediction uncertainty using MC Dropout
- Confidence calibration
- Grad-CAM explanations
- Explanation faithfulness
- Explanation stability
- Explanation uncertainty
- Integrated reliability

V8 remains separate and unchanged.

## 2. DATASET PREPARATION — COMPLETED

| Step | What we did | Result |
|---|---|---|
| 00 | Checked Python, libraries and dataset paths | PASS |
| 01 | Counted original MRI images | 7,200 |
| 02 | Identified exact duplicate copies | 322 excluded |
| 02 | Retained exact-unique images | 6,878 |
| 03 | Identified possible near-duplicate pairs | 1,238 |
| 03 | Created near-similarity groups | 6,171 |
| 04 | Excluded ambiguous-label groups | 2 groups / 6 images |
| 04 | Created new train/val/test splits | 6,872 images |
| 05 | Verified dataset and split integrity | PASS |
| 06 | Previewed sample MRI images | Completed |

NOTE:
The near-duplicate candidates were grouped, NOT deleted.
The original MRI files were not changed.

## 3. WHY WERE SIX IMAGES EXCLUDED?

We found two groups containing nearly identical MRI images
with conflicting labels: Glioma and Meningioma.

Because we could not determine which labels were correct,
we excluded all six images in those groups from the V9
experiment.

We did NOT delete the original files.

## 4. FINAL V9 DATASET

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| Glioma | 1,138 | 284 | 355 |
| Meningioma | 1,075 | 269 | 336 |
| No Tumor | 1,061 | 265 | 331 |
| Pituitary | 1,125 | 281 | 352 |
| TOTAL | 4,399 | 1,099 | 1,374 |

Final total: 6,872 MRI images

Training = 64.01%
Validation = 15.99%
Testing = 19.99%

Step 05 confirmed:
- Zero detected exact-duplicate overlap across splits
- Zero flagged near-similarity group overlap across splits
- All six ambiguous images excluded
- Dataset files and hashes verified

LIMITATION:
Patient identity is unknown. Patient-independent splitting
has NOT been established. Undetected similarities may remain.

## 5. STEP 06 — HOW TO JUDGE MRI IMAGE QUALITY

We previewed 12 sample images: one per class in each split.

CHECK 1 — VISIBILITY
GOOD: Brain anatomy can be seen.
BAD: Image is completely black, white or unreadable.
V9: All 12 preview images were visible.

CHECK 2 — IMAGE INTEGRITY
GOOD: Image opens correctly.
BAD: File is broken, corrupted or fails to open.
V9: All 7,200 images passed the inventory readability check.

CHECK 3 — FRAMING
GOOD: Relevant brain anatomy remains in the image.
WARNING: Brain appears very small or is accidentally cropped.
V9: Some images have different framing and image scales.

CHECK 4 — ORIENTATION
Axial = View from above/below.
Sagittal = View from the side.
Coronal = View from the front.
V9: Different MRI orientations are present.
This is not automatically an error, but it can affect learning.

CHECK 5 — LABELS
The dataset contains Glioma, Meningioma, No Tumor
and Pituitary classes.

IMPORTANT:
Visual inspection cannot prove that the medical diagnosis
assigned to an MRI is correct.

## 6. WHAT DOES STEP 06 ACTUALLY PROVE?

It provides an initial check that sample images can be
loaded, displayed and reviewed.

It does NOT prove:
- Every MRI has a medically correct label
- All images have equally good quality
- The model will classify tumors correctly
- The AI explanations will be reliable
- The dataset is patient-independent

## 7. NEXT STEP — STEP 07: TRAINING

Use 4,399 MRI images to train EfficientNet-B4 with
dual attention.

Use 1,099 validation MRIs to monitor performance
and select the best checkpoint.

Keep 1,374 test MRIs separate for evaluation.

Run this command from the V9 project folder:

../Brain_Tumor_Reproduction_V8/.venv/bin/python -m src.step07_train

After training, examine:
- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Best saved checkpoint
- Signs of overfitting

Then continue to Step 08: Test Evaluation.

## 8. IMPORTANT RESEARCH REMINDER

A good dataset preview does not automatically mean
a good AI model.

Step 06: Are the images technically usable?
Step 07: Can the AI learn from the training images?
Step 08: How accurately does it classify held-out MRIs?
Steps 09–16: How reliable are its predictions and explanations?

This remains a RESEARCH PROTOTYPE, not a medically
validated diagnostic system.
