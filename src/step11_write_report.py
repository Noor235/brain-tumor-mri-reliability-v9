"""STEP 11: combine reproducible, measured metrics into an editable Markdown draft."""
import json
from . import config as C
from .common import output_dir

def get(name):
    path=C.OUTPUT/name
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

def main():
    C.heading('STEP 11 / THESIS METHODOLOGY & RESULTS REPORT')
    inv=get('01_inventory/inventory_summary.json');exact=get('02_exact_duplicates/exact_summary.json')
    near=get('03_near_duplicates/near_summary.json');splits=get('04_split/split_summary.json')
    verified=get('05_verification/verification.json');training=get('07_training/training_summary.json')
    evaluation=get('08_test/test_metrics.json');mc=get('09_uncertainty/mc_summary.json')
    lines=['# Kaggle V9 — Leakage-Controlled Brain Tumor MRI Classification','',
      '## Dataset and provenance','',
      'Dataset 2: newer Kaggle Brain Tumor MRI four-class version (nominal 7,200 images); verify precise Kaggle version/URL for final thesis citation.',
      '',f'Original V9 input: {inv["image_count"] if inv else "NOT RUN"} images. Source images are never modified.',
      '', '## Duplicate control and experimental design','',
      'Exact duplicate detection uses SHA256 of original file bytes and SHA256 of EXIF-corrected decoded grayscale pixels (dimensions included).',
      'One representative per exact-duplicate group is retained in the V9 manifest; original files remain untouched.',
      f'Excluded additional exact copies: {exact["exact_duplicate_extra_copies_excluded"] if exact else "NOT RUN"}.',
      'Near-duplicate screening uses perceptual hash (Hamming <= 8) plus 96x96-image SSIM >= 0.985.',
      'Potentially similar images are grouped together for splitting, NOT deleted or asserted to be the same patient.',
      f'Similarity candidate pairs: {near["potential_near_duplicate_pairs"] if near else "NOT RUN"}; similarity groups: {near["image_similarity_groups"] if near else "NOT RUN"}.',
      f'Mixed-label near-similarity groups excluded from all V9 splits: {splits.get("excluded_mixed_label_groups","NOT RUN") if splits else "NOT RUN"}; total excluded representative images: {splits.get("excluded_mixed_label_images","NOT RUN") if splits else "NOT RUN"}. This is a conservative data-quality exclusion, not a verified correction of tumor diagnoses.',
      'Train/test separation uses stratified group fivefold partitioning. Test holds approximately 20%; validation holds approximately 20% of the remaining images.',
      f'Actual split counts: {splits["counts"] if splits else "NOT RUN"}.',
      f'Verification status: {verified["status"] if verified else "NOT RUN"}.',
      '', '## Model architecture and training','',
      'EfficientNet-B4 (ImageNet pretrained) adapted to 1-channel grayscale by averaging first-convolution RGB kernels. Sequential channel attention and 7x7 spatial attention; dropout p=0.4; four-class linear classifier.',
      'Preprocessing: grayscale conversion, 224x224 bicubic, mean 0.5/std 0.5. Training-only flip p=.5, rotation ±15 degrees, brightness and contrast ±20%, Gaussian noise SD=.02.',
      'AdamW, lr 1e-4, weight decay 0.01, batch size 16, max 20 epochs. BatchNorm running statistics remain frozen, as a documented deviation from unspecified paper practice.',
      'Checkpoint selected on validation accuracy (then macro F1). Held-out V9 test labels are never consulted during training or selection.',
      f'Best validation accuracy: {training["best_validation_accuracy"]:.4%}.' if training else 'Best validation accuracy: NOT TRAINED.',
      '', '## Final held-out test results','',
      (f'Test accuracy: {evaluation["accuracy"]:.4%} ({evaluation["correct"]}/{evaluation["test_count"]}); macro-F1: {evaluation["macro_f1"]:.4f}.') if evaluation else 'Test evaluation: NOT RUN.',
      (f'Per-class recall: {evaluation["recalls"]}.') if evaluation else '',
      '', '## Monte Carlo prediction uncertainty','',
      (f'MC dropout ({mc["mc_passes"]} passes) test accuracy: {mc["mc_test_accuracy"]:.4%}; mean entropy correct: {mc["mean_entropy_correct"]}; incorrect: {mc["mean_entropy_incorrect"]}.') if mc else 'MC uncertainty: NOT RUN.',
      '', '## Comparison scope and dataset distinction','',
      'V9 uses a newer source dataset than V8. Cross-dataset exact and near duplicate overlap must be checked before asserting external independence.',
      'V8 and V9 use different source datasets and test sets; comparing their accuracy does not isolate the cause of performance differences.',
      'Original published accuracy belongs to its specific dataset and protocol. This V9 reproduction uses a different data version and grouped split.',
      '', '## Important limitations','',
      '- No verified Kaggle patient IDs; zero identified duplicate overlap is NOT proof of patient-level independence.',
      '- Similarity thresholds are heuristic: false-positive grouping can occur, and undetected near duplicates can remain.',
      '- Test samples can resemble prior evaluation samples; accuracy comparisons after repeated experimentation are exploratory.',
      '- No claim is made that comparable numeric accuracy verifies the exact published experiment or clinical generalizability.',
      '- Grad-CAM localization and prediction faithfulness require separate quantitative testing.',
      '', '## Reproducibility artifacts','',
      'results/01_inventory through results/11_report contain image manifests, duplicate reports, near-match montage, fixed split, verification, training history, classification metrics, uncertainty data, Grad-CAM images, and this methodology summary.',
      '']
    dest=output_dir('11_report')/'V9_methodology_and_results.md'
    dest.write_text('\n'.join(lines),encoding='utf-8')
    print('Saved editable report draft:',dest,flush=True)
    print('NOTE: all metrics only appear after running the corresponding experiment stages.',flush=True)
if __name__=='__main__':main()
