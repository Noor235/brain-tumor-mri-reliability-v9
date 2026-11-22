"""STEP 17: Merge measured Kaggle V9 outputs into an editable research report.
The script does NOT invent missing results. Produces Markdown and optional Word document.
"""
import json
from pathlib import Path
from . import config as C
from .common import output_dir


def read(path):
    p=C.OUTPUT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None

def fmt_percent(v):return f'{v:.2%}' if v is not None else 'NOT AVAILABLE'

def main():
    C.heading('STEP 17 / V9 FINAL EXPERIMENT REPORT — ONLY VERIFIED MEASUREMENTS')
    exact=read('02_exact_duplicates/exact_summary.json')
    split=read('04_split/split_summary.json')
    val=read('07_training/training_summary.json')
    test=read('08_test/test_metrics.json')
    mc=read('09_uncertainty/mc_summary.json')
    cal=read('12_calibration/calibration_summary.json')
    faith=read('13_faithfulness_val/faithfulness_summary.json')
    stable=read('14_stability_val/stability_summary.json')
    expl=read('15_explanation_uncertainty_test_final/uncertainty_summary.json')
    integrated=read('16_integrated_test_final/integrated_summary.json')
    lines=['# Kaggle V9: Uncertainty-Aware Reliability Assessment of Brain Tumour MRI Classification','',
        '## Aim and scope','',
        'Assess predictive performance, predictive uncertainty, confidence calibration, Grad-CAM faithfulness, explanation stability, explanation uncertainty, and their relationships.',
        '', '## Data preparation and methodological safeguards','',
        'Source: newer four-class Kaggle MRI dataset, nominal 7,200 input images (exact dataset version/provenance to confirm).',
        f'Exact duplicate copies excluded from manifest: {exact.get("exact_duplicate_extra_copies_excluded","NOT AVAILABLE") if exact else "NOT AVAILABLE"}. Original image files untouched.',
        f'New group-preserving split: {split.get("counts", "NOT AVAILABLE") if split else "NOT AVAILABLE"}.',
        f'Mixed-label near-similarity groups excluded: {split.get("excluded_mixed_label_groups","NOT AVAILABLE") if split else "NOT AVAILABLE"}; representative images excluded: {split.get("excluded_mixed_label_images","NOT AVAILABLE") if split else "NOT AVAILABLE"}. Labels were not changed; excluded original images were not deleted.',
        'Near duplicates were screened with perceptual hash and SSIM and grouped across train/validation/test, not automatically deleted.',
        'Exact-image and detected near-group overlap were verified absent; patient-disjoint splitting was NOT verified.',
        '', '## Classification architecture and training','',
        'ImageNet EfficientNet-B4 with one-channel adaptation, sequential channel/spatial attention and dropout 0.4.',
        'Input 224×224; grayscale and bicubic resize; AdamW learning rate 1e-4; weight decay 0.01; batch size 16; 20 epochs.',
        'BatchNorm running statistics frozen; checkpoint selected by validation (not test).',
        f'Best validation accuracy: {fmt_percent(val.get("best_validation_accuracy") if val else None)}.',
        f'One-time V9 test accuracy: {fmt_percent(test.get("accuracy") if test else None)}' + (f' ({test["correct"]}/{test["test_count"]}).' if test else '.'),
        f'Test macro-F1: {test["macro_f1"]:.4f}.' if test else 'Test macro-F1: NOT AVAILABLE.',
        '', '## Prediction uncertainty','',
        f'MC dropout passes: {mc.get("mc_passes", "NOT AVAILABLE") if mc else "NOT AVAILABLE"}.',
        f'MC accuracy: {fmt_percent(mc.get("mc_test_accuracy") if mc else None)}.',
        (f'Mean predictive entropy: correct={mc["mean_entropy_correct"]:.6f}, incorrect={mc["mean_entropy_incorrect"]:.6f}.') if mc and mc.get('mean_entropy_correct') is not None and mc.get('mean_entropy_incorrect') is not None else 'Mean entropy by correctness: NOT AVAILABLE.',
        'Predictive entropy represents total predictive uncertainty; dropout mutual information is a proxy for model uncertainty, not a calibrated medical-risk estimate.',
        '', '## Confidence calibration','',
        (f'Validation-fitted temperature: {cal["temperature"]:.4f}. Test ECE before: {cal["calibration"]["test_raw"]["ece"]:.5f}, after: {cal["calibration"]["test_temperature_scaled"]["ece"]:.5f}.') if cal else 'Temperature scaling and ECE: NOT RUN.',
        (f'Test Brier before/after: {cal["calibration"]["test_raw"]["brier"]:.5f}/{cal["calibration"]["test_temperature_scaled"]["brier"]:.5f}; NLL before/after: {cal["calibration"]["test_raw"]["nll"]:.5f}/{cal["calibration"]["test_temperature_scaled"]["nll"]:.5f}.') if cal else 'Brier score and NLL: NOT RUN.',
        '', '## Explanation faithfulness','',
        f'Validation Grad-CAM deletion/insertion test: {faith["examples"]} images, compared high-importance against low-importance and random controls.' if faith else 'Quantitative faithfulness: NOT RUN.',
        'Interpretation: positive high-minus-random deletion score suggests high-CAM patches influence the fixed target margin more than random patches; not sufficient for medical localization.',
        '', '## Explanation stability','',
        f'Validation explanation stability: {stable["examples"]} images with noise, intensity and aligned translations.' if stable else 'Quantitative explanation stability: NOT RUN.',
        '', '## Explanation uncertainty','',
        (f'Test MC Grad-CAM exploratory subset: {expl["examples"]} images, {expl["mc_passes"]} passes per image; mean normalized attribution-map standard deviation={expl["mean_normalized_spatial_attribution_std"]:.6f}.') if expl else 'Stochastic Grad-CAM uncertainty: NOT RUN.',
        'Per-pass Grad-CAM maps were normalized before spatial variation was summarized; this is a descriptive shape-uncertainty measurement rather than a validated Bayesian uncertainty measure.',
        '', '## Integrated prediction–explanation reliability','',
        f'Per-image integrated profiles: {integrated["matched_examples"]} matched cases.' if integrated else 'Integrated reliability profiles: NOT RUN.',
        'Interpret correlations cautiously; the number of matched cases and repeat testing limit any inference.',
        '', '## Main limitations and thesis alignment','',
        '1. V9 uses a different input dataset and cleaned split from V8; any accuracy differences do not isolate a single cause.',
        '2. Patient IDs and patient independence were not verified.',
        '3. Repeated analysis on the same reserved test set is post-hoc and exploratory, not evidence of new independent validation.',
        '4. A Grad-CAM heatmap that visually overlaps anatomy does not guarantee explanation faithfulness.',
        '5. Calibration, faithfulness, stability, and explanation-uncertainty stages should retain their chosen analysis protocols before looking at test outcomes.',
        '', '## Files for evidence and reproducibility','',
        'See results/02_exact_duplicates, 04_split, 05_verification, 07_training, 08_test, 09_uncertainty, 10_gradcam, 12_calibration, 13_faithfulness_val, 14_stability_val, 15_explanation_uncertainty_test_final, 16_integrated_test_final.',
    ]
    content='\n'.join(lines)+'\n'
    out=output_dir('17_thesis_report')
    path=out/'Kaggle_V9_thesis_results.md';path.write_text(content,encoding='utf-8')
    print('Saved Markdown:',path,flush=True)
    try:
        from docx import Document
        from docx.shared import Pt
        doc=Document()
        normal=doc.styles['Normal'];normal.font.name='Aptos';normal.font.size=Pt(10)
        for line in lines:
            if not line:continue
            if line.startswith('# '):doc.add_heading(line[2:],0)
            elif line.startswith('## '):doc.add_heading(line[3:],1)
            else:doc.add_paragraph(line)
        docx=out/'Kaggle_V9_thesis_results.docx';doc.save(docx)
        print('Saved editable Word file:',docx,flush=True)
    except ImportError:
        print('Optional Word export skipped: install python-docx to also generate .docx.',flush=True)
    print('STEP 17 COMPLETE — report entries are tagged NOT RUN/NOT AVAILABLE when missing.',flush=True)
if __name__=='__main__':main()
