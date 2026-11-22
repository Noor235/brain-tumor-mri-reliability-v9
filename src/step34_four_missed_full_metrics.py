"""Step 34: Re-run all FIVE V9 reliability measurements on the four known
MC-dropout false negatives, using the FINAL HiResCAM layer and Step 33 protocol.

Run from the V9 project root:
    python3 -m src.step34_four_missed_full_metrics

Optional quick saved-results inspection, without re-running inference:
    python3 -m src.step34_four_missed_full_metrics --saved-only

This is a diagnostic demonstration, NOT a new accuracy evaluation, training,
clinical validation, or validated five-metric trust decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from . import config as C
from .common import source_image
from .research_v9 import environment, get_layer
from .transforms_v9 import transform
from . import step22_full_test_reliability as s22
from . import step33_hirescam_full_test as s33

# These are EXACT IDs in the original V9 1,374-MRI TEST split; the original
# Kaggle Training/Testing directory names are NOT the V9 split assignments.
FOUR = [
    'Testing/glioma/Te-gl_159.jpg',
    'Testing/glioma/Te-gl_58.jpg',
    'Training/meningioma/Tr-me_289.jpg',
    'Training/meningioma/Tr-me_964.jpg',
]
# From Step 24: selected using validation MC entropy; DO NOT refit on 4 test cases.
REFERRAL_THRESHOLD = 0.00148676
# Whole V9 full-test HiResCAM reference mean, DESCRIPTIVE only.
HIRESCAM_SPATIAL_STD_REFERENCE = 0.008284
PROFILE_PATH = C.OUTPUT / '33_hirescam_full_test' / 'earlier' / 'per_image_reliability_profile.csv'
FIELDS = [
    'file', 'true_class', 'predicted_class', 'correct',
    'calibrated_confidence', 'mc_predictive_entropy', 'mc_mutual_information',
    'faithfulness_del_high_minus_random_20',
    'faithfulness_ins_high_minus_random_20',
    'mean_stability_cosine', 'mean_stability_top20_iou',
    'explanation_shape_uncertainty', 'mc_referral',
]


def saved_cases():
    if not PROFILE_PATH.is_file():
        raise FileNotFoundError('Final Step 33 profile not found: ' + str(PROFILE_PATH))
    with PROFILE_PATH.open(newline='', encoding='utf-8') as fp:
        all_rows = {r['file']: r for r in csv.DictReader(fp)}
    if len(all_rows) != 1374:
        raise RuntimeError(f'Expected 1374 Step 33 profiles; found {len(all_rows)}')
    missing = set(FOUR) - set(all_rows)
    if missing:
        raise RuntimeError(f'Missing images in Step 33: {sorted(missing)}')
    result = []
    for file in FOUR:
        r = all_rows[file]
        if int(r['deterministic_correct']) != 0:
            raise RuntimeError('Expected a known wrong prediction: ' + file)
        result.append({
            'file': file,
            'true_class': r['true_class'],
            'predicted_class': r['deterministic_predicted_class'],
            'correct': 0,
            'calibrated_confidence': float(r['calibrated_confidence']),
            'mc_predictive_entropy': float(r['mc_predictive_entropy']),
            'mc_mutual_information': float(r['mc_mutual_information']),
            'faithfulness_del_high_minus_random_20': float(r['faithfulness_del_high_minus_random_20']),
            'faithfulness_ins_high_minus_random_20': float(r['faithfulness_ins_high_minus_random_20']),
            'mean_stability_cosine': float(r['mean_stability_cosine']),
            'mean_stability_top20_iou': float(r['mean_stability_top20_iou']),
            'explanation_shape_uncertainty': float(r['explanation_shape_uncertainty']),
            'mc_referral': 'REFER' if float(r['mc_predictive_entropy']) > REFERRAL_THRESHOLD else 'NO WARNING',
        })
    return result


def probability(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def show(title, rows):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)
    for n, r in enumerate(rows, 1):
        print(f'\n[{n}/{len(rows)}] {r["file"]}')
        print(f'  Known label:                    {r["true_class"]}')
        print(f'  V9 predicted:                  {r["predicted_class"]}  -- WRONG')
        print(f'  (1) Calibrated confidence:     {100*r["calibrated_confidence"]:.4f}%')
        print(f'  (2) MC predictive entropy:     {r["mc_predictive_entropy"]:.8f} '
              f'(threshold {REFERRAL_THRESHOLD:.8f})')
        mi = r['mc_mutual_information']
        print(f'      MC mutual information:     {float(mi):.8f}' if mi is not None else
              '      MC mutual information:     see SAVED Step 09 (not recalculated in fresh CAM passes)')
        print(f'      MC warning:                {r["mc_referral"]}')
        print(f'  (3) HiResCAM faithfulness:     deletion advantage '
              f'{r["faithfulness_del_high_minus_random_20"]:+.4f}, '
              f'insertion {r["faithfulness_ins_high_minus_random_20"]:+.4f}')
        print(f'  (4) HiResCAM stability:        cosine {r["mean_stability_cosine"]:.6f}, '
              f'top-20% overlap {r["mean_stability_top20_iou"]:.6f}')
        print(f'  (5) HiResCAM expl. uncertainty: {r["explanation_shape_uncertainty"]:.6f} '
              f'(dataset mean ~{HIRESCAM_SPATIAL_STD_REFERENCE:.6f})')
        print('  CONCLUSION: known incorrect label; 5 metrics cannot verify correctness.')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: r[k] for k in FIELDS} for r in rows])


def create_figure(rows, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 2, figsize=(13, 15), gridspec_kw={'width_ratios': [1, 2.4]})
    for i, r in enumerate(rows):
        with Image.open(source_image(r['file'])) as orig:
            image = ImageOps.exif_transpose(orig).convert('L')
            data = np.asarray(image)
        a, b = axs[i]
        a.imshow(data, cmap='gray')
        a.set_title(Path(r['file']).name, fontsize=11)
        a.axis('off')
        b.axis('off')
        content = '\n'.join([
            f'ACTUAL: {r["true_class"].upper()}  /  PREDICTED: {r["predicted_class"].upper()}  [WRONG]',
            f'1. Confidence (calibrated): {r["calibrated_confidence"]:.2%}',
            f'2. MC predictive entropy: {r["mc_predictive_entropy"]:.6f}  -- {r["mc_referral"]}',
            f'3. HiResCAM faithfulness: deletion {r["faithfulness_del_high_minus_random_20"]:+.3f}, '
            f'insertion {r["faithfulness_ins_high_minus_random_20"]:+.3f}',
            f'4. HiResCAM stability (cosine): {r["mean_stability_cosine"]:.4f}',
            f'5. HiResCAM explanation std: {r["explanation_shape_uncertainty"]:.6f}',
            'Known-label comparison: WRONG. No metric is a diagnosis verifier.',
        ])
        b.text(0.02, .95, content, va='top', ha='left', fontsize=11,
               linespacing=1.6, transform=b.transAxes, family='monospace', wrap=True)
    fig.suptitle(f'V9 — Four known errors missed by saved MC Dropout | {label}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path, saved, live=None):
    lines = [
        '# V9 — Four MC-dropout false negatives: five-measurement audit',
        '',
        f'Warning threshold from V9 validation: predictive entropy > {REFERRAL_THRESHOLD:.8f}.',
        'The four cases were selected *because* they are wrong and below the saved MC threshold.',
        'This is a targeted case study, not a new independent evaluation of error detection.',
        '',
        '## Saved Step 33 values',
        '',
        '| MRI | True → predicted | Conf. | Saved MC entropy | Del. | Ins. | Stability | Explanation std |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in saved:
        lines.append(f'| {Path(r["file"]).name} | {r["true_class"]} → {r["predicted_class"]} | '
                     f'{r["calibrated_confidence"]:.2%} | {r["mc_predictive_entropy"]:.6f} | '
                     f'{r["faithfulness_del_high_minus_random_20"]:+.3f} | '
                     f'{r["faithfulness_ins_high_minus_random_20"]:+.3f} | '
                     f'{r["mean_stability_cosine"]:.4f} | {r["explanation_shape_uncertainty"]:.6f} |')
    if live is not None:
        lines += ['', '## Fresh checkpoint inference (20 fresh MC passes per MRI)', '',
                  'Fresh MC entropy may differ slightly from previously saved Step 09 values because dropout is stochastic.',
                  'Confidence is recalculated using the saved validation-fit temperature.',
                  'HiResCAM faithfulness, stability and explanation uncertainty use the original Step 33 protocol.', '',
                  '| MRI | Live conf. | Live entropy | Live MC warning | Deletion | Insertion | Stability | Explanation std |',
                  '|---|---:|---:|---|---:|---:|---:|---:|']
        for r in live:
            lines.append(f'| {Path(r["file"]).name} | {r["calibrated_confidence"]:.2%} | '
                         f'{r["mc_predictive_entropy"]:.6f} | {r["mc_referral"]} | '
                         f'{r["faithfulness_del_high_minus_random_20"]:+.3f} | '
                         f'{r["faithfulness_ins_high_minus_random_20"]:+.3f} | '
                         f'{r["mean_stability_cosine"]:.4f} | {r["explanation_shape_uncertainty"]:.6f} |')
    lines += ['', '## Interpret carefully', '',
              '- **Confidence** is an estimated probability, not a probability that the true diagnosis was independently verified.',
              '- **MC predictive entropy** is used for the validation-selected referral rule; below threshold does not imply correct.',
              '- **HiResCAM faithfulness** measures influence under fixed blur perturbations relative to random regions, not tumour location.',
              '- **Stability** measures map similarity under limited small perturbations, not clinical repeatability.',
              '- **Explanation uncertainty** measures normalized map variability across classifier dropout passes; below the dataset mean is descriptive, not a validated safety threshold.',
              '- All four are **known classification errors** when compared against their dataset labels.',
              '- The V9 test dataset has been examined repeatedly; no independent clinical/patient-level validation is claimed.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-only', action='store_true', help='Only inspect existing Step 33 results (no model inference)')
    args = parser.parse_args(argv)
    saved = saved_cases()
    show('SAVED STEP 33 PROFILES', saved)
    if args.saved_only:
        return

    # Full inference: same V9 checkpoint, split, image preprocessing, faithfulness
    # masks, stability perturbations, and 20-pass HiResCAM protocol as Step 33.
    rows, fingerprint, model, dev = environment('test')
    test_rows = sorted((r for r in rows if r['split'] == 'test'), key=lambda r: r['file'])
    original = s22.load_original(test_rows, fingerprint)
    by_file = {r['file']: r for r in test_rows}
    if any(f not in by_file for f in FOUR):
        raise RuntimeError('One of the four images is not in the verified V9 test split')
    summary_path = C.OUTPUT / '12_calibration' / 'calibration_summary.json'
    temperature = float(json.loads(summary_path.read_text())['temperature'])
    if temperature <= 0 or not math.isfinite(temperature):
        raise RuntimeError('Invalid saved calibration temperature')
    out = C.OUTPUT / '34_four_missed_five_metrics' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    print(f'\nLIVE INFERENCE: 4 images, device={dev}, checkpoint={C.CHECKPOINT}')
    print(f'Using validation-fit temperature T={temperature:.6f} and 20 MC passes/image.')
    print('Images are read-only. New output:', out)
    model.eval()
    tx = transform(False)
    cam = s33.HiResCAM(model, get_layer(model, 'earlier'))
    live = []
    try:
        for i, file in enumerate(FOUR, 1):
            r = by_file[file]
            with Image.open(source_image(file)) as im:
                im = ImageOps.exif_transpose(im).convert('L')
                x = tx(im).unsqueeze(0).to(dev)
            with torch.inference_mode():
                logits = model(x).detach().cpu().numpy()[0]
            pred = C.CLASSES[int(np.argmax(logits))]
            if pred != original['test'][file]['predicted_class']:
                raise RuntimeError(f'Live prediction differs from saved V9 prediction for {file}: {pred}')
            conf = float(probability(logits, temperature).max())
            print(f'\nRunning full five-measurement HiResCAM evaluation for case {i}/4: {file}', flush=True)
            computed = s22.record_one(r, 'earlier', i, model, dev, cam, tx, original, out, set(FOUR))
            combined = s22.combine_one(computed, original)
            fresh_ent = float(computed['explanation']['predictive_entropy_from_same_passes'])
            mc_probs_note = 'REFER' if fresh_ent > REFERRAL_THRESHOLD else 'NO WARNING'
            result = {
                'file': file, 'true_class': r['class_name'], 'predicted_class': pred,
                'correct': int(pred == r['class_name']), 'calibrated_confidence': conf,
                'mc_predictive_entropy': fresh_ent,
                # Original Step 09 mutual information stays in saved profiles;
                # it is not recalculated by Step 33's image-level CAM routine.
                'mc_mutual_information': None,
                'faithfulness_del_high_minus_random_20': combined['faithfulness_del_high_minus_random_20'],
                'faithfulness_ins_high_minus_random_20': combined['faithfulness_ins_high_minus_random_20'],
                'mean_stability_cosine': combined['mean_stability_cosine'],
                'mean_stability_top20_iou': combined['mean_stability_top20_iou'],
                'explanation_shape_uncertainty': combined['explanation_shape_uncertainty'],
                'mc_referral': mc_probs_note,
            }
            live.append(result)
            show('NEWLY CALCULATED RESULT', [result])
            # Save intermediate progress in case the Mac interrupts the experiment.
            write_csv(out / 'live_four_metrics.csv', live)
    finally:
        cam.close()
    write_csv(out / 'saved_step33_four_metrics.csv', saved)
    create_figure(live, out / 'four_real_mris_five_metrics.png', 'fresh model inference')
    write_report(out / 'FOUR_MRI_FIVE_METRICS_REPORT.md', saved, live)
    print('\nSUCCESS. Four live cases complete. See:', out)
    print('  four_real_mris_five_metrics.png')
    print('  live_four_metrics.csv')
    print('  saved_step33_four_metrics.csv')
    print('  FOUR_MRI_FIVE_METRICS_REPORT.md')
    print('  example_maps/earlier/*.png (HiResCAM and uncertainty heatmaps)')
    print('All four remain known wrong labels. A low uncertainty score is NOT verification.')


if __name__ == '__main__':
    main()
