"""STEP 26: Fresh, unbiased random MRI from the V9 split + LIVE MC uncertainty.

Designed to run inside the existing Brain_Tumor_Reproduction_V9 project with
src.step25_uncertainty_visual_demo already installed from the previous lab.

Run from project root:
    python3 -m src.step26_random_mri_uncertainty
Optional reproducible image selection:
    python3 -m src.step26_random_mri_uncertainty --seed 20261119

Uses the V9 split_manifest.csv, *not* the original Kaggle Testing folder.
Only the classifier's dropout is activated; it does not retrain or edit images.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
from datetime import datetime
from pathlib import Path

from . import config as C
from .common import source_image
from .step25_uncertainty_visual_demo import (
    VALIDATION_ENTROPY_CUTOFF, get_device, load_v9_model,
    predict_twenty_times, save_case_figure,
)


def select_random_test_row(manifest: Path, split: str, seed: int | None):
    if not manifest.is_file():
        raise FileNotFoundError(
            f'Cannot find the V9 split manifest: {manifest}\n'
            'This file was created by Step 04. Please use your original V9 project.'
        )
    with manifest.open(newline='', encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r.get('split') == split]
    if not rows:
        raise ValueError(f'No MRI images found for split={split!r} in {manifest}')
    # When seed is omitted this uses the operating system's unpredictable randomness.
    idx = secrets.randbelow(len(rows)) if seed is None else random.Random(seed).randrange(len(rows))
    row = rows[idx]
    if row.get('class_name') not in C.CLASSES:
        raise ValueError(f'Unexpected label in V9 manifest: {row.get("class_name")}')
    return row, idx, len(rows)


def main():
    parser = argparse.ArgumentParser(description='Pick a genuinely random V9 test MRI and visualize 20 MC dropout passes.')
    parser.add_argument('--split', choices=['test', 'val'], default='test', help='Use V9 held-out test by default; not original Kaggle Testing folder.')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed to REPRODUCE the random image selection.')
    parser.add_argument('--mc-seed', type=int, default=42, help='Seed for stochastic dropout forward passes.')
    parser.add_argument('--passes', type=int, default=20, help='20 was used in your original V9 experiments.')
    parser.add_argument('--device', choices=['auto','mps','cpu','cuda'], default='auto')
    parser.add_argument('--checkpoint', type=Path, default=C.CHECKPOINT)
    parser.add_argument('--out', type=Path, default=C.OUTPUT/'26_random_mri_uncertainty')
    args = parser.parse_args()
    if args.passes < 2:
        parser.error('--passes must be >= 2')

    row, idx, count = select_random_test_row(C.SPLIT, args.split, args.seed)
    file_path = source_image(row['file'])
    print(f'\nRANDOM V9 MRI: index {idx+1} / {count} in the {args.split} split')
    print(f'Original source file: {row["file"]}')
    print(f'Random seed: {args.seed if args.seed is not None else "unseeded (fresh each run)"}')
    print('Loading the original V9 checkpoint...')
    device = get_device(args.device)
    model = load_v9_model(args.checkpoint, device)
    print(f'Running {args.passes} real MC Dropout passes on {device}...')
    result = predict_twenty_times(model, file_path, device, passes=args.passes, seed=args.mc_seed)
    label = row['class_name']
    predicted = result['det_prediction']
    correct = predicted == label

    # A separate folder per run avoids overwriting previous random experiments.
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3)
    output = args.out.expanduser().resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    case_info = {'true_class': label, 'source_file': row['file']}
    figure = save_case_figure('random_v9_mri', case_info, result, output)

    columns = ['pass'] + [f'p_{c}' for c in C.CLASSES]
    with (output/'20_MC_passes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for n, p in enumerate(result['mc_probs'], 1):
            writer.writerow([n] + [float(v) for v in p])

    report = {
        'file': row['file'], 'V9_split': args.split,
        'selected_index_one_based': idx+1, 'split_size': count,
        'selection_seed': args.seed, 'mc_seed': args.mc_seed,
        'checkpoint': str(args.checkpoint), 'passes': args.passes,
        'known_class': label, 'deterministic_prediction': predicted,
        'correct': correct, 'deterministic_confidence': result['det_confidence'],
        'mc_prediction': result['mc_prediction'],
        'mc_mean_confidence': result['mc_mean_confidence'],
        'mean_mc_probabilities': {k:float(p) for k,p in zip(C.CLASSES,result['mc_mean'])},
        'predictive_entropy': result['predictive_entropy'],
        'mutual_information': result['mutual_information'],
        'validation_entropy_cutoff': VALIDATION_ENTROPY_CUTOFF,
        'refer_for_review': result['referral'],
        'interpretation': 'Uncertainty cannot confirm a diagnosis or prediction correctness.',
    }
    (output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n--- REAL RANDOM MRI RESULT ---')
    print(f'Known class:             {label}')
    print(f'V9 predicted:            {predicted}')
    print(f'Correct for this image:  {correct}')
    print(f'Deterministic confidence:{result["det_confidence"]:>9.2%}')
    print(f'MC mean confidence:      {result["mc_mean_confidence"]:>9.2%}')
    print(f'Predictive entropy:      {result["predictive_entropy"]:.6f}')
    print(f'Mutual information:      {result["mutual_information"]:.6f}')
    print('Uncertainty warning:    ' + ('REFER FOR REVIEW' if result['referral'] else 'NO UNCERTAINTY REFERRAL (not verified)'))
    print(f'\nOpen your visualization: {figure}')
    print(f'Full details:            {output / "result.json"}')
    print('Low uncertainty does not prove correctness. Not for clinical use.')


if __name__ == '__main__':
    main()
