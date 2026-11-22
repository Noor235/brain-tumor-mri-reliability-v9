"""STEP 14B — One REAL, randomly selected V9 validation MRI, explained visually.

Read-only companion to Step 14. Uses the EXACT same V9 model, split and perturbations.
Generates side-by-side MRI, Grad-CAM and overlay for Original / Noise / Brightness / 2px shift.
Calculates fixed-predicted-class CAM cosine, correlation and top-20% IoU.
No retraining; the original source images, checkpoint and Step 14 files are unchanged.

Run from project root:
    ./.venv/bin/python -m src.step14b_real_mri_stability_example
Optional:
    ./.venv/bin/python -m src.step14b_real_mri_stability_example --class meningioma --seed 24 --layer earlier
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from . import config as C
from .common import source_image, output_dir
from .research_v9 import (
    environment, ResearchCAM, get_layer, normalize_cam,
    cosine, pearson, top_iou,
)
from .step14_stability import variations  # Use EXACT Step 14 changes
from .transforms_v9 import transform


def norm_to_grayscale(x):
    # Input tensors are normalized into [-1,1] by V9 preprocessing.
    return np.uint8(np.clip((np.asarray(x) + 1.0) * 127.5, 0, 255).round())


def color_heatmap(heat):
    # No background masking; use same colormap across conditions.
    colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def probability_of(logits, idx):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    exp = np.exp(z)
    return float(exp[idx] / exp.sum())


def readable(val):
    return 'undefined (empty CAM)' if not np.isfinite(val) else f'{val:.4f}'


def main():
    p = argparse.ArgumentParser(description='Show Step 14 with one randomly chosen real V9 validation MRI')
    p.add_argument('--class', dest='target_class', choices=C.CLASSES, default=None,
                   help='Optional: select a specific true class; otherwise any class')
    p.add_argument('--layer', choices=('final', 'earlier'), default='final',
                   help='Which Grad-CAM layer to visualize')
    p.add_argument('--seed', type=int, default=2026,
                   help='Controls reproducible random selection and noise realization')
    args = p.parse_args()

    C.heading('STEP 14B / VISUAL EXAMPLE: SAME REAL MRI, SMALL CHANGES, GRAD-CAM STABILITY')
    rows, fingerprint, model, dev = environment('val')
    candidates = sorted((r for r in rows if r['split'] == 'val'
                         and (args.target_class is None or r['class_name'] == args.target_class)),
                        key=lambda r: r['file'])
    if not candidates:
        raise RuntimeError('No matching images found in V9 validation split.')
    rng = np.random.default_rng(args.seed)
    chosen = candidates[int(rng.integers(0, len(candidates)))]

    print('Dataset: V9 validation images ONLY (not test)')
    print('Random seed:', args.seed, '| class selection:', args.target_class or 'any')
    print('MRI file:', chosen['file'], '| true class:', chosen['class_name'])
    print('Layer:', args.layer, '| device:', dev)
    with Image.open(source_image(chosen['file'])) as im:
        gray = ImageOps.exif_transpose(im).convert('L')
    x = transform(False)(gray).unsqueeze(0).to(dev)
    raw_x = x.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    cam = ResearchCAM(model, get_layer(model, args.layer))
    try:
        with torch.no_grad():
            orig_logits = model(x).detach().cpu().numpy()[0]
        target = int(np.argmax(orig_logits))
        original_pred = C.CLASSES[target]
        base_raw, _ = cam.get(x, target)
        base = normalize_cam(base_raw)
        inputs = [
            ('Original', raw_x, 0, None),
            *[(name, changed, inverse_dx, None)
              for name, changed, inverse_dx in variations(raw_x, args.seed + 1)]
        ]
        entries = []
        for label, arr, inverse_dx, _ in inputs:
            xx = torch.from_numpy(np.asarray(arr, dtype=np.float32)[None, None]).to(dev)
            heat_raw, logits = cam.get(xx, target)
            heat = normalize_cam(heat_raw)
            # The displayed heatmap stays in the MODIFIED image's coordinates.
            # For measuring similarity, align a translated heatmap back to original coordinates.
            comparison_heat = heat
            if inverse_dx:
                comparison_heat = cv2.warpAffine(
                    heat, np.float32([[1, 0, inverse_dx], [0, 1, 0]]),
                    (C.IMAGE_SIZE, C.IMAGE_SIZE), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            predicted = int(np.argmax(logits))
            entries.append(dict(
                label=label, x=np.asarray(arr, dtype=np.float32), heat=heat,
                predicted=C.CLASSES[predicted], flip=predicted != target,
                original_class_probability=probability_of(logits, target),
                top_class_probability=probability_of(logits, predicted),
                cosine_similarity=cosine(base, comparison_heat),
                pearson_correlation=pearson(base, comparison_heat),
                top20_iou=top_iou(base, comparison_heat, .20),
            ))
    finally:
        cam.close()

    folder = output_dir('14b_real_mri_stability_example')
    part_name = f"{chosen['class_name']}_{args.layer}_seed{args.seed}"
    dest = folder / f'{part_name}.png'
    js = folder / f'{part_name}.json'
    if dest.exists() or js.exists():
        raise FileExistsError(f'Example already exists: {dest}\nUse a different --seed to produce another random MRI.')

    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(4, 3, figsize=(11.7, 17.0), dpi=140)
    fig.suptitle('KAGGLE V9  |  REAL MRI — GRAD-CAM EXPLANATION STABILITY',
                 fontsize=17, weight='bold', y=.991)
    fig.text(.5, .969,
             f"Validation MRI  •  True: {chosen['class_name']}  •  "
             f"Original prediction: {original_pred}  •  Layer: {args.layer}",
             ha='center', fontsize=11)
    column_names = ['MRI INPUT', 'GRAD-CAM HEATMAP', 'GRAD-CAM OVERLAY']
    for k, title in enumerate(column_names):
        axes[0, k].set_title(title, fontsize=12, fontweight='bold', pad=18)

    descriptions = {
        'Original': 'Original MRI (no changes)',
        'noise_std_0.01': 'Tiny noise added (std 0.01 on 0–1 scale)',
        'brightness_plus_0.02': 'Brightness increased by 0.02 (0–1 scale)',
        'shift_2px_right': 'MRI shifted 2 pixels right',
    }
    for idx, entry in enumerate(entries):
        grayscale = norm_to_grayscale(entry['x'])
        rgb = np.stack([grayscale] * 3, axis=-1)
        colored = color_heatmap(entry['heat'])
        overlay = np.uint8(np.clip(.60 * rgb.astype(np.float32) + .40 * colored.astype(np.float32), 0, 255))
        for j, picture in enumerate((grayscale, colored, overlay)):
            axes[idx, j].imshow(picture, cmap='gray' if j == 0 else None, vmin=0 if j == 0 else None, vmax=255 if j == 0 else None)
            axes[idx, j].set_axis_off()
        display = descriptions[entry['label']]
        if idx == 0:
            metric_line = f"Predicted: {entry['predicted']} | P(original class)={entry['original_class_probability']:.3%}"
        else:
            metric_line = (f"Predicted: {entry['predicted']} | Cosine={readable(entry['cosine_similarity'])}"
                           f" | Top-20% IoU={readable(entry['top20_iou'])}")
        axes[idx, 0].text(.5, -0.10, f'{display}\n{metric_line}',
                          transform=axes[idx, 0].transAxes, ha='center', va='top',
                          fontsize=9, color='black', clip_on=False)
        if idx > 0:
            axes[idx, 2].text(.5, -0.095, f"P(original class): {entry['original_class_probability']:.3%}"
                                   f"  |  Class changed: {'YES' if entry['flip'] else 'NO'}",
                              transform=axes[idx, 2].transAxes, ha='center', va='top',
                              fontsize=9, color='black', clip_on=False)

    fig.subplots_adjust(left=.045, right=.955, top=.925, bottom=.145, hspace=.54, wspace=.035)
    fig.text(.5, .037,
             'Interpretation: Higher cosine (closer to 1) and higher top-20% IoU mean more similar Grad-CAM maps.\n'
             'For the 2-pixel shift, the heatmap is aligned back BEFORE scoring similarity (not in the displayed overlay).\n'
             'An unchanged explanation can still be medically wrong. This is ONE randomly selected MRI, not an average.',
             ha='center', va='center', fontsize=9, color='#333333')
    fig.savefig(dest, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    summary = {
        'source': 'One randomly selected REAL Kaggle V9 validation MRI',
        'source_image': chosen['file'], 'true_class': chosen['class_name'],
        'original_predicted_class': original_pred, 'random_seed': args.seed,
        'layer': args.layer, 'split_fingerprint': fingerprint,
        'fixed_gradcam_target': original_pred,
        'rows': [{k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                          else float(v) if isinstance(v, np.floating) else v)
                  for k, v in entry.items() if k not in ('x','heat')}
                 for entry in entries],
        'note': 'Changes from Step 14, no image/source/checkpoint modified. Shift CAM aligned for metrics. Single image; not representative.'
    }
    with js.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    print('\nREAL MRI stability result (same Grad-CAM target for all conditions):')
    for e in entries:
        if e['label'] == 'Original':
            print(f"  Original    prediction={e['predicted']}, confidence={e['top_class_probability']:.3%}")
        else:
            print(f"  {e['label']:<25} cosine={readable(e['cosine_similarity'])}  "
                  f"IoU={readable(e['top20_iou'])}  class_flip={'YES' if e['flip'] else 'NO'}")
    print('Saved actual MRI + heatmaps:', dest)
    print('Saved per-image measurements:', js)
    print('No training, dataset edits, or model checkpoint changes.')


if __name__ == '__main__':
    main()
