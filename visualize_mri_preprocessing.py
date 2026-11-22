#!/usr/bin/env python3
"""Create a thesis-ready preprocessing/augmentation figure from an ACTUAL MRI image.

Usage:
  python3 visualize_mri_preprocessing.py /path/to/actual_mri.jpg
  python3 visualize_mri_preprocessing.py /path/to/actual_mri.png --output figure_3_3.png

Required: pillow, numpy, matplotlib.

Transforms illustrate the DOCUMENTED Chapter 3 settings:
  grayscale, bicubic 224 x 224, pixel intensity in [0, 1], normalize (x - 0.5)/0.5;
  training-only demonstrations of flip, rotation +/-15 degrees, brightness/contrast
  up to 20%, and Gaussian noise SD 0.02 (on [0, 1] tensor before normalization).
For exact order and probability of your model's training transforms, consult
and cite your saved source code; this image is a reproducible illustration.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', help='Path to a real MRI .jpg/.png from your dataset')
    p.add_argument('--output', default='figure_3_3_real_preprocessing.png', help='Output PNG file')
    args = p.parse_args()
    src = Path(args.image).expanduser()
    if not src.is_file():
        p.error(f'MRI image not found: {src}')

    with Image.open(src) as im:
        original = im.copy()
    grayscale = original.convert('L')
    resized = grayscale.resize((224, 224), resample=Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    normalized = (array - 0.5) / 0.5

    flip = ImageOps.mirror(resized)
    rotation = resized.rotate(15, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
    brighter = ImageEnhance.Contrast(ImageEnhance.Brightness(resized).enhance(1.20)).enhance(1.20)
    noise = np.clip(array + np.random.default_rng(42).normal(0, 0.02, array.shape), 0.0, 1.0)

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 7.2), constrained_layout=True)
    items = [
        ('(a) Original MRI\n' + f'{original.width} x {original.height}', original),
        ('(b) Grayscale conversion\n1 channel', grayscale),
        ('(c) Bicubic resize\n224 x 224', resized),
        ('(d) Normalized tensor\n(x - 0.5) / 0.5', normalized),
        ('(e) Horizontal flip\ntraining augmentation', flip),
        ('(f) Rotation +15 deg\ntraining augmentation', rotation),
        ('(g) Brightness / contrast +20%\ntraining augmentation', brighter),
        ('(h) Gaussian noise SD = 0.02\ntraining augmentation', noise),
    ]
    for ax, (heading, im) in zip(axes.flat, items):
        if isinstance(im, Image.Image):
            ax.imshow(im, cmap='gray', vmin=0, vmax=255 if im.mode == 'L' else None)
        elif heading.startswith('(d)'):
            ax.imshow(im, cmap='gray', vmin=-1, vmax=1)
        else:
            ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        ax.set_title(heading, fontsize=10, fontweight='semibold')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor('#666666')
            s.set_linewidth(0.7)

    fig.suptitle('MRI Preprocessing and Training Augmentation — Real Dataset Image', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.005,
             'Panels (a–d): deterministic preparation. Panels (e–h): examples of training-only transforms. '
             'Normalized values change numerically, not visibly.',
             ha='center', fontsize=8)
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.18)
    plt.close(fig)
    print(f'Saved: {out.resolve()}')
    print(f'Source image: {src.resolve()} | original: {original.width} x {original.height} | resized: 224 x 224')
    print(f'Pixel values [0,1]: min={array.min():.4f}, max={array.max():.4f}')
    print(f'Normalized values: min={normalized.min():.4f}, max={normalized.max():.4f}')


if __name__ == '__main__':
    main()
