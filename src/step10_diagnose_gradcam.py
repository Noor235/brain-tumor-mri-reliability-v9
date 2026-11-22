"""Kaggle V9 — independent Grad-CAM layer diagnostic (NO retraining).

Put at src/step10_diagnose_gradcam.py, run from the Kaggle V9 project root:
  ./.venv/bin/python -m src.step10_diagnose_gradcam

Compares the exact V9 Step 10 layer and interpolation against an earlier,
higher-resolution backbone block and the output of spatial attention.
All heatmaps are kept UNMASKED: no cosmetic removal of background activation.
Original checkpoints, source MRIs, and previous outputs are untouched.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from . import config as C
from .common import output_dir, source_image
from .model import build_model
from .train_utils import device
from .transforms_v9 import transform
from .verify import verify

OUTPUT_NAME = '10_gradcam_diagnostic'
CASE_COUNTS = {'notumor': 3, 'glioma': 1, 'meningioma': 1, 'pituitary': 1}


def gradcam_for_layer(model, layer, x, target, interpolation):
    """Capture layer output and differentiate chosen logit wrt that output."""
    captured = {}
    def hook(_module, _inputs, output):
        captured['features'] = output
    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            if 'features' not in captured:
                raise RuntimeError('Failed to capture the requested feature layer')
            activation = captured['features']
            gradient = torch.autograd.grad(logits[0, target], activation)[0]
            # Channel weights are mean gradients over the layer's spatial axes.
            weight = gradient.mean(dim=(-2, -1), keepdim=True)
            cam = torch.relu((weight * activation).sum(dim=1))[0]
            original_shape = tuple(cam.shape)
            cam_np = cam.detach().float().cpu().numpy()
    finally:
        handle.remove()

    # Preserve the raw scale for mass calculations: a zero CAM stays zero.
    cam_np = cv2.resize(cam_np, (C.IMAGE_SIZE, C.IMAGE_SIZE), interpolation=interpolation)
    cam_np = np.maximum(cam_np, 0.0)
    scale = float(cam_np.max())
    cam_np = cam_np / scale if scale > 1e-12 else np.zeros_like(cam_np)
    return cam_np.astype(np.float32), original_shape


def approx_foreground(gray):
    """Approximate head/body silhouette, NOT a tumor segmentation mask."""
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = np.uint8(smoothed > 20) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    return mask > 0


def colorize(heat):
    col = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)
    return cv2.cvtColor(col, cv2.COLOR_BGR2RGB)


def metrics(heat, foreground):
    # Hotspot/foreground measures quantify visualization, NOT explanation faithfulness.
    total = float(heat.sum())
    h, w = heat.shape
    corner = np.zeros((h, w), dtype=bool)
    corner[:h // 4, 3 * w // 4:] = True
    peak_y, peak_x = np.unravel_index(np.argmax(heat), heat.shape)
    if total <= 1e-12:
        return {'corner_mass_fraction': 0., 'outside_foreground_fraction': 0.,
                'peak_x': int(peak_x), 'peak_y': int(peak_y), 'peak_outside_foreground': False,
                'heatmap_all_zero': True}
    return {
        'corner_mass_fraction': float(heat[corner].sum() / total),
        'outside_foreground_fraction': float(heat[~foreground].sum() / total),
        'peak_x': int(peak_x), 'peak_y': int(peak_y),
        'peak_outside_foreground': bool(not foreground[peak_y, peak_x]),
        'heatmap_all_zero': False,
    }


def montage(gray, maps, title, path):
    base = np.stack((gray, gray, gray), axis=-1)
    titles = ['Original MRI', 'Final block / cubic', 'Final block / linear',
              'Earlier block / linear', 'After spatial attention']
    width = len(titles) * C.IMAGE_SIZE
    h = 30 + C.IMAGE_SIZE + 25
    canvas = Image.new('RGB', (width, h), 'white')
    painter = ImageDraw.Draw(canvas)
    painter.text((6, 5), title, fill=(0, 0, 0))
    canvas.paste(Image.fromarray(base), (0, 30))
    for j, heat in enumerate(maps, 1):
        overlay = np.uint8(np.clip(0.60 * base + 0.40 * colorize(heat), 0, 255))
        canvas.paste(Image.fromarray(overlay), (j * C.IMAGE_SIZE, 30))
    for j, name in enumerate(titles):
        painter.text((j * C.IMAGE_SIZE + 5, 30 + C.IMAGE_SIZE + 4), name, fill=(0, 0, 0))
    canvas.save(path)


def main():
    C.heading('GRAD-CAM DIAGNOSTIC / Is the corner artifact layer-dependent?')
    rows, _ = verify(rehash=False, detail=False)
    if not C.CHECKPOINT.is_file():
        raise FileNotFoundError(f'V9 model checkpoint missing: {C.CHECKPOINT}')
    dev = device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT, map_location='cpu', weights_only=True), strict=True)
    model = model.to(dev).eval()
    tx = transform(False)
    out = output_dir(OUTPUT_NAME)
    records = []
    layers = [
        ('final_cubic', model.features[-1], cv2.INTER_CUBIC),
        ('final_linear', model.features[-1], cv2.INTER_LINEAR),
        ('earlier_linear', model.features[-4], cv2.INTER_LINEAR),
        ('attention_linear', model.spatial_attention, cv2.INTER_LINEAR),
    ]
    print('Comparing final layer + interpolation, earlier layer, and attention output.', flush=True)
    print('No pixels or heatmaps are artificially masked. All results are diagnostic.', flush=True)
    for cls, n in CASE_COUNTS.items():
        picks = sorted((r for r in rows if r['split'] == 'test' and r['class_name'] == cls),
                       key=lambda r: r['file'])[:n]
        if len(picks) < n:
            print(f'WARNING: Only {len(picks)} test images found for {cls}.', flush=True)
        for idx, entry in enumerate(picks, 1):
            with Image.open(source_image(entry['file'])) as src:
                img = src.convert('L')
            x = tx(img).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = torch.softmax(model(x), dim=1)[0]
                target = int(p.argmax().item())
                confidence = float(p[target].item())
            gray = np.asarray(img.resize((C.IMAGE_SIZE, C.IMAGE_SIZE), Image.Resampling.BICUBIC),
                              dtype=np.uint8)
            foreground = approx_foreground(gray)
            heatmaps = []
            for name, layer, interpolation in layers:
                heat, shape = gradcam_for_layer(model, layer, x, target, interpolation)
                heatmaps.append(heat)
                record = dict(image=entry['file'], true_class=cls,
                              predicted_class=C.CLASSES[target], confidence=round(confidence, 6),
                              layer=name, feature_size=f'{shape[0]}x{shape[1]}', **metrics(heat, foreground))
                records.append(record)
                print(f'  {cls}_{idx} {name}: {shape[0]}x{shape[1]} '
                      f'corner mass={record["corner_mass_fraction"]:.1%} '
                      f'outside silhouette={record["outside_foreground_fraction"]:.1%}', flush=True)
            montage(gray, heatmaps,
                    f'True: {cls} | Predicted: {C.CLASSES[target]} ({confidence:.1%})',
                    out / f'{cls}_{idx}_layer_comparison.png')
    with (out / 'cam_diagnostic_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    summary = {'checkpoint': str(C.CHECKPOINT),
               'images_evaluated': len({row['image'] for row in records}),
               'gradcam_variants_per_image': len(layers),
               'mean_corner_mass_by_class_layer': {},
               'cautions': [
                   'Grad-CAM maps are not direct evidence of class-score faithfulness.',
                   'Foreground is automatically approximated, not manually annotated.',
                   'Layer comparisons do not establish whether the model actually uses background.',
                   'No patient-level independence is guaranteed.'
               ]}
    for cls in CASE_COUNTS:
        summary['mean_corner_mass_by_class_layer'][cls] = {}
        for name, _, _ in layers:
            values = [r['corner_mass_fraction'] for r in records
                      if r['true_class'] == cls and r['layer'] == name]
            if values:
                summary['mean_corner_mass_by_class_layer'][cls][name] = round(float(np.mean(values)), 6)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nCOMPLETED:', out, flush=True)
    print('Check the *_layer_comparison.png files first (especially all three notumor images).', flush=True)
    print('Same corner across layers: investigate model background dependence with perturbation tests.', flush=True)
    print('Corner only on final block: explanation layer/resolution is a plausible contributor.', flush=True)
    print('No checkpoint, source image, original CAM, or test report was modified.', flush=True)


if __name__ == '__main__':
    main()
