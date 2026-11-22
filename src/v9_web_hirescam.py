"""BrainLens V9: exact Step 33-style *live* HiResCAM five-metric MRI audit.

Research-only tool. No training, checkpoint edits, or medical verification.
Uses same 20%-area blur perturbation, Step14 small transformations, HiResCAM
at features[-4], and 20 classifier-dropout passes as the final V9 analysis.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps

from src import config as C
from src.research_v9 import get_layer, normalize_cam, target_margin, cosine, top_iou
from src.step33_hirescam_full_test import HiResCAM
from src.step13_faithfulness import patch_masks, perturb, score_many
from src.step14_stability import variations
from src.step15_explanation_uncertainty import softmax_1
from src.transforms_v9 import transform
from src.model import build_model
from src.train_utils import device as research_device
import json


MC_REFERRAL_ENTROPY = 0.00148676  # Step 24, selected using validation data; NOT a diagnosis rule.


def load_v9():
    """Read-only loader for the original V9 checkpoint and Step 12 temperature."""
    if not C.CHECKPOINT.is_file():
        raise FileNotFoundError(f"V9 checkpoint missing: {C.CHECKPOINT}")
    cal = C.OUTPUT / '12_calibration' / 'calibration_summary.json'
    if not cal.is_file():
        raise FileNotFoundError(f"Step 12 calibration summary missing: {cal}")
    temperature = float(json.loads(cal.read_text(encoding='utf-8'))['temperature'])
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError('Invalid temperature saved from validation')
    device = research_device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    return model, device, temperature


def seed_for_image(image: Image.Image) -> int:
    grey = ImageOps.exif_transpose(image).convert('L')
    digest = hashlib.sha256(grey.tobytes()).digest()
    return int.from_bytes(digest[:4], 'little') % (2**31)


def normalize_input(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert('L')
    if min(image.size) < 64:
        raise ValueError('Image must be at least 64 × 64 pixels.')
    if max(image.size) > 8000:
        raise ValueError('Image dimensions are unusually large; resize before uploading.')
    return image


def _overlay(image: Image.Image, heat: np.ndarray) -> Image.Image:
    grey = np.array(image.resize((C.IMAGE_SIZE, C.IMAGE_SIZE), Image.Resampling.BICUBIC), dtype=np.uint8)
    base = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    color = cv2.cvtColor(cv2.applyColorMap((np.clip(heat, 0, 1)*255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    alpha = (np.clip(heat, 0, 1)**0.8 * .53)[:, :, None]
    mixed = np.clip(base * (1-alpha) + color * alpha, 0, 255).astype('uint8')
    return Image.fromarray(mixed)


def light_heatmap(image: Image.Image, model: torch.nn.Module, dev: torch.device):
    """For random test MRIs: saved Step 33 metrics plus a fresh HiResCAM visual.

    This does NOT recalculate the five metrics or overwrite saved profiles.
    """
    grey = normalize_input(image)
    x = transform(False)(grey).unsqueeze(0).to(dev)
    model.eval()
    with torch.inference_mode():
        logits = model(x)[0].detach().cpu().numpy()
    target = int(np.argmax(logits))
    cam = HiResCAM(model, get_layer(model, 'earlier'))
    try:
        raw, _ = cam.get(x, target)
        mask = normalize_cam(raw)
    finally:
        cam.close()
    return _overlay(grey, mask), C.CLASSES[target]


def analyze_new_mri(image: Image.Image, model: torch.nn.Module, dev: torch.device,
                    temperature: float):
    """Recompute every dimension from actual checkpoint inference for a new MRI."""
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError('Invalid validation-fitted calibration temperature')
    grey = normalize_input(image)
    seed = seed_for_image(grey)
    model.eval()
    tx = transform(False)
    x = tx(grey).unsqueeze(0).to(dev)
    with torch.inference_mode():
        logits = model(x)[0].detach().cpu().numpy()
    target = int(np.argmax(logits))
    pred = C.CLASSES[target]
    uncal = softmax_1(logits)
    calibrated = softmax_1(logits / temperature)
    cam = HiResCAM(model, get_layer(model, 'earlier'))
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if len(drops) != 1:
        cam.close()
        raise RuntimeError(f'Expected one classifier dropout, found {len(drops)}')
    try:
        raw, _ = cam.get(x, target)
        reference = normalize_cam(raw)
        overlay = _overlay(grey, reference)
        pixels = x[0, 0].detach().cpu().numpy().astype(np.float32)
        blurred = cv2.GaussianBlur(pixels, (0, 0), sigmaX=10).astype(np.float32)
        baseline = float(target_margin(logits, target)[0])
        blurred_margin = float(score_many(model, dev, [torch.from_numpy(blurred)[None, ...]], target)[0])

        # Step 33 20% high vs random region importance, both deletion and insertion.
        masks = patch_masks(reference, .2, seed + 20)
        altered, keys = [], []
        for kind in ('deletion', 'insertion'):
            for area, mask in masks.items():
                altered.append(perturb(x, blurred, mask, kind))
                keys.append((kind, area))
        margins = score_many(model, dev, altered, target)
        faith = {}
        for (kind, area), margin in zip(keys, margins):
            val = baseline - float(margin) if kind == 'deletion' else float(margin) - blurred_margin
            faith[(kind, area)] = val
        deletion = float(faith['deletion', 'high'] - faith['deletion', 'random'])
        insertion = float(faith['insertion', 'high'] - faith['insertion', 'random'])

        # Small MRI changes, aligned back after translation, as in Step 33.
        similarities, overlaps, flips = [], [], []
        for _, changed, dx in variations(pixels, seed):
            xx = torch.from_numpy(changed[None, None, ...]).to(dev)
            altered_raw, _ = cam.get(xx, target)
            heat = normalize_cam(altered_raw)
            if dx:
                heat = cv2.warpAffine(heat, np.float32([[1, 0, dx], [0, 1, 0]]),
                                      (C.IMAGE_SIZE, C.IMAGE_SIZE),
                                      flags=cv2.INTER_LINEAR, borderValue=0)
            similarities.append(cosine(reference, heat))
            overlaps.append(top_iou(reference, heat, .2))
            with torch.inference_mode():
                flips.append(int(model(xx).argmax(dim=1).item() != target))

        # Same classifier dropout mechanism: 20 stochastic class distributions
        # AND 20 fixed-target HiResCAM maps from those shared forward passes.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cam_shapes, probabilities = [], []
        try:
            for m in drops:
                m.train()
            for _ in range(C.MC_PASSES):
                stochastic_raw, stochastic_logits = cam.get(x, target)
                cam_shapes.append(normalize_cam(stochastic_raw))
                probabilities.append(softmax_1(stochastic_logits))
        finally:
            for m in drops:
                m.eval()
        probabilities = np.stack(probabilities)
        avg = probabilities.mean(axis=0)
        entropy = float(-np.sum(avg * np.log(np.clip(avg, 1e-12, 1))))
        individual_entropy = float(np.mean(-np.sum(probabilities *
                    np.log(np.clip(probabilities, 1e-12, 1)), axis=1)))
        mutual_info = max(0., entropy - individual_entropy)
        variability = float(np.stack(cam_shapes).std(axis=0).mean())
        return {
            'predicted_class': pred,
            'calibrated_confidence': float(calibrated[target]),
            'raw_confidence': float(uncal[target]),
            'probabilities': {c: float(calibrated[i]) for i, c in enumerate(C.CLASSES)},
            'mc_predictive_entropy': entropy,
            'mc_mutual_information': mutual_info,
            'mc_mean_confidence': float(avg.max()),
            'faithfulness_del_high_minus_random_20': deletion,
            'faithfulness_ins_high_minus_random_20': insertion,
            'mean_stability_cosine': float(np.nanmean(similarities)),
            'mean_stability_top20_iou': float(np.nanmean(overlaps)),
            'stability_prediction_flip_count': int(sum(flips)),
            'explanation_shape_uncertainty': variability,
            'mc_passes': int(C.MC_PASSES),
            'overlay': overlay,
            'heatmap': Image.fromarray((np.clip(reference, 0, 1) * 255).astype('uint8')),
        }
    finally:
        for m in drops:
            m.eval()
        cam.close()
