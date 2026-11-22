"""Kaggle V9 — Step 10B: paired corner perturbation diagnostic.

Run from the root of the existing Kaggle V9 project:
    ./.venv/bin/python -m src.step10b_background_dependence
Optional larger diagnostic (still on the previously used test set):
    ./.venv/bin/python -m src.step10b_background_dependence --all-notumor

Purpose: distinguish *Grad-CAM highlighting* from *actual classifier sensitivity*
to the suspicious upper-right corner in the No Tumor test MRIs. Compare equal
upper-right and upper-left corner patches with paired perturbations.

- Uses the UNCHANGED, trained V9 checkpoint and frozen existing test split.
- Never modifies original MRIs, model, previous test scores, or Grad-CAM maps.
- Blackout may cause NO CHANGE if a corner is already black; such trials cannot
  establish whether the network depends on the corner. Mild additive noise
  is diagnostic, not a clinically realistic new MRI acquisition.
- Low sensitivity does not establish faithfulness or absence of shortcuts.
- A high sensitivity is evidence for follow-up, not proof of the root cause.
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import config as C
from .common import output_dir, source_image, save_json, check_no_overwrite
from .model import build_model
from .train_utils import device
from .transforms_v9 import transform
from .verify import verify

CLASS = 'notumor'
NOISE_STD = 0.04  # in [0,1] image-intensity units; small, but not a clinical acquisition model
REPEATS = 3
PATCH_FRAC = 0.25
REGION_NAMES = ('upper_right', 'upper_left')
MODE_NAMES = ('blackout', 'mild_noise')


def square_regions(side):
    """Equal-sized image-coordinate patch positions (top two corners)."""
    patch = max(1, round(side * PATCH_FRAC))
    return {
        'upper_right': (slice(0, patch), slice(side - patch, side)),
        'upper_left': (slice(0, patch), slice(0, patch)),
    }


def make_perturbed(x, patch_slices, method, shared_noise=None):
    """Return modified, normalized 1x1xHxW tensor; NEVER modify input in-place."""
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 1:
        raise ValueError('Expected one 1-channel V9 MRI with shape 1x1xHxW')
    ysl, xsl = patch_slices
    img01 = (x.detach().clone() + 1.0) / 2.0
    old = img01[:, :, ysl, xsl].clone()
    if method == 'blackout':
        replacement = torch.zeros_like(old)
    elif method == 'mild_noise':
        if shared_noise is None:
            raise ValueError('Mild noise needs a deterministic shared patch noise array')
        noise = torch.as_tensor(shared_noise, dtype=img01.dtype, device=img01.device)
        if tuple(noise.shape) != tuple(old.shape[-2:]):
            raise ValueError(f'Noise patch shape {tuple(noise.shape)} != corner patch {tuple(old.shape[-2:])}')
        replacement = (old + noise[None, None, :, :]).clamp(0.0, 1.0)
    else:
        raise ValueError(f'Unexpected perturbation method: {method}')
    img01[:, :, ysl, xsl] = replacement
    return img01 * 2.0 - 1.0


def predict(model, x, target_idx):
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().float().cpu().numpy()
        lg = logits[0].detach().float().cpu().numpy()
    runner_up = np.max(np.delete(lg, target_idx))
    return {
        'probs': probs,
        'class_index': int(np.argmax(probs)),
        'target_prob': float(probs[target_idx]),
        'target_logit_margin': float(lg[target_idx] - runner_up),
    }


def image_from_tensor(x):
    image = ((x[0, 0].detach().float().cpu().numpy() + 1.0) / 2.0 * 255.0)
    return Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), mode='L').convert('RGB')


def view_montage(path, sample, display, patches, baseline_prob):
    side = sample.shape[-1]
    columns = [('Original', sample, None)]
    for name, method in [('UR blackout', 'blackout'), ('UL blackout', 'blackout'),
                         ('UR mild noise', 'mild_noise'), ('UL mild noise', 'mild_noise')]:
        region = 'upper_right' if name.startswith('UR') else 'upper_left'
        columns.append((name, display[(method, region)], region))
    canvas = Image.new('RGB', (side * len(columns), side + 47), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((5, 3), f'No Tumor case — original P(no tumor): {baseline_prob:.4f}', fill='black')
    for i, (label, tensor, region) in enumerate(columns):
        img = image_from_tensor(tensor)
        if region is not None:
            ysl, xsl = patches[region]
            ImageDraw.Draw(img).rectangle((xsl.start, ysl.start, xsl.stop - 1, ysl.stop - 1),
                                           outline=(255, 210, 0), width=2)
        canvas.paste(img, (side * i, 20))
        d.text((side * i + 4, side + 25), label, fill='black')
    canvas.save(path)


def summarize(records):
    out = {}
    for method in MODE_NAMES:
        out[method] = {}
        for region in REGION_NAMES:
            subset = [r for r in records if r['method'] == method and r['region'] == region]
            if not subset:
                continue
            out[method][region] = {
                'trials': len(subset),
                'mean_absolute_target_probability_change': float(np.mean([abs(r['delta_target_prob']) for r in subset])),
                'mean_absolute_target_margin_change': float(np.mean([abs(r['delta_target_margin']) for r in subset])),
                'mean_total_variation_probability': float(np.mean([r['prob_total_variation'] for r in subset])),
                'prediction_flips': int(sum(r['prediction_flipped'] for r in subset)),
                'effectively_unchanged_trials': int(sum(r['changed_pixel_fraction'] < 1e-6 for r in subset)),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description='Kaggle V9 corner-dependence diagnostic; no model training')
    parser.add_argument('--all-notumor', action='store_true',
                        help='Check every No Tumor test MRI (default: the first three used in Step 10)')
    args = parser.parse_args()

    C.heading('STEP 10B / BACKGROUND-DEPENDENCE TEST: upper-right vs matched upper-left')
    rows, fingerprint = verify(rehash=False, detail=False)
    if not C.CHECKPOINT.is_file():
        raise FileNotFoundError(f'V9 checkpoint not found: {C.CHECKPOINT}')
    config_path = C.OUTPUT / '07_training' / 'experiment_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing V9 training metadata: {config_path}')
    meta = json.loads(config_path.read_text(encoding='utf-8'))
    if meta.get('split_fingerprint') != fingerprint:
        raise RuntimeError('Current split differs from the split used to train V9. Aborting.')

    folder = '10_background_dependence_all' if args.all_notumor else '10_background_dependence'
    target_summary = C.OUTPUT / folder / 'summary.json'
    check_no_overwrite(target_summary)
    cases = sorted((r for r in rows if r['split'] == 'test' and r['class_name'] == CLASS),
                   key=lambda r: r['file'])
    if not args.all_notumor:
        cases = cases[:3]  # same image order as src.step10_gradcam
    if not cases:
        raise RuntimeError('No No Tumor test images found in frozen V9 split')

    dev = device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT, map_location='cpu', weights_only=True), strict=True)
    model = model.to(dev).eval()  # dropout OFF for controlled deterministic comparisons
    target_idx = C.CLASSES.index(CLASS)
    out_dir = output_dir(folder)
    tx = transform(False)
    patches = square_regions(C.IMAGE_SIZE)
    rng = np.random.default_rng(C.SEED)

    print(f'Device: {dev} | cases: {len(cases)} | model: V9 best checkpoint', flush=True)
    print('Only original TEST images are used. The saved checkpoint and dataset are READ-ONLY.', flush=True)
    print(f'Each corner patch: {round(C.IMAGE_SIZE * PATCH_FRAC)}x{round(C.IMAGE_SIZE * PATCH_FRAC)} pixels.', flush=True)
    print(f'Conditions: blackout, and additive Gaussian noise std={NOISE_STD} on [0,1] intensity; ',
          f'{REPEATS} paired noise repeats per image.', flush=True)
    print('NOTE: Blackout of an already-black corner can have no pixel effect.', flush=True)

    records = []
    image_summaries = []
    for case_idx, entry in enumerate(cases, 1):
        with Image.open(source_image(entry['file'])) as src:
            im = src.convert('L')
        x = tx(im).unsqueeze(0).to(dev)
        original = predict(model, x, target_idx)
        display = {}
        trial_records = []

        patch_sz = round(C.IMAGE_SIZE * PATCH_FRAC)
        noises = [rng.normal(0.0, NOISE_STD, size=(patch_sz, patch_sz)).astype(np.float32)
                  for _ in range(REPEATS)]
        for method in MODE_NAMES:
            for repeat in (range(REPEATS) if method == 'mild_noise' else range(1)):
                # Same generated noise patch applied to UR and UL: paired control.
                noise = noises[repeat] if method == 'mild_noise' else None
                for region in REGION_NAMES:
                    altered = make_perturbed(x, patches[region], method, noise)
                    result = predict(model, altered, target_idx)
                    change = (altered - x).abs().detach().float().cpu().numpy()
                    portion = change[0, 0]
                    ys, xs = patches[region]
                    within_patch = portion[ys, xs]
                    changed_fraction = float(np.mean(within_patch > 1e-5))
                    mean_change01 = float(np.mean(within_patch) / 2.0)
                    baseline_label = C.CLASSES[original['class_index']]
                    predicted_label = C.CLASSES[result['class_index']]
                    record = {
                        'file': entry['file'], 'true_class': CLASS, 'case_number': case_idx,
                        'original_prediction': baseline_label,
                        'original_notumor_probability': original['target_prob'],
                        'original_notumor_logit_margin': original['target_logit_margin'],
                        'method': method, 'region': region, 'repeat': repeat + 1,
                        'altered_prediction': predicted_label,
                        'altered_notumor_probability': result['target_prob'],
                        'delta_target_prob': result['target_prob'] - original['target_prob'],
                        'altered_notumor_logit_margin': result['target_logit_margin'],
                        'delta_target_margin': result['target_logit_margin'] - original['target_logit_margin'],
                        'prob_total_variation': float(0.5 * np.abs(result['probs'] - original['probs']).sum()),
                        'prediction_flipped': int(result['class_index'] != original['class_index']),
                        'changed_pixel_fraction': changed_fraction,
                        'mean_absolute_pixel_change_01': mean_change01,
                    }
                    records.append(record)
                    trial_records.append(record)
                    if (method, region) not in display:
                        display[(method, region)] = altered.detach().cpu()
        if not args.all_notumor:
            view_montage(out_dir / f'notumor_{case_idx}_perturbations.png', x.cpu(), display,
                         patches, original['target_prob'])
        s = summarize(trial_records)
        image_summaries.append({'file': entry['file'], 'original_prediction': C.CLASSES[original['class_index']],
                                'original_notumor_probability': original['target_prob'], 'effect_summary': s})
        print(f"  Case {case_idx}/{len(cases)} | original: {C.CLASSES[original['class_index']]} "
              f"P(no tumor)={original['target_prob']:.6f} | "
              f"UR noise mean |Delta P|={s['mild_noise']['upper_right']['mean_absolute_target_probability_change']:.6f} | "
              f"UL control={s['mild_noise']['upper_left']['mean_absolute_target_probability_change']:.6f}",
              flush=True)
        for region in REGION_NAMES:
            if s['blackout'][region]['effectively_unchanged_trials']:
                print(f'    {region} blackout changed no pixels: NOT an informative ablation.', flush=True)

    columns = ['file','true_class','case_number','original_prediction',
               'original_notumor_probability','original_notumor_logit_margin','method','region','repeat',
               'altered_prediction','altered_notumor_probability','delta_target_prob',
               'altered_notumor_logit_margin','delta_target_margin','prob_total_variation',
               'prediction_flipped','changed_pixel_fraction','mean_absolute_pixel_change_01']
    csv_path = out_dir / 'perturbation_trials.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    payload = {
        'title': 'Kaggle V9 Step 10B — No Tumor background-dependence diagnostic',
        'dataset_fingerprint': fingerprint,
        'checkpoint': str(C.CHECKPOINT),
        'cases_evaluated': len(cases),
        'no_tumor_test_total': sum(r['split']=='test' and r['class_name']==CLASS for r in rows),
        'patch_location': 'upper-right (suspicious) versus upper-left (same-size control)',
        'patch_size': [patch_sz, patch_sz],
        'methods': {'blackout': 'set ROI intensities to zero (may have no effect if already black)',
                    'mild_noise': f'add clipped Gaussian noise sigma {NOISE_STD} to [0,1] intensities; '
                                  'same noise matrix for right and left comparison'},
        'stochastic_noise_repeats': REPEATS,
        'aggregate_results': summary,
        'individual_cases': image_summaries,
        'limitations': [
            'Exploratory perturbation diagnostic, not a clinical faithfulness validation.',
            'Artificial corner noise is a distribution shift; predictions may change for reasons unrelated to meaningful anatomy.',
            'Blackout of a black corner gives no effective perturbation and cannot rule out reliance on background.',
            'A small effect does not rule out shortcuts; a large effect does not prove why a Grad-CAM map has a corner hotspot.',
            'Three example images do not represent all No Tumor test MRIs.' if not args.all_notumor else
            'All No Tumor MRIs belong to the already-evaluated test set; this is not new independent validation.',
            'Patient-level independence has not been established for Kaggle V9.'
        ],
    }
    save_json(target_summary, payload)
    print('\nAGGREGATE (all tested No Tumor images):', flush=True)
    for method in MODE_NAMES:
        for region in REGION_NAMES:
            r = summary[method][region]
            print(f"  {method:10s} {region:11s} | mean |delta P(no tumor)|="
                  f"{r['mean_absolute_target_probability_change']:.6f} | "
                  f"mean |delta margin|={r['mean_absolute_target_margin_change']:.5f} | "
                  f"flips={r['prediction_flips']} | unchanged={r['effectively_unchanged_trials']}", flush=True)
    print(f'CSV: {csv_path}', flush=True)
    print(f'Summary: {target_summary}', flush=True)
    print('No MRI, trained checkpoint, Grad-CAM, or original evaluation was changed.', flush=True)
    print('INTERPRETATION: Compare UR and UL effects; a small difference is NOT definitive proof.', flush=True)


if __name__ == '__main__':
    main()
