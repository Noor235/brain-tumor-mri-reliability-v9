"""V9 Step 33: final full held-out test reliability with HiResCAM at features[-4].

Reuses Step 22's existing faithfulness, stability, MC explanation-uncertainty,
per-image integration, validation gates, and resumable SQLite database. Only the
explanation map formula changes from Grad-CAM to HiResCAM. Does not retrain.

From original V9 project root:
    python3 -m src.step33_hirescam_full_test --max-images 2
    python3 -m src.step33_hirescam_full_test

All new files go under results/33_hirescam_full_test/; no old results overwritten.
Use EXACT same previous V9 checkpoint, test split and Step 22 dependencies.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config as C
from .common import sha_file
from .research_v9 import environment, ResearchCAM, get_layer
from .transforms_v9 import transform
from . import step22_full_test_reliability as previous

OUT_NAME = '33_hirescam_full_test'
METHOD = 'HiResCAM'
LAYER = 'earlier'  # In this V9 codebase, earlier means model.features[-4].
REFERENCE = 'gradcam_features4_full_test/earlier/per_image_reliability_profile.csv'
COMPARABLE = (
    ('faithfulness_del_high_minus_random_20', 'Deletion advantage', 'higher'),
    ('faithfulness_ins_high_minus_random_20', 'Insertion advantage', 'higher'),
    ('mean_stability_cosine', 'Mean stability cosine', 'higher'),
    ('mean_stability_top20_iou', 'Stability top-20% overlap', 'higher'),
    ('explanation_shape_uncertainty', 'MC CAM spatial std', 'descriptive'),
)


class HiResCAM(ResearchCAM):
    """HiResCAM: ReLU(sum_channels(activation * spatial_gradient)).

    Same target, hooks, resize, nonnegative clipping, and preprocessing as the
    original V9 ResearchCAM, but does NOT spatially average the gradients.
    The formula matches Steps 30-32's HiResCAM maps at features[-4].
    """
    def get(self, x, target, interpolation=cv2.INTER_LINEAR):
        self.activation = None
        self.gradient = None
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, int(target)].backward()
        if self.activation is None or self.gradient is None:
            raise RuntimeError('HiResCAM forward/backward hooks failed')
        activation = self.activation[0]
        grad = self.gradient[0]
        raw = torch.relu((activation * grad).sum(dim=0))
        arr = raw.detach().cpu().numpy().astype(np.float32)
        arr = cv2.resize(arr, (C.IMAGE_SIZE, C.IMAGE_SIZE), interpolation=interpolation)
        return np.maximum(arr, 0), logits.detach().cpu().numpy()[0]


def make_figures(rows, out):
    figures = out / 'figures' / LAYER
    figures.mkdir(parents=True, exist_ok=True)
    pairs = [
        ('mc_predictive_entropy', 'explanation_shape_uncertainty',
         '01_prediction_vs_explanation_uncertainty.png',
         'Prediction vs HiResCAM explanation uncertainty',
         'MC predictive entropy', 'Mean per-pixel HiResCAM shape std'),
        ('faithfulness_del_high_minus_random_20', 'explanation_shape_uncertainty',
         '02_faithfulness_vs_explanation_uncertainty.png',
         'Faithfulness vs HiResCAM explanation uncertainty',
         'High minus random deletion margin drop (20%)', 'Mean per-pixel HiResCAM shape std'),
    ]
    for x, y, filename, title, xlabel, ylabel in pairs:
        previous.plot_scatter(rows, x, y, figures / filename, title, xlabel, ylabel)
    groups = [
        ('explanation_shape_uncertainty', '03_uncertainty_by_class.png',
         'HiResCAM uncertainty by true class', 'Mean per-pixel shape std', 'true_class'),
        ('faithfulness_del_high_minus_random_20', '04_faithfulness_by_class.png',
         'HiResCAM deletion faithfulness by true class', 'Deletion advantage (margin)', 'true_class'),
        ('mean_stability_top20_iou', '05_stability_by_class.png',
         'HiResCAM stability by true class', 'Mean top-20% overlap', 'true_class'),
        ('explanation_shape_uncertainty', '06_uncertainty_correct_vs_wrong.png',
         'HiResCAM uncertainty: correct vs incorrect', 'Mean per-pixel shape std', 'correct'),
        ('faithfulness_del_high_minus_random_20', '07_faithfulness_correct_vs_wrong.png',
         'HiResCAM deletion faithfulness: correct vs incorrect', 'Deletion advantage (margin)', 'correct'),
        ('mc_predictive_entropy', '08_prediction_uncertainty_correct_vs_wrong.png',
         'MC predictive entropy: correct vs incorrect', 'MC predictive entropy', 'correct'),
        ('mean_stability_cosine', '09_stability_correct_vs_wrong.png',
         'HiResCAM stability: correct vs incorrect', 'Mean cosine similarity', 'correct'),
    ]
    for field, filename, title, ylabel, group in groups:
        previous.plot_by_group(rows, field, figures / filename, title, ylabel, group=group)


def update_summary(out, meta):
    path = out / LAYER / 'integrated_summary.json'
    if not path.is_file():
        return
    obj = json.loads(path.read_text(encoding='utf-8'))
    obj['explanation_method'] = METHOD
    obj['explanation_layer'] = 'EfficientNet-B4 features[-4]'
    obj['measurement_protocol'] = 'Original V9 Step 22, with HiResCAM replacing Grad-CAM only'
    obj['limitations'] = [
        'Exploratory post-hoc analysis on previously inspected V9 test set; not an untouched external validation.',
        'Near-duplicate grouping was performed, but patient identities are unavailable; patient independence is not established.',
        'Faithfulness compares fixed-class margin changes under 16x16 patch blur deletion/insertion; masking creates artificial inputs.',
        'Higher deletion/insertion scores do not prove tumour localization or clinical correctness.',
        'The HiResCAM theoretical guarantees do not directly establish faithfulness for this particular EfficientNet-B4 architecture.',
        'Stability uses fixed small image changes, not comprehensive robustness to clinical acquisition variation.',
        'Explanation uncertainty is per-pixel variation of independently max-normalized maps over 20 classifier-dropout passes.',
        'Calibration and prediction uncertainty are re-used from saved V9 outputs and have not been re-fitted.',
        'No clinical decision threshold or composite reliability score is validated; favourable measurements cannot verify correctness.',
    ]
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def read_profiles(path):
    with path.open(newline='', encoding='utf-8') as fp:
        rows = list(csv.DictReader(fp))
    ids = [r['file'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate files in: {path}')
    return {r['file']: r for r in rows}


def paired_ci(values, seed=333, draws=1500):
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.choice(arr, size=(draws, len(arr)), replace=True).mean(axis=1)
    return [float(arr.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]


def compare_gradcam(out, expected=1374):
    """Descriptive paired comparison, run only when both full 1374-image profiles exist."""
    old_path = C.OUTPUT / REFERENCE
    new_path = out / LAYER / 'per_image_reliability_profile.csv'
    if not old_path.is_file():
        print(f'  Original corrected Grad-CAM profile not found: {old_path}', flush=True)
        print('  HiResCAM report remains valid; comparison can be made later.', flush=True)
        return
    old, new = read_profiles(old_path), read_profiles(new_path)
    if len(old) != expected or len(new) != expected or set(old) != set(new):
        print('  Paired comparison deferred: both profiles must have the same 1374 MRI IDs.', flush=True)
        return
    for file in old:
        if (old[file]['true_class'] != new[file]['true_class'] or
            old[file]['deterministic_predicted_class'] != new[file]['deterministic_predicted_class']):
            raise RuntimeError(f'Existing Grad-CAM test predictions differ for {file}; refusing comparison')
    comparison = []
    paired_rows = []
    for file in sorted(old):
        line = {'file': file,
                'true_class': new[file]['true_class'],
                'correct': int(new[file]['deterministic_correct'])}
        for key, label, orient in COMPARABLE:
            a = float(old[file][key]); b = float(new[file][key]);
            line['gradcam_' + key] = a
            line['hirescam_' + key] = b
            line['hirescam_minus_gradcam_' + key] = b - a
        paired_rows.append(line)
    for key, label, orient in COMPARABLE:
        previous_values = np.array([float(old[f][key]) for f in old])
        current_values = np.array([float(new[f][key]) for f in old])
        diffs = current_values - previous_values
        comparison.append({
            'metric': key,
            'description': label,
            'interpretation': orient,
            'n': len(diffs),
            'gradcam_mean': float(previous_values.mean()),
            'hirescam_mean': float(current_values.mean()),
            'hirescam_minus_gradcam_bootstrap_95ci': paired_ci(diffs),
            'correct_mean_difference': float(diffs[np.array([int(old[f]['deterministic_correct']) for f in old]) == 1].mean()),
            'incorrect_mean_difference': float(diffs[np.array([int(old[f]['deterministic_correct']) for f in old]) == 0].mean()),
        })
    info = {
        'comparison': 'Existing corrected Grad-CAM vs HiResCAM, V9 1374-image exploratory held-out test',
        'n': expected,
        'source_gradcam': str(old_path),
        'source_hirescam': str(new_path),
        'paired_metric_differences': comparison,
        'cautions': [
            'Do not treat test-data comparisons as independent model selection or external confirmation.',
            'Bootstrap intervals resample images, not patients; repeated patient identities are unknown.',
            'The original Step 22 patchwise protocol is held fixed here; its scores are not directly comparable with Step 30 validation pixel-mask scores.',
            'Lower explanation shape std is not proof of tumour localization or fewer diagnosis errors.',
        ],
    }
    (out / 'comparison_vs_corrected_gradcam.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    previous.write_csv(out / 'paired_cam_comparison.csv', paired_rows, list(paired_rows[0]))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = [r['description'] for r in comparison]
        a = [r['gradcam_mean'] for r in comparison]
        b = [r['hirescam_mean'] for r in comparison]
        fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 4))
        for ax, label, x, y in zip(axes, labels, a, b):
            ax.bar(['Grad-CAM', 'HiResCAM'], [x, y])
            ax.set_title(label, fontsize=10)
            ax.tick_params(axis='x', rotation=30)
        fig.suptitle('V9 full-test, same images and perturbation protocol (n=1374)')
        fig.tight_layout()
        fig.savefig(out / 'comparison_vs_corrected_gradcam.png', dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f'  Optional comparison figure failed: {exc}', flush=True)
    print('\nPAIRED ORIGINAL GRAD-CAM vs HIRESCAM:', flush=True)
    for row in comparison:
        d = row['hirescam_minus_gradcam_bootstrap_95ci']
        print(f'  {row["description"]:<28}: Grad-CAM={row["gradcam_mean"]:.6f} '
              f'HiResCAM={row["hirescam_mean"]:.6f} delta={d[0]:+.6f} '
              f'CI=[{d[1]:+.6f}, {d[2]:+.6f}]', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-images', type=int, default=None,
                    help='Smoke check first N test images; subsequent full run resumes from these results')
    ap.add_argument('--out-name', default=OUT_NAME,
                    help='Output subdirectory under results/. Choose a fresh name for a new run.')
    ap.add_argument('--examples-per-class', type=int, default=2,
                    help='Save heatmap panels for first N of each class and for incorrect predictions')
    ap.add_argument('--skip-figures', action='store_true',
                    help='Skip plots during an initial smoke test (CSV and summary still created)')
    args = ap.parse_args(argv)
    if args.max_images is not None and args.max_images < 1:
        ap.error('--max-images must be >=1')
    if args.examples_per_class < 0:
        ap.error('--examples-per-class must be >=0')
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.out_name):
        ap.error('--out-name must contain only letters, digits, underscores')
    if args.out_name in ('v9_full_reliability', 'gradcam_features4_full_test'):
        ap.error('Refusing to overwrite an existing Grad-CAM experiment')

    C.heading('V9 STEP 33 — FULL TEST HIRESCAM / FIVE RELIABILITY MEASUREMENTS')
    rows, fingerprint, model, device = environment('test')
    test_rows = sorted((r for r in rows if r['split'] == 'test'),
                       key=lambda r: (C.CLASSES.index(r['class_name']), r['file']))
    original = previous.load_original(test_rows, fingerprint)
    output = C.OUTPUT / args.out_name
    metadata = dict(
        schema=33, dataset='V9', method=METHOD, layer='features[-4]',
        previous_protocol='Step22_full_test_reliability',
        split='test', n_test=len(test_rows), fingerprint=fingerprint,
        checkpoint_sha256=sha_file(C.CHECKPOINT),
        mc_passes=C.MC_PASSES,
        faithfulness_fractions=list(previous.FRACTIONS),
        faithfulness_grid=16, blur_sigma=10,
        perturbations=['noise_original_code_std_0.02_normalized',
                       'brightness_plus_0.02', 'shift_2px_right'],
        original_output_root=str(C.OUTPUT.resolve()),
    )
    con = previous.open_db(output, metadata)
    tx = transform(False)
    model.eval()
    chosen = set()
    for cls in C.CLASSES:
        pool = [r for r in test_rows if r['class_name'] == cls]
        chosen.update(r['file'] for r in pool[:args.examples_per_class])
    chosen.update(r['file'] for r in test_rows if not int(original['test'][r['file']]['correct']))
    limit = min(len(test_rows), args.max_images or len(test_rows))
    completed_count = 0
    print(f'Device: {device} | total held-out test MRIs: {len(test_rows)} | '
          f'this invocation: {limit}', flush=True)
    print('HiResCAM at features[-4]; 20 classifier-dropout passes per image.', flush=True)
    print('Reusing saved V9 MC uncertainty, temperature-scaled confidence, and deterministic predictions.', flush=True)
    print('Resumable output:', output, flush=True)
    cam = HiResCAM(model, get_layer(model, LAYER))
    try:
        for i, row in enumerate(test_rows[:limit], 1):
            file = row['file']
            already = con.execute('SELECT 1 FROM completed WHERE file=? AND layer=?',
                                  (file, LAYER)).fetchone()
            if already:
                continue
            result = previous.record_one(row, LAYER, i, model, device, cam, tx,
                                         original, output, chosen)
            con.execute('INSERT INTO completed(file,layer,payload) VALUES(?,?,?)',
                        (file, LAYER, json.dumps(previous.native(result), allow_nan=False)))
            con.commit()
            completed_count += 1
            print(f'  {i:4}/{len(test_rows)} HiResCAM  {row["class_name"]:11s} '
                  f'{"CORRECT" if result["deterministic_correct"] else "WRONG"}  '
                  f'explanation_std={result["explanation"]["mean_shape_attribution_std"]:.6f}',
                  flush=True)
            if i % 25 == 0:
                print(f'  Saved progress after index {i}. It is safe to resume.', flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted safely. Rerun the exact same command to resume.', flush=True)
    except Exception:
        print('\nFailed after last saved MRI; progress is preserved. Check the exception.', flush=True)
        raise
    finally:
        cam.close()
        print('\nExporting per-image profiles, trial CSVs, and summary...', flush=True)
        previous.export(con, output, test_rows, original, fingerprint,
                        [LAYER], metadata, render=False)
        con.close()
        update_summary(output, metadata)
        if not args.skip_figures:
            try:
                profiles = read_profiles(output / LAYER / 'per_image_reliability_profile.csv')
                make_figures([{k: (float(v) if k in previous.NUM_FIELDS and v not in ('',None)
                                  else int(v) if k == 'deterministic_correct' else v)
                               for k, v in row.items()}
                              for row in profiles.values()], output)
            except Exception as exc:
                print(f'Optional figures could not be rendered: {exc}', flush=True)
        profpath = output / LAYER / 'per_image_reliability_profile.csv'
        if profpath.exists():
            profiles = read_profiles(profpath)
            n = len(profiles)
            if n == len(test_rows):
                print('\nCOMPLETE: 1374/1374 HiResCAM test images.', flush=True)
                correct = sum(int(row['deterministic_correct']) for row in profiles.values())
                print(f'Predictions: {correct} correct / {n-correct} incorrect.', flush=True)
                compare_gradcam(output, expected=n)
            else:
                print(f'PARTIAL: {n}/{len(test_rows)}. Resume using same command.', flush=True)
        print('Saved results under:', output, flush=True)


if __name__ == '__main__':
    main()
