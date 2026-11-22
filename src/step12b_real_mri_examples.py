"""Step 12B - Real MRI examples for confidence calibration.

Reads only existing V9 Step 12 test predictions and *actual* original MRI files.
No inference, no retraining, no test-set tuning, no changes to the dataset.

Run from the V9 project root:
    ./.venv/bin/python -m src.step12b_real_mri_examples
or:
    ./.venv/bin/python -m src.step12b_real_mri_examples --per-class 2 --seed 42

Output: results/12b_real_mri_examples/
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from . import config as C
from .common import source_image, save_json


def read_outputs():
    base = C.OUTPUT / '12_calibration'
    predictions = base / 'per_image_test_calibration.csv'
    summary_path = base / 'calibration_summary.json'
    if not predictions.is_file() or not summary_path.is_file():
        raise FileNotFoundError('Step 12 outputs not found. Run: ./.venv/bin/python -m src.step12_calibration')
    with predictions.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = {'file','true_class','raw_prediction','raw_confidence','calibrated_confidence'}
    expected.update('raw_prob_'+c for c in C.CLASSES)
    expected.update('calibrated_prob_'+c for c in C.CLASSES)
    missing = expected.difference(rows[0] if rows else {})
    if missing: raise ValueError(f'Missing Step 12 CSV columns: {sorted(missing)}')
    if not rows: raise ValueError('Step 12 per-image prediction CSV is empty')
    expected = int(summary.get('calibration', {}).get('test_raw', {}).get('count', len(rows)))
    if expected != len(rows):
        print(f'WARNING: Saved calibration count {expected}; found {len(rows)}. Check Step 12 inputs.')
    for r in rows:
        if r['true_class'] not in C.CLASSES:
            raise ValueError(f"Invalid MRI class {r['true_class']} in {r['file']}")
        raw = np.array([float(r['raw_prob_'+c]) for c in C.CLASSES], dtype=np.float64)
        cal = np.array([float(r['calibrated_prob_'+c]) for c in C.CLASSES], dtype=np.float64)
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(cal)):
            raise ValueError(f'Invalid probability value for {r["file"]}')
        if not np.isclose(raw.sum(),1,atol=1e-5) or not np.isclose(cal.sum(),1,atol=1e-5):
            raise ValueError(f'Probabilities do not sum to one for {r["file"]}')
        y = C.CLASSES.index(r['true_class'])
        target = np.zeros(len(C.CLASSES)); target[y] = 1
        r['_raw_p'] = raw; r['_cal_p'] = cal
        r['_pred_idx'] = int(raw.argmax())
        r['_correct'] = (int(raw.argmax()) == y)
        r['_raw_brier'] = float(np.square(raw - target).sum())
        r['_cal_brier'] = float(np.square(cal - target).sum())
        r['_raw_nll'] = float(-math.log(max(float(raw[y]),1e-15)))
        r['_cal_nll'] = float(-math.log(max(float(cal[y]),1e-15)))
    return rows, summary


def choose(rows, per_class, seed):
    rng = random.Random(seed)
    random_examples = []
    interesting_examples = []
    for cls in C.CLASSES:
        pool = sorted((r for r in rows if r['true_class']==cls), key=lambda x:x['file'])
        picks = rng.sample(pool, k=min(len(pool), per_class))
        for r in picks: random_examples.append((r, 'Random test MRI'))
        # One diagnostic MRI per true class, preferring a misclassified example.
        errors = [r for r in pool if not r['_correct']]
        if errors:
            chosen = min(errors, key=lambda r: r['_raw_p'][C.CLASSES.index(cls)])
            label = 'Purposefully selected error (not random)'
        else:
            chosen = min(pool, key=lambda r: r['_raw_p'][C.CLASSES.index(cls)])
            label = 'Purposefully selected uncertain case (not random)'
        interesting_examples.append((chosen,label))
    return random_examples,interesting_examples


def summarize_image(row, kind):
    raw=row['_raw_p'];cal=row['_cal_p']
    return {
        'selection':kind,'file':row['file'],'true_class':row['true_class'],
        'raw_prediction':C.CLASSES[int(raw.argmax())],
        'calibrated_prediction':C.CLASSES[int(cal.argmax())],
        'correct':row['_correct'],
        'raw_confidence':float(raw.max()),'calibrated_confidence':float(cal.max()),
        'raw_true_class_probability':float(raw[C.CLASSES.index(row['true_class'])]),
        'calibrated_true_class_probability':float(cal[C.CLASSES.index(row['true_class'])]),
        'raw_brier':row['_raw_brier'],'calibrated_brier':row['_cal_brier'],
        'raw_nll':row['_raw_nll'],'calibrated_nll':row['_cal_nll'],
    }


def render(items, path, title, summary):
    n=len(items)
    fig, axes=plt.subplots(n,2,figsize=(12.6,max(3.2*n,5)),
                           gridspec_kw={'width_ratios':[1.0,2.4]})
    if n==1:axes=np.asarray([axes])
    # Full test metrics; ECE is not meaningful for a single MRI image.
    cr=summary['calibration']['test_raw']
    cc=summary['calibration']['test_temperature_scaled']
    fig.suptitle(title + '\n'
        f'Full test set (n=1,374): ECE {cr["ece"]:.5f} → {cc["ece"]:.5f} | '
        f'Brier {cr["brier"]:.5f} → {cc["brier"]:.5f} | '
        f'NLL {cr["nll"]:.5f} → {cc["nll"]:.5f}',
        y=0.997,fontsize=13,fontweight='bold')
    for i,(row,label) in enumerate(items):
        axim, axp=axes[i]
        with Image.open(source_image(row['file'])) as im:
            im=ImageOps.exif_transpose(im).convert('L')
            pixels=np.asarray(im)
        axim.imshow(pixels,cmap='gray',vmin=0,vmax=255)
        axim.set_title(f'{i+1}. TRUE: {row["true_class"]}\n{label}',fontsize=10)
        axim.axis('off')
        raw=row['_raw_p']; cal=row['_cal_p']
        y=np.arange(len(C.CLASSES))
        axp.barh(y-0.18,raw,height=0.33,label='Before calibration',alpha=0.9)
        axp.barh(y+0.18,cal,height=0.33,label='After calibration',alpha=0.9)
        axp.set_yticks(y, C.CLASSES, fontsize=9)
        axp.invert_yaxis();axp.set_xlim(0,1.01)
        axp.set_xlabel('Probability (0 to 1)',fontsize=9)
        axp.grid(axis='x',alpha=0.2)
        if i==0:axp.legend(loc='upper right',fontsize=8)
        correct='CORRECT' if row['_correct'] else 'INCORRECT'
        axp.set_title(
            f'{correct} | Prediction: {C.CLASSES[int(raw.argmax())]}\n'
            f'Confidence: {100*raw.max():.6f}% → {100*cal.max():.6f}%\n'
            f'Brier: {row["_raw_brier"]:.6f} → {row["_cal_brier"]:.6f}   '
            f'NLL: {row["_raw_nll"]:.6f} → {row["_cal_nll"]:.6f}',
            fontsize=10,loc='left',pad=8)
    fig.text(0.5,0.011,'Per-image Brier and NLL are computed for each displayed MRI. '
        'ECE above is calculated over the full test set, not individually.',
        ha='center',fontsize=9)
    fig.subplots_adjust(top=0.94 if n>=8 else 0.87,bottom=0.055,hspace=0.95,wspace=0.3,left=0.12,right=0.98)
    fig.savefig(path,dpi=135,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='Show actual random V9 test MRI images and calibration metrics.')
    parser.add_argument('--per-class',type=int,default=2,help='Random test MRIs per class (default 2)')
    parser.add_argument('--seed',type=int,default=42,help='Reproducible selection seed (default 42)')
    args=parser.parse_args()
    if not 1<=args.per_class<=5:parser.error('--per-class must be 1 through 5')
    C.heading('STEP 12B / REAL TEST MRI EXAMPLES — BEFORE AND AFTER CALIBRATION')
    rows,summary=read_outputs()
    chosen,challenging=choose(rows,args.per_class,args.seed)
    out=C.OUTPUT/'12b_real_mri_examples'
    out.mkdir(parents=True,exist_ok=True)
    # Save under distinct names so existing experiment reports remain unchanged.
    random_path=out/'random_test_MRIs_before_after.png'
    challenging_path=out/'challenging_test_MRIs_before_after.png'
    samples=chosen+challenging
    csv_path=out/'selected_mri_metrics.csv'
    rendered=[summarize_image(row,kind) for row,kind in samples]
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rendered[0]));w.writeheader();w.writerows(rendered)
    render(chosen,random_path,'Random REAL Kaggle V9 test MRIs — calibration comparison',summary)
    render(challenging,challenging_path,'Intentionally chosen challenging V9 test MRIs',summary)
    save_json(out/'selected_mri_summary.json',{
        'source':'V9 Step12 actual per-image test predictions, without model rerun',
        'random_sampling_seed':args.seed,'random_images_per_class':args.per_class,
        'test_set_size':len(rows),'full_test_calibration':{
            'raw':summary['calibration']['test_raw'],
            'temperature_scaled':summary['calibration']['test_temperature_scaled']},
        'selected_images':rendered,
        'interpretation':'ECE needs a dataset/group; individual images have Brier and NLL only. '
                         'The challenging selection is deliberate, not random. '
                         'The plots are examples, not new independent validation.'})
    print('Actual MRI files loaded from:',C.SOURCE)
    print(f'Step12 test prediction records: {len(rows)}')
    print(f'Random examples: {len(chosen)}; challenging examples: {len(challenging)}')
    print('\nOne actual random MRI per class (first sampled):')
    for cls in C.CLASSES:
        row=next(row for row,lab in chosen if row['true_class']==cls)
        print(f'  {cls:11} -> {C.CLASSES[row["_pred_idx"]]:11} | confidence '
              f'{row["_raw_p"].max():.6%} -> {row["_cal_p"].max():.6%} | '
              f'Brier {row["_raw_brier"]:.5f}->{row["_cal_brier"]:.5f} | '
              f'NLL {row["_raw_nll"]:.5f}->{row["_cal_nll"]:.5f}')
    print('\nFull-test ECE: '
          f'{summary["calibration"]["test_raw"]["ece"]:.5f} -> '
          f'{summary["calibration"]["test_temperature_scaled"]["ece"]:.5f}')
    print('Random MRI comparison:',random_path)
    print('Challenging MRI comparison:',challenging_path)
    print('Per-image CSV:',csv_path)
    print('DONE. No MRIs, checkpoints, dataset splits, or original Step12 reports were modified.')


if __name__=='__main__':main()
