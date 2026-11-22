"""Step 28: audit V9 deterministic probability calibration from ORIGINAL Step 12 output.

Reads existing test per-image probabilities and original validation-fitted temperature.
No retraining, refitting, threshold selection or inference. The saved Step 12
calibration is the source of truth; every test metric is independently recomputed.

Run in V9 project: python3 -m src.step28_confidence_calibration_audit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ('glioma', 'meningioma', 'notumor', 'pituitary')
N_BINS = 15


def metric_values(probs: np.ndarray, truth: np.ndarray, n_bins: int = N_BINS):
    n, k = probs.shape
    y = np.asarray(truth, dtype=int)
    guess = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (guess == y)
    nll = float(np.mean(-np.log(np.clip(probs[np.arange(n), y], 1e-15, 1.0))))
    brier = float(np.mean(np.sum((probs - np.eye(k)[y]) ** 2, axis=1)))
    edges = np.linspace(0., 1., n_bins + 1)
    details, ece = [], 0.0
    for idx in range(n_bins):
        lo, hi = float(edges[idx]), float(edges[idx+1])
        mask = (confidence >= lo) & ((confidence < hi) if idx < n_bins-1 else (confidence <= hi))
        count = int(mask.sum())
        average_conf = float(confidence[mask].mean()) if count else np.nan
        average_acc = float(correct[mask].mean()) if count else np.nan
        if count: ece += (count/n) * abs(average_acc-average_conf)
        details.append({'bin':idx+1,'lower':lo,'upper':hi,'n':count,
                        'mean_confidence':average_conf,'observed_accuracy':average_acc})
    return {'accuracy':float(correct.mean()), 'ece':float(ece),
            'brier':brier,'nll':nll}, pd.DataFrame(details)


def load_and_validate(csv_path: Path, summary: dict):
    df = pd.read_csv(csv_path)
    expected = ['file', 'true_class', 'raw_prediction', 'raw_confidence',
                'calibrated_confidence', 'correct']
    for prefix in ('raw','calibrated'):
        expected += [f'{prefix}_prob_{c}' for c in CLASSES]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in Step 12 file: {missing}')
    if df.file.duplicated().any():
        raise ValueError('Duplicate MRI filenames in saved Step 12 results.')
    if df.true_class.isna().any() or not set(df.true_class).issubset(set(CLASSES)):
        raise ValueError('Unexpected true-class values in the saved CSV.')
    y = df.true_class.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(dtype=int)
    found = {}
    for key, prefix in (('test_raw','raw'),('test_temperature_scaled','calibrated')):
        p = df[[f'{prefix}_prob_{c}' for c in CLASSES]].to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p < -1e-9).any():
            raise ValueError(f'Invalid probability values in {prefix} data.')
        if not np.allclose(p.sum(axis=1), 1., atol=1e-5):
            raise ValueError(f'{prefix} class probabilities do not add up to 1.')
        if not np.allclose(np.max(p,axis=1), df[f'{prefix}_confidence'].to_numpy(), atol=1e-6):
            raise ValueError(f'{prefix} confidences mismatch saved class probabilities.')
        metrics, bins = metric_values(p, y)
        for name in ('accuracy','ece','brier','nll'):
            ref = float(summary['calibration'][key][name])
            if not np.isclose(metrics[name],ref,rtol=1e-5,atol=1e-6):
                raise ValueError(f'{key}.{name} differs from original Step 12 summary: {metrics[name]:.8f} vs {ref:.8f}')
        found[prefix] = (p, metrics, bins)
    guess = np.argmax(found['raw'][0],axis=1)
    guess_scaled = np.argmax(found['calibrated'][0],axis=1)
    if not np.array_equal(guess, guess_scaled):
        raise ValueError('Temperature scaling changed the predicted class labels; investigate.')
    if not np.array_equal((guess == y).astype(int), df.correct.astype(int).to_numpy()):
        raise ValueError('Saved correct labels do not match predictions.')
    if not np.array_equal([CLASSES[i] for i in guess],df.raw_prediction.to_numpy()):
        raise ValueError('Saved predicted labels do not match raw probabilities.')
    return df, found


def figure_reliability(found, output):
    fig, ax = plt.subplots(figsize=(9,7))
    ax.plot([0,1],[0,1],ls='--',lw=1.5,color='gray',label='Perfect calibration')
    for prefix, label, color in [('raw','Before calibration','tab:orange'),
                                  ('calibrated','After temperature scaling','tab:blue')]:
        bins = found[prefix][2]
        bins = bins[bins.n > 0]
        ax.plot(bins.mean_confidence,bins.observed_accuracy,'-',lw=1.5,color=color,alpha=.7)
        sizes = 28+20*np.log1p(bins.n)
        ax.scatter(bins.mean_confidence,bins.observed_accuracy,s=sizes,color=color,alpha=.8,label=label)
    ax.set(xlim=(0,1.01),ylim=(0,1.01),xlabel='Mean confidence in each bin',
           ylabel='Actual accuracy in each bin', title='V9 test-set calibration (15 equal-width bins)')
    ax.legend(loc='lower right')
    ax.grid(alpha=.15)
    fig.text(.5,.025,'Marker size reflects number of MRIs in a bin. Sparse bins have highly uncertain accuracy estimates.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(output,dpi=180);plt.close(fig)


def figure_metrics(found,out):
    names = ['ece','nll','brier'];labels=['ECE ↓','NLL ↓','Multiclass Brier ↓']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,name,label in zip(axes,names,labels):
        a,b=found['raw'][1][name],found['calibrated'][1][name]
        bars=ax.bar(['Before','After'],[a,b],color=['tab:orange','tab:blue'],width=.55)
        ax.set(title=label,ylabel='Lower is better',ylim=(0,max(a,b)*1.30))
        for bar,value in zip(bars,[a,b]):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(a,b)*.03,
                    f'{value:.5f}',ha='center',va='bottom',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('V9 held-out test set (1,374 MRIs): same accuracy before and after (97.89%)')
    fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)


def figure_wrong_confidence(df,out):
    wrong=df.loc[df.correct.eq(0)].sort_values('raw_confidence',ascending=False).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(11,6))
    xs=np.arange(len(wrong))
    ax.scatter(xs,wrong.raw_confidence*100,label='Before',color='tab:orange',s=45)
    ax.scatter(xs,wrong.calibrated_confidence*100,label='After scaling',color='tab:blue',s=45)
    for x,a,b in zip(xs,wrong.raw_confidence*100,wrong.calibrated_confidence*100):
        ax.plot([x,x],[a,b],color='grey',alpha=.28,lw=1)
    ax.axhline(95,color='tab:red',linestyle='--',lw=1.2,label='95% confidence')
    ax.set(xlabel='29 incorrectly classified MRIs (sorted by raw confidence)',
           ylabel='Predicted-class confidence (%)',ylim=(20,102),
           title='Temperature scaling lowers confidence; it does not change incorrect labels')
    ax.legend(loc='lower left');ax.grid(alpha=.2,axis='y')
    fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)


def locate_data_image(root:Path,file:str):
    candidates = [root/'data'/'brain-tumor-mri'/file,root/'data'/file,root/file]
    return next((p for p in candidates if p.is_file()),None)


def figure_examples(df,root,out):
    wrong=df.loc[df.correct.eq(0)].sort_values('raw_confidence',ascending=False).head(4)
    images=[]
    try:
        from PIL import Image
    except ImportError:
        print('Pillow not installed; skipping MRI image gallery.')
        return False
    for _,r in wrong.iterrows():
        path=locate_data_image(root,str(r.file))
        if path is None:
            print(f'Could not find MRI image {r.file}; skipping gallery.')
            return False
        with Image.open(path) as im:
            images.append(np.array(im.convert('L')))
    fig,axes=plt.subplots(2,2,figsize=(11,11))
    for ax,(_,r),im in zip(axes.ravel(),wrong.iterrows(),images):
        ax.imshow(im,cmap='gray');ax.axis('off')
        ax.set_title(f'ACTUAL: {r.true_class}  |  WRONG PREDICTION: {r.raw_prediction}\n'
                     f'Before: {r.raw_confidence*100:.3f}%   →   After: {r.calibrated_confidence*100:.3f}%',
                     fontsize=10)
    fig.suptitle('Four real V9 test images: confident errors remain errors after calibration',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out,dpi=180);plt.close(fig)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.project_root.resolve()
    src=root/'results'/'12_calibration'
    datafile=src/'per_image_test_calibration.csv'
    summaryfile=src/'calibration_summary.json'
    if not datafile.is_file() or not summaryfile.is_file():
        raise SystemExit('Missing Step 12 outputs. Expected results/12_calibration/\n'
                         '  per_image_test_calibration.csv\n  calibration_summary.json\n'
                         'Restore your ORIGINAL V9 results first; no recalculation is required.')
    summary=json.loads(summaryfile.read_text(encoding='utf-8'))
    if summary.get('temperature_fit_on')!='validation_only':
        raise RuntimeError('Expected validation-only temperature fitting; examine saved summary.')
    temp=float(summary['temperature'])
    df,found=load_and_validate(datafile,summary)
    out=root/'results'/'28_confidence_calibration_audit';out.mkdir(parents=True,exist_ok=True)
    n=len(df);wrong=df.loc[df.correct.eq(0)].copy(); right=df.loc[df.correct.eq(1)].copy()
    if n != 1374 or len(wrong) != 29:
        print(f'NOTE: Current saved data contain {n} test images, {len(wrong)} wrong (historical V9 was 1374 / 29).')
    figure_reliability(found,out/'01_reliability_diagram.png')
    figure_metrics(found,out/'02_ece_nll_brier.png')
    figure_wrong_confidence(df,out/'03_wrong_predictions_confidence.png')
    has_images=figure_examples(df,root,out/'04_real_confidently_wrong_mris.png')
    wrong.to_csv(out/'all_29_wrong_predictions_before_after.csv',index=False)
    wrong.sort_values('raw_confidence',ascending=False).head(15).to_csv(out/'top_15_confidently_wrong.csv',index=False)
    for prefix in ('raw','calibrated'):
        found[prefix][2].to_csv(out/f'{prefix}_15_bin_table.csv',index=False)
    counts={str(int(t*100)): {'raw': int((wrong.raw_confidence >= t).sum()),
                              'calibrated': int((wrong.calibrated_confidence >= t).sum())}
           for t in (.90,.95,.99)}
    report={
        'source':'Step 12 original saved V9 results (no retraining or refitting)',
        'temperature':temp,'temperature_fitted_on':'validation only',
        'test_count':n,'test_wrong':len(wrong),'test_correct':len(right),
        'test_before':found['raw'][1],'test_after':found['calibrated'][1],
        'validation_before':{k:summary['calibration']['val_raw'][k] for k in ('accuracy','ece','brier','nll')},
        'validation_after':{k:summary['calibration']['val_temperature_scaled'][k] for k in ('accuracy','ece','brier','nll')},
        'wrong_prediction_confidence_threshold_counts':counts,
        'mean_wrong_confidence_before':float(wrong.raw_confidence.mean()),
        'mean_wrong_confidence_after':float(wrong.calibrated_confidence.mean()),
        'n_example_images_found':4 if has_images else 0,
        'caveats':[
            'Calibration estimates population-level reliability in this dataset, not correctness of any individual image.',
            'Temperature scaling changes probabilities, not predicted classes or classification accuracy.',
            'ECE depends on bin choice and can understate localized overconfidence.',
            'Original V9 test set has been revisited; do not interpret repeated explorations as fresh independent external validation.',
            'No patient-independent or clinical validation has been established.'
        ]}
    (out/'step28_calibration_audit_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=[
        '# Step 28 — V9 confidence calibration audit',
        '',f'Original saved source: `results/12_calibration/`; temperature T={temp:.6f} (validation fitted).',
        f'Test images: {n}; correct: {len(right)}; wrong: {len(wrong)}.',
        '', '| Metric | Before | After |', '|---|---:|---:|',
        *[f'| {key.upper()} | {found["raw"][1][key]:.6f} | {found["calibrated"][1][key]:.6f} |'
          for key in ('accuracy','ece','nll','brier')],
        '', '| Wrong predictions with confidence ≥ | Before | After |', '|---|---:|---:|',
        *[f'| {t}% | {counts[str(t)]["raw"]} | {counts[str(t)]["calibrated"]} |' for t in (90,95,99)],
        '', '**Interpretation:** Better average calibration, but no change in which MRI classes are predicted.',
        'A high-confidence wrong prediction can remain high-confidence even after scaling.',
        'ECE uses 15 equal-width bins; Brier is the *sum over four classes* per example.',
        'These results are retrospective V9 test-set diagnostics, not clinical validation.',
        '', 'See generated figures and CSVs alongside this report.'
    ]
    (out/'STEP28_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'\nSTEP 28 / V9 CALIBRATION — {n} TEST IMAGES; {len(wrong)} INCORRECT')
    print(f'Validation-fitted temperature: {temp:.6f}')
    for metric in ('accuracy','ece','nll','brier'):
        print(f'  {metric:8}: before={found["raw"][1][metric]:.6f} | after={found["calibrated"][1][metric]:.6f}')
    for t in (90,95,99):
        v=counts[str(t)];print(f'  Wrong with confidence >= {t}%: before={v["raw"]} | after={v["calibrated"]}')
    print('NOTE: Temperature scaling improves calibration, but DOES NOT fix a wrong predicted class.')
    print('RESULTS:',out)
    if not has_images: print('Optional MRI gallery skipped (images missing). The numerical audit is complete.')
    print('DONE. No training or MC Dropout rerun.')

if __name__=='__main__':
    main()
