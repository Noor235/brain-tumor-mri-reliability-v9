"""V9 STEP 27 -- audit all 29 deterministic errors with MC Dropout uncertainty.

Run in the original V9 project folder:
    python3 -m src.step27_audit_29_errors
Optional: repeat 20 *fresh* MC dropout passes on the 29 errors using the original checkpoint:
    python3 -m src.step27_audit_29_errors --live

Saved-score mode (default) uses original V9 full test MC results from Step 09
and original Step 08 deterministic predictions, matched by exact file path.
The threshold is recomputed from Step 21 validation entropy, not from test labels.
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
from PIL import Image, ImageOps

from . import config as C
from .common import source_image


def _read(path: Path, expected: int, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {name}: {path}')
    df = pd.read_csv(path)
    if len(df) != expected or df.file.nunique() != expected:
        raise ValueError(f'{name}: expected {expected} distinct files, found {len(df)} rows')
    return df


def load_original_v9_results():
    p = C.OUTPUT
    det = _read(p/'08_test'/'per_image_predictions.csv', 1374, 'deterministic test results')
    mc = _read(p/'09_uncertainty'/'uncertainty.csv', 1374, 'test MC results')
    val = _read(p/'21_validation_mc'/'uncertainty.csv', 1099, 'validation MC results')
    split = _read(p/'04_split'/'split_manifest.csv', 6872, 'V9 split manifest')

    for name, df, split_name in [('deterministic',det,'test'),('MC test',mc,'test'),('MC validation',val,'val')]:
        joined = df[['file','true_class']].merge(split[['file','class_name','split']],on='file',validate='one_to_one')
        if len(joined)!=len(df) or set(joined['split'])!={split_name} or not (joined.true_class==joined.class_name).all():
            raise ValueError(f'{name} rows do not match V9 split manifest')

    merged = det[['file','true_class','predicted_class','confidence','correct']].merge(
        mc[['file','true_class','predicted_class','mean_confidence','predictive_entropy', 'mutual_information']],
        on='file',validate='one_to_one',suffixes=('_det','_mc'))
    if len(merged)!=1374 or not (merged.true_class_det==merged.true_class_mc).all():
        raise ValueError('MC and deterministic predictions are not aligned by file')
    if not (merged.correct.astype(bool)==(merged.true_class_det==merged.predicted_class_det)).all():
        raise ValueError('Deterministic prediction correctness mismatch')
    if not np.isfinite(merged[['predictive_entropy','mutual_information','mean_confidence']].to_numpy(float)).all():
        raise ValueError('Nonfinite MC scores')

    threshold = float(np.quantile(val['predictive_entropy'].to_numpy(float),0.9))
    merged['refer_for_review'] = merged.predictive_entropy > threshold
    merged['is_error'] = ~merged.correct.astype(bool)
    errors = merged[merged.is_error].copy().sort_values('predictive_entropy',ascending=False)
    if len(errors)!=29:
        raise AssertionError(f'Expected 29 deterministic errors, found {len(errors)}')
    return merged,errors,threshold


def plot_entropy(errors:pd.DataFrame, threshold:float, output:Path, label='Saved Step 09'):
    a = errors.sort_values('predictive_entropy',ascending=False).reset_index(drop=True)
    x = np.arange(1,len(a)+1)
    fig, ax = plt.subplots(figsize=(13,6.5))
    for referred, symbol, series_label in [(True,'o','Referred'),(False,'X','Missed')]:
        take = a.refer_for_review.to_numpy(bool)==referred
        ax.scatter(x[take], np.maximum(a.predictive_entropy.to_numpy(float)[take],1e-12), marker=symbol,
                   s=75, label=series_label, zorder=3)
    ax.plot(x, np.maximum(a.predictive_entropy.to_numpy(float),1e-12),alpha=.32)
    ax.axhline(threshold,ls='--',lw=1.7, label=f'Validation threshold: {threshold:.7f}')
    ax.set(yscale='log',xlabel='Incorrect test MRI (sorted from highest to lowest uncertainty)',
           ylabel='MC predictive entropy (log scale)',title=f'All 29 wrong V9 predictions | {label}')
    ax.set_xticks(x)
    ax.set_ylim(bottom=max(1e-7,min(a.predictive_entropy.min(),threshold)/5))
    ax.grid(axis='y',alpha=.2)
    ax.legend()
    fig.tight_layout()
    dest = output/'29_wrong_mri_uncertainty_threshold.png'
    fig.savefig(dest,dpi=180)
    plt.close(fig)
    return dest


def plot_gallery(errors:pd.DataFrame, output:Path):
    a = errors.sort_values(['refer_for_review','predictive_entropy'],ascending=[True,True]).reset_index(drop=True)
    fig, axs = plt.subplots(6,5, figsize=(15.5,18),layout='constrained')
    for i,ax in enumerate(axs.flat):
        ax.axis('off')
        if i>=len(a):continue
        row=a.iloc[i]
        try:
            path = source_image(str(row['file']))
            with Image.open(path) as im:
                arr=np.asarray(ImageOps.exif_transpose(im).convert('L'))
            ax.imshow(arr,cmap='gray')
        except (FileNotFoundError, OSError):
            ax.text(.5,.5,'Image unavailable',transform=ax.transAxes,ha='center',va='center')
        base=Path(str(row['file'])).stem
        status='REFERRED' if bool(row.refer_for_review) else 'MISSED ERROR'
        ax.set_title(f'{i+1:02d}. {base}\n{row.true_class_det} → {row.predicted_class_det}\n'
                     f'H={row.predictive_entropy:.6f} | {status}',fontsize=8.8)
    fig.suptitle('Actual V9 held-out MRIs that were classified incorrectly\n'
                 'Sorted: four uncertainty misses first, then the 25 flagged errors', fontsize=14)
    dest=output/'29_wrong_mri_gallery.png'
    fig.savefig(dest,dpi=130,bbox_inches='tight')
    plt.close(fig)
    return dest


def plot_missed(errors:pd.DataFrame,output:Path):
    missed=errors[~errors.refer_for_review].copy()
    if len(missed)==0:return None
    fig, axs=plt.subplots(1,len(missed),figsize=(3.8*len(missed),4.4),layout='constrained')
    for i,(ax,(_,row)) in enumerate(zip(np.array(axs).flat,missed.iterrows()),1):
        path=source_image(row['file'])
        with Image.open(path) as im:
            arr=np.asarray(ImageOps.exif_transpose(im).convert('L'))
        ax.imshow(arr,cmap='gray');ax.axis('off')
        ax.set_title(f'{Path(row.file).stem}\nTrue: {row.true_class_det}\nPredicted: {row.predicted_class_det}\n'
                     f'MC entropy: {row.predictive_entropy:.6f}',fontsize=10)
    fig.suptitle('Four WRONG predictions missed by MC entropy referral',fontsize=15)
    dest=output/'4_missed_wrong_mris.png'
    fig.savefig(dest,dpi=160,bbox_inches='tight')
    plt.close(fig)
    return dest


def write_report(output,all_test,errors,threshold, mode, mc_seed=None):
    flagged = int(errors.refer_for_review.sum())
    missed = len(errors)-flagged
    false_alerts=int((all_test.correct.astype(bool)&all_test.refer_for_review).sum()) if all_test is not None else None
    result = dict(experiment='V9 Step 27 — all 29 deterministic wrong MRI predictions',
        data_mode=mode, validation_target_coverage=0.9, validation_threshold=threshold,
        total_wrong=29, error_referred=flagged, error_missed=missed,
        sensitivity_on_wrong=flagged/29, correct_cases_referred_in_full_test=false_alerts,
        checkpoint_retrained=False,
        notes=['The original V9 deterministic prediction defines correct/wrong.',
               'MC entropy is not a probability that the prediction is wrong.',
               'Threshold was selected from validation data, not test correctness.',
               'This is not clinically validated.'])
    if mc_seed is not None:result['new_mc_dropout_seed']=mc_seed
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fmt = ['file','true_class_det','predicted_class_det','confidence','mean_confidence',
           'predictive_entropy','mutual_information','refer_for_review']
    errors[fmt].to_csv(output/'29_wrong_mri_audit.csv',index=False)
    lines=['# V9 — Audit of all 29 misclassified MRIs','',
        f'- Mode: **{mode}**',f'- Validation-derived entropy threshold: **{threshold:.8f}**',
        f'- Errors referred: **{flagged}/29 ({flagged/29:.2%})**',
        f'- Wrong predictions missed: **{missed}/29**']
    if false_alerts is not None:lines.append(f'- Correct predictions also referred in the full test: **{false_alerts}**')
    lines.extend(['','## Missed wrong predictions',''])
    for r in errors[~errors.refer_for_review].itertuples():
        lines.append(f'- `{r.file}` — true **{r.true_class_det}**, predicted **{r.predicted_class_det}**, entropy **{r.predictive_entropy:.6f}**')
    lines.extend(['','## Interpretation',
        'MC dropout is useful for referral, but some misclassifications receive confident, low-entropy scores.',
        'A referral is a risk warning, not proof that the classification is incorrect.',
        'A missing referral is not proof that the classification is correct.',
        'All numbers are retrospective and require further independent validation.'])
    (output/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def live_refresh(errors:pd.DataFrame,seed:int,passes:int, device:str, checkpoint:Path)->pd.DataFrame:
    # Import lazily; saved-mode requires no torch/checkpoint/GPU.
    from .step25_uncertainty_visual_demo import get_device, load_v9_model, predict_twenty_times
    dev=get_device(device)
    model=load_v9_model(checkpoint,dev)
    live=errors.copy()
    for j,idx in enumerate(live.index,1):
        row=live.loc[idx]
        src=source_image(row['file'])
        # Reusing identical seed mirrors Step 25 but produces stochastic sampled head outputs;
        # Step 09's original seed/RNG sequence may differ. This is a new *run*, not bitwise recreation.
        res=predict_twenty_times(model,src,dev,passes=passes,seed=seed+j)
        if res['det_prediction']!=row['predicted_class_det']:
            raise RuntimeError(f'Deterministic checkpoint mismatch for {row.file}: '
                               f'{res["det_prediction"]} != {row.predicted_class_det}')
        live.at[idx,'predictive_entropy']=res['predictive_entropy']
        live.at[idx,'mutual_information']=res['mutual_information']
        live.at[idx,'mean_confidence']=res['mc_mean_confidence']
        live.at[idx,'refer_for_review']=bool(res['predictive_entropy']>live.attrs['threshold'])
        print(f'  {j:02d}/29 | {row.file} | H={res["predictive_entropy"]:.6f} '
              f'| {"REFER" if bool(live.at[idx,"refer_for_review"]) else "MISSED"}',flush=True)
    return live


def main():
    ap=argparse.ArgumentParser(description='Visually audit ALL 29 wrong V9 test MRIs, original MC Dropout uncertainty scores.')
    ap.add_argument('--out',type=Path,default=C.OUTPUT/'27_audit_29_errors')
    ap.add_argument('--live',action='store_true',help='Also recompute 20 MC passes for each error using your checkpoint (optional and slower)')
    ap.add_argument('--passes',type=int,default=20)
    ap.add_argument('--mc-seed',type=int,default=42)
    ap.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    ap.add_argument('--checkpoint',type=Path,default=C.CHECKPOINT)
    args=ap.parse_args()
    if args.passes<2:ap.error('passes must be >=2')
    all_test,errors,threshold=load_original_v9_results()
    mode='saved original Step 09 MC Dropout (20 passes per MRI)'
    if args.live:
        # Live outputs are kept separate to avoid replacing the original V9 summary.
        errors.attrs['threshold']=threshold
        errors=live_refresh(errors,args.mc_seed,args.passes,args.device,args.checkpoint)
        mode=f'new {args.passes}-pass MC Dropout run on same 29 MRI images'
    out=args.out.expanduser().resolve()/('live_29_errors' if args.live else 'saved_29_errors')
    out.mkdir(parents=True,exist_ok=True)
    result=write_report(out,all_test if not args.live else None,errors,threshold,mode,
                        args.mc_seed if args.live else None)
    plot_entropy(errors,threshold,out,mode)
    plot_gallery(errors,out)
    plot_missed(errors,out)
    print('\n'+'='*58)
    print(f'29 wrong test MRIs | {mode}')
    print(f'Validation entropy threshold: {threshold:.8f}')
    print(f'ERRORS FLAGGED: {result["error_referred"]}/29')
    print(f'ERRORS MISSED: {result["error_missed"]}/29')
    if not args.live: print(f'CORRECT PREDICTIONS ALSO FLAGGED: {result["correct_cases_referred_in_full_test"]}')
    print('Four missed errors (or current run missed errors):')
    for r in errors[~errors.refer_for_review].itertuples():
        print(f'  {r.file:48s} {r.true_class_det:11s} -> {r.predicted_class_det:11s} H={r.predictive_entropy:.6f}')
    print(f'\nOpen the figures, CSV and report here: {out}')
    print('No model retraining. This is research-only uncertainty assessment.')

if __name__=='__main__':
    main()
