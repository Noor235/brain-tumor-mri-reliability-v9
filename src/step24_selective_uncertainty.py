"""STEP 24: MC-Dropout selective referral, independently thresholded on V9 validation.

Research-only. The 90% validation-coverage threshold is selected without test labels.
Predictions assessed for correctness are deterministic V9 Step 08 predictions,
matching the model's main displayed class (29 errors). MC Dropout (Step 09)
provides only the uncertainty score; its own argmax differs on one image.
This script never re-trains the classifier or modifies prior outputs.

Usage from ORIGINAL V9 project root:
    python3 -m src.step24_selective_uncertainty
Optionally:
    python3 -m src.step24_selective_uncertainty --out-dir results/24_uncertainty_selective_new
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

DEFAULT_OUTPUT='results/24_uncertainty_selective'
SCORES={'MC predictive entropy':('predictive_entropy',1),
        'MC mutual information':('mutual_information',1),
        'MC mean confidence':('mean_confidence',-1)}
TARGET_COVERAGE=0.90
SECONDARY_COVERAGE=[0.80, 0.85, 0.95]

def csv_file(root, rel):
    p=root/rel
    if not p.is_file():raise FileNotFoundError(f'Missing required V9 result: {p}')
    return pd.read_csv(p)

def load_v9(root):
    val=csv_file(root,'results/21_validation_mc/uncertainty.csv')
    test=csv_file(root,'results/09_uncertainty/uncertainty.csv')
    det=csv_file(root,'results/08_test/per_image_predictions.csv')
    split=csv_file(root,'results/04_split/split_manifest.csv')
    def check(df,name,n,split_type):
        if len(df)!=n or df.file.nunique()!=n:
            raise ValueError(f'{name}: expected {n} unique image rows, got {len(df)}')
        m=df.merge(split[['file','class_name','near_group','split']],on='file',validate='one_to_one')
        if len(m)!=n or set(m['split'])!={split_type} or (m.true_class!=m.class_name).any():
            raise AssertionError(f'{name}: split or true-class mismatch')
    check(val,'MC validation',1099,'val')
    check(test,'MC test',1374,'test')
    check(det,'Deterministic test',1374,'test')
    t=test.merge(det[['file','predicted_class','correct']],on='file',validate='one_to_one',suffixes=('_mc','_det'))
    assert len(t)==1374 and not t.isna().any().any()
    assert (t.true_class==t.predicted_class_det).eq(t.correct_det.astype(bool)).all()
    if int((1-t.correct_det).sum())!=29:raise AssertionError('Expected 29 deterministic test errors')
    if not all(np.isfinite(val[col]).all() and np.isfinite(t[col]).all() for col,_ in SCORES.values()):
        raise ValueError('Missing or nonfinite uncertainty scores')
    return val,t

def selection(score, val_score, det_correct, coverage):
    # The cut-off is the TRAINING-FREE quantile of validation uncertainty only.
    threshold=float(np.quantile(val_score,coverage))
    accept=np.asarray(score)<=threshold
    correct=np.asarray(det_correct,dtype=bool)
    rejected=~accept
    error=~correct
    if not accept.any():raise ValueError('No retained cases')
    return {
      'validation_target_coverage':float(coverage),'validation_threshold':threshold,
      'test_coverage':float(accept.mean()),'test_n_retained':int(accept.sum()),
      'test_n_referred':int(rejected.sum()),
      'test_retained_accuracy':float(correct[accept].mean()),
      'test_retained_error_rate':float(error[accept].mean()),
      'test_error_capture_fraction':float((error&rejected).sum()/error.sum()),
      'test_errors_referred':int((error&rejected).sum()),
      'test_errors_missed':int((error&accept).sum()),
      'test_correct_referred':int((correct&rejected).sum()),
      'test_correct_retained':int((correct&accept).sum()),
      'test_referred_error_fraction':float(error[rejected].mean()) if rejected.any() else 0.,
      'test_accuracy_all':float(correct.mean()),
    }

def risk_at_coverage(score,wrong):
    order=np.argsort(score,kind='stable')
    errors=np.asarray(wrong,dtype=int)[order]
    k=np.arange(1,len(errors)+1)
    risk=np.cumsum(errors)/k
    # Trapezoid with 0 risk at 0 coverage. AURC, lower is better.
    x=np.r_[0,k/len(errors)]; y=np.r_[0,risk]
    aurc=float(np.trapezoid(y,x))
    return x,y,aurc

def run(root:Path,out_dir:Path):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'Won\'t overwrite: {out_dir}')
    val,test=load_v9(root)
    wrong=(1-test.correct_det.astype(int)).to_numpy()
    comparison={}; threshold_specs={}
    for name,(col,orientation) in SCORES.items():
        v=np.asarray(val[col],dtype=float)*orientation
        s=np.asarray(test[col],dtype=float)*orientation
        op=selection(s,v,test.correct_det, TARGET_COVERAGE)
        rows={str(round(k,2)):selection(s,v,test.correct_det,k)
              for k in [*SECONDARY_COVERAGE,TARGET_COVERAGE]}
        _,_,aurc=risk_at_coverage(s,wrong)
        comparison[name]={
           'roc_auc_error':float(roc_auc_score(wrong,s)),
           'average_precision_error':float(average_precision_score(wrong,s)),
           'risk_coverage_area_aurc':aurc,
           'primary_validation_90pct_coverage':op,
           'secondary_validation_coverage_sensitivity':rows,
        }
        threshold_specs[name]={'source':'V9 validation MC uncertainty (1099 images)',
           'feature':col,'higher_score_means_higher_uncertainty':True,
           'orientation':orientation,'threshold_q90':op['validation_threshold'],
           'target_validation_coverage':TARGET_COVERAGE}
    out_dir.mkdir(parents=True,exist_ok=False)
    primary=comparison['MC predictive entropy']['primary_validation_90pct_coverage']
    summary={
       'experiment':'STEP24 validation-thresholded MC-dropout selective referral',
       'dataset':'original V9 Kaggle 2026 cleaned group-split',
       'model':'unchanged trained V9 EfficientNet-B4',
       'n_validation':len(val),'n_test':len(test),
       'n_deterministic_test_errors':int(wrong.sum()),
       'n_mc_test_errors':int((1-test.correct_mc).sum()),
       'n_mc_vs_deterministic_label_disagreements':int((test.predicted_class_mc!=test.predicted_class_det).sum()),
       'threshold_selection':'90th percentile of validation MC predictive entropy, without inspecting test labels',
       'primary_score':'MC predictive entropy','primary_result':primary,
       'score_comparison':comparison,
       'limitations':[
           'Previously examined V9 test data and previously chosen method; exploratory retrospective evaluation, not an independent prospective test.',
           'Near-duplicate groups were split by V9 original manifest; real patient IDs unavailable.',
           'Only 29 deterministic mistakes; wide uncertainty in error-capture estimates.',
           'No out-of-distribution or independent hospital external validation.',
           'High certainty / non-referred status does not confirm correctness or clinical safety.',
           'Using validation entropy quantiles controls coverage approximately, not individual-case risk.',
           'MC-dropout entropy does not outperform calibrated confidence by default; compare baselines.',
       ]}
    (out_dir/'thresholds.json').write_text(json.dumps({
       'description':'Research-only V9 selective referral; no clinical threshold or individual diagnostic guarantee',
       'threshold_selection':'validation-only quantile',
       'primary_method':'MC predictive entropy',
       'primary_threshold':primary['validation_threshold'],
       'target_validation_coverage':TARGET_COVERAGE,
       'source_validation_n':len(val),
       'score_thresholds':threshold_specs
       },indent=2))
    (out_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    test_rows=test[['file','true_class','predicted_class_det','correct_det','predicted_class_mc','correct_mc',
                    'predictive_entropy','mutual_information','mean_confidence']].copy()
    for name,(col,sgn) in SCORES.items():
        key={'MC predictive entropy':'referral_entropy','MC mutual information':'referral_mutual_information',
             'MC mean confidence':'referral_mean_confidence'}[name]
        thr=comparison[name]['primary_validation_90pct_coverage']['validation_threshold']
        test_rows[key]=((test[col].astype(float)*sgn)>thr).astype(int)
    test_rows.to_csv(out_dir/'per_image_uncertainty_referral.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.3,5.7))
    for name,(col,sgn) in SCORES.items():
        x,y,aurc=risk_at_coverage(test[col].to_numpy()*sgn,wrong)
        ax.plot(x*100,y*100,label=f'{name} (AURC {aurc:.4f})',lw=2)
    ax.axhline(wrong.mean()*100,color='gray',ls='--',alpha=.7,label='No referral (baseline test error rate)')
    ax.plot([primary['test_coverage']*100],[primary['test_retained_error_rate']*100],
            marker='o',markersize=7,color='black',label='Primary 90%-validation entropy threshold')
    ax.set(xlabel='Coverage — retained images (%)',ylabel='Error rate among retained images (%)',
           title='V9 selective prediction: empirical risk–coverage (1,374 test MRIs)',xlim=(0,100))
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left');fig.tight_layout()
    fig.savefig(out_dir/'risk_coverage.png',dpi=200);plt.close(fig)
    # Fixed primary operating point at validation quantile 90, grouped summary plot
    a=primary
    fig,ax=plt.subplots(figsize=(6.7,3.7))
    bars=ax.barh(['No referral: all test MRIs','Uncertainty-accepted MRIs'],
                 [100*a['test_accuracy_all'],100*a['test_retained_accuracy']])
    ax.set_xlim(95,100.5);ax.set_xlabel('Classification accuracy (%)')
    ax.set_title('V9 selective accuracy: reported together with coverage')
    for bar,cov in zip(bars,[1.,a['test_coverage']]):
        ax.text(bar.get_width()+.08,bar.get_y()+bar.get_height()/2,
                f'{bar.get_width():.2f}%  |  coverage {cov:.1%}',va='center',fontsize=9)
    fig.tight_layout();fig.savefig(out_dir/'selective_accuracy.png',dpi=200);plt.close(fig)
    print('STEP24 selective uncertainty: validation-only threshold, 90% target coverage')
    print(json.dumps(primary,indent=2))
    print('Saved:',out_dir)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',type=Path,default=Path.cwd())
    ap.add_argument('--out-dir',type=Path,default=Path(DEFAULT_OUTPUT))
    args=ap.parse_args()
    root=args.project_root.resolve()
    dest=args.out_dir if args.out_dir.is_absolute() else root/args.out_dir
    run(root,dest)
