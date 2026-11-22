"""STEP 23 — Exploratory cross-fitted error warning based on completed V9 Step 22.

Uses only saved, already-computed test-set measurements. No MRI reprocessing,
checkpoint editing, classifier training, or model architecture modifications.

IMPORTANT: V9 test-set findings/layer choice were inspected in earlier exploratory
work. Cross-fitted performance is INTERNAL and NOT an untouched external test.
Samples in detected near-duplicate groups remain in the same fold.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from . import config as C

PROFILE_REL = 'gradcam_features4_full_test/earlier/per_image_reliability_profile.csv'
F_COLS = ['calibrated_confidence', 'mc_predictive_entropy',
          'faithfulness_del_high_minus_random_20', 'mean_stability_cosine',
          'explanation_shape_uncertainty']
VIEWS = {
  'Confidence only': ['calibrated_confidence'],
  'Confidence + MC uncertainty': F_COLS[:2],
  'All five measurements': F_COLS,
}
FEATURE_DESCRIPTIONS = {
    'calibrated_confidence': 'Calibrated confidence: lower can indicate error',
    'mc_predictive_entropy': 'MC predictive entropy: higher means less decisive prediction',
    'faithfulness_del_high_minus_random_20': 'Deletion faithfulness advantage over random patches',
    'mean_stability_cosine': 'Mean Grad-CAM similarity under three small image changes',
    'explanation_shape_uncertainty': 'MC Grad-CAM mean shape standard deviation',
}
SEED = 42


def csv_rows(p):
    with open(p, newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def load_profile():
    prof = C.OUTPUT / PROFILE_REL
    manifest = C.SPLIT
    if not prof.is_file() or not manifest.is_file():
        raise FileNotFoundError(f'Missing {prof} or {manifest}; use original V9 results.')
    rows = csv_rows(prof)
    if len(rows) != 1374 or len({r['file'] for r in rows}) != 1374:
        raise ValueError('Expected 1374 unique matched test MRI rows')
    split = {r['file']:r for r in csv_rows(manifest) if r['split']=='test'}
    if set(split) != {r['file'] for r in rows}:
        raise ValueError('Test manifest and corrected reliability profiles do not match')
    for r in rows:
        if r['true_class'] != split[r['file']]['class_name']:
            raise ValueError('Class mismatch: '+r['file'])
        if int(r['deterministic_correct']) != int(r['true_class']==r['deterministic_predicted_class']):
            raise ValueError('Incorrect correctness flag: '+r['file'])
    X = np.array([[float(r[k]) for k in F_COLS] for r in rows], dtype=float)
    if not np.isfinite(X).all():
        raise ValueError('Missing or non-finite metric; no silent imputation')
    y = np.array([1-int(r['deterministic_correct']) for r in rows], dtype=int)
    groups = np.array([split[r['file']]['near_group'] for r in rows])
    if y.sum()!=29:
        raise ValueError(f'Expected 29 wrong predictions, got {y.sum()}')
    return rows,X,y,groups


def pipeline():
    # Not a calibrated probability! Balanced class weights change intercept/base rate.
    return make_pipeline(StandardScaler(), LogisticRegression(
        class_weight='balanced', max_iter=3000, C=0.5, random_state=SEED))


def cv_predict(X,y,groups):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {key:np.full(len(y), np.nan) for key in VIEWS}
    oof['Direct confidence (no training)'] = 1 - X[:,0]
    oof['MC entropy alone (no training)'] = X[:,1].copy()
    fold_of = np.zeros(len(y),dtype=int)
    fold_meta=[]
    for k,(tr,te) in enumerate(cv.split(X,y,groups),1):
        if y[tr].sum()==0 or y[te].sum()==0:
            raise ValueError('A fold has no errors: reduce folds or use new random seed')
        if set(groups[tr]).intersection(groups[te]):
            raise AssertionError('Near-duplicate leakage between warning-model folds')
        fold_of[te]=k
        fold_meta.append({'fold':k,'train_n':len(tr),'evaluation_n':len(te),
                          'train_errors':int(y[tr].sum()),'evaluation_errors':int(y[te].sum())})
        for name,cols in VIEWS.items():
            idx=[F_COLS.index(col) for col in cols]
            mod=pipeline().fit(X[tr][:,idx],y[tr])
            oof[name][te]=mod.predict_proba(X[te][:,idx])[:,1]
    if any(not np.isfinite(v).all() for v in oof.values()):
        raise AssertionError('Incomplete CV predictions')
    return oof,fold_of,fold_meta


def bootstrap_delta(y, new, old, seed=42, n=1000):
    rng = np.random.default_rng(seed)
    neg=np.flatnonzero(y==0);pos=np.flatnonzero(y==1)
    diffs=[]
    for _ in range(n):
        ids=np.concatenate([rng.choice(neg,len(neg),replace=True),
                            rng.choice(pos,len(pos),replace=True)])
        diffs.append(roc_auc_score(y[ids],new[ids])-roc_auc_score(y[ids],old[ids]))
    return {'estimate':float(roc_auc_score(y,new)-roc_auc_score(y,old)),
            'stratified_bootstrap_95ci':[float(q) for q in np.percentile(diffs,[2.5,97.5])]}


def save_plot(y,oof,out):
    fig,ax=plt.subplots(figsize=(7.8,6))
    for name,p in oof.items():
        fpr,tpr,_=roc_curve(y,p)
        ax.plot(fpr,tpr,lw=2,label=f'{name} (AUC {roc_auc_score(y,p):.3f})')
    ax.plot([0,1],[0,1],':',color='grey',label='Chance')
    ax.set(xlabel='False warning rate among correct MRIs',
           ylabel='Wrong-prediction detection rate',
           title='V9 internal 5-fold cross-fitted error-warning comparison',
           xlim=(0,1),ylim=(0,1))
    ax.legend(loc='lower right',fontsize=8)
    ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out,dpi=190);plt.close(fig)


def main(outname='23_reliability_warning'):
    out=C.OUTPUT/outname
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Existing output found: {out}. Won\'t overwrite past results.')
    rows,X,y,groups=load_profile()
    oof,folds,meta=cv_predict(X,y,groups)
    summary={'study':'V9 error-warning retrospective internal cross-fitting',
      'original_image_classification_checkpoint':'07_training/best_v9.pt',
      'corrected_gradcam_layer':'features[-4]',
      'n':len(y),'errors':int(y.sum()),'correct':int((1-y).sum()),
      'folds':meta,'fold_group':'detected near_group',
      'definition_of_error':'deterministic predicted class differs from Kaggle class label',
      'warning_score_interpretation':'relative prioritization for review, NOT calibrated individual error probability',
      'limitations':['Exploratory test-set reuse and earlier model/layer inspection; not an untouched test.',
                     'Only 29 errors. Uncertainty in conclusions is substantial.',
                     'Near-image groups are separated across CV folds; patient identity is unknown.',
                     'Correct/incorrect refer to dataset labels, not confirmed clinical truth.',
                     'Five metrics cannot guarantee correctness or medical reliability.'],
      'comparisons':{},'paired_auc_differences':{}}
    for name,v in oof.items():
        summary['comparisons'][name]={
            'roc_auc':float(roc_auc_score(y,v)),
            'average_precision':float(average_precision_score(y,v)),
            'base_error_prevalence':float(y.mean())}
    summary['paired_auc_differences']['five_minus_direct_confidence']=bootstrap_delta(y,oof['All five measurements'],oof['Direct confidence (no training)'])
    summary['paired_auc_differences']['five_minus_confidence']=bootstrap_delta(y,oof['All five measurements'],oof['Confidence only'])
    summary['paired_auc_differences']['five_minus_confidence_mc']=bootstrap_delta(y,oof['All five measurements'],oof['Confidence + MC uncertainty'])

    # Save the fitted demonstration model. Coefficients are Python-independent JSON.
    learned={}
    for name,cols in VIEWS.items():
        ids=[F_COLS.index(col) for col in cols]
        sc=StandardScaler().fit(X[:,ids])
        mod=LogisticRegression(class_weight='balanced',max_iter=3000,C=0.5,random_state=SEED).fit(sc.transform(X[:,ids]),y)
        learned[name]={
            'features':cols,'means':[float(z) for z in sc.mean_],
            'scales':[float(z) for z in sc.scale_],
            'coefficients':[float(z) for z in mod.coef_[0]],
            'intercept':float(mod.intercept_[0]),
        }
    # Exploratory thresholds on FULL retrospective data (never call calibrated).
    # At the 95th percentile of correctly classified cases for each model.
    for name,m in learned.items():
        s=score_with_model(X, m)
        m['illustrative_high_review_threshold']=float(np.percentile(s[y==0],95))
    learned['metadata']={'trained_on':'previously examined 1,374 V9 test records',
         'use':'demonstration and exploratory research only',
         'warning_score_not_probability':True,
         'sampling_validation':'5-fold StratifiedGroupKFold by near-duplicate group',
         'features_description':FEATURE_DESCRIPTIONS,
         'class_counts':{'correct':int((y==0).sum()),'incorrect':int((y==1).sum())}}
    out.mkdir(parents=True,exist_ok=False)
    reference_quantiles={field:{str(q):float(np.quantile(X[:,i],q)) for q in [0.1,0.25,0.5,0.75,0.9,0.95]} for i,field in enumerate(F_COLS)}
    (out/'reference_quantiles.json').write_text(json.dumps({'description':'Unlabelled descriptive distribution of the retrospectively examined V9 test cohort, not validated clinical thresholds','quantiles':reference_quantiles},indent=2))
    with (out/'oof_error_warning_scores.csv').open('w',newline='',encoding='utf8') as f:
        cols=['file','true_class','predicted_class','wrong','fold','calibrated_confidence','mc_predictive_entropy',
              'faithfulness_del_high_minus_random_20','mean_stability_cosine','explanation_shape_uncertainty']+[
              'score_confidence','score_confidence_mc','score_five','score_direct_confidence','score_mc_entropy']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for i,r in enumerate(rows):
            d={'file':r['file'],'true_class':r['true_class'],'predicted_class':r['deterministic_predicted_class'],
               'wrong':int(y[i]),'fold':int(folds[i])}
            d.update({k:float(r[k]) for k in F_COLS})
            d.update({'score_confidence':float(oof['Confidence only'][i]),
                      'score_confidence_mc':float(oof['Confidence + MC uncertainty'][i]),
                      'score_five':float(oof['All five measurements'][i]),
                      'score_direct_confidence':float(oof['Direct confidence (no training)'][i]),
                      'score_mc_entropy':float(oof['MC entropy alone (no training)'][i])})
            w.writerow(d)
    (out/'warning_model.json').write_text(json.dumps(learned,indent=2))
    (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
    save_plot(y,oof,out/'cross_fitted_roc.png')
    print('STEP 23 research warning analysis completed — no classifier training, no MRIs reprocessed.')
    print('n=',len(y),'classification errors=',int(y.sum()))
    for n,c in summary['comparisons'].items():
        print(f' {n}: ROC-AUC={c["roc_auc"]:.4f}, AP={c["average_precision"]:.4f}')
    print('Paired AUC differences:',summary['paired_auc_differences'])
    print('Saved:',out)
    return summary


def score_with_model(X, model_spec):
    ids=[F_COLS.index(f) for f in model_spec['features']]
    x=(X[:,ids]-np.asarray(model_spec['means']))/np.asarray(model_spec['scales'])
    logits=x@np.asarray(model_spec['coefficients']) + float(model_spec['intercept'])
    logits=np.clip(logits,-40,40)
    return 1/(1+np.exp(-logits))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out-name',default='23_reliability_warning')
    a=p.parse_args();main(a.out_name)
