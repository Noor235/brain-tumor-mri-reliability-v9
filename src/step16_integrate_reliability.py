"""STEP 16: Combine per-image confidence, uncertainty, faithfulness, stability, explanation
uncertainty on matching split/file IDs. No invented composite reliability score.
Spearman correlations reported with sample sizes; analyses are descriptive/exploratory.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from . import config as C
from .common import output_dir,save_json,check_no_overwrite
from .research_v9 import environment


def readcsv(file):
    if not file.is_file():return []
    with file.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def findone(rows,split):return {r['file']:r for r in rows if r.get('file')}

def combine(args):
    out=C.OUTPUT/f'16_integrated_{args.split}_{args.layer}';check_no_overwrite(out/'integrated_summary.json')
    source=C.OUTPUT
    ex=readcsv(source/f'15_explanation_uncertainty_{args.split}_{args.layer}'/'explanation_uncertainty.csv')
    if not ex:raise FileNotFoundError('Run Step 15 on the desired split and layer first.')
    ux=readcsv(source/'09_uncertainty'/'uncertainty.csv') if args.split=='test' else []
    cal=readcsv(source/'12_calibration'/'per_image_test_calibration.csv') if args.split=='test' else []
    f=readcsv(source/f'13_faithfulness_{args.split}'/'faithfulness_trials.csv')
    st=readcsv(source/f'14_stability_{args.split}'/'stability_trials.csv')
    um=findone(ux,args.split);cm=findone(cal,args.split)
    # Faithfulness at 20% area: high CAM vs random control, deletion (target margin drop).
    byfaith={}
    for r in f:
        if r['layer']==args.layer and r['method']=='deletion' and abs(float(r['fraction'])-.2)<1e-6:
            byfaith.setdefault(r['file'],{})[r['region']]=float(r['score_change'])
    bystable={}
    for r in st:
        if r['layer']==args.layer:
            bystable.setdefault(r['file'],[]).append(float(r['cosine_similarity']))
    merged=[]
    for r in ex:
        name=r['file'];u=um.get(name,{});c=cm.get(name,{});fr=byfaith.get(name,{});stab=bystable.get(name,[])
        merged.append(dict(file=name,true_class=r['true_class'],prediction_correct=r['original_predicted_correct'],
            predictive_entropy=(u.get('predictive_entropy') or r.get('predictive_entropy_from_same_passes') or ''),
            mc_mutual_information=u.get('mutual_information',''),
            raw_confidence=c.get('raw_confidence',''),calibrated_confidence=c.get('calibrated_confidence',''),
            faithfulness_deletion_high_minus_random=(fr['high']-fr['random'] if 'high' in fr and 'random' in fr else ''),
            mean_stability_cosine=(float(np.nanmean(stab)) if stab else ''),
            explanation_shape_uncertainty=r.get('mean_shape_attribution_std',''),
            mc_explanation_flip_rate=r.get('mc_class_flip_rate','')))
    numerical=['predictive_entropy','mc_mutual_information','raw_confidence','calibrated_confidence',
        'faithfulness_deletion_high_minus_random','mean_stability_cosine','explanation_shape_uncertainty','mc_explanation_flip_rate']
    relationships=[('predictive_entropy','explanation_shape_uncertainty'),('predictive_entropy','faithfulness_deletion_high_minus_random'),
        ('predictive_entropy','mean_stability_cosine'),('explanation_shape_uncertainty','mean_stability_cosine'),
        ('explanation_shape_uncertainty','faithfulness_deletion_high_minus_random')]
    correlations=[]
    for x,y in relationships:
        pairs=[]
        for r in merged:
            try:
                a=float(r[x]);b=float(r[y])
                if np.isfinite(a) and np.isfinite(b):pairs.append((a,b))
            except (ValueError,TypeError):pass
        if len(pairs)<5 or np.std([z[0] for z in pairs])==0 or np.std([z[1] for z in pairs])==0:
            corr=None;p=None
        else:
            res=spearmanr(np.asarray(pairs)[:,0],np.asarray(pairs)[:,1]);corr=float(res.statistic);p=float(res.pvalue)
        correlations.append(dict(x=x,y=y,n=len(pairs),spearman_rho=corr,unadjusted_p_value=p))
        print(f'  {x} vs {y}: n={len(pairs)} rho={corr}',flush=True)
    out=output_dir(f'16_integrated_{args.split}_{args.layer}')
    with (out/'per_image_reliability_profile.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(merged[0]));w.writeheader();w.writerows(merged)
    save_json(out/'integrated_summary.json',dict(split=args.split,layer=args.layer,matched_examples=len(merged),
        presence={k:sum(r[k]!='' for r in merged) for k in numerical},correlations=correlations,
        limitation='Pilot subset, post-hoc test inspection may bias interpretation; no patient IDs, no causal conclusions, no arbitrary composite reliability index. Uncorrected p-values descriptive only.'))
    return merged


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['val','test'],default='test');ap.add_argument('--layer',choices=['final','earlier'],default='final')
    args=ap.parse_args()
    C.heading('STEP 16 / INTEGRATED PREDICTION AND EXPLANATION RELIABILITY')
    environment(args.split)
    combine(args)
    print('STEP 16 COMPLETE. Missing analyses left blank rather than fabricated.',flush=True)
if __name__=='__main__':main()
