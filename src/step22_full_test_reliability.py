"""V9 STEP 22 — complete, resumable held-out TEST reliability extension.

Place inside original V9 project's src/ folder and run from project root:
    python -m src.step22_full_test_reliability --layers final earlier

Requires existing Steps 01-09, 12 and the original MRI files. Does not train, tune,
modify input files, checkpoints, split manifests, or any prior result directory.

Analysis is an *exploratory, post-hoc* test-set extension. Layer choices are
reported separately, NOT selected by test-set performance. No composite score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config as C
from .common import source_image, sha_file
from .research_v9 import (
    environment, ResearchCAM, get_layer, normalize_cam,
    cosine, pearson, top_iou, target_margin
)
from .step13_faithfulness import FRACTIONS, patch_masks, score_many, perturb
from .step14_stability import variations
from .step15_explanation_uncertainty import image_panel, softmax_1
from .transforms_v9 import transform

STAGE_NAME = '22_full_test_reliability'
REQUIRED = {
    'test': '08_test/per_image_predictions.csv',
    'mc': '09_uncertainty/uncertainty.csv',
    'cal': '12_calibration/per_image_test_calibration.csv',
}
RELATIONS = [
    ('mc_predictive_entropy', 'explanation_shape_uncertainty'),
    ('mc_predictive_entropy', 'faithfulness_del_high_minus_random_20'),
    ('mc_predictive_entropy', 'mean_stability_cosine'),
    ('explanation_shape_uncertainty', 'mean_stability_cosine'),
    ('explanation_shape_uncertainty', 'faithfulness_del_high_minus_random_20'),
]
NUM_FIELDS = [
    'raw_confidence','calibrated_confidence','mc_mean_confidence',
    'mc_predictive_entropy','mc_mutual_information','mc_variance_predicted_class',
    'faithfulness_del_high_20','faithfulness_del_random_20','faithfulness_del_low_20',
    'faithfulness_del_high_minus_random_20','faithfulness_ins_high_20',
    'faithfulness_ins_random_20','faithfulness_ins_high_minus_random_20',
    'mean_stability_cosine','mean_stability_pearson','mean_stability_top20_iou',
    'stability_prediction_flip_count','explanation_shape_uncertainty',
    'explanation_raw_attribution_std','explanation_mc_class_flip_rate',
    'explanation_mean_cosine_to_first',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv_index(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing existing results file: {path}')
    with path.open(newline='', encoding='utf-8') as fp:
        rows = list(csv.DictReader(fp))
    if any(not r.get('file') for r in rows):
        raise ValueError(f'Missing file ID in {path}')
    result = {r['file']:r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate MRI IDs in existing file: {path}')
    return result


def load_original(test_rows, fingerprint):
    expected_ids = {r['file'] for r in test_rows}
    if len(expected_ids) != len(test_rows):
        raise ValueError('Duplicate files in manifest test split')
    outputs = {k:read_csv_index(C.OUTPUT / relative) for k,relative in REQUIRED.items()}
    for key, mapping in outputs.items():
        if set(mapping) != expected_ids:
            raise RuntimeError(f'{key}: MRI file IDs do not match the verified test split: '
                               f'missing={len(expected_ids-set(mapping))}, extra={len(set(mapping)-expected_ids)}')
    summary_paths = ['08_test/test_metrics.json','09_uncertainty/mc_summary.json',
                     '12_calibration/calibration_summary.json']
    for relative in summary_paths:
        path = C.OUTPUT / relative
        if not path.is_file(): raise FileNotFoundError(path)
        obj = json.loads(path.read_text(encoding='utf-8'))
        if obj.get('split_fingerprint') != fingerprint:
            raise RuntimeError(f'Fingerprint mismatch in {path}; refusing mixed results')
    if len(test_rows) != 1374:
        raise RuntimeError(f'Expected the verified V9 1,374-image test split; found {len(test_rows)}')
    if sum(int(r['correct']) for r in outputs['test'].values()) != 1345:
        raise RuntimeError('Existing V9 deterministic results do not match the 1,345/1,374 reference run')
    for row in test_rows:
        name = row['file']; cls = row['class_name']
        for key in outputs:
            if outputs[key][name]['true_class'] != cls:
                raise RuntimeError(f'Original {key} labels disagree for {name}')
        if outputs['test'][name]['predicted_class'] != outputs['cal'][name]['raw_prediction']:
            raise RuntimeError(f'Test predictions/calibration disagree for {name}')
        if int(outputs['test'][name]['correct']) != int(outputs['cal'][name]['correct']):
            raise RuntimeError(f'Test/calibration correctness differs for {name}')
    return outputs


def project_seed(path, layer):
    h=hashlib.sha256(f'V9-STEP22|{path}|{layer}|{C.SEED}'.encode()).digest()
    return int.from_bytes(h[:4],'little') % (2**31)


def clean_number(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def safe_mean(numbers):
    vals = [clean_number(v) for v in numbers]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def native(obj):
    if isinstance(obj,dict): return {k:native(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)): return [native(v) for v in obj]
    if isinstance(obj,np.generic):return native(obj.item())
    if isinstance(obj,float) and not math.isfinite(obj):return None
    return obj


def open_db(out, meta):
    out.mkdir(parents=True,exist_ok=True)
    dbpath=out/'progress.sqlite3'
    if any(out.iterdir()) and not dbpath.exists():
        raise FileExistsError(f'Nonempty output directory without a Step22 resume database: {out}. '
                              'Use --out-name SOME_NEW_NAME; no files were overwritten.')
    con=sqlite3.connect(dbpath)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
    con.execute('CREATE TABLE IF NOT EXISTS completed (file TEXT NOT NULL, layer TEXT NOT NULL, payload TEXT NOT NULL, '
                'PRIMARY KEY(file,layer))')
    existing=con.execute('SELECT payload FROM metadata WHERE id=1').fetchone()
    if existing:
        if json.loads(existing[0]) != meta:
            con.close()
            raise RuntimeError(f'Existing Step22 run has different parameters/checkpoint/manifest in {dbpath}. '
                               'Use --out-name NEW_NAME for a new run; old results were not overwritten.')
    else:
        con.execute('INSERT INTO metadata(id,payload) VALUES(1,?)',(json.dumps(meta,sort_keys=True),))
    con.commit()
    return con


def record_one(row, layer, index, model, dev, cam, tx, original, output, examples):
    name=row['file']; truth=row['class_name']
    with Image.open(source_image(name)) as im:
        image=ImageOps.exif_transpose(im).convert('L')
        x=tx(image).unsqueeze(0).to(dev)
    with torch.no_grad():
        original_logits=model(x)[0].detach().cpu().numpy()
    target=int(original_logits.argmax())
    predicted=C.CLASSES[target]
    if predicted != original['test'][name]['predicted_class']:
        raise RuntimeError(f'Deterministic prediction mismatch for {name}: model={predicted}, '
                           f'saved={original["test"][name]["predicted_class"]}. '
                           'Check your original checkpoint/dependencies; refusing to combine different model outputs.')
    true_correct = int(truth==predicted)
    raw, _ = cam.get(x,target)
    reference=normalize_cam(raw)
    original_margin=float(target_margin(original_logits,target)[0])
    pixels=x[0,0].detach().cpu().numpy().astype(np.float32)
    blurred=cv2.GaussianBlur(pixels,(0,0),sigmaX=10).astype(np.float32)
    base_margin=float(score_many(model,dev,[torch.from_numpy(blurred)[None,...]],target)[0])
    seed=project_seed(name,layer)
    faith=[]
    for fraction in FRACTIONS:
        masks=patch_masks(reference,fraction,seed+int(fraction*100))
        alternatives=[];keys=[]
        for method in ('deletion','insertion'):
            for kind,mask in masks.items():
                alternatives.append(perturb(x,blurred,mask,method))
                keys.append((method,kind))
        vals=score_many(model,dev,alternatives,target)
        for (method,kind),margin in zip(keys,vals):
            change=original_margin-float(margin) if method=='deletion' else float(margin)-base_margin
            faith.append(dict(fraction=fraction,method=method,region=kind,
                              score_change=change,modified_margin=float(margin)))
    stable=[]
    for title,modified,inverse_dx in variations(pixels,seed):
        xx=torch.from_numpy(modified[None,None,...]).to(dev)
        new_raw,_=cam.get(xx,target)
        compare=normalize_cam(new_raw)
        if inverse_dx:
            compare=cv2.warpAffine(compare,np.float32([[1,0,inverse_dx],[0,1,0]]),
                       (C.IMAGE_SIZE,C.IMAGE_SIZE),flags=cv2.INTER_LINEAR,borderValue=0)
        with torch.no_grad(): new_target=int(model(xx).argmax(1).item())
        # Original step14's variation function actually uses std=0.02 in [-1,1]
        # for 'noise_std_0.01'. Keep its perturbation precisely, clarify the label.
        label='noise_std_0.02_normalized' if title=='noise_std_0.01' else title
        stable.append(dict(perturbation=label,cosine_similarity=clean_number(cosine(reference,compare)),
                           pearson_correlation=clean_number(pearson(reference,compare)),
                           top20_iou=clean_number(top_iou(reference,compare,.2)),
                           prediction_flip=int(new_target!=target)))
    # MC Grad-CAM: same fixed class target. Only explicit classifier dropout enabled.
    dropout=[m for m in model.modules() if isinstance(m,nn.Dropout)]
    if len(dropout)!=1:
        raise RuntimeError(f'Expected exactly one V9 nn.Dropout, found {len(dropout)}')
    # Stable per-file seed allows uninterrupted runs and resumed runs to agree.
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    mc_shapes=[];mc_raws=[];mc_predictions=[];mc_probabilities=[]
    try:
        for m in dropout:m.train()
        for _ in range(C.MC_PASSES):
            mraw, logits=cam.get(x,target)
            mc_shapes.append(normalize_cam(mraw))
            mc_raws.append(mraw)
            mc_predictions.append(int(np.asarray(logits).argmax()))
            mc_probabilities.append(softmax_1(logits))
    finally:
        for m in dropout:m.eval()
    shapes=np.stack(mc_shapes)
    rawstack=np.stack(mc_raws)
    stdshape=shapes.std(axis=0)
    mean_shape=shapes.mean(axis=0)
    mm=dict(mean_shape_attribution_std=float(stdshape.mean()),
            mean_raw_attribution_std=float(rawstack.std(axis=0).mean()),
            mc_class_flip_rate=float(np.mean(np.asarray(mc_predictions)!=target)),
            cosine_to_first_mc_cam=safe_mean([cosine(shapes[0],z) for z in shapes[1:]]),
            predictive_entropy_from_same_passes=float(
                -np.sum(np.mean(mc_probabilities,axis=0)*
                        np.log(np.clip(np.mean(mc_probabilities,axis=0),1e-12,1)))) )
    if name in examples:
        folder=output/'example_maps'/layer
        folder.mkdir(parents=True,exist_ok=True)
        short=hashlib.sha256(name.encode()).hexdigest()[:12]
        filename=f'{truth}_{"WRONG" if not true_correct else "correct"}_{short}.png'
        image_panel(pixels,mean_shape,stdshape,target,folder/filename)
    return dict(file=name,true_class=truth,deterministic_predicted_class=predicted,
                deterministic_correct=true_correct,layer=layer,faithfulness_trials=faith,
                stability_trials=stable,explanation=mm)


def combine_one(computed, original):
    file=computed['file']; m=original['mc'][file]; c=original['cal'][file]
    faith={}
    for entry in computed['faithfulness_trials']:
        if abs(entry['fraction']-.2)<1e-6:
            faith[(entry['method'],entry['region'])]=entry['score_change']
    def getf(method,region):return faith.get((method,region))
    def diff(method):
        h,r=getf(method,'high'),getf(method,'random')
        return h-r if h is not None and r is not None else None
    s=computed['stability_trials'];x=computed['explanation']
    return dict(file=file,true_class=computed['true_class'],layer=computed['layer'],
        deterministic_predicted_class=computed['deterministic_predicted_class'],
        deterministic_correct=computed['deterministic_correct'],
        mc_predicted_class=m['predicted_class'],mc_correct=int(m['correct']),
        raw_confidence=clean_number(c['raw_confidence']),
        calibrated_confidence=clean_number(c['calibrated_confidence']),
        mc_mean_confidence=clean_number(m['mean_confidence']),
        mc_predictive_entropy=clean_number(m['predictive_entropy']),
        mc_mutual_information=clean_number(m['mutual_information']),
        mc_variance_predicted_class=clean_number(m['variance_predicted_class']),
        faithfulness_del_high_20=getf('deletion','high'),
        faithfulness_del_random_20=getf('deletion','random'),
        faithfulness_del_low_20=getf('deletion','low'),
        faithfulness_del_high_minus_random_20=diff('deletion'),
        faithfulness_ins_high_20=getf('insertion','high'),
        faithfulness_ins_random_20=getf('insertion','random'),
        faithfulness_ins_high_minus_random_20=diff('insertion'),
        mean_stability_cosine=safe_mean([k['cosine_similarity'] for k in s]),
        mean_stability_pearson=safe_mean([k['pearson_correlation'] for k in s]),
        mean_stability_top20_iou=safe_mean([k['top20_iou'] for k in s]),
        stability_prediction_flip_count=sum(k['prediction_flip'] for k in s),
        explanation_shape_uncertainty=x['mean_shape_attribution_std'],
        explanation_raw_attribution_std=x['mean_raw_attribution_std'],
        explanation_mc_class_flip_rate=x['mc_class_flip_rate'],
        explanation_mean_cosine_to_first=x['cosine_to_first_mc_cam'])


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    temp.replace(path)


def correlation_data(rows):
    analyses=[]
    for x,y in RELATIONS:
        pairs=[(float(r[x]),float(r[y])) for r in rows if r.get(x) is not None and r.get(y) is not None
               and np.isfinite(r[x]) and np.isfinite(r[y])]
        if len(pairs)<5 or np.std(np.asarray(pairs),axis=0).min()<1e-12:
            rho=p=None
        else:
            result=spearmanr(np.asarray(pairs)[:,0],np.asarray(pairs)[:,1])
            rho=clean_number(result.statistic)
            p=clean_number(result.pvalue)
        analyses.append(dict(x=x,y=y,n=len(pairs),spearman_rho=rho,unadjusted_p=p))
    # Benjamini-Hochberg FDR correction, for transparency (still exploratory).
    valid=sorted([(i,r['unadjusted_p']) for i,r in enumerate(analyses) if r['unadjusted_p'] is not None],
                 key=lambda kv:kv[1])
    corrected=[None]*len(analyses)
    smallest=1.0
    for rank in range(len(valid),0,-1):
        i,p=valid[rank-1]
        smallest=min(smallest,p*len(valid)/rank)
        corrected[i]=min(1.0,smallest)
    for i,r in enumerate(analyses): r['bh_fdr_p']=corrected[i]
    return analyses


def plot_scatter(rows,x,y,out,heading,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(8,6))
    for correct,label,col in [(1,'Correct','#218a76'),(0,'Incorrect','#c15b45')]:
        sub=[r for r in rows if r['deterministic_correct']==correct and r.get(x) is not None
             and r.get(y) is not None and np.isfinite(r[x]) and np.isfinite(r[y])]
        if sub:
            ax.scatter([r[x] for r in sub],[r[y] for r in sub],s=24,alpha=.6,
                       color=col,label=f'{label} (n={len(sub)})',edgecolors='none')
    ax.set(xlabel=xlabel,ylabel=ylabel,title=heading)
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out,dpi=180);plt.close(fig)


def plot_by_group(rows,field,out,title,ylabel,group='true_class'):
    labels=C.CLASSES if group=='true_class' else ['Correct','Incorrect']
    data=[];clean=[]
    for label in labels:
        arr=[r[field] for r in rows if (r[group]==label if group=='true_class'
             else r['deterministic_correct']==int(label=='Correct')) and r.get(field) is not None
             and np.isfinite(r[field])]
        if arr:data.append(arr);clean.append(label)
    if not data:return
    fig,ax=plt.subplots(figsize=(8,5))
    ax.boxplot(data,tick_labels=[f'{x}\nn={len(d)}' for x,d in zip(clean,data)],showfliers=True)
    ax.set(title=title,ylabel=ylabel)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)


def make_figure_set(rows,out,layer):
    charts=out/'figures'/layer;charts.mkdir(parents=True,exist_ok=True)
    plot_scatter(rows,'mc_predictive_entropy','explanation_shape_uncertainty',
        charts/'01_prediction_vs_explanation_uncertainty.png',
        'V9 full test: prediction vs explanation uncertainty',
        'MC predictive entropy','Mean normalized Grad-CAM shape standard deviation')
    plot_scatter(rows,'faithfulness_del_high_minus_random_20','explanation_shape_uncertainty',
        charts/'02_faithfulness_vs_explanation_uncertainty.png',
        'V9 full test: faithfulness vs explanation uncertainty',
        'High minus random deletion margin drop (20%)','Mean normalized Grad-CAM shape standard deviation')
    plot_by_group(rows,'explanation_shape_uncertainty',charts/'03_uncertainty_by_class.png',
        'Explanation shape uncertainty by true class','Mean spatial attribution std')
    plot_by_group(rows,'faithfulness_del_high_minus_random_20',charts/'04_faithfulness_by_class.png',
        'Deletion faithfulness advantage by true class','High minus random margin drop (20%)')
    plot_by_group(rows,'mean_stability_top20_iou',charts/'05_stability_by_class.png',
        'Explanation stability by true class','Mean top-20% Grad-CAM IoU')
    plot_by_group(rows,'explanation_shape_uncertainty',charts/'06_uncertainty_correct_vs_wrong.png',
        'Explanation uncertainty: correct vs incorrect','Mean spatial attribution std',group='correct')
    plot_by_group(rows,'faithfulness_del_high_minus_random_20',charts/'07_faithfulness_correct_vs_wrong.png',
        'Faithfulness: correct vs incorrect','High minus random margin drop (20%)',group='correct')
    plot_by_group(rows,'mc_predictive_entropy',charts/'08_prediction_uncertainty_correct_vs_wrong.png',
        'Prediction uncertainty: correct vs incorrect','MC predictive entropy',group='correct')
    plot_by_group(rows,'mean_stability_cosine',charts/'09_stability_correct_vs_wrong.png',
        'Stability: correct vs incorrect','Mean Grad-CAM cosine similarity',group='correct')


def summarize(rows,fp,layer,metadata,complete):
    correct=[r for r in rows if r['deterministic_correct']]
    wrong=[r for r in rows if not r['deterministic_correct']]
    def stats(records):
        return {k:dict(n=sum(r.get(k) is not None for r in records),mean=safe_mean([r.get(k) for r in records]),
                       median=float(np.median([r[k] for r in records if r.get(k) is not None]))
                       if any(r.get(k) is not None for r in records) else None)
                for k in NUM_FIELDS}
    correlations=correlation_data(rows)
    roc={}
    for field,orientation in [
        ('mc_predictive_entropy',1),('calibrated_confidence',-1),
        ('explanation_shape_uncertainty',1),('mean_stability_cosine',-1),
        ('faithfulness_del_high_minus_random_20',-1)]:
        pairs=[r for r in rows if r.get(field) is not None]
        truth=[int(not r['deterministic_correct']) for r in pairs]
        roc[field] = (float(roc_auc_score(truth,[orientation*r[field] for r in pairs]))
                      if len(set(truth))==2 else None)
    return native(dict(analysis='Post-hoc exploratory full TEST reliability profiling',
        completed_utc=now(),complete=complete,matched_images=len(rows),
        test_split_count=1374,layer=layer,split_fingerprint=fp,
        correct=len(correct),incorrect=len(wrong),
        overall=stats(rows),correct_group=stats(correct),incorrect_group=stats(wrong),
        by_true_class={cls:dict(n=sum(r['true_class']==cls for r in rows),
            stats=stats([r for r in rows if r['true_class']==cls])) for cls in C.CLASSES},
        spearman_correlations=correlations,
        exploratory_error_discrimination_auc=roc,
        source_checkpoint_sha256=metadata['checkpoint_sha256'],
        limitations=[
            'Post-hoc: test set was previously inspected; all correlations and error AUCs are exploratory.',
            'Patient IDs unavailable; no proof of patient-independent testing.',
            'No validated composite reliability index or clinical safety claim.',
            'Both Grad-CAM layers are reported separately, not chosen using test performance.',
            'Faithfulness patch blur is off-distribution; high-vs-random is not proof of tumour localization.',
            'Final-layer Grad-CAM can exhibit upper-right corner artefact.',
            'MC Grad-CAM shape std uses per-pass max normalization, not calibrated pixel uncertainty.',
            'Only predetermined intensity/noise/translation stability perturbations were tested.',
        ]))


def export(con, out, manifest, original, fingerprint, layers, meta, render=True):
    testids={r['file'] for r in manifest}
    total=0
    for layer in layers:
        items=[];faith=[];stability=[]
        for file,layer_in,contents in con.execute('SELECT file,layer,payload FROM completed WHERE layer=? ORDER BY file',(layer,)):
            if file not in testids or layer_in!=layer:
                raise RuntimeError('Unexpected cached result file/layer')
            result=json.loads(contents)
            items.append(combine_one(result,original))
            for v in result['faithfulness_trials']:
                faith.append(dict(file=file,true_class=result['true_class'],layer=layer,**v))
            for v in result['stability_trials']:
                stability.append(dict(file=file,true_class=result['true_class'],layer=layer,**v))
        total+=len(items)
        # Always save partial progress CSVs. Do NOT label pilot/partial results as complete.
        if not items:continue
        layer_out=out/layer
        write_csv(layer_out/'per_image_reliability_profile.csv',items,list(items[0]))
        write_csv(layer_out/'faithfulness_trials.csv',faith,list(faith[0]))
        write_csv(layer_out/'stability_trials.csv',stability,list(stability[0]))
        complete=len(items)==len(manifest)
        summary=summarize(items,fingerprint,layer,meta,complete)
        (layer_out/'integrated_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
        if render:
            make_figure_set(items,out,layer)
        print(f'  Exported {layer}: {len(items)}/{len(manifest)} images; complete={complete}',flush=True)
    return total


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layers',nargs='+',choices=['final','earlier'],default=['final','earlier'],
                    help='Run final and earlier CAM layers separately (default: both)')
    ap.add_argument('--out-name',default=STAGE_NAME,
                    help='NEW results subfolder name (only within original V9 results root)')
    ap.add_argument('--max-images',type=int,default=None,
                    help='Optional test run on first N MRIs; does not create a full-cohort claim')
    ap.add_argument('--examples-per-class',type=int,default=2,
                    help='Save heatmap panels for first N examples of each class and all errors')
    args=ap.parse_args(argv)
    layers=list(dict.fromkeys(args.layers))
    if args.max_images is not None and args.max_images<1:ap.error('--max-images must be >=1')
    if args.examples_per_class<0:ap.error('--examples-per-class must be >=0')
    if args.out_name in ('.','..') or not args.out_name.isidentifier():
        ap.error('--out-name must be a new simple folder name (letters, digits, underscores)')
    C.heading('STEP 22 / FULL HELD-OUT TEST RELIABILITY — RESUMABLE; NO RETRAINING')
    rows,fp,model,dev=environment('test')
    test_rows=sorted((r for r in rows if r['split']=='test'),
                     key=lambda r:(C.CLASSES.index(r['class_name']),r['file']))
    original=load_original(test_rows,fp)
    checkpoint_sha256=sha_file(C.CHECKPOINT)
    out=C.OUTPUT/args.out_name
    metadata=dict(schema=2,split='test',n_test=len(test_rows),fingerprint=fp,
                  checkpoint_sha256=checkpoint_sha256,layers=layers,
                  mc_passes=C.MC_PASSES,faithfulness_fractions=list(FRACTIONS),
                  faithfulness_grid=16,blur_sigma=10,
                  perturbations=['noise_original_code_std_0.02_normalized','brightness_plus_0.02',
                                 'shift_2px_right'],original_output_root=str(C.OUTPUT.resolve()))
    con=open_db(out,metadata)
    tx=transform(False)
    model.eval()
    images_to_save=set()
    for cls in C.CLASSES:
        pool=[r for r in test_rows if r['class_name']==cls]
        images_to_save.update(r['file'] for r in pool[:args.examples_per_class])
    images_to_save.update(r['file'] for r in test_rows if not int(original['test'][r['file']]['correct']))
    limit=min(len(test_rows),args.max_images or len(test_rows))
    print(f'  Device={dev} | total test images={len(test_rows)} | this invocation={limit} | layers={layers}',flush=True)
    print(f'  Output: {out} (existing stages unchanged)',flush=True)
    print(f'  Checkpoint SHA256: {checkpoint_sha256}',flush=True)
    try:
        for i,row in enumerate(test_rows[:limit],1):
            file=row['file']
            for layer in layers:
                exists=con.execute('SELECT 1 FROM completed WHERE file=? AND layer=?',(file,layer)).fetchone()
                if exists:continue
                cam=ResearchCAM(model,get_layer(model,layer))
                try:
                    result=record_one(row,layer,i,model,dev,cam,tx,original,out,images_to_save)
                finally:
                    cam.close()
                con.execute('INSERT INTO completed(file,layer,payload) VALUES(?,?,?)',
                    (file,layer,json.dumps(native(result),allow_nan=False)))
                con.commit()   # Safe resume point after each completed MRI/layer.
                print(f'  {i:4}/{len(test_rows)} [{layer:7}] {row["class_name"]:11} '
                      f'{"CORRECT" if result["deterministic_correct"] else "WRONG"} '
                      f'CAM-shape-std={result["explanation"]["mean_shape_attribution_std"]:.6f}',flush=True)
            if i%25==0:
                print(f'  Checkpoint: {i} images considered; run may be resumed with identical command.',flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted safely. MRI/layer records completed so far are committed; rerun the same command.',flush=True)
    except Exception:
        print('\nAn error occurred. Completed records are preserved; no existing V9 result was overwritten.',flush=True)
        raise
    finally:
        print('\nExporting available results and figures...',flush=True)
        export(con,out,test_rows,original,fp,layers,metadata)
        con.close()
    print('\nSTEP 22: If every layer has 1,374 rows, the full-cohort extension is complete.',flush=True)
    print(f'Results directory: {out}',flush=True)

if __name__=='__main__': main()
