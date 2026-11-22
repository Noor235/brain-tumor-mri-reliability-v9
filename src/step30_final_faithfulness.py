"""V9 Step 30 FINAL: paired validation-only CAM comparison and ROAD-style audit.

Methods: Grad-CAM, HiResCAM, Grad-CAM++ on the SAME features[-4] layer.
Blurring test: existing V9 Step13 matched 20% patch deletion/insertion.
ROAD-style test: 20% pixel-level most/least/random removal with sparse
neighbor-weighted noisy linear imputation (Rong et al., ICML 2022).

CAVEAT: ROAD-style high-vs-random *logit-margin* is a diagnostic variation,
NOT the official ROADCombined metric or a tumour localization score.
The model, dataset and prior results are never changed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from PIL import Image, ImageOps, ImageDraw
from . import config as C
from .common import source_image
from .research_v9 import environment, sample_split, normalize_cam, target_margin
from .transforms_v9 import transform
from .step13_faithfulness import patch_masks, perturb, score_many

class CAMAlternatives:
    """Three CAM maps from a single forward and backward pass, same layer/target."""
    def __init__(self, model, layer):
        self.model=model;self.activation=None;self.gradient=None
        self.h1=layer.register_forward_hook(self._fw)
        self.h2=layer.register_full_backward_hook(self._bw)
    def _fw(self,m,args,out):self.activation=out.detach()
    def _bw(self,m,gi,go):self.gradient=go[0].detach()
    def close(self):self.h1.remove();self.h2.remove()
    def compute_all(self,x,class_id):
        self.activation=None;self.gradient=None
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits=self.model(x);logits[0,class_id].backward()
        if self.activation is None or self.gradient is None:
            raise RuntimeError('Grad-CAM forward/backward hook failed')
        a=self.activation[0];g=self.gradient[0]
        with torch.no_grad():
            classic=torch.relu((a*g.mean((1,2),keepdim=True)).sum(0))
            hires=torch.relu((a*g).sum(0))
            g2=g*g;g3=g2*g
            denom=2*g2 + a.sum((1,2),keepdim=True)*g3
            alpha=torch.where(torch.abs(denom)>1e-12,g2/torch.where(torch.abs(denom)>1e-12,denom,torch.ones_like(denom)),torch.zeros_like(denom))
            weight=(alpha*torch.relu(g)).sum((1,2),keepdim=True)
            plusplus=torch.relu((weight*a).sum(0))
        maps={}
        for name,v in [('gradcam',classic),('hirescam',hires),('gradcampp',plusplus)]:
            arr=v.detach().float().cpu().numpy()
            maps[name]=normalize_cam(np.maximum(cv2.resize(arr,(C.IMAGE_SIZE,C.IMAGE_SIZE),interpolation=cv2.INTER_LINEAR),0))
        return maps,logits.detach().cpu().numpy()[0]

def overlay_for(pil,heat):
    base=np.asarray(pil.resize((C.IMAGE_SIZE,C.IMAGE_SIZE),Image.Resampling.BICUBIC),dtype=np.uint8)
    base=np.stack([base]*3,axis=2)
    cmap=cv2.cvtColor(cv2.applyColorMap(np.uint8(255*np.clip(heat,0,1)),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(base*.58+cmap*.42,0,255))

METHODS = ('gradcam', 'hirescam', 'gradcampp')
# Weighted 8-connected neighborhood; nearest axial pixels count twice diagonals.
NEIGHBORS = ((-1,-1,1/12),(-1,0,1/6),(-1,1,1/12),
             (0,-1,1/6),(0,1,1/6),(1,-1,1/12),(1,0,1/6),(1,1,1/12))


def pixel_masks(heat: np.ndarray, fraction: float, seed: int):
    """Identical mask area for most, least and random pixels."""
    flat = np.asarray(heat, dtype=np.float32).ravel()
    n = max(1, int(math.ceil(flat.size*fraction)))
    ranked = np.argsort(flat, kind='stable')
    rng = np.random.default_rng(seed)
    return {kind: np.isin(np.arange(flat.size), chosen).reshape(heat.shape)
            for kind, chosen in [('high', ranked[-n:]), ('low', ranked[:n]),
                                 ('random', rng.choice(flat.size, size=n, replace=False))]}


def road_style_impute(image: np.ndarray, removed: np.ndarray, seed: int,
                      noise: float=0.01) -> np.ndarray:
    """Weighted linear imputation on 8-connected pixel neighbors + fixed-seed noise.

    Reimplements ROAD paper's noisy neighbor interpolation idea as a sparse
    Dirichlet interpolation. Output is a *ROAD-style* perturbation: it is NOT
    represented as the official pytorch-grad-cam ROADCombined estimator.
    Input: normalized single channel (H,W) array and bool removed mask.
    """
    if image.ndim!=2 or removed.shape!=image.shape:
        raise ValueError('Expected single-channel image and matching mask')
    H, W = image.shape
    missing = np.flatnonzero(removed.ravel())
    if len(missing)==0: return image.copy()
    if len(missing)==H*W: raise ValueError('Cannot impute an entirely removed image')
    lookup = np.full(H*W, -1, dtype=np.int32)
    lookup[missing] = np.arange(len(missing), dtype=np.int32)
    yy, xx = np.divmod(missing, W)
    row_ids = [np.arange(len(missing), dtype=np.int32)]
    col_ids = [np.arange(len(missing), dtype=np.int32)]
    diagonal = np.zeros(len(missing), dtype=np.float64)
    weights = [np.zeros(len(missing), dtype=np.float64)]
    rhs = np.zeros(len(missing), dtype=np.float64)
    pix = image.reshape(-1).astype(np.float64)
    for dy, dx, weight in NEIGHBORS:
        y2, x2 = yy+dy, xx+dx
        valid = (y2>=0)&(y2<H)&(x2>=0)&(x2<W)
        local_row = np.nonzero(valid)[0]
        if local_row.size==0: continue
        neighbors = y2[valid]*W + x2[valid]
        diagonal[local_row] += weight
        known = lookup[neighbors] < 0
        rhs[local_row[known]] += weight * pix[neighbors[known]]
        unknown_rows = local_row[~known]
        unknown_cols = lookup[neighbors[~known]]
        if unknown_rows.size:
            row_ids.append(unknown_rows.astype(np.int32))
            col_ids.append(unknown_cols.astype(np.int32))
            weights.append(np.full(len(unknown_rows), -weight, dtype=np.float64))
    weights[0]=diagonal
    A=coo_matrix((np.concatenate(weights),(np.concatenate(row_ids),np.concatenate(col_ids))),
                 shape=(len(missing),len(missing))).tocsc()
    values=spsolve(A, rhs)
    if not np.all(np.isfinite(values)):
        raise RuntimeError('ROAD-style sparse linear interpolation returned non-finite pixels')
    restored=pix.copy()
    restored[missing]=values+np.random.default_rng(seed).normal(0,noise,len(missing))
    return restored.reshape(H,W).astype(np.float32)


def report_progress(out, records, completed, meta):
    if records:
        tmp=out/'trials.tmp.csv'
        with tmp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        tmp.replace(out/'trials.csv')
    info={**meta, 'completed_images':len(completed),'completed_files':completed,
          'records':len(records),'complete':len(completed)==meta['planned_images']}
    p=out/'progress.tmp.json'
    p.write_text(json.dumps(info,indent=2),encoding='utf-8')
    p.replace(out/'progress.json')


def from_csv(out):
    csv_path=out/'trials.csv'
    if not csv_path.is_file(): return []
    with csv_path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def mean_ci(values, seed=311, draws=1000):
    a=np.asarray(values,dtype=float)
    if len(a)==0: return [None,None,None]
    rng=np.random.default_rng(seed)
    means=np.empty(draws)
    for j in range(draws):
        means[j]=rng.choice(a,size=len(a),replace=True).mean()
    return [float(a.mean()), float(np.quantile(means,.025)), float(np.quantile(means,.975))]


def summarize(records, meta, out):
    by_case={}
    for r in records:
        k=(r['file'],r['method'],r['protocol'],r['intervention'])
        by_case.setdefault(k,{})[r['region']]=float(r['score_change'])
    paired=[]
    for (file,method,protocol,intervention), d in by_case.items():
        if 'high' not in d or 'random' not in d:
            continue
        match=next(x for x in records if x['file']==file)
        paired.append({'file':file,'true_class':match['true_class'],
          'predicted_class':match['predicted_class'],'correct':match['correct'],
          'method':method,'protocol':protocol,'intervention':intervention,
          'high_minus_random':d['high']-d['random'],
          'high_minus_low':d['high']-d['low'] if 'low' in d else '',
          'high':d['high'],'random':d['random']})
    with (out/'per_image_paired.csv').open('w',newline='',encoding='utf-8') as f:
        if paired:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    metrics=[]
    for protocol,intervention in [('blur','deletion'),('blur','insertion'),('road_style','deletion')]:
        subset=[r for r in paired if r['protocol']==protocol and r['intervention']==intervention]
        for name in METHODS:
            chosen=[r for r in subset if r['method']==name]
            vals=[float(r['high_minus_random']) for r in chosen]
            metrics.append({'protocol':protocol,'intervention':intervention,'method':name,
               'n':len(vals),'mean_and_bootstrap_95ci':mean_ci(vals),
               'percent_positive':float(100*np.mean(np.asarray(vals)>0)) if vals else None,
               'class_means':{cls:float(np.mean([float(r['high_minus_random']) for r in chosen if r['true_class']==cls]))
                              if any(r['true_class']==cls for r in chosen) else None for cls in C.CLASSES},
               'correct_mean':float(np.mean([float(r['high_minus_random']) for r in chosen if str(r['correct']).lower()=='true']))
                              if any(str(r['correct']).lower()=='true' for r in chosen) else None,
               'incorrect_mean':float(np.mean([float(r['high_minus_random']) for r in chosen if str(r['correct']).lower()=='false']))
                                if any(str(r['correct']).lower()=='false' for r in chosen) else None})
        reference={r['file']:r for r in subset if r['method']=='gradcam'}
        for name in ('hirescam','gradcampp'):
            other={r['file']:r for r in subset if r['method']==name}
            same=sorted(set(reference)&set(other))
            difference=[float(other[f]['high_minus_random'])-float(reference[f]['high_minus_random']) for f in same]
            metrics.append({'protocol':protocol,'intervention':intervention,
                 'comparison':f'{name}_minus_gradcam','n':len(same),
                 'paired_difference_mean_and_bootstrap_95ci':mean_ci(difference)})
    info={'methodology':meta,'metrics':metrics,
          'notes':['These are V9 validation images; test set was already explored in earlier project stages.',
                   'Evaluation uses the original fixed predicted class, not the known true class.',
                   'ROAD-style sparse imputation is based on noisy linear interpolation; the target-margin metric is not official ROADCombined.',
                   'A high faithfulness score cannot establish that a heatmap localizes the actual tumour.',
                   'Select a replacement only if evidence agrees across protocols and classes; retain the baseline for ambiguous results.']}
    (out/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        panels=[('blur','deletion'),('blur','insertion'),('road_style','deletion')]
        fig,ax=plt.subplots(figsize=(10,5))
        names=['Grad-CAM','HiResCAM','Grad-CAM++']
        data=[]
        for proto,inter in panels:
            data.append([next(x for x in metrics if x.get('protocol')==proto and x.get('intervention')==inter and x.get('method')==m)['mean_and_bootstrap_95ci'][0] for m in METHODS])
        X=np.arange(3);w=.25
        for i,n in enumerate(names):ax.bar(X+(i-1)*w,[data[j][i] for j in range(3)],w,label=n)
        ax.axhline(0,color='black',linewidth=.7); ax.set_xticks(X,['Blur deletion','Blur insertion','ROAD-style removal']);ax.set_ylabel('Mean target-margin advantage: high minus random');ax.set_title(f'V9 Step 30: paired validation comparison (n={len(set(r["file"] for r in paired))})');ax.legend();fig.tight_layout()
        fig.savefig(out/'method_comparison.png',dpi=170);plt.close(fig)
    except Exception as exc:
        print(f'Plot creation skipped: {exc}',flush=True)
    print('\nPaired validation summary:')
    for m in metrics:
        if 'method' in m:
            print(f"  {m['protocol']:<10} {m['intervention']:<10} {m['method']:<10} n={m['n']} advantage={m['mean_and_bootstrap_95ci'][0]:+.4f}")
        else:
            print(f"  DELTA {m['protocol']:<10} {m['comparison']:<22} {m['paired_difference_mean_and_bootstrap_95ci'][0]:+.4f}")
    return info


def run():
    p=argparse.ArgumentParser(description='V9 FINAL validation CAM comparison with ROAD-style imputation')
    p.add_argument('--n-per-class',type=int,default=50,help='50 each -> 200 MRI comparison')
    p.add_argument('--fraction',type=float,default=.20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out-name',default='30_final_faithfulness_validation')
    p.add_argument('--resume',action='store_true',help='Resume after interruption; verifies experiment metadata')
    p.add_argument('--no-road',action='store_true',help='Temporary debugging only; not for final result')
    p.add_argument('--gallery-max',type=int,default=8)
    args=p.parse_args()
    if args.n_per_class<1 or not 0<args.fraction<1:
        p.error('Use --n-per-class >=1, 0 < --fraction < 1')
    if not args.out_name.startswith('30_') or args.out_name != Path(args.out_name).name:
        p.error('Output name must be simple and start with 30_')
    out=C.OUTPUT / args.out_name
    if out.exists() and not args.resume:
        raise FileExistsError(f'{out} already exists. Choose --resume or a new --out-name.')
    rows, fp, model, dev = environment('val')
    selected=sample_split(rows,'val',args.n_per_class,args.seed)
    meta={'split':'val','split_fingerprint':fp,'checkpoint':str(C.CHECKPOINT),
          'layer':'features[-4]','seed':args.seed,'n_per_class':args.n_per_class,
          'planned_images':len(selected),'fraction':args.fraction,
          'methods':list(METHODS),'include_road_style':not args.no_road,
          'target':'original predicted class','protocol':'20%-area blur patch deletion/insertion and noisy linear imputation ROAD-style',
          'selected_files':[r['file'] for r in selected]}
    if args.resume and (out/'progress.json').exists():
        saved=json.loads((out/'progress.json').read_text(encoding='utf-8'))
        for key in ('split_fingerprint','seed','n_per_class','planned_images','fraction','include_road_style','selected_files'):
            if saved.get(key)!=meta.get(key):
                raise RuntimeError('Cannot resume: experimental setup changed: '+key)
        completed=set(saved['completed_files'])
        records=from_csv(out)
    else:
        completed=set();records=[]
    out.mkdir(parents=True,exist_ok=True)
    cams=CAMAlternatives(model,model.features[-4]);tx=transform(False)
    print(f'STEP 30 REVISED FINAL | images={len(selected)} | device={dev} | road={not args.no_road}',flush=True)
    print(f'Progress is saved per MRI; restart with --resume if interrupted. Folder: {out}',flush=True)
    examples=[]
    try:
        for i,row in enumerate(selected,1):
            if row['file'] in completed:continue
            tic=time.monotonic()
            with Image.open(source_image(row['file'])) as original:
                pil=ImageOps.exif_transpose(original).convert('L')
            x=tx(pil).unsqueeze(0).to(dev)
            with torch.no_grad():raw_logits=model(x).detach().cpu().numpy()
            target=int(raw_logits.argmax(1)[0]);predicted=C.CLASSES[target]
            original_margin=float(target_margin(raw_logits,target)[0])
            maps,_=cams.compute_all(x,target)
            source=x.detach().cpu().numpy()[0,0].astype(np.float32)
            blurred=cv2.GaussianBlur(source,(0,0),sigmaX=10).astype(np.float32)
            blur_base=float(score_many(model,dev,[torch.from_numpy(blurred)[None,...]],target)[0])
            local=[]
            for index,method in enumerate(METHODS):
                heat=maps[method]
                masks=patch_masks(heat,args.fraction,args.seed+1000+i)
                variants=[];labels=[]
                for intervention in ('deletion','insertion'):
                    for region in ('high','low','random'):
                        variants.append(perturb(x,blurred,masks[region],intervention))
                        labels.append((intervention,region))
                margins=score_many(model,dev,variants,target)
                for (intervention,region),value in zip(labels,margins):
                    val=(original_margin-value) if intervention=='deletion' else value-blur_base
                    local.append({'file':row['file'],'true_class':row['class_name'],
                     'predicted_class':predicted,'correct':predicted==row['class_name'],
                     'method':method,'protocol':'blur','intervention':intervention,
                     'region':region,'fraction':args.fraction,'score_change':float(val),
                     'cam_zero':bool(heat.max()<=1e-12)})
                if not args.no_road:
                    rmasks=pixel_masks(heat,args.fraction,args.seed+2000+i)
                    perturbed=[]
                    for ridx,region in enumerate(('high','low','random')):
                        modified=road_style_impute(source,rmasks[region],
                                             seed=args.seed + 100000*i + 10*ridx,
                                             noise=.01)
                        perturbed.append(torch.from_numpy(modified)[None,...])
                    road_margins=score_many(model,dev,perturbed,target)
                    for region,value in zip(('high','low','random'),road_margins):
                        local.append({'file':row['file'],'true_class':row['class_name'],
                         'predicted_class':predicted,'correct':predicted==row['class_name'],
                         'method':method,'protocol':'road_style','intervention':'deletion',
                         'region':region,'fraction':args.fraction,'score_change':float(original_margin-value),
                         'cam_zero':bool(heat.max()<=1e-12)})
            records.extend(local)
            completed.add(row['file'])
            report_progress(out,records,sorted(completed),meta)
            if len(examples)<args.gallery_max:
                examples.append((pil,maps,row,predicted))
            print(f'  {i}/{len(selected)} {Path(row["file"]).name} -> {predicted} | {time.monotonic()-tic:.1f}s | saved',flush=True)
    finally:
        cams.close()
    if examples:
        w=4*224;h=250*len(examples)
        canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
        for i,(pil,maps,row,pred) in enumerate(examples):
            y=i*250;draw.text((7,y+3),f"{row['class_name']} -> {pred}  | {Path(row['file']).name}",fill='black')
            panels=[np.asarray(pil.resize((224,224)).convert('RGB'))]+[overlay_for(pil,maps[m]) for m in METHODS]
            for j,arr in enumerate(panels):canvas.paste(Image.fromarray(arr),(j*224,y+23))
        canvas.save(out/'actual_mri_cam_gallery.png')
    info=summarize(records,meta,out)
    (out/'STEP30_FINDINGS.md').write_text('''# V9 Step 30 — revised faithfulness validation\n\n'''+
       f"**Validation MRIs:** {len(completed)}/{len(selected)}; **Layer:** `features[-4]`; **seed:** {args.seed}.\n\n"+
       '**Endpoints:** paired high-minus-random target-logit-margin changes under blur deletion, blur insertion, and ROAD-style noisy-linear-imputed removal.\n\n'+
       '\n'.join(f"- {m['protocol']} / {m['intervention']} / {m.get('method',m.get('comparison'))}: "
              f"{m.get('mean_and_bootstrap_95ci',m.get('paired_difference_mean_and_bootstrap_95ci'))}" for m in info['metrics'])+
       '\n\nThese metrics assess model attribution under synthetic perturbations, not tumour localization. Validation is not independent external evidence.\n',encoding='utf-8')
    print(f'\nFINISHED. Figures and report: {out}',flush=True)
    print('Original checkpoint, old Grad-CAM, original Steps 22/30 results unchanged.',flush=True)

if __name__=='__main__':run()
