"""V9 Step 32: paired MC explanation uncertainty for Grad-CAM / HiResCAM.

Both maps are created using the SAME 20 classifier-dropout draws on each MRI.
Uses the validated Step 30/31 selection: 50 validation MRIs per class, seed=42.
No weights, train/test datasets, or prior reports are modified.

V9 has one dropout after spatial/channel attention and global pooling. It is
therefore mathematically equivalent to evaluate deterministic features once,
apply independent classifier dropout masks, and compute all corresponding
feature gradients using vector-Jacobian products. Doing this is much faster
than repeating 20 identical EfficientNet feature forward passes.

Interpretation: per-image map shape variability (independently max normalized),
NOT tumour localization and NOT calibrated per-pixel Bayesian uncertainty.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from . import config as C
from .common import source_image
from .research_v9 import environment, sample_split, normalize_cam, cosine, top_iou
from .transforms_v9 import transform

METHODS=('gradcam','hirescam')
FIELDS=('file','true_class','predicted_class','correct','method','passes',
        'mean_shape_std','mean_shape_std_top20','cosine_to_mean',
        'top20_iou_to_mean','empty_maps','predicted_class_flip_rate',
        'mc_mean_confidence','predictive_entropy','mutual_information')


def _softmax_np(z):
    z=np.asarray(z,dtype=np.float64)
    e=np.exp(z-z.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def _entropy(p):
    p=np.asarray(p,dtype=np.float64)
    return -np.sum(p*np.log(np.clip(p,1e-14,1.0)),axis=-1)


def _cams_from_maps(a,grads):
    """a:[C,H,W] features[-4] activations; grads:[P,C,H,W] per mask."""
    with torch.no_grad():
        classic=torch.relu((a[None,:,:,:]*grads.mean((-1,-2),keepdim=True)).sum(1))
        hires=torch.relu((a[None,:,:,:]*grads).sum(1))
    results={}
    for method,tensor in (('gradcam',classic),('hirescam',hires)):
        arr=tensor.detach().float().cpu().numpy()
        resized=np.stack([cv2.resize(z,(C.IMAGE_SIZE,C.IMAGE_SIZE),interpolation=cv2.INTER_LINEAR)
                          for z in arr])
        results[method]=np.stack([normalize_cam(np.maximum(z,0)) for z in resized])
    return results


def _gradients(model,pooled,activation,derivatives,mode='auto'):
    """derivatives [P,D] gradients of selected-class logits w.r.t pooled features."""
    if mode=='auto' and activation.device.type=='mps':
        mode='loop'  # Apple MPS vectorized VJPs can be unsupported.
    if mode!='loop':
        try:
            return torch.autograd.grad(pooled,activation,
                    grad_outputs=derivatives[:,None,:],is_grads_batched=True)[0][:,0]
        except (RuntimeError,NotImplementedError) as e:
            if mode=='vectorized':raise
            print('  Vectorized gradients unavailable; switching to sequential VJPs:',
                  str(e).splitlines()[0][:150],flush=True)
    output=[]
    for i in range(derivatives.shape[0]):
        grad=torch.autograd.grad(pooled,activation,
                    grad_outputs=derivatives[i:i+1],
                    retain_graph=i<derivatives.shape[0]-1)[0]
        output.append(grad[0].detach())
    return torch.stack(output)


def compute_one(model,x,filename,seed,passes,grad_mode):
    """Exact model head; 20 statistically independent MC-dropout masks."""
    # Only classifier dropout is stochastic in original V9 inference.
    with torch.no_grad():
        f=model.features[:-3](x)
    a=f.detach().requires_grad_(True)
    z=model.features[-3:](a)
    z=model.channel_attention(z)
    z=model.spatial_attention(z)
    pooled=model.pool(z).flatten(1)
    with torch.no_grad():
        deterministic_logits=model.classifier(pooled)[0]
        target=int(deterministic_logits.argmax())
        # A fixed per-file RNG prevents needing to re-run earlier files for resume.
        digest=hashlib.sha256(filename.encode('utf-8')).digest()
        image_seed=(seed+int.from_bytes(digest[:8],'little'))%(2**63-1)
        rng=torch.Generator(device='cpu').manual_seed(image_seed)
        mask=(torch.rand((passes,pooled.shape[-1]),generator=rng)>=model.dropout.p)
        mask=mask.to(device=x.device,dtype=pooled.dtype)/(1.0-model.dropout.p)
        logits=torch.nn.functional.linear(pooled.detach().expand(passes,-1)*mask,
                         model.classifier.weight.detach(),model.classifier.bias.detach())
        probs=_softmax_np(logits.detach().cpu().numpy())
        mean_p=probs.mean(0)
        ent=float(_entropy(mean_p))
        mi=max(0.,float(ent-_entropy(probs).mean()))
        # For each stochastic forward pass, d target-logit / d pooled.
        derivative=mask*model.classifier.weight.detach()[target][None,:]
    grad=_gradients(model,pooled,a,derivative,grad_mode)
    maps=_cams_from_maps(a[0].detach(),grad)
    meta={
      'target':target,'prediction':C.CLASSES[target],
      'mc_mean_confidence':float(mean_p.max()),
      'predictive_entropy':ent,'mutual_information':mi,
      'predicted_class_flip_rate':float(np.mean(probs.argmax(1)!=target)),
    }
    return maps,meta


def stats(stack):
    """Per-image uncertainty: normalized spatial dispersion of MC explanations."""
    arr=np.asarray(stack,dtype=np.float32)
    mu=arr.mean(axis=0)
    std=arr.std(axis=0,ddof=0)
    valid=[z for z in arr if z.max()>1e-12]
    if mu.max()>1e-12 and valid:
        cos=np.mean([cosine(z,mu) for z in valid])
        iou=np.mean([top_iou(z,mu,.20) for z in valid])
    else:
        cos=float('nan');iou=float('nan')
    # spatial standard deviation concentrated on most-highlighted mean-map pixels
    n=max(1,int(np.ceil(.20*mu.size)))
    top=np.argsort(mu.ravel(),kind='stable')[-n:]
    return {
      'mean_shape_std':float(std.mean()),
      'mean_shape_std_top20':float(std.ravel()[top].mean()),
      'cosine_to_mean':float(cos),
      'top20_iou_to_mean':float(iou),
      'empty_maps':int(sum(z.max()<=1e-12 for z in arr)),
    },mu,std


def overlay(pil,heat):
    gray=np.asarray(pil.resize((C.IMAGE_SIZE,C.IMAGE_SIZE),Image.Resampling.BICUBIC),dtype=np.uint8)
    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    colored=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255),cv2.COLORMAP_JET),
                         cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(.6*image+.4*colored,0,255))


def write_example(out,pil,maps,meta,stats_map,idx):
    """Display mean map and spatial uncertainty map for both CAM methods."""
    size=C.IMAGE_SIZE
    canvas=Image.new('RGB',(3*size,2*(size+25)+38),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((9,9),f'{meta["true_class"]} | predicted {meta["prediction"]} | {Path(meta["file"]).name}',fill='black')
    scaled=pil.resize((size,size),Image.Resampling.BICUBIC)
    original=np.stack([np.asarray(scaled)]*3,axis=-1)
    for y,method in enumerate(METHODS):
        mu,sd=maps[method]
        uncertainty=sd/(float(sd.max())+1e-12)
        heat=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(uncertainty,0,1)*255),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
        tiles=[original,overlay(pil,mu),heat]
        for x,arr in enumerate(tiles):
            canvas.paste(Image.fromarray(arr),(x*size,38+y*(size+25)))
            draw.text((x*size+4,38+y*(size+25)+size+4),
                      ['Original MRI',f'{method} mean',f'{method} MC std'][x],fill='black')
    example_dir=out/'examples';example_dir.mkdir(exist_ok=True)
    canvas.save(example_dir/f'{idx:02}_{Path(meta["file"]).stem}.png')


def atomic_csv(file,rows):
    temp=file.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    temp.replace(file)


def bootstrap(x,seed=3200,draws=3000):
    x=np.asarray(x,dtype=np.float64)
    x=x[np.isfinite(x)]
    if not len(x):return {'n':0,'mean':None,'ci95':[None,None]}
    rng=np.random.default_rng(seed)
    indexes=rng.integers(0,len(x),(draws,len(x)))
    v=x[indexes].mean(1)
    return {'n':len(x),'mean':float(x.mean()),'ci95':[float(np.quantile(v,.025)),float(np.quantile(v,.975))]}


def summarize(rows,meta,out):
    summary={'metadata':meta,'results':{},'paired_hirescam_minus_gradcam':{},
       'caveats':[
         'These figures measure *variability of attribution shape*, not diagnosis accuracy or anatomical localization.',
         'Lower spatial variation may indicate greater consistency but not greater faithfulness or clinical correctness.',
         'Independently max-normalized MC maps remove amplitude variation and make values scale-dependent.',
         'Both methods use the same dropout masks, predicted target class, images and features[-4] layer.',
         'Only classifier dropout is active, matching original V9 MC implementation; sampling does not cover other uncertainty sources.',
         'Validation subset only. Previous test-set Grad-CAM uncertainty numbers cannot be reused for HiResCAM.',
         'Few validation misclassifications limit conclusions about error discrimination.',
         'A different random seed or MC sample count may change numerical values slightly.',
       ]}
    for method in METHODS:
        group=[r for r in rows if r['method']==method]
        summary['results'][method]={}
        for key in ('mean_shape_std','mean_shape_std_top20','cosine_to_mean','top20_iou_to_mean'):
            summary['results'][method][key]=bootstrap([float(r[key]) for r in group])
        summary['results'][method]['empty_maps']=sum(int(r['empty_maps']) for r in group)
        summary['results'][method]['n']=len(group)
        summary['results'][method]['by_class']={}
        for cls in C.CLASSES:
            subset=[r for r in group if r['true_class']==cls]
            summary['results'][method]['by_class'][cls]={
              'n':len(subset),'mean_shape_std':bootstrap([float(r['mean_shape_std']) for r in subset])}
    pairs={}
    for r in rows:
        pairs.setdefault(r['file'],{})[r['method']]=r
    for key in ('mean_shape_std','mean_shape_std_top20','cosine_to_mean','top20_iou_to_mean'):
        diffs=[float(pair['hirescam'][key])-float(pair['gradcam'][key])
               for pair in pairs.values() if all(m in pair for m in METHODS)]
        summary['paired_hirescam_minus_gradcam'][key]=bootstrap(diffs)
    summary['correct_predictions']=sum(int(r['correct']) for r in rows if r['method']=='gradcam')
    summary['wrong_predictions']=sum(1-int(r['correct']) for r in rows if r['method']=='gradcam')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12.0,4.8))
        metrics=(('mean_shape_std','Mean per-pixel CAM shape std\n(lower = more consistent)'),
                 ('cosine_to_mean','Mean cosine to MC mean CAM\n(higher = more consistent)'))
        for ax,(key,title) in zip(axes,metrics):
            vals=[summary['results'][m][key]['mean'] for m in METHODS]
            b=ax.bar(['Grad-CAM','HiResCAM'], vals, color=['#5675A7','#D48B44'],width=.5)
            for bar,v in zip(b,vals):
                if v is not None:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(vals)*.02,
                                        f'{v:.5f}',ha='center',fontsize=10)
            ax.set_title(title);ax.set_ylim(0,max(vals)*1.20 if max(vals)>0 else 1)
            ax.grid(axis='y',alpha=.18)
        fig.suptitle(f'V9 validation | {meta["planned_images"]} MRIs | {meta["passes"]} shared MC-dropout passes | features[-4]')
        fig.tight_layout()
        fig.savefig(out/'explanation_uncertainty_comparison.png',dpi=170)
        plt.close(fig)
    except Exception as exc:print('Plot failed:',repr(exc),flush=True)
    lines=['# V9 Step 32 — Explanation uncertainty: Grad-CAM vs HiResCAM','','Validation-only, paired experiment on identical MRI cases and dropout masks.','',
      '| Metric | Grad-CAM | HiResCAM | HiResCAM minus Grad-CAM (95% bootstrap CI) |',
      '|---|---:|---:|---:|']
    for key in ('mean_shape_std','mean_shape_std_top20','cosine_to_mean','top20_iou_to_mean'):
        a=summary['results']['gradcam'][key]['mean']
        b=summary['results']['hirescam'][key]['mean']
        ci=summary['paired_hirescam_minus_gradcam'][key]
        lines.append(f'| {key} | {a:.6f} | {b:.6f} | {ci["mean"]:+.6f} [{ci["ci95"][0]:+.6f}, {ci["ci95"][1]:+.6f}] |')
    lines += ['',f'Correct validation predictions: **{summary["correct_predictions"]}**; wrong: **{summary["wrong_predictions"]}**.','',
      'These measurements reflect **heatmap variation under stochastic classifier dropout**. They do not verify tumour localization, clinical correctness, or that a low-variation explanation is faithful.','',
      'Final implementation note: if HiResCAM is selected, Step 22 full-test faithfulness, stability and explanation uncertainty must be recomputed for HiResCAM. Do not relabel old Grad-CAM numbers.','']
    (out/'STEP32_FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('V9 Step32 summary:',flush=True)
    for key in ('mean_shape_std','cosine_to_mean','top20_iou_to_mean'):
        z=summary['results'];ci=summary['paired_hirescam_minus_gradcam'][key]
        print(f' {key}: GradCAM {z["gradcam"][key]["mean"]:.6f} | HiResCAM {z["hirescam"][key]["mean"]:.6f} '
              f'| paired difference {ci["mean"]:+.6f} [{ci["ci95"][0]:+.6f},{ci["ci95"][1]:+.6f}]',flush=True)
    return summary


def run():
    pa=argparse.ArgumentParser(description='Step 32 paired MC explanation uncertainty on V9 validation')
    pa.add_argument('--n-per-class',type=int,default=50,help='50 per class = 200 validation MRIs, matching Steps 30 and 31')
    pa.add_argument('--seed',type=int,default=42)
    pa.add_argument('--passes',type=int,default=20)
    pa.add_argument('--out-name',default='32_hirescam_explanation_uncertainty')
    pa.add_argument('--resume',action='store_true')
    pa.add_argument('--examples',type=int,default=4)
    pa.add_argument('--gradient-mode',choices=['auto','vectorized','loop'],default='auto')
    args=pa.parse_args()
    if args.n_per_class<1 or args.passes<2:pa.error('n-per-class must be >=1; passes >=2')
    if Path(args.out_name).name != args.out_name or not args.out_name.startswith('32_'):
        pa.error('out-name must be a simple directory name beginning with 32_')
    out=C.OUTPUT/args.out_name
    if out.exists() and not args.resume:
        raise FileExistsError(f'{out} exists. Use --resume to reuse its saved results.')
    print('Loading trained V9 checkpoint and verifying the split...',flush=True)
    rows,fp,model,dev=environment('val')
    selection=sample_split(rows,'val',args.n_per_class,args.seed)
    if not isinstance(model.dropout,torch.nn.Dropout):
        raise RuntimeError('V9 classifier dropout is not nn.Dropout; fast stochastic head equivalence unverified.')
    if model.dropout.p<=0 or model.dropout.p>=1:
        raise RuntimeError('Invalid dropout probability')
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    meta={'split':'val','split_fingerprint':fp,'checkpoint_sha256':hashlib.sha256(C.CHECKPOINT.read_bytes()).hexdigest(),
      'layer':'features[-4]','seed':args.seed,'passes':args.passes,
      'dropout_probability':float(model.dropout.p),'n_per_class':args.n_per_class,
      'planned_images':len(selection),'selected_files':[r['file'] for r in selection],
      'methods':list(METHODS),'normalize':'resize then per-pass max normalize',
      'target':'fixed original deterministic predicted class',
      'note':'Only classifier dropout is sampled; same independent masks for both methods.'}
    if args.resume and (out/'progress.json').exists():
        prev=json.loads((out/'progress.json').read_text())
        for k in ('split_fingerprint','checkpoint_sha256','layer','seed','passes','dropout_probability',
                  'n_per_class','planned_images','selected_files','methods','normalize','target'):
            if prev.get(k)!=meta[k]:raise ValueError('Resume mismatch: '+k)
        done=set(prev['completed_files'])
        with (out/'uncertainty_trials.csv').open(encoding='utf-8',newline='') as f:
            records=list(csv.DictReader(f))
    else:done=set();records=[]
    out.mkdir(parents=True,exist_ok=True)
    tx=transform(False)
    saved_classes=set()
    print(f'Step32 | {len(selection)} validation MRIs | {args.passes} MC draws | device={dev} | dropout={model.dropout.p}',flush=True)
    for index,row in enumerate(selection,1):
        if row['file'] in done:continue
        t=time.monotonic()
        with Image.open(source_image(row['file'])) as im:
            pil=ImageOps.exif_transpose(im).convert('L')
        x=tx(pil).unsqueeze(0).to(dev)
        maps, info=compute_one(model,x,row['file'],args.seed,args.passes,args.gradient_mode)
        correct=int(info['prediction']==row['class_name'])
        shape_maps={}
        for method in METHODS:
            r,mu,sd=stats(maps[method]);shape_maps[method]=(mu,sd)
            records.append({'file':row['file'],'true_class':row['class_name'],
                'predicted_class':info['prediction'],'correct':correct,
                'method':method,'passes':args.passes,**r,
                'predicted_class_flip_rate':info['predicted_class_flip_rate'],
                'mc_mean_confidence':info['mc_mean_confidence'],
                'predictive_entropy':info['predictive_entropy'],
                'mutual_information':info['mutual_information']})
        if args.examples>0 and len(saved_classes)<args.examples and row['class_name'] not in saved_classes:
            write_example(out,pil,shape_maps,{**info,'file':row['file'],'true_class':row['class_name']},None,len(saved_classes)+1)
            saved_classes.add(row['class_name'])
        done.add(row['file'])
        atomic_csv(out/'uncertainty_trials.csv',records)
        progress={**meta,'completed_images':len(done),'completed_files':sorted(done),
                   'complete':len(done)==len(selection)}
        tmp=out/'progress.tmp.json';tmp.write_text(json.dumps(progress,indent=2),encoding='utf-8');tmp.replace(out/'progress.json')
        if index%10==0 or index==1:
            print(f' {len(done):3}/{len(selection)} images | last {time.monotonic()-t:.2f}s '
                  f'| GradCAM std={stats(maps["gradcam"])[0]["mean_shape_std"]:.5f}'
                  f' HiResCAM std={stats(maps["hirescam"])[0]["mean_shape_std"]:.5f}',flush=True)
    if len(done)<len(selection):
        print(f'Incomplete {len(done)}/{len(selection)}; rerun with --resume',flush=True);return
    summarize(records,meta,out)
    print(f'COMPLETE: {out}',flush=True)
    print('No V9 classifier retraining; previous Steps 22, 30 and 31 preserved.',flush=True)

if __name__=='__main__':run()
