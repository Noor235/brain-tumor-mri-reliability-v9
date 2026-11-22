"""STEP 13: Grad-CAM faithfulness via patchwise deletion and insertion.
Validation samples by default. Class target fixed at ORIGINAL deterministic prediction.
High-CAM, low-CAM, and random patch controls of identical area. Blur replacement.
Run on test only after design choices have been frozen; do not tune on test.
"""
import argparse
import csv
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from . import config as C
from .common import output_dir, save_json, check_no_overwrite, source_image
from .research_v9 import environment,sample_split,ResearchCAM,get_layer,normalize_cam,target_margin
from .transforms_v9 import transform

FRACTIONS=(0.1,0.2,0.3)
GRID=16   # 16x16 non-overlapping patches, 14x14 pixels each at 224x224


def score_many(model,dev,images,target):
    preds=[]
    with torch.no_grad():
        for start in range(0,len(images),8):
            batch=torch.stack(images[start:start+8],dim=0).to(dev)
            logits=model(batch)
            preds.extend(target_margin(logits,target).tolist())
    return preds


def patch_masks(cam,fraction,seed):
    patches=cv2.resize(cam.astype(np.float32),(GRID,GRID),interpolation=cv2.INTER_AREA).reshape(-1)
    n=max(1,int(np.ceil(GRID*GRID*fraction)))
    ranked=np.argsort(patches,kind='stable')
    rng=np.random.default_rng(seed)
    picks={'high':ranked[-n:], 'low':ranked[:n], 'random':rng.choice(GRID*GRID,size=n,replace=False)}
    return {k:np.kron(np.isin(np.arange(GRID*GRID),ids).reshape(GRID,GRID),np.ones((C.IMAGE_SIZE//GRID,C.IMAGE_SIZE//GRID),dtype=bool)) for k,ids in picks.items()}


def perturb(x,base,mask,kind):
    source=x.squeeze(0).squeeze(0).detach().cpu().numpy()
    if kind=='deletion':modified=np.where(mask,base,source)
    else:modified=np.where(mask,source,base)
    return torch.from_numpy(modified.astype(np.float32))[None,...]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['val','test'],default='val')
    ap.add_argument('--n-per-class',type=int,default=8)
    args=ap.parse_args()
    if args.n_per_class<1:raise ValueError('n-per-class must be >=1')
    C.heading('STEP 13 / GRAD-CAM FAITHFULNESS — DELETION AND INSERTION')
    out=C.OUTPUT/f'13_faithfulness_{args.split}'
    check_no_overwrite(out/'faithfulness_summary.json')
    rows,fp,model,dev=environment(args.split)
    sampled=sample_split(rows,args.split,args.n_per_class)
    transform_img=transform(False)
    records=[]
    print(f'Split={args.split}, images={len(sampled)}, identical patch counts for high/low/random; background=blurred image',flush=True)
    for layer_name in ('final','earlier'):
        cam_calc=ResearchCAM(model,get_layer(model,layer_name))
        try:
            for index,row in enumerate(sampled,1):
                with Image.open(source_image(row['file'])) as img:pil=ImageOps.exif_transpose(img).convert('L')
                x=transform_img(pil).unsqueeze(0).to(dev)
                with torch.no_grad():
                    original_logits=model(x).detach().cpu().numpy()
                target=int(original_logits.argmax(axis=1)[0]);original_margin=float(target_margin(original_logits,target)[0])
                raw,_=cam_calc.get(x,target); heat=normalize_cam(raw)
                pixels=x.squeeze(0).squeeze(0).detach().cpu().numpy()
                blurred=cv2.GaussianBlur(pixels,(0,0),sigmaX=10).astype(np.float32)
                blurred_logits=score_many(model,dev,[torch.from_numpy(blurred)[None,...]],target)
                baseline_margin=float(blurred_logits[0])
                for frac in FRACTIONS:
                    masks=patch_masks(heat,frac,C.SEED+index+int(frac*100))
                    alterations=[];keys=[]
                    for mode in ('deletion','insertion'):
                        for kind,mask in masks.items():
                            alterations.append(perturb(x,blurred,mask,mode));keys.append((mode,kind))
                    scored=score_many(model,dev,alterations,target)
                    for (mode,kind),margin in zip(keys,scored):
                        change=(original_margin-margin) if mode=='deletion' else (margin-baseline_margin)
                        records.append(dict(file=row['file'],true_class=row['class_name'],predicted_class=C.CLASSES[target],
                            layer=layer_name,split=args.split,fraction=frac,method=mode,region=kind,
                            baseline_target_margin=original_margin,modified_target_margin=float(margin),
                            score_change=float(change)))
                if index%8==0:print(f'  {layer_name}: {index}/{len(sampled)} images',flush=True)
        finally:cam_calc.close()
    out=output_dir(f'13_faithfulness_{args.split}')
    cols=list(records[0]);
    with (out/'faithfulness_trials.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(records)
    result=[]
    for layer in ('final','earlier'):
        for frac in FRACTIONS:
            for method in ('deletion','insertion'):
                group=[r for r in records if r['layer']==layer and r['fraction']==frac and r['method']==method]
                by_region={k:[q['score_change'] for q in group if q['region']==k] for k in ('high','low','random')}
                means={k:float(np.mean(v)) for k,v in by_region.items()}
                result.append(dict(layer=layer,area_fraction=frac,method=method,mean_change=means,
                    high_minus_random=float(means['high']-means['random']),sample_count=len(by_region['high'])))
                print(f'  {layer:7} {method:9} area={frac:.0%} | high={means["high"]:+.4f} low={means["low"]:+.4f} random={means["random"]:+.4f} | high-random={means["high"]-means["random"]:+.4f}',flush=True)
    save_json(out/'faithfulness_summary.json',dict(split=args.split,examples=len(sampled),comparison=result,
        split_fingerprint=fp,limitation='Blur patch replacement may create off-distribution inputs. CAM is low resolution. A high-vs-random margin difference is a diagnostic, not validated tumor localization. Test split must not guide choices.'))
    print('STEP 13 COMPLETE — no checkpoint changed.',flush=True)
if __name__=='__main__':main()
