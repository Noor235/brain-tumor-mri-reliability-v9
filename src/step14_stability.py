"""STEP 14: Explanation stability under small input noise, intensity shift and translation.
Class target is FIXED to original deterministic prediction. Translation CAM is inverse aligned.
Default: fixed validation subset for development; test only after plan frozen.
"""
import argparse
import csv
import numpy as np
import cv2
import torch
from PIL import Image,ImageOps
from . import config as C
from .common import output_dir,save_json,check_no_overwrite,source_image
from .research_v9 import environment,sample_split,ResearchCAM,get_layer,normalize_cam,cosine,pearson,top_iou
from .transforms_v9 import transform


def variations(x,seed):
    rng=np.random.default_rng(seed)
    im=x.astype(np.float32);h,w=im.shape
    noise=np.clip(im+rng.normal(0,.02,im.shape).astype(np.float32),-1,1)
    brighter=np.clip(im+.04,-1,1)
    # warp +2 pixels horizontally; inverse warp applied to CAM before similarity
    shift=np.float32([[1,0,2],[0,1,0]])
    translated=cv2.warpAffine(im,shift,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-1)
    return [('noise_std_0.01',noise,0),('brightness_plus_0.02',brighter,0),('shift_2px_right',translated,-2)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['val','test'],default='val');ap.add_argument('--n-per-class',type=int,default=8)
    args=ap.parse_args()
    if args.n_per_class<1:raise ValueError('--n-per-class must be >=1')
    C.heading('STEP 14 / EXPLANATION STABILITY — FIXED TARGET, ALIGNED MAPS')
    out=C.OUTPUT/f'14_stability_{args.split}';check_no_overwrite(out/'stability_summary.json')
    rows,fp,model,dev=environment(args.split)
    sample=sample_split(rows,args.split,args.n_per_class)
    tx=transform(False);records=[]
    for layer_name in ('final','earlier'):
        cam=ResearchCAM(model,get_layer(model,layer_name))
        try:
            for index,row in enumerate(sample,1):
                with Image.open(source_image(row['file'])) as src:pil=ImageOps.exif_transpose(src).convert('L')
                x=tx(pil).unsqueeze(0).to(dev)
                with torch.no_grad():target=int(model(x).argmax(1).item())
                base_raw,_=cam.get(x,target);base=normalize_cam(base_raw)
                raw=x.squeeze(0).squeeze(0).cpu().numpy()
                for name,modified,inverse_dx in variations(raw,C.SEED+index):
                    xx=torch.from_numpy(modified[None,None,...]).to(dev)
                    heat,_=cam.get(xx,target);heat=normalize_cam(heat)
                    if inverse_dx:
                        heat=cv2.warpAffine(heat,np.float32([[1,0,inverse_dx],[0,1,0]]),(C.IMAGE_SIZE,C.IMAGE_SIZE),flags=cv2.INTER_LINEAR,borderValue=0)
                    with torch.no_grad():other_pred=int(model(xx).argmax(1).item())
                    records.append(dict(file=row['file'],true_class=row['class_name'],original_predicted_class=C.CLASSES[target],
                        modified_predicted_class=C.CLASSES[other_pred],prediction_flip=int(other_pred!=target),layer=layer_name,
                        perturbation=name,cosine_similarity=cosine(base,heat),pearson_correlation=pearson(base,heat),top20_iou=top_iou(base,heat,.2)))
                if index%8==0:print(f'  {layer_name}: {index}/{len(sample)}',flush=True)
        finally:cam.close()
    out=output_dir(f'14_stability_{args.split}')
    with (out/'stability_trials.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for layer in ('final','earlier'):
        for pert in ('noise_std_0.01','brightness_plus_0.02','shift_2px_right'):
            subset=[r for r in records if r['layer']==layer and r['perturbation']==pert]
            def av(k):
                v=np.array([r[k] for r in subset],dtype=float);return float(np.nanmean(v)) if np.isfinite(v).any() else None
            m=dict(layer=layer,perturbation=pert,n=len(subset),mean_cosine=av('cosine_similarity'),mean_pearson=av('pearson_correlation'),mean_top20_iou=av('top20_iou'),prediction_flips=sum(r['prediction_flip'] for r in subset))
            summary.append(m)
            print(f'  {layer:7} {pert:25} cosine={m["mean_cosine"]} IoU={m["mean_top20_iou"]} flips={m["prediction_flips"]}',flush=True)
    save_json(out/'stability_summary.json',dict(split=args.split,examples=len(sample),results=summary,split_fingerprint=fp,
        limitation='Fixed small perturbations chosen for diagnostic purposes. MRI clinical plausibility not formally validated. Shifted Grad-CAM inversely aligned; background interpolation may affect similarity.'))
    print('STEP 14 COMPLETE — original images unchanged.',flush=True)
if __name__=='__main__':main()
