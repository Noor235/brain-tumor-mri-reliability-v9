"""STEP 15: Explanation uncertainty from 20 MC-dropout Grad-CAM maps per MRI.
Target class fixed using original deterministic prediction; BatchNorm remains EVAL.
Per-pass CAM is independently max-normalized for SHAPE variability; raw maps also recorded.
Test sample defaults to 4/class: exploratory subset, NOT a 1,319-image full-cohort claim.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageOps
from . import config as C
from .common import output_dir,save_json,check_no_overwrite,source_image
from .research_v9 import environment,sample_split,ResearchCAM,get_layer,normalize_cam,cosine
from .transforms_v9 import transform


def softmax_1(logits):
    a=np.asarray(logits,dtype=np.float64);a=a-a.max();p=np.exp(a);return p/p.sum()

def image_panel(x,mean_map,std_map,target,filename):
    raw=((x+1)/2*255).clip(0,255).astype(np.uint8)
    base=cv2.cvtColor(raw,cv2.COLOR_GRAY2RGB)
    hm=cv2.cvtColor(cv2.applyColorMap((255*np.clip(mean_map,0,1)).astype(np.uint8),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    s=std_map/max(float(std_map.max()),1e-9)
    unc=cv2.cvtColor(cv2.applyColorMap((255*np.clip(s,0,1)).astype(np.uint8),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
    overlay=np.uint8(np.clip(.55*base+.45*hm,0,255))
    fig=Image.new('RGB',(4*C.IMAGE_SIZE,250),'white');d=ImageDraw.Draw(fig)
    for k,(pic,label) in enumerate(zip([base,hm,unc,overlay],['MRI','Mean MC Grad-CAM','CAM standard deviation','Mean overlay'])):
        fig.paste(Image.fromarray(pic),(k*C.IMAGE_SIZE,25));d.text((k*C.IMAGE_SIZE+5,6),label,fill='black')
    fig.save(filename)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['val','test'],default='test');ap.add_argument('--n-per-class',type=int,default=4)
    ap.add_argument('--layer',choices=['final','earlier'],default='final');ap.add_argument('--passes',type=int,default=20)
    args=ap.parse_args()
    if args.n_per_class<1 or args.passes<2:raise ValueError('Need >=1 image per class and >=2 passes.')
    C.heading('STEP 15 / EXPLANATION UNCERTAINTY — STOCHASTIC GRAD-CAM MAPS')
    outpath=C.OUTPUT/f'15_explanation_uncertainty_{args.split}_{args.layer}'/'uncertainty_summary.json';check_no_overwrite(outpath)
    rows,fp,model,dev=environment(args.split)
    selected=sample_split(rows,args.split,args.n_per_class)
    model.eval()
    dps=[m for m in model.modules() if isinstance(m,nn.Dropout)]
    if len(dps)!=1:raise RuntimeError(f'Expected exactly 1 dropout layer, got {len(dps)}')
    out=output_dir(f'15_explanation_uncertainty_{args.split}_{args.layer}')
    image_dir=out/'figures';image_dir.mkdir(parents=True,exist_ok=True)
    tx=transform(False);records=[]
    cam=ResearchCAM(model,get_layer(model,args.layer))
    try:
        for i,row in enumerate(selected,1):
            with Image.open(source_image(row['file'])) as src:pil=ImageOps.exif_transpose(src).convert('L')
            x=tx(pil).unsqueeze(0).to(dev)
            with torch.no_grad():
                det_logits=model(x)[0].detach().cpu().numpy();target=int(det_logits.argmax())
            normalized=[];raws=[];predicted=[];probs=[]
            try:
                for m in dps:m.train()
                for k in range(args.passes):
                    raw,logits=cam.get(x,target)
                    normalized.append(normalize_cam(raw));raws.append(raw)
                    probs.append(softmax_1(logits));predicted.append(int(np.asarray(logits).argmax()))
            finally:
                for m in dps:m.eval()
            stack=np.stack(normalized,axis=0)
            rawstack=np.stack(raws,axis=0)
            mean=stack.mean(axis=0);sd=stack.std(axis=0)
            mean_p=np.stack(probs).mean(axis=0)
            ent=float(-(mean_p*np.log(np.clip(mean_p,1e-12,1))).sum())
            avg_cos=float(np.mean([cosine(stack[0],stack[k]) for k in range(1,len(stack))]))
            mean_sd=float(sd.mean())
            basename=f'{row["class_name"]}_{i:03}_{args.layer}'
            image_panel(x.squeeze(0).squeeze(0).cpu().numpy(),mean,sd,target,image_dir/(basename+'.png'))
            np.savez_compressed(image_dir/(basename+'.npz'),mean_shape=mean,sd_shape=sd,raw_mean=rawstack.mean(axis=0),raw_sd=rawstack.std(axis=0))
            record=dict(file=row['file'],true_class=row['class_name'],predicted_class=C.CLASSES[target],split=args.split,layer=args.layer,
                passes=args.passes,mc_class_flip_rate=float(np.mean(np.asarray(predicted)!=target)),
                mean_shape_attribution_std=mean_sd,mean_raw_attribution_std=float(rawstack.std(axis=0).mean()),
                cosine_to_first_mc_cam=avg_cos,predictive_entropy_from_same_passes=ent,
                original_predicted_correct=int(target==C.CLASSES.index(row['class_name'])))
            records.append(record)
            print(f'  {i}/{len(selected)} {row["class_name"]:10} mean normalized CAM std={mean_sd:.6f} MC class-flip rate={record["mc_class_flip_rate"]:.1%}',flush=True)
    finally:
        for m in dps:m.eval()
        cam.close()
    with (out/'explanation_uncertainty.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=dict(split=args.split,layer=args.layer,examples=len(selected),mc_passes=args.passes,split_fingerprint=fp,
        mean_normalized_spatial_attribution_std=float(np.mean([x['mean_shape_attribution_std'] for x in records])),
        caution='Uses individually max-normalized MC Grad-CAM maps to assess spatial shape variability; does not equate to calibrated Bayesian pixel uncertainty. Raw attribution variation also saved. Small selected sample, not full test cohort.')
    save_json(outpath,summary)
    print('STEP 15 COMPLETE — saved mean and uncertainty maps; weights unchanged.',flush=True)
if __name__=='__main__':main()
