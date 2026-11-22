"""Step 12. Prediction calibration: validation-only temperature fit, test diagnostic.
ECE (15 equal-width bins), multiclass Brier, NLL, accuracy, reliability diagram.
No model training, no overwrite of Step 08. Re-reads V9 checkpoint and fixed split.
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize_scalar
from . import config as C
from .common import output_dir, save_json, check_no_overwrite
from .dataset_v9 import loader
from .research_v9 import environment


def softmax(logits):
    z=np.asarray(logits,dtype=np.float64)
    z=z-z.max(axis=1,keepdims=True)
    v=np.exp(z);return v/v.sum(axis=1,keepdims=True)

def metrics(logits,targets,temperature=1.0,bins=15):
    p=softmax(logits/temperature)
    y=np.asarray(targets,dtype=int)
    confidence=p.max(axis=1);guess=p.argmax(axis=1);correct=(guess==y)
    nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean())
    brier=float(((p-np.eye(len(C.CLASSES))[y])**2).sum(axis=1).mean())
    edges=np.linspace(0,1,bins+1);ece=0.;detail=[]
    for i in range(bins):
        included=(confidence>=edges[i]) & (confidence<(edges[i+1]) if i<bins-1 else confidence<=1)
        n=int(included.sum());acc=float(correct[included].mean()) if n else None
        conf=float(confidence[included].mean()) if n else None
        if n:ece+=n/len(y)*abs(acc-conf)
        detail.append(dict(lower=float(edges[i]),upper=float(edges[i+1]),count=n,empirical_accuracy=acc,mean_confidence=conf))
    return dict(accuracy=float(correct.mean()),ece=float(ece),brier=brier,nll=nll,bins=detail),p


def fetch(model,device,rows,split):
    ds=loader(rows,split,False)
    ys=[];scores=[]
    with torch.inference_mode():
        for i,(x,y) in enumerate(ds,1):
            scores.append(model(x.to(device)).cpu().numpy());ys.extend(y.tolist())
            if i%15==0:print(f'  {split}: {min(i*C.BATCH_SIZE,len(ds.dataset))}/{len(ds.dataset)} MRI images',flush=True)
    return np.concatenate(scores,axis=0),np.asarray(ys,dtype=int),ds.dataset.rows


def main():
    C.heading('STEP 12 / CONFIDENCE CALIBRATION — ECE, BRIER, NLL, TEMPERATURE SCALING')
    out_file=C.OUTPUT/'12_calibration'/'calibration_summary.json';check_no_overwrite(out_file)
    rows,fp,model,dev=environment('test')
    val_logits,val_y,_=fetch(model,dev,rows,'val')
    print('Fitting one temperature on validation labels ONLY (not test).',flush=True)
    result=minimize_scalar(lambda lt:metrics(val_logits,val_y,np.exp(lt))[0]['nll'],bounds=(-2.0,2.0),method='bounded',options={'xatol':1e-5})
    temperature=float(np.exp(result.x))
    test_logits,test_y,test_rows=fetch(model,dev,rows,'test')
    values={};profiles={}
    for split,lg,y in [('val',val_logits,val_y),('test',test_logits,test_y)]:
        for label,T in [('raw',1.0),('temperature_scaled',temperature)]:
            values[f'{split}_{label}'],p=metrics(lg,y,T)
            profiles[f'{split}_{label}']=p
            m=values[f'{split}_{label}']
            print(f'  {split:4} {label:19} accuracy={m["accuracy"]:.4%} ECE={m["ece"]:.5f} Brier={m["brier"]:.5f} NLL={m["nll"]:.5f}',flush=True)
    out=output_dir('12_calibration')
    with (out/'per_image_test_calibration.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['file','true_class','raw_prediction','raw_confidence','calibrated_confidence','correct',*[f'raw_prob_{x}' for x in C.CLASSES],*[f'calibrated_prob_{x}' for x in C.CLASSES]])
        raw=profiles['test_raw'];scaled=profiles['test_temperature_scaled']
        for r,y,a,b in zip(test_rows,test_y,raw,scaled):
            w.writerow([r['file'],C.CLASSES[y],C.CLASSES[int(a.argmax())],float(a.max()),float(b.max()),int(a.argmax()==y),*map(float,a),*map(float,b)])
    fig,ax=plt.subplots(figsize=(7,6));ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
    for label in ['raw','temperature_scaled']:
        points=[(b['mean_confidence'],b['empirical_accuracy']) for b in values['test_'+label]['bins'] if b['count']]
        if points:ax.plot(*zip(*points),'o-',label=label.replace('_',' '))
    ax.set(xlabel='Mean predicted confidence',ylabel='Observed accuracy',title='Kaggle V9 — test reliability diagram',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/'reliability_diagram.png',dpi=175);plt.close(fig)
    save_json(out_file,dict(temperature_fit_on='validation_only',temperature=temperature,calibration=values,split_fingerprint=fp,
        limitation='Test data were evaluated previously; this is post-hoc diagnosis. Temperature is fitted only on validation labels. ECE bin count=15, Brier summed over 4 classes.'))
    print('STEP 12 COMPLETE — checkpoint untouched; temperature is an external calibration parameter.',flush=True)
if __name__=='__main__':main()
