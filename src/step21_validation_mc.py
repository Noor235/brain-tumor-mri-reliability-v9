"""STEP 21: Validation 20-pass MC Dropout + predictive variance, entropy, mutual information.

Only the classifier dropout is activated. BatchNorm stays in eval mode.
Uncertainty is not proof of calibration or medical reliability.
"""
import csv
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score,roc_auc_score
from . import config as C
from .common import output_dir,save_json,check_no_overwrite
from .verify import verify
from .dataset_v9 import loader
from .model import build_model
from .train_utils import device

def entropy(p):return -np.sum(p*np.log(np.clip(p,1e-12,1)),axis=-1)

def main():
    C.heading('STEP 21 / VALIDATION PREDICTION UNCERTAINTY: 20 MC passes PER validation MRI')
    if not (C.OUTPUT/'08_test'/'test_metrics.json').is_file():raise RuntimeError('Run final V9 evaluation (step08) first.')
    rows,fingerprint=verify(rehash=False,detail=False)
    outpath=C.OUTPUT/'21_validation_mc'/'uncertainty.csv';check_no_overwrite(outpath)
    dl=loader(rows,'val',False)
    dev=device();model=build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT,map_location='cpu',weights_only=True),strict=True)
    model.to(dev).eval()
    dropout_count=0
    for m in model.modules():
        if isinstance(m,nn.Dropout):m.train();dropout_count+=1
    print(f'Model: {C.MC_PASSES} MC passes | activated {dropout_count} dropout layer(s); BatchNorm EVAL.',flush=True)
    if dropout_count!=1:raise RuntimeError('Expected exactly 1 classifier dropout layer.')
    records=[];targets=[];predictions=[];entropies=[]
    with torch.inference_mode():
        for b,(images,y) in enumerate(dl,1):
            x=images.to(dev)
            ps=np.stack([torch.softmax(model(x),dim=1).cpu().numpy() for _ in range(C.MC_PASSES)],axis=0)
            mean=ps.mean(axis=0);variance=ps.var(axis=0)
            avg_entropy=entropy(ps).mean(axis=0)
            pe=entropy(mean);mi=np.maximum(pe-avg_entropy,0)
            guess=mean.argmax(1)
            start=(b-1)*C.BATCH_SIZE
            for i,truth in enumerate(y.tolist()):
                row=dl.dataset.rows[start+i];pred=int(guess[i])
                records.append({'file':row['file'],'true_class':C.CLASSES[truth],
                    'predicted_class':C.CLASSES[pred],'correct':int(pred==truth),
                    'mean_confidence':f'{mean[i,pred]:.7f}',
                    'variance_predicted_class':f'{variance[i,pred]:.9f}',
                    'predictive_entropy':f'{pe[i]:.8f}',
                    'expected_entropy':f'{avg_entropy[i]:.8f}',
                    'mutual_information':f'{mi[i]:.8f}'})
                targets.append(truth);predictions.append(pred);entropies.append(float(pe[i]))
            if b%15==0:print(f'  MC evaluated {min(b*C.BATCH_SIZE,len(dl.dataset))}/{len(dl.dataset)} MRIs',flush=True)
    targets=np.asarray(targets);predictions=np.asarray(predictions);entropies=np.asarray(entropies)
    out=output_dir('21_validation_mc')
    from .common import write_csv
    fields=['file','true_class','predicted_class','correct','mean_confidence','variance_predicted_class','predictive_entropy','expected_entropy','mutual_information']
    write_csv(outpath,records,fields)
    accuracy=float(accuracy_score(targets,predictions))
    is_wrong=(targets!=predictions).astype(int)
    try:uncertainty_auc=float(roc_auc_score(is_wrong,entropies)) if len(set(is_wrong))>1 else None
    except ValueError:uncertainty_auc=None
    summary={'mc_passes':C.MC_PASSES,'mc_validation_accuracy':accuracy,'validation_count':len(records),
       'wrong_predictions':int(is_wrong.sum()),
       'mean_entropy_correct':float(entropies[is_wrong==0].mean()) if (is_wrong==0).any() else None,
       'mean_entropy_incorrect':float(entropies[is_wrong==1].mean()) if (is_wrong==1).any() else None,
       'entropy_discrimination_auc_for_errors':uncertainty_auc,'split_fingerprint':fingerprint,
       'caution':'Entropy of mean probability includes multiple uncertainty sources; MC mutual information is a commonly used epistemic uncertainty proxy, not a guarantee of clinical safety.'}
    save_json(out/'mc_summary.json',summary)
    print('MC accuracy:',f'{accuracy:.4%}','| wrong predictions:',summary['wrong_predictions'],flush=True)
    print('Mean predictive entropy — correct:',summary['mean_entropy_correct'],flush=True)
    print('Mean predictive entropy — incorrect:',summary['mean_entropy_incorrect'],flush=True)
    print('No original checkpoints or images were modified.',flush=True)
if __name__=='__main__':main()
