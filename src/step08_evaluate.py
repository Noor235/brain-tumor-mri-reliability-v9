"""STEP 08: one-time deterministic V9 test. NO training and no test-based tuning."""
import csv
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix,classification_report
from . import config as C
from .common import output_dir,save_json,check_no_overwrite
from .verify import verify
from .dataset_v9 import loader
from .model import build_model
from .train_utils import device

def main():
    C.heading('STEP 08 / FINAL V9 TEST: ONE deterministic evaluation on held-out test images')
    rows,fingerprint=verify(rehash=False,detail=True)
    if not C.CHECKPOINT.is_file():raise FileNotFoundError(f'No trained V9 checkpoint: {C.CHECKPOINT}')
    config=C.OUTPUT/'07_training'/'experiment_config.json'
    import json
    meta=json.loads(config.read_text(encoding='utf-8'))
    if meta['split_fingerprint']!=fingerprint:raise RuntimeError('Split has changed since training!')
    dest=C.OUTPUT/'08_test'/'test_metrics.json';check_no_overwrite(dest)
    dl=loader(rows,'test',False)
    dev=device();model=build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT,map_location='cpu',weights_only=True),strict=True)
    model=model.to(dev).eval()
    print(f'Device: {dev} | samples: {len(dl.dataset)} | single-pass, dropout OFF.',flush=True)
    y=[];pred=[];pr=[]
    with torch.inference_mode():
        for x,t in dl:
            probs=torch.softmax(model(x.to(dev)),dim=1).cpu().numpy()
            y.extend(t.tolist());pr.extend(probs.tolist());pred.extend(probs.argmax(1).tolist())
    y=np.asarray(y);pred=np.asarray(pred);pr=np.asarray(pr)
    labels=list(range(len(C.CLASSES)))
    acc=accuracy_score(y,pred);f1=f1_score(y,pred,average='macro',labels=labels,zero_division=0)
    recalls=recall_score(y,pred,average=None,labels=labels,zero_division=0)
    cm=confusion_matrix(y,pred,labels=labels)
    report=classification_report(y,pred,labels=labels,target_names=C.CLASSES,zero_division=0,digits=4)
    summary=f'KAGGLE V9 DEDUPLICATED TEST\nAccuracy {acc:.4%} ({(y==pred).sum()}/{len(y)})\nMacro-F1 {f1:.4f}\nGlioma recall {recalls[0]:.4%}\n\n{report}\nConfusion matrix (true rows, predicted columns):\n{cm}\n'
    print(summary,flush=True)
    out=output_dir('08_test');(out/'test_report.txt').write_text(summary,encoding='utf-8')
    with (out/'per_image_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['file','true_class','predicted_class','confidence','correct','prob_glioma','prob_meningioma','prob_notumor','prob_pituitary'])
        for r,t,p,probs in zip(dl.dataset.rows,y,pred,pr):
            writer.writerow([r['file'],C.CLASSES[t],C.CLASSES[p],f'{max(probs):.7f}',int(t==p),*map(lambda z:f'{z:.7f}',probs)])
    with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['actual / predicted',*C.CLASSES])
        for cls,row in zip(C.CLASSES,cm):writer.writerow([cls,*row.tolist()])
    save_json(dest,dict(test_count=len(y),correct=int((y==pred).sum()),accuracy=float(acc),macro_f1=float(f1),
        recalls=dict(zip(C.CLASSES,recalls.tolist())),classes=C.CLASSES,confusion_matrix=cm.tolist(),
        split_fingerprint=fingerprint,warning='Patient independence cannot be established without patient IDs; this test split was built after detected duplicate grouping.'))
    print('Original source V7 experiment unchanged. V9 test evaluated once.',flush=True)
if __name__=='__main__':main()
