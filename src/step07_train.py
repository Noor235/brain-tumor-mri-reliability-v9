"""STEP 07: 20 epoch standalone V9 training. Train/validation ONLY, never test."""
import csv
from datetime import datetime,timezone
import torch
from torch import nn
from . import config as C
from .common import output_dir,check_no_overwrite,save_json
from .verify import verify
from .dataset_v9 import loader
from .model import build_model
from .train_utils import set_seed,device,run_epoch

def main():
    C.heading('STEP 07 / TRAIN V9: paper-aligned EfficientNet-B4 and leakage-controlled splits')
    checked=C.OUTPUT/'05_verification'/'verification.json'
    if not checked.is_file():raise RuntimeError('STEP 05 must pass BEFORE training. Run src.step05_verify.')
    rows,fingerprint=verify(rehash=False,detail=True)
    import json
    check=json.loads(checked.read_text(encoding='utf-8'))
    if check['status']!='PASS' or check['split_fingerprint']!=fingerprint:
        raise RuntimeError('Split differs from verified manifest. Run Step 05 again.')
    out=output_dir('07_training');check_no_overwrite(C.CHECKPOINT)
    history=out/'train_history.csv';check_no_overwrite(history)
    set_seed();dev=device()
    train_loader=loader(rows,'train',True);val_loader=loader(rows,'val',False)
    print('Device:',dev,flush=True)
    print(f'Input counts: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}. TEST SET NOT USED.',flush=True)
    print('Original source dataset is READ-ONLY. Near-similar candidates grouped; exact repeat copies excluded.',flush=True)
    print('Model: EfficientNet-B4 ImageNet 1-channel + Channel Attention -> Spatial Attention.',flush=True)
    print(f'Training: {C.EPOCHS} epochs, batch {C.BATCH_SIZE}, AdamW lr {C.LR}, weight decay {C.WEIGHT_DECAY}, dropout {C.DROPOUT}.',flush=True)
    print('BatchNorm running statistics FROZEN (V7 stability choice, not specified in original paper).',flush=True)
    print('NO brain cropping. Checkpoint selected on validation accuracy; no test tuning.',flush=True)
    model=build_model(pretrained=True).to(dev)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=C.LR,weight_decay=C.WEIGHT_DECAY)
    fields=['epoch','train_loss','train_accuracy','val_loss','val_accuracy','val_macro_f1','val_glioma_recall','saved_best']
    best_acc,best_f1=-1,-1
    save_json(out/'experiment_config.json',dict(experiment='Kaggle V9',source=str(C.SOURCE),
        split_fingerprint=fingerprint,train_count=len(train_loader.dataset),val_count=len(val_loader.dataset),
        optimizer='AdamW',lr=C.LR,weight_decay=C.WEIGHT_DECAY,epochs=C.EPOCHS,batch_size=C.BATCH_SIZE,
        loss='unweighted cross entropy',initialization='ImageNet EfficientNet-B4',
        batchnorm_running_stats='FROZEN',preprocessing='V7 grayscale 224 bicubic +/-0.5, train-only augmentation',
        study_design='exact copy removal + near-candidate grouping + new stratified group split',
        started_utc=datetime.now(timezone.utc).isoformat()))
    with history.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();f.flush()
        for epoch in range(1,C.EPOCHS+1):
            tr=run_epoch(model,train_loader,loss_fn,dev,optimizer=optimizer)
            val=run_epoch(model,val_loader,loss_fn,dev,optimizer=None)
            improved=val['accuracy']>best_acc or (val['accuracy']==best_acc and val['macro_f1']>best_f1)
            if improved:
                best_acc,best_f1=val['accuracy'],val['macro_f1']
                torch.save(model.state_dict(),C.CHECKPOINT)
            writer.writerow(dict(epoch=epoch,train_loss=round(tr['loss'],7),train_accuracy=round(tr['accuracy'],7),
                val_loss=round(val['loss'],7),val_accuracy=round(val['accuracy'],7),
                val_macro_f1=round(val['macro_f1'],7),val_glioma_recall=round(val['glioma_recall'],7),
                saved_best=int(improved)))
            f.flush()
            print(f"Epoch {epoch:02d}/{C.EPOCHS} | Train {tr['accuracy']:.2%} | Val {val['accuracy']:.2%} | "
                  f"Macro-F1 {val['macro_f1']:.4f} | Glioma recall {val['glioma_recall']:.2%}" +
                  (' | SAVED BEST' if improved else ''),flush=True)
    save_json(out/'training_summary.json',dict(best_validation_accuracy=best_acc,best_validation_macro_f1=best_f1,
        epochs_completed=C.EPOCHS,checkpoint=str(C.CHECKPOINT),split_fingerprint=fingerprint))
    print('TRAINING COMPLETE. BEST validation accuracy:',f'{best_acc:.4%}',flush=True)
    print('Checkpoint:',C.CHECKPOINT,flush=True)
    print('NEXT: only after training, run step08_evaluate once on the NEW V9 test set.',flush=True)
if __name__=='__main__':main()
