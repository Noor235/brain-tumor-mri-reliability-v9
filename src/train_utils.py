"""Reproducibility and training loop helpers."""
import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score,recall_score
from . import config as C

def set_seed():
    random.seed(C.SEED);np.random.seed(C.SEED);torch.manual_seed(C.SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(C.SEED)

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def freeze_bn(model):
    for layer in model.modules():
        if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()

def run_epoch(model,load,criterion,dev,optimizer=None):
    training=optimizer is not None
    model.train(training)
    if training:freeze_bn(model)
    y_true=[];y_pred=[];loss_sum=0.0
    for x,y in load:
        x,y=x.to(dev),y.to(dev)
        if training:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits=model(x);loss=criterion(logits,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss detected')
            if training:loss.backward();optimizer.step()
        loss_sum+=float(loss.detach().cpu())*len(y)
        y_true.extend(y.detach().cpu().tolist());y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    truth=np.asarray(y_true);guess=np.asarray(y_pred);labels=list(range(len(C.CLASSES)))
    recalls=recall_score(y_true,y_pred,labels=labels,average=None,zero_division=0)
    return dict(loss=loss_sum/len(truth),accuracy=float(np.mean(truth==guess)),
        macro_f1=float(f1_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)),
        glioma_recall=float(recalls[C.CLASSES.index('glioma')]))
