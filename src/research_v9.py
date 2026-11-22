"""Common read-only research helpers for V9 post-hoc analyses (Steps 12-17).
Never edits source MRIs, manifests, prior results, or trained checkpoints.
"""
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from . import config as C
from .verify import verify
from .model import build_model
from .train_utils import device


def environment(split='val'):
    rows, fingerprint = verify(rehash=False, detail=False)
    meta_path = C.OUTPUT/'07_training'/'experiment_config.json'
    if not C.CHECKPOINT.is_file() or not meta_path.is_file():
        raise FileNotFoundError('Run Step 07 training first; trained checkpoint and experiment config required.')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('split_fingerprint') != fingerprint:
        raise RuntimeError('Dataset split fingerprint has CHANGED since model training. Refusing evaluation.')
    if split == 'test':
        test_file = C.OUTPUT/'08_test'/'test_metrics.json'
        if not test_file.is_file():
            raise FileNotFoundError('Run Step 08 before post-hoc test analyses.')
        test_meta = json.loads(test_file.read_text(encoding='utf-8'))
        if test_meta.get('split_fingerprint') != fingerprint:
            raise RuntimeError('Test evaluation uses a different dataset fingerprint.')
    dev = device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT, map_location='cpu', weights_only=True), strict=True)
    model = model.to(dev).eval()
    return rows, fingerprint, model, dev


def sample_split(rows, split, n_per_class=8, seed=None):
    if split not in ('train','val','test'):
        raise ValueError('Split must be train, val or test')
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    chosen = []
    for cls in C.CLASSES:
        pool = sorted((r for r in rows if r['split']==split and r['class_name']==cls),key=lambda r:r['file'])
        if not pool: raise ValueError(f'No {cls} images in {split} split')
        ids = sorted(rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False).tolist())
        chosen.extend(pool[i] for i in ids)
    return chosen


def target_margin(logits, target):
    """Target logit minus logsumexp of competing logits, stable at p near 1."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim == 1: a = a[None, :]
    other = np.delete(a, target, axis=1)
    m = other.max(axis=1)
    logsum = m + np.log(np.exp(other-m[:,None]).sum(axis=1))
    return a[:,target] - logsum


class ResearchCAM:
    """Grad-CAM computed on exact fixed class target; exposes same-pass logits for MC runs."""
    def __init__(self, model, layer):
        self.activation = None
        self.gradient = None
        self.h1 = layer.register_forward_hook(self._fw)
        self.h2 = layer.register_full_backward_hook(self._bw)
        self.model = model
    def _fw(self, layer, args, output):
        self.activation = output.detach()
    def _bw(self, layer, grad_input, grad_output):
        self.gradient = grad_output[0].detach()
    def close(self):
        self.h1.remove(); self.h2.remove()
    def get(self, x, target, interpolation=cv2.INTER_LINEAR):
        self.activation = None; self.gradient = None
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, int(target)].backward()
        if self.activation is None or self.gradient is None:
            raise RuntimeError('Grad-CAM hooks failed; inspect feature layer and torch version.')
        a = self.activation[0]
        g = self.gradient[0]
        cam = torch.relu((a * g.mean((1,2), keepdim=True)).sum(0))
        arr = cam.detach().cpu().numpy().astype(np.float32)
        arr = cv2.resize(arr, (C.IMAGE_SIZE,C.IMAGE_SIZE), interpolation=interpolation)
        return np.maximum(arr,0), logits.detach().cpu().numpy()[0]


def normalize_cam(raw):
    v = np.asarray(raw,dtype=np.float32)
    ma = float(np.max(v))
    return v/ma if ma>1e-16 else np.zeros_like(v)


def get_layer(model, name):
    if name=='final':return model.features[-1]
    if name=='earlier':return model.features[-4]
    raise ValueError('Layer must be final or earlier')


def fixed_top_mask(cam, fraction):
    """Exactly ceil(fraction * H * W) pixels, deterministic even with ties."""
    flat = np.asarray(cam).reshape(-1)
    count = max(1,int(np.ceil(len(flat)*fraction)))
    ids = np.argsort(flat, kind='stable')[-count:]
    mask = np.zeros(len(flat),dtype=bool);mask[ids] = True
    return mask.reshape(np.asarray(cam).shape)


def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    da=np.linalg.norm(a);db=np.linalg.norm(b)
    return float(np.dot(a,b)/(da*db)) if da>1e-12 and db>1e-12 else float('nan')


def pearson(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean()
    return cosine(a,b)


def top_iou(a,b,proportion=.2):
    one=fixed_top_mask(a,proportion);two=fixed_top_mask(b,proportion)
    n=int(np.logical_or(one,two).sum())
    return float(np.logical_and(one,two).sum()/n) if n else float('nan')
