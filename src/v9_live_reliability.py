"""V9 research demonstration: compute the existing five metrics on a new MRI.

Uses the user's immutable EfficientNet-B4 checkpoint, preprocessing, and the
corrected features[-4] Grad-CAM layer. No model retraining or source mutation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image,ImageOps
from . import config as C
from .model import build_model
from .train_utils import device
from .research_v9 import ResearchCAM, get_layer, normalize_cam, cosine, top_iou, target_margin
from .transforms_v9 import transform
from .step13_faithfulness import patch_masks, perturb, score_many
from .step14_stability import variations
from .step15_explanation_uncertainty import softmax_1


def load_v9():
    if not C.CHECKPOINT.is_file():
        raise FileNotFoundError(f'Original trained V9 checkpoint missing: {C.CHECKPOINT}')
    dev=device()
    model=build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT,map_location='cpu',weights_only=True),strict=True)
    model.to(dev).eval()
    cal=C.OUTPUT/'12_calibration'/'calibration_summary.json'
    if not cal.is_file():raise FileNotFoundError(f'Missing validation temperature: {cal}')
    temp=float(json.loads(cal.read_text())['temperature'])
    if not temp>0: raise ValueError('Temperature must be positive')
    return model,dev,temp


def overlay_pic(image,heat):
    arr=np.array(ImageOps.exif_transpose(image).convert('L').resize(
        (C.IMAGE_SIZE,C.IMAGE_SIZE),Image.Resampling.BICUBIC))
    rgb=cv2.cvtColor(arr,cv2.COLOR_GRAY2RGB)
    cmap=cv2.cvtColor(cv2.applyColorMap((np.clip(heat,0,1)*255).astype('uint8'),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    result=np.clip(0.6*rgb+0.4*cmap,0,255).astype('uint8')
    return Image.fromarray(result)


def analyze(image:Image.Image, model,dev,temperature, mc_passes=None,seed=None):
    model.eval()
    passes=C.MC_PASSES if mc_passes is None else int(mc_passes)
    if passes<2:raise ValueError('At least two MC passes required')
    pil=ImageOps.exif_transpose(image).convert('L')
    if seed is None:
        seed=int.from_bytes(hashlib.sha256(pil.tobytes()).digest()[:4],'little')%2**31
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    tx=transform(False)
    x=tx(pil).unsqueeze(0).to(dev)
    with torch.no_grad(): logits=model(x)[0].detach().cpu().numpy()
    target=int(logits.argmax())
    rawprob=softmax_1(logits)
    calprob=softmax_1(logits/temperature)
    dropouts=[m for m in model.modules() if isinstance(m,nn.Dropout)]
    if len(dropouts)!=1:raise RuntimeError('Expected exactly 1 classifier dropout module')
    # Prediction uncertainty matches the research MC approach: stochastic softmax
    # probabilities with BatchNorm frozen in evaluation mode.
    try:
        for d in dropouts:d.train()
        with torch.no_grad():
            mcps=np.stack([torch.softmax(model(x)[0],dim=0).cpu().numpy() for _ in range(passes)])
    finally:
        for d in dropouts:d.eval()
    mc_mean=mcps.mean(axis=0)
    ent=float(-np.sum(mc_mean*np.log(np.clip(mc_mean,1e-12,1))))
    variance=float(mcps[:,target].var())
    mc_class=C.CLASSES[int(mc_mean.argmax())]
    mc_disagreement=float(np.mean(mcps.argmax(axis=1)!=target))

    cam=ResearchCAM(model,get_layer(model,'earlier'))  # IMPORTANT: corrected features[-4]
    try:
        raw,_=cam.get(x,target)
        reference=normalize_cam(raw)
        pixels=x[0,0].detach().cpu().numpy().astype(np.float32)
        blur=cv2.GaussianBlur(pixels,(0,0),sigmaX=10).astype(np.float32)
        original_margin=float(target_margin(logits,target)[0])
        # Same fraction, 16x16 equal-area patch masks, blurred replacement,
        # and target-logit margin as the finished Step 22 experiment.
        masks=patch_masks(reference,0.2,seed+20)
        alter=[perturb(x,blur,masks[k],'deletion') for k in ('high','random','low')]
        margins=score_many(model,dev,alter,target)
        drops=[original_margin-float(t) for t in margins]
        faith_adv=float(drops[0]-drops[1])
        stability=[];flips=[]
        for title,modified,inverse_dx in variations(pixels,seed):
            xx=torch.from_numpy(modified[None,None,...]).to(dev)
            new_raw, _ = cam.get(xx,target)
            h=normalize_cam(new_raw)
            if inverse_dx:
                h=cv2.warpAffine(h,np.float32([[1,0,inverse_dx],[0,1,0]]),
                    (C.IMAGE_SIZE,C.IMAGE_SIZE),flags=cv2.INTER_LINEAR,borderValue=0)
            similarity=float(cosine(reference,h))
            stability.append(similarity)
            with torch.no_grad(): flips.append(int(model(xx).argmax(dim=1).item()!=target))
        mean_stability=float(np.nanmean(stability)) if np.isfinite(stability).any() else float('nan')
        # Same fixed target class on every stochastic CAM pass.
        # No modified input MRI; only dropout masks change.
        shapes=[];mc_classes=[]
        torch.manual_seed(seed)  # Step 22 uses a fresh per-MRI seed before stochastic CAMs
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        try:
            for d in dropouts:d.train()
            for _ in range(passes):
                r,ls=cam.get(x,target)
                shapes.append(normalize_cam(r));mc_classes.append(int(np.asarray(ls).argmax()))
        finally:
            for d in dropouts:d.eval()
        shapes=np.stack(shapes)
        shape_unc=float(shapes.std(axis=0).mean())
        mean_shape=shapes.mean(axis=0)
        corner=float(mean_shape[:C.IMAGE_SIZE//4, 3*C.IMAGE_SIZE//4:].sum() /
                     max(float(mean_shape.sum()),1e-12))
        return {
          'predicted_class':C.CLASSES[target],
          'probabilities':{label:float(calprob[i]) for i,label in enumerate(C.CLASSES)},
          'raw_confidence':float(rawprob[target]),
          'calibrated_confidence':float(calprob[target]),
          'mc_predicted_class':mc_class,
          'mc_predictive_entropy':ent,
          'mc_variance_predicted_class':variance,
          'mc_class_disagreement_rate':mc_disagreement,
          'faithfulness_del_high_minus_random_20':faith_adv,
          'faithfulness_del_high_20':float(drops[0]),
          'faithfulness_del_random_20':float(drops[1]),
          'mean_stability_cosine':mean_stability,
          'stability_prediction_flip_count':int(sum(flips)),
          'explanation_shape_uncertainty':shape_unc,
          'explanation_mc_class_flip_rate':float(np.mean(np.asarray(mc_classes)!=target)),
          'gradcam_corner_mass_example':corner,
          'gradcam_layer':'features[-4]',
          'mc_passes':passes,
          'overlay':overlay_pic(pil,reference),
          'heatmap':Image.fromarray((reference*255).astype('uint8')),
        }
    finally:
        for d in dropouts:d.eval()
        cam.close()


def explain_metrics(m,refstats=None):
    """Descriptive research indicators; NOT medical pass/fail cutoffs."""
    confidence=float(m['calibrated_confidence'])
    entropy=float(m['mc_predictive_entropy'])
    faith=float(m['faithfulness_del_high_minus_random_20'])
    stab=float(m['mean_stability_cosine'])
    shape=float(m['explanation_shape_uncertainty'])
    thresholds=(refstats or {}).get('quantiles',{})
    shape_p75=float(thresholds.get('explanation_shape_uncertainty',{}).get('0.75',0.020))
    result={
      'Calibrated confidence': ('Caution' if confidence<.9 else 'No strong warning',
           f'{confidence:.1%}. Temperature-scaled probability; confidence is not proof of correctness.'),
      'Prediction uncertainty': ('Caution' if entropy>.05 else 'No strong warning',
           f'Predictive entropy {entropy:.5f}. Higher means the mean MC prediction is less decisive.'),
      'Explanation faithfulness': ('Caution' if faith<=0 else 'Favourable deletion result',
           f'High-minus-random deletion margin {faith:+.3f}. A positive value means highlighted patches affected the selected class more than random patches.'),
      'Explanation stability': ('Caution' if not np.isfinite(stab) or stab<.95 else 'Favourable under tested changes',
           f'Mean heatmap cosine {stab:.3f} under noise, brightness and shift. Stability does not establish tumour location.'),
      'Explanation uncertainty': ('Above cohort 75th percentile' if shape>shape_p75 else 'Within cohort typical range',
           f'Mean CAM shape standard deviation {shape:.5f} (cohort 75th percentile {shape_p75:.5f}). This is not a probability of error.'),
    }
    return result


def warning_message(m,expected_label=None,refstats=None):
    metrics=explain_metrics(m,refstats)
    concerns=[name for name,(status,_) in metrics.items() if status in ('Caution','Above cohort 75th percentile')]
    if expected_label in C.CLASSES:
        if expected_label!=m['predicted_class']:
            return ('Known label mismatch',
                f'The predicted class {m["predicted_class"]} does not match the supplied reference label {expected_label}. '+
                'The class is incorrect relative to that label. Do not rely on the prediction.',concerns)
        return ('Matches supplied label',
            f'The prediction matches the supplied dataset/reference label ({expected_label}). '+
            'This does not establish a clinical diagnosis; independent verification is still required.',concerns)
    if concerns:
        return ('Research caution',
          'One or more reliability measurements raised a concern: '+', '.join(concerns)+'. '+
          'The system cannot determine whether this classification is correct without an independent reference label.', concerns)
    return ('No strong research warning',
       'No major concern appeared under these exploratory checks. This does NOT verify correctness, '+
       'rule out a tumour, or establish a clinical diagnosis. Independent review remains necessary.', concerns)
