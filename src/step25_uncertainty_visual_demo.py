"""V9 uncertainty teaching lab: LIVE MC Dropout inference (not saved-CSV plots).

Unzip this lab into the root of the original Brain_Tumor_Reproduction_V9 project.
Run: python3 -m src.step25_uncertainty_visual_demo --case all

Does NOT retrain or modify the checkpoint. The included examples are from the
user's known V9 held-out test split; labels are used only after prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save PNGs reliably from VS Code / Mac Terminal
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from . import config as C
from .model import build_model
from .transforms_v9 import transform

CLASSES = list(C.CLASSES)
# Derived in previous V9 Step 24 from the 90th percentile of validation MC entropy.
# Not a per-image error probability and not a clinical decision threshold.
VALIDATION_ENTROPY_CUTOFF = 0.00148676
CASES = {
    'correct': {
        'file': '01_correct_low_uncertainty.jpg',
        'true_class': 'notumor',
        'source_file': 'Testing/notumor/Te-no_347.jpg',
        'why': 'Correct with comparatively low uncertainty (saved Step 09 result)',
        'saved_entropy': 0.00060588,
    },
    'uncertain': {
        'file': '02_wrong_high_uncertainty.jpg',
        'true_class': 'glioma',
        'source_file': 'Testing/glioma/Te-gl_204.jpg',
        'why': 'Wrong, high uncertainty (saved Step 09 result)',
        'saved_entropy': 1.07149708,
    },
    'confidently_wrong': {
        'file': '03_wrong_low_uncertainty.jpg',
        'true_class': 'glioma',
        'source_file': 'Testing/glioma/Te-gl_58.jpg',
        'why': 'Wrong despite low uncertainty (saved Step 09 result)',
        'saved_entropy': 0.00025635,
    },
}


def entropy(p):
    """Entropy in NATURAL-LOG units, same calculation as original V9 Step 09."""
    v = np.asarray(p, dtype=np.float64)
    return -np.sum(v * np.log(np.clip(v, 1e-12, 1.0)), axis=-1)


def get_device(requested='auto'):
    if requested != 'auto':
        if requested == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable; use --device cpu')
        if requested == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu')
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def load_v9_model(checkpoint, device):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f'Checkpoint not found: {checkpoint}\n'
            'Unzip this ZIP inside your ORIGINAL V9 project (which has results/07_training/best_v9.pt), '
            'or pass --checkpoint /path/to/best_v9.pt.'
        )
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    # CHECK: Only the classifier dropout will be switched on for MC passes.
    d = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if len(d) != 1:
        raise RuntimeError(f'Expected exactly one dropout layer, found {len(d)}')
    return model


def predict_twenty_times(model, image_path, device, passes=20, seed=42):
    """Actual full V9 model inference; 20 distinct stochastic classifier dropout passes."""
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert('L')
        display_image = np.asarray(im.resize((224, 224), Image.Resampling.BICUBIC))
        x = transform(training=False)(im).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        det_p = torch.softmax(model(x), 1)[0].cpu().numpy().astype(float)

    # Keep the whole network, including BatchNorm, in evaluation mode.
    # Turn on only our one dropout layer to sample slightly different heads.
    model.eval()
    model.dropout.train()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    samples = []
    try:
        with torch.inference_mode():
            for _ in range(passes):
                probs = torch.softmax(model(x), 1)[0].cpu().numpy().astype(float)
                samples.append(probs)
    finally:
        model.eval()

    samples = np.stack(samples)
    mc_mean = samples.mean(axis=0)
    pred_idx = int(det_p.argmax())
    pe = float(entropy(mc_mean))
    ee = float(np.mean(entropy(samples)))
    mi = max(pe - ee, 0.0)
    result = {
        'image': display_image,
        'mc_probs': samples,
        'mc_mean': mc_mean,
        'det_probs': det_p,
        'det_prediction': CLASSES[pred_idx],
        'det_confidence': float(det_p[pred_idx]),
        'mc_prediction': CLASSES[int(mc_mean.argmax())],
        'mc_mean_confidence': float(mc_mean.max()),
        'predictive_entropy': pe,
        'expected_entropy': ee,
        'mutual_information': mi,
        'variance_of_det_pred_class': float(samples[:, pred_idx].var()),
        'referral': bool(pe > VALIDATION_ENTROPY_CUTOFF),
        'passes': passes,
    }
    return result


def fmt(value):
    return f'{float(value):.6f}'


def save_case_figure(key, info, result, out_dir):
    samples = result['mc_probs']
    mean = result['mc_mean']
    x = np.arange(1, len(samples) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), constrained_layout=True,
                             gridspec_kw={'width_ratios':[1.0, 1.5], 'height_ratios':[1.0, 1.1]})
    ax_img, ax_lines = axes[0]
    ax_bar, ax_txt = axes[1]
    ax_img.imshow(result['image'], cmap='gray', vmin=0, vmax=255)
    ax_img.set_title('Original MRI (test example)')
    ax_img.set_axis_off()
    for i,c in enumerate(CLASSES):
        ax_lines.plot(x, samples[:, i], marker='.', markersize=4.5, linewidth=1.7, label=c)
    ax_lines.set(xlabel='MC Dropout pass', ylabel='Predicted probability', ylim=(-0.03, 1.03),
                 xlim=(0.8, len(samples)+0.2), title=f'{len(samples)} stochastic model predictions')
    ax_lines.grid(alpha=.2)
    ax_lines.legend(loc='best', fontsize=9)
    bars=ax_bar.barh(CLASSES, mean)
    ax_bar.set_xlim(0, 1.07)
    ax_bar.invert_yaxis()
    ax_bar.set_title('Mean probability from all MC passes')
    ax_bar.set_xlabel('Mean predicted probability')
    for b,v in zip(bars,mean):
        ax_bar.text(min(.98,v+.02),b.get_y()+b.get_height()/2, f'{v:.2%}', va='center', fontsize=9)
    ax_txt.axis('off')
    if info.get('true_class'):
        correct = result['det_prediction'] == info['true_class']
        truth=f"True class: {info['true_class']}\nPrediction: {result['det_prediction']} ({'CORRECT' if correct else 'WRONG'})"
    else:
        truth=f"Prediction: {result['det_prediction']} (true label unknown)"
    status='REFER FOR REVIEW' if result['referral'] else 'NO UNCERTAINTY REFERRAL'
    txt=(f"{truth}\n\n"
         f"Deterministic confidence: {result['det_confidence']:.2%}\n"
         f"MC predictive entropy: {result['predictive_entropy']:.6f}\n"
         f"Validation-derived cutoff: {VALIDATION_ENTROPY_CUTOFF:.8f}\n"
         f"MC mutual information: {result['mutual_information']:.6f}\n\n"
         f"Decision: {status}\n\n"
         "Low uncertainty does NOT mean correct.\n"
         "A referral is NOT proof of error.")
    ax_txt.text(.03,.97,txt,va='top',ha='left',fontsize=11,
                bbox=dict(boxstyle='round,pad=.6',facecolor='#f5f5f5',edgecolor='#dddddd'),
                transform=ax_txt.transAxes,linespacing=1.5)
    fig.suptitle(f'V9 uncertainty example: {key.replace("_", " ")}',fontsize=15,fontweight='bold')
    dest=out_dir/f'{key}_20_mc_passes.png'
    fig.savefig(dest,dpi=160,bbox_inches='tight')
    plt.close(fig)
    return dest


def save_comparison_figure(results, out_dir):
    if len(results)<2:
        return None
    keys=list(results)
    fig, axs=plt.subplots(len(keys),3,figsize=(13.8,len(keys)*3.15),constrained_layout=True,
                         gridspec_kw={'width_ratios':[1,2,1.3]})
    axs=np.atleast_2d(axs)
    for i,key in enumerate(keys):
        info,r=results[key]['info'],results[key]['result']
        a_img,a_lines,a_text=axs[i]
        a_img.imshow(r['image'],cmap='gray',vmin=0,vmax=255)
        a_img.axis('off')
        a_img.set_title(key.replace('_',' '),fontsize=11)
        for j,label in enumerate(CLASSES):
            a_lines.plot(range(1,r['passes']+1),r['mc_probs'][:,j],label=label,linewidth=1.5)
        a_lines.set_ylim(-.02,1.02)
        a_lines.set_xlim(.8,r['passes']+.2)
        a_lines.set_ylabel('Probability')
        a_lines.grid(alpha=.15)
        if i==0:a_lines.legend(ncol=4,loc='upper center',fontsize=8, bbox_to_anchor=(.5,1.33))
        if i==len(keys)-1:a_lines.set_xlabel('MC Dropout pass (20 total)')
        a_text.axis('off')
        yes=r['det_prediction']==info['true_class']
        a_text.text(0,.9, f"True: {info['true_class']}\nPredict: {r['det_prediction']}\n"
                              f"{'CORRECT' if yes else 'WRONG'}\n\n"
                              f"Entropy: {r['predictive_entropy']:.6f}\n"
                              f"{'REFER FOR REVIEW' if r['referral'] else 'NOT REFERRED'}",
                              transform=a_text.transAxes,va='top',fontsize=11,linespacing=1.45)
    fig.suptitle('V9: why uncertainty helps, and why it can miss a wrong prediction',fontweight='bold',fontsize=15)
    dest=out_dir/'00_compare_three_real_mris.png'
    fig.savefig(dest,dpi=160,bbox_inches='tight')
    plt.close(fig)
    return dest


def save_csv_json(results, out_dir):
    cols=['example','original_test_file','true_class','det_prediction','correct','det_confidence',
          'mc_prediction','mc_mean_confidence','predictive_entropy','expected_entropy',
          'mutual_information','variance_of_det_pred_class','validation_entropy_cutoff','refer_for_review']
    with (out_dir/'uncertainty_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        for k,entry in results.items():
            info,r=entry['info'],entry['result']
            w.writerow(dict(example=k, original_test_file=info['source_file'],true_class=info['true_class'],
                  det_prediction=r['det_prediction'],correct=(int(r['det_prediction']==info['true_class']) if info.get('true_class') else ''),
                  det_confidence=r['det_confidence'],mc_prediction=r['mc_prediction'],
                  mc_mean_confidence=r['mc_mean_confidence'],predictive_entropy=r['predictive_entropy'],
                  expected_entropy=r['expected_entropy'],mutual_information=r['mutual_information'],
                  variance_of_det_pred_class=r['variance_of_det_pred_class'],
                  validation_entropy_cutoff=VALIDATION_ENTROPY_CUTOFF,
                  refer_for_review=int(r['referral'])))
    with (out_dir/'mc_20_pass_probabilities.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['example','pass']+[f'prob_{k}' for k in CLASSES])
        for k,entry in results.items():
            for i,row in enumerate(entry['result']['mc_probs'],1):
                w.writerow([k,i]+[f'{v:.12g}' for v in row])
    (out_dir/'README_RESULTS.txt').write_text(
        'Actual V9 inference with classifier dropout ON during 20 passes.\n'
        'CNN & BatchNorm eval; original V9 checkpoint; uncalibrated MC probabilities.\n'
        'Entropy = -sum(mean_prob * natural_log(mean_prob)); higher means more predictive uncertainty.\n'
        'Mutual information = predictive entropy - mean(per-pass entropy); epistemic proxy.\n'
        'Referral threshold = 0.00148676, derived earlier from V9 validation entropy.\n'
        'True labels are for demonstration only: unknown labels cannot be inferred from uncertainty.\n'
        'The small selection of three cases is illustrative, not an unbiased performance study.\n'
        '20 stochastic passes can vary slightly from the originally saved Step 09 results.\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description='Visual Studio Code V9 Monte Carlo Dropout lab (real checkpoint).')
    p.add_argument('--case',choices=['all']+list(CASES),default='all',help='Choose one of three real test examples (default: all).')
    p.add_argument('--image',type=Path,help='Analyze any MRI image (no known ground truth). Overrides --case.')
    p.add_argument('--label',choices=CLASSES,help='Optional independently known true label for --image.')
    p.add_argument('--passes',type=int,default=20,help='Number of MC samples; use 20 for thesis-consistent results.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--checkpoint',type=Path,default=C.CHECKPOINT)
    p.add_argument('--out',type=Path,default=C.ROOT/'results'/'25_uncertainty_visual_demo')
    args=p.parse_args()
    if args.passes < 2: p.error('Use at least 2 passes; original thesis uses 20.')
    if args.label and not args.image: p.error('--label requires --image')
    if args.image and not args.image.is_file():p.error(f'Image not found: {args.image}')
    dev=get_device(args.device)
    print(f'V9 uncertainty lab | device={dev} | passes={args.passes}',flush=True)
    model=load_v9_model(args.checkpoint,dev)
    print('Checkpoint loaded; only classifier dropout is activated during MC inference.',flush=True)
    out=args.out.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    samples_root=C.ROOT/'uncertainty_lab_samples'
    selected = ({'uploaded': {'file':str(args.image),'true_class':args.label,'source_file':str(args.image)}}
                if args.image else (CASES if args.case=='all' else {args.case:CASES[args.case]}))
    res={}
    for k,info in selected.items():
        path=Path(info['file']) if args.image else samples_root/info['file']
        if not path.is_file():
            raise FileNotFoundError(f'Example image not found: {path}\nMake sure uncertainty_lab_samples from the ZIP was extracted.')
        print(f'\nAnalyzing: {k}: {path}',flush=True)
        r=predict_twenty_times(model,path,dev,passes=args.passes,seed=args.seed)
        res[k]={'info':info,'result':r}
        path_img=save_case_figure(k,info,r,out)
        truth=info.get('true_class')
        print(f"  Deterministic: {r['det_prediction']} | confidence: {r['det_confidence']:.2%}")
        if truth:print(f"  Known true label: {truth} | correct: {r['det_prediction']==truth}")
        print(f"  MC entropy: {r['predictive_entropy']:.6f} | mutual information: {r['mutual_information']:.6f}")
        print(f"  {'REFER FOR REVIEW' if r['referral'] else 'NO UNCERTAINTY REFERRAL (not verified)'}")
        print(f'  Figure: {path_img}',flush=True)
    comparison=save_comparison_figure(res,out)
    save_csv_json(res,out)
    if comparison:print(f'\nComparison figure: {comparison}')
    print(f'\nDONE. Open images and CSV files in VS Code under: {out}',flush=True)
    print('IMPORTANT: Low uncertainty does NOT prove correctness. No patient/clinical validation.',flush=True)

if __name__=='__main__':main()
