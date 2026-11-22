"""One REAL Kaggle V9 validation MRI: visualize Grad-CAM deletion/insertion with actual model scores.

Run from V9 root:
    ./.venv/bin/python -m src.step13b_real_mri_explainer
    ./.venv/bin/python -m src.step13b_real_mri_explainer --class-name glioma

Read-only w.r.t. model, MRI dataset, and existing evaluation outputs.
This is a teaching illustration of a SINGLE MRI, not a clinical localization test.
"""
import argparse
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from . import config as C
from .common import source_image, output_dir, save_json
from .research_v9 import environment, sample_split, ResearchCAM, get_layer, normalize_cam, target_margin
from .transforms_v9 import transform
from .step13_faithfulness import patch_masks, perturb, score_many


def display_img(tensor):
    """Un-normalize [-1,1] -> [0,1] for a grayscale visual only."""
    arr = tensor.detach().cpu().numpy().squeeze()
    return np.clip((arr + 1) / 2, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--class-name', choices=C.CLASSES, default='meningioma')
    p.add_argument('--split', choices=['val'], default='val')
    p.add_argument('--seed', type=int, default=C.SEED)
    p.add_argument('--layer', choices=['final','earlier'], default='final')
    args = p.parse_args()

    C.heading('STEP 13B / REAL MRI EXAMPLE: WHAT DELETION & INSERTION MEAN')
    rows, fingerprint, model, dev = environment(args.split)
    chosen = sample_split(rows, args.split, n_per_class=1, seed=args.seed)
    r = next(item for item in chosen if item['class_name'] == args.class_name)
    with Image.open(source_image(r['file'])) as im:
        pil = ImageOps.exif_transpose(im).convert('L')
    x = transform(False)(pil).unsqueeze(0).to(dev)
    with torch.inference_mode():
        original_logits = model(x)
        probs = original_logits.softmax(dim=1)[0].detach().cpu().numpy()
    target = int(probs.argmax())
    original_score = float(target_margin(original_logits, target)[0])

    gradcam = ResearchCAM(model,get_layer(model,args.layer))
    try:
        raw,_ = gradcam.get(x, target)
        heatmap = normalize_cam(raw)
    finally:
        gradcam.close()

    fraction = .20
    masks = patch_masks(heatmap, fraction, C.SEED + 1 + int(fraction * 100))
    pixels = x.squeeze(0).squeeze(0).detach().cpu().numpy()
    blur = cv2.GaussianBlur(pixels, (0,0), sigmaX=10).astype(np.float32)
    blur_t = torch.from_numpy(blur)[None,...]
    base_score = float(score_many(model,dev,[blur_t],target)[0])
    scores, changed = {}, {}
    for mode in ('deletion','insertion'):
        for region in ('high','random','low'):
            edited = perturb(x,blur,masks[region],mode)
            key = (mode,region)
            changed[key] = edited
            scores[key] = float(score_many(model,dev,[edited],target)[0])
    changes = {f'{mode}_{region}':float((original_score - scores[(mode,region)]) if mode=='deletion' else (scores[(mode,region)] - base_score)) for mode in ('deletion','insertion') for region in ('high','random','low')}

    out = output_dir('13b_real_mri_example')
    figure_path = out / f'{args.split}_{args.class_name}_{args.layer}_seed{args.seed}.png'
    metadata_path = out / f'{args.split}_{args.class_name}_{args.layer}_seed{args.seed}.json'
    if figure_path.exists() or metadata_path.exists():
        raise FileExistsError('Example already generated. To avoid overwriting it, change --seed or --layer.')

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5), constrained_layout=True)
    gray_kw = {'cmap':'gray', 'vmin':0, 'vmax':1}
    axes[0,0].imshow(display_img(x),**gray_kw)
    axes[0,0].set_title(f'Original real MRI\nTrue: {r["class_name"]} | predicted: {C.CLASSES[target]}\nConfidence: {float(probs[target]):.2%}, margin: {original_score:+.2f}',fontsize=12)
    axes[0,1].imshow(display_img(x),**gray_kw)
    axes[0,1].imshow(heatmap,cmap='jet',alpha=.5, vmin=0,vmax=1)
    axes[0,1].set_title(f'Grad-CAM on real MRI\n{args.layer} feature layer')
    axes[0,2].imshow(display_img(blur_t),**gray_kw)
    axes[0,2].set_title(f'Fully blurred baseline\nMargin: {base_score:+.2f}')
    for j,region in enumerate(('high','random','low')):
        for row_index,mode in ((1,'deletion'),(2,'insertion')):
            ax = axes[row_index,j]
            ax.imshow(display_img(changed[(mode,region)]),**gray_kw)
            change = changes[f'{mode}_{region}']
            mode_label='DELETE (blur)' if mode=='deletion' else 'INSERT (restore)'
            ax.set_title(f'{mode_label} {region.upper()} 20%\nMargin: {scores[(mode,region)]:+.2f} | effect: {change:+.2f}', fontsize=12)
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle('V9 Grad-CAM faithfulness | ONE real validation MRI\nA positive effect means high importance regions changed the original-class margin as expected',fontsize=15,weight='bold')
    fig.savefig(figure_path,dpi=160,bbox_inches='tight')
    plt.close(fig)
    payload=dict(split=args.split, filename=r['file'], true_class=r['class_name'], predicted_class=C.CLASSES[target],
                 predicted_confidence=float(probs[target]), original_margin=original_score, blurred_baseline_margin=base_score,
                 region_fraction=fraction, layer=args.layer, score_changes=changes, dataset_fingerprint=fingerprint,
                 note='Single sampled validation MRI; scores are class-logit margins, NOT probabilities. This is not ground-truth tumor localization. Blur may yield artificial images.')
    save_json(metadata_path,payload)
    print(f'Real MRI: {r["file"]}')
    print(f'True class: {r["class_name"]} | predicted: {C.CLASSES[target]} ({float(probs[target]):.2%})')
    print(f'Original class margin: {original_score:+.3f}')
    print('20% DELETION: how much margin DROPS after blurring a region')
    for region in ('high','random','low'):
        print(f'  {region:6}: {changes["deletion_"+region]:+.3f}')
    print('20% INSERTION: how much margin RISES after restoring a region')
    for region in ('high','random','low'):
        print(f'  {region:6}: {changes["insertion_"+region]:+.3f}')
    print(f'Saved explanation figure: {figure_path}')
    print(f'Saved actual values: {metadata_path}')
    print('No retraining; original MRI images and model checkpoint are untouched.')

if __name__=='__main__':main()
