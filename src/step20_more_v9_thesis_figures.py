"""Step 20: additional V9 research figure pack (including standard and MC confusion matrices), from SAVED experiment outputs only.

Install in V9/src/ and run:
    ./.venv/bin/python -m src.step20_more_v9_thesis_figures

No model loaded; no training; no saved results or checkpoint modified.
Outputs results/20_more_thesis_figures/*.png and FIGURE_INDEX.txt.
"""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C = {'high':'#C8553D','random':'#7486A1','low':'#269B80',
     'raw':'#C8553D','temperature_scaled':'#269B80',
     'final':'#246B9B','earlier':'#C68F32'}
CLASSES = ('glioma','meningioma','notumor','pituitary')


def rows(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def data(path):
    if not path.is_file(): return None
    with path.open(encoding='utf-8') as f:return json.load(f)


def num(v, default=float('nan')):
    try:return float(v)
    except (TypeError,ValueError):return default


def valid(vals):
    a=np.asarray(vals,dtype=float)
    return a[np.isfinite(a)]


def save(fig, dst, filename, index, source, note):
    p=dst/filename
    fig.savefig(p, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    index.append((filename,source,note))
    print('[SAVED]',p)


def basic(title, xlabel, ylabel):
    fig,ax=plt.subplots(figsize=(8.8,5.4))
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.22,axis='y')
    return fig,ax



def actual_test_count(results):
    metrics=data(results/'08_test'/'test_metrics.json')
    if not metrics or not isinstance(metrics.get('test_count'),int) or metrics['test_count']<1:
        raise FileNotFoundError('Need Step 08 test_metrics.json with positive test_count to plot V9 figures.')
    return int(metrics['test_count'])

def calibration_plots(results,dst,index):
    summ=data(results/'12_calibration'/'calibration_summary.json')
    if not summ:
        print('[SKIP] calibration: calibration_summary.json missing');return
    entries=summ.get('calibration',{})
    vals=[('raw','Before calibration'),('temperature_scaled','After calibration')]
    fig,ax=basic('V9 Test Reliability Diagram — Before vs After Calibration','Mean predicted confidence','Observed fraction correct')
    ax.plot([0,1],[0,1],ls='--',color='#555555',label='Perfect calibration')
    count=0
    for k,label in vals:
        e=entries.get('test_'+k,{})
        pts=[(b.get('mean_confidence'),b.get('empirical_accuracy')) for b in e.get('bins',[]) if b.get('count',0)>0]
        if pts:
            ax.plot([p[0] for p in pts],[p[1] for p in pts],'o-',label=f'{label} (ECE={num(e.get("ece")):.4f})',color=C[k],lw=2)
            count+=1
    ax.set(xlim=(0,1),ylim=(0,1));ax.legend()
    if count:save(fig,dst,'01_calibration_reliability_before_after.png',index,'12_calibration/calibration_summary.json','Post-hoc test diagnostic, bins populated only')
    else:plt.close(fig)
    for idx,(metric,display) in enumerate([('ece','Expected Calibration Error'),('brier','Multiclass Brier Score'),('nll','Negative Log Likelihood')],2):
        x=['Raw','Scaled'];v=[]
        for k in ['raw','temperature_scaled']:
            v.append(num(entries.get('test_'+k,{}).get(metric)))
        if not np.all(np.isfinite(v)): continue
        fig,ax=basic(f'V9 Test — {display}', 'Calibration','Lower is better')
        bars=ax.bar(x,v,color=[C['raw'],C['temperature_scaled']],width=.48)
        ax.set_ylim(0,max(v)*1.28 if max(v)>0 else 1)
        ax.bar_label(bars,fmt='%.5f',padding=4)
        save(fig,dst,f'{idx:02d}_calibration_{metric}_comparison.png',index,'12_calibration/calibration_summary.json','Test metrics only; no correction of wrong classes')


def faithfulness_plots(results,dst,index):
    for split in ['val','test']:
        path=results/f'13_faithfulness_{split}'/'faithfulness_summary.json'
        summ=data(path)
        if not summ:
            print(f'[SKIP] faithfulness {split}: summary missing');continue
        entries=summ.get('comparison',[])
        for layer in ['final','earlier']:
            for method in ['deletion','insertion']:
                rec=sorted([r for r in entries if r.get('layer')==layer and r.get('method')==method],key=lambda r:float(r.get('area_fraction',-1)))
                if not rec: continue
                x=[100*num(r.get('area_fraction')) for r in rec]
                fig,ax=basic(f'V9 Grad-CAM {method.title()} — {layer.title()} Layer ({split}, n={summ.get("examples","?")})',
                             'Fraction of image region modified (%)','Mean change in original-class logit margin')
                for k,label in [('high','High Grad-CAM'),('random','Random control'),('low','Low Grad-CAM')]:
                    y=[num(r.get('mean_change',{}).get(k)) for r in rec]
                    if np.all(np.isfinite(y)):
                        ax.plot(x,y,'o-',lw=2.4,label=label,color=C[k])
                ax.axhline(0,color='#888888',lw=1)
                ax.set_xticks(x);ax.legend()
                save(fig,dst,f'05_faithfulness_{split}_{layer}_{method}.png',index,str(path.relative_to(results)),
                     'Higher positive effect means more original-class margin removed/restored; logit margins not accuracy')
    # paired per-MRI high-minus-random, on test subset
    trialfile=results/'13_faithfulness_test'/'faithfulness_trials.csv'
    trials=rows(trialfile)
    if trials:
        grouped=defaultdict(dict)
        for r in trials:
            if r.get('layer')=='final' and r.get('method')=='deletion' and abs(num(r.get('fraction'))-.2)<.001:
                grouped[r['file']][r.get('region')]=num(r.get('score_change'))
        pairs=[(f,v['high']-v['random']) for f,v in grouped.items() if 'high' in v and 'random' in v]
        if pairs:
            fig,ax=basic(f'V9 Test Pilot — Deletion Advantage by MRI (n={len(pairs)})','Test pilot MRI index','High minus random decrease in target margin')
            idx=np.arange(1,len(pairs)+1)
            ys=np.asarray([v for _,v in pairs]);ax.bar(idx,ys,color=np.where(ys>=0,C['low'],C['raw']))
            ax.axhline(0,color='#555555',lw=1.2);ax.set_xticks(idx)
            save(fig,dst,'06_faithfulness_20pct_per_image_test.png',index,str(trialfile.relative_to(results)),
                 'Descriptive 16-image pilot; each bar corresponds to one matched test MRI')


def stability_plots(results,dst,index):
    labels={'noise_std_0.01':'Small noise','brightness_plus_0.02':'Brightness +0.02','shift_2px_right':'Shift right 2px'}
    for split in ['test','val']:
        path=results/f'14_stability_{split}'/'stability_summary.json'
        summ=data(path)
        if not summ:continue
        rs=summ.get('results',[])
        for metric,display,scale in [('mean_top20_iou','Top-20% Grad-CAM overlap',100),('mean_cosine','Cosine similarity',1)]:
            fig,ax=basic(f'V9 Explanation Stability — {display} ({split}, n={summ.get("examples","?")})',
                         'Small MRI change','Overlap (%)' if scale==100 else 'Similarity (1 = identical)')
            xs=np.arange(len(labels));barw=.33; plotted=False
            for j,layer in enumerate(['final','earlier']):
                lookup={r.get('perturbation'):num(r.get(metric))*scale for r in rs if r.get('layer')==layer}
                yy=np.asarray([lookup.get(q,float('nan')) for q in labels])
                if np.any(np.isfinite(yy)):
                    ax.bar(xs+(j-.5)*barw,yy,width=barw,label=f'{layer.title()} layer',color=C[layer]);plotted=True
            ax.set_xticks(xs,list(labels.values()))
            ax.set_ylim(0,105 if scale==100 else 1.08)
            ax.legend()
            if plotted:save(fig,dst,f'07_stability_{split}_{metric}.png',index,str(path.relative_to(results)),
                            'Averaged over selected MRIs; heatmaps inverse-aligned for 2px shift')
            else:plt.close(fig)


def uncertainty_plots(results,dst,index):
    mcfile=results/'09_uncertainty'/'uncertainty.csv';mc=rows(mcfile)
    if mc:
        yes=valid([num(r.get('predictive_entropy')) for r in mc if str(r.get('correct')) in ('1','True','true')]);
        no=valid([num(r.get('predictive_entropy')) for r in mc if str(r.get('correct')) in ('0','False','false')])
        if len(yes) and len(no):
            fig,ax=basic('V9 Predictive Entropy — Correct vs Wrong MC Predictions','MC prediction outcome','Predictive entropy')
            ax.boxplot([yes,no],showfliers=True,
                       patch_artist=True,boxprops={'facecolor':'#D7E6F1'})
            ax.set_xticks([1,2],[f'Correct (n={len(yes)})',f'Wrong (n={len(no)})'])
            ax.set_ylim(bottom=0)
            save(fig,dst,'08_predictive_entropy_correct_vs_wrong.png',index,str(mcfile.relative_to(results)),
                 'Entropy is of mean 20-pass class probabilities; boxplots, not percentages')
        truewrong=np.asarray([int(str(r.get('correct')) in ('0','False','false')) for r in mc])
        pred=valid([num(r.get('predictive_entropy')) for r in mc])
        if len(pred)==len(truewrong) and len(np.unique(truewrong))==2:
            from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score
            fpr,tpr,_=roc_curve(truewrong,pred)
            fig,ax=basic('V9 Prediction-Uncertainty Detection of Classification Errors',
                         'False positive rate','True positive rate')
            ax.plot(fpr,tpr,lw=2.3,color=C['final'],label=f'Entropy ROC AUC = {auc(fpr,tpr):.3f}')
            ax.plot([0,1],[0,1],ls='--',color='gray');ax.set(xlim=(0,1),ylim=(0,1));ax.legend()
            save(fig,dst,'09_entropy_error_detection_roc.png',index,str(mcfile.relative_to(results)),
                 'Post-hoc diagnostic for detecting model errors on test set; not a validated screening threshold')
            pr,rec,_=precision_recall_curve(truewrong,pred)
            fig,ax=basic('V9 Error Detection — Precision/Recall from Predictive Entropy',
                         'Recall (proportion of errors found)','Precision (fraction of flags that were errors)')
            ax.plot(rec,pr,lw=2.3,color=C['final'],label=f'Average precision = {average_precision_score(truewrong,pred):.3f}')
            ax.axhline(truewrong.mean(),ls='--',color='gray',label=f'Error prevalence = {truewrong.mean():.1%}')
            ax.set(xlim=(0,1),ylim=(0,1));ax.legend()
            save(fig,dst,'10_entropy_error_detection_pr.png',index,str(mcfile.relative_to(results)),
                 'Evaluation on previously inspected test set; not patient-independent external validation')
    exfile=results/'15_explanation_uncertainty_test_final'/'explanation_uncertainty.csv'
    ex=rows(exfile)
    if ex:
        gp=defaultdict(list)
        for r in ex:
            x=num(r.get('mean_shape_attribution_std'))
            if np.isfinite(x):gp[r.get('true_class','unknown')].append(x)
        order=[x for x in CLASSES if gp.get(x)]
        if order:
            fig,ax=basic('V9 MC Grad-CAM Shape Uncertainty — 16 Test MRIs', 'MRI class',
                         'Mean standard deviation of normalized attribution')
            means=[np.mean(gp[k]) for k in order]
            xs=np.arange(len(order))
            ax.bar(xs,means,width=.55,color='#4D83B9',alpha=.8)
            for i,k in enumerate(order):
                samples=np.asarray(gp[k]);ax.scatter(np.full(len(samples),i)+np.linspace(-.12,.12,len(samples)),samples,
                                                     color='#C8553D',zorder=3,s=30)
            ax.set_xticks(xs,['No Tumor' if k=='notumor' else k.title() for k in order])
            ax.set_ylim(bottom=0)
            save(fig,dst,'11_explanation_uncertainty_by_class.png',index,str(exfile.relative_to(results)),
                 'Four MRIs per class; bar=class mean, dots=individual MRI results')


def integrated_plots(results,dst,index):
    path=results/'16_integrated_test_final'/'per_image_reliability_profile.csv'
    rs=rows(path)
    if not rs:return
    keypairs=[('predictive_entropy','explanation_shape_uncertainty',
               'Prediction Entropy vs Explanation Shape Uncertainty','Predictive entropy','CAM shape standard deviation'),
              ('faithfulness_deletion_high_minus_random','explanation_shape_uncertainty',
               'Faithfulness vs Explanation Shape Uncertainty','High minus random deletion effect','CAM shape standard deviation')]
    for j,(kx,ky,title,xlab,ylab) in enumerate(keypairs,12):
        pairs=[(num(r.get(kx)),num(r.get(ky))) for r in rs]
        pairs=[(x,y) for x,y in pairs if np.isfinite(x) and np.isfinite(y)]
        if len(pairs)<3:continue
        x=np.asarray([p[0] for p in pairs]);y=np.asarray([p[1] for p in pairs])
        from scipy.stats import spearmanr
        rho=float(spearmanr(x,y).statistic) if len(np.unique(x))>1 and len(np.unique(y))>1 else float('nan')
        fig,ax=basic(f'V9 Pilot — {title}',xlab,ylab)
        ax.scatter(x,y,s=65,color='#246B9B',edgecolor='white',lw=.7)
        ax.text(.98,.96,f'n={len(pairs)}\nSpearman rho={rho:+.3f}',transform=ax.transAxes,
                ha='right',va='top',bbox={'boxstyle':'round','fc':'#EEF2F6','ec':'none'})
        save(fig,dst,f'{j:02d}_integrated_{kx}_vs_{ky}.png',index,str(path.relative_to(results)),
             'Exploratory 16-MRI pilot; correlations do not establish causal or clinical usefulness')


def roc_pr_by_class(results,dst,index):
    path=results/'12_calibration'/'per_image_test_calibration.csv'
    rs=rows(path)
    if not rs:return
    colnames=[f'raw_prob_{c}' for c in CLASSES]
    if not all(k in rs[0] for k in colnames):
        print('[SKIP] class ROC/PR: saved probabilities missing');return
    y=np.asarray([r.get('true_class') for r in rs]);score=np.array([[num(r.get(k)) for k in colnames] for r in rs])
    if not np.all(np.isfinite(score)) or not set(y).issubset(set(CLASSES)):
        print('[SKIP] class ROC/PR: invalid saved per-MRI data');return
    n_test=actual_test_count(results)
    if len(rs)!=n_test:
        raise ValueError(f'V9 test calibration should have {n_test} images, found {len(rs)}. Do not mix datasets.')
    from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score
    fig,ax=basic('V9 Test — One-vs-Rest ROC Curves (Raw Probabilities)','False positive rate','True positive rate')
    vals=[]
    for i,c in enumerate(CLASSES):
        t=(y==c).astype(int)
        if len(np.unique(t))<2:continue
        fpr,tpr,_=roc_curve(t,score[:,i]);v=auc(fpr,tpr);vals.append(v)
        ax.plot(fpr,tpr,lw=2,label=f'{c.title()} AUC={v:.3f}')
    ax.plot([0,1],[0,1],ls='--',color='gray');ax.set(xlim=(0,1),ylim=(0,1));ax.legend()
    save(fig,dst,'14_v9_classification_roc.png',index,str(path.relative_to(results)),
         f'Post-hoc one-versus-rest ROC, macro AUC={np.mean(vals):.4f}; based on the V9 test images')
    fig,ax=basic('V9 Test — One-vs-Rest Precision–Recall Curves','Recall','Precision')
    for i,c in enumerate(CLASSES):
        t=(y==c).astype(int)
        if len(np.unique(t))<2:continue
        prec,rec,_=precision_recall_curve(t,score[:,i]);ap=average_precision_score(t,score[:,i])
        ax.plot(rec,prec,lw=2,label=f'{c.title()} AP={ap:.3f}')
    ax.set(xlim=(0,1),ylim=(0,1));ax.legend(loc='lower left')
    save(fig,dst,'15_v9_classification_pr.png',index,str(path.relative_to(results)),
         'Test-set one-versus-rest precision–recall; AP is not classification accuracy')



def comparison_confusion_plots(results,dst,index):
    """Standard vs 20-pass MC: per-MRI files ONLY, paired by exact MRI ID.

    Never infer the MC confusion matrix from accuracy; it must be calculated
    from 09_uncertainty/uncertainty.csv. No model inference or retraining.
    """
    single_path=results/'08_test'/'per_image_predictions.csv'
    mc_path=results/'09_uncertainty'/'uncertainty.csv'
    single=rows(single_path)
    mc=rows(mc_path)
    needed={'file','true_class','predicted_class'}
    if not single or not mc:
        print('[SKIP] paired confusion matrices: require both '+str(single_path)+' AND '+str(mc_path))
        return
    if not needed.issubset(single[0]) or not needed.issubset(mc[0]):
        print('[SKIP] paired confusion matrices: expected file, true_class, predicted_class columns in both CSVs')
        return
    def to_dict(rs,label):
        out={}
        for r in rs:
            file=r['file'].strip()
            if not file or file in out:
                raise ValueError(f'{label}: blank/duplicate MRI file identity: {file!r}')
            truth=r['true_class'].strip().lower()
            pred=r['predicted_class'].strip().lower()
            if truth not in CLASSES or pred not in CLASSES:
                raise ValueError(f'{label}: unsupported class for MRI {file}: {truth}/{pred}')
            out[file]=(truth,pred)
        return out
    a=to_dict(single,'Standard model');b=to_dict(mc,'MC Dropout')
    n_test=actual_test_count(results)
    if len(a)!=n_test or len(b)!=n_test:
        raise ValueError(f'Model comparison requires V9 {n_test} test MRIs; standard={len(a)}, MC={len(b)}. '
                         'Do not mix with other experiments.')
    if set(a)!=set(b):
        raise ValueError(f'Standard and MC files do not contain identical MRIs: '
                         f'only in standard={len(set(a)-set(b))}, only in MC={len(set(b)-set(a))}')
    wrong_truth=[f for f in a if a[f][0]!=b[f][0]]
    if wrong_truth:
        raise ValueError(f'True labels mismatch between two test predictions for {len(wrong_truth)} MRIs; '
                         'cannot fairly compare.')
    from sklearn.metrics import confusion_matrix
    ids=sorted(a)
    true=[a[f][0] for f in ids]
    std=[a[f][1] for f in ids]
    stochastic=[b[f][1] for f in ids]
    cm_standard=confusion_matrix(true,std,labels=list(CLASSES))
    cm_mc=confusion_matrix(true,stochastic,labels=list(CLASSES))
    std_correct=int(np.trace(cm_standard))
    mc_correct=int(np.trace(cm_mc))
    saved_std=data(results/'08_test'/'test_metrics.json')
    if saved_std:
        expected=saved_std.get('confusion_matrix')
        if expected is not None and not np.array_equal(cm_standard,np.asarray(expected)):
            raise ValueError('Saved standard Step 08 confusion matrix differs from per-image CSV!')
        if saved_std.get('correct') is not None and int(saved_std['correct'])!=std_correct:
            raise ValueError('Step 08 correct count differs from saved per-image predictions!')
    saved_mc=data(results/'09_uncertainty'/'mc_summary.json')
    if saved_mc:
        if saved_mc.get('wrong_predictions') is not None and int(saved_mc['wrong_predictions'])!=n_test-mc_correct:
            raise ValueError('Step 09 MC wrong count differs from saved per-image predictions!')
    from matplotlib.colors import Normalize
    names=['Glioma','Meningioma','No Tumor','Pituitary']
    def draw_cm(ax,cm,title,percent):
        mat=cm / np.maximum(cm.sum(axis=1,keepdims=True),1)*100 if percent else cm
        vmax=100 if percent else max(int(cm_standard.max()),int(cm_mc.max()))
        im=ax.imshow(mat,cmap='Blues',norm=Normalize(vmin=0,vmax=vmax))
        ax.set_xticks(range(4),names,rotation=32,ha='right',fontsize=9)
        ax.set_yticks(range(4),names,fontsize=9)
        ax.set_xlabel('Predicted class');ax.set_ylabel('Actual class')
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                value=mat[i,j]
                ax.text(j,i,f'{value:.1f}%' if percent else str(int(value)),
                        ha='center',va='center',fontsize=10,
                        color='white' if value>vmax*.54 else '#17283A',fontweight='bold' if i==j else 'normal')
        return im

    for cm,title,filename in [(cm_standard,f'V9 Standard Classification — Confusion Matrix (n={n_test})',
                               '16_v9_standard_confusion_counts.png'),
                              (cm_mc,f'V9 20-pass MC Dropout — Confusion Matrix (n={n_test})',
                               '17_v9_mc_dropout_confusion_counts.png')]:
        fig,ax=plt.subplots(figsize=(8.1,6.7)); im=draw_cm(ax,cm,title,False)
        fig.colorbar(im,ax=ax,fraction=.045,pad=.045,label='MRI count')
        save(fig,dst,filename,index,
             '08_test/per_image_predictions.csv; 09_uncertainty/uncertainty.csv',
             'Rows are actual labels, columns are predictions; these are DIFFERENT inference methods on the SAME test MRIs')

    fig,axes=plt.subplots(1,2,figsize=(14.8,6.3))
    draw_cm(axes[0],cm_standard,f'Standard: {std_correct}/{n_test} ({std_correct/n_test:.2%})',True)
    im=draw_cm(axes[1],cm_mc,f'MC Dropout: {mc_correct}/{n_test} ({mc_correct/n_test:.2%})',True)
    fig.colorbar(im,ax=axes.tolist(),fraction=.027,pad=.03,label='Percent of each true class')
    save(fig,dst,'18_v9_standard_vs_mc_confusion_percent.png',index,
         '08_test/per_image_predictions.csv; 09_uncertainty/uncertainty.csv',
         'Paired row-normalized confusion matrices; same test images, with matched IDs and truth labels')

    fig,ax=basic('V9 Test — Standard vs 20-pass MC Dropout Accuracy','Prediction method','Test MRI images (count)')
    nstd_wrong=n_test-std_correct;nmc_wrong=n_test-mc_correct
    xx=np.arange(2)
    ax.bar(xx,[std_correct,mc_correct],color='#269B80',label='Correct')
    ax.bar(xx,[nstd_wrong,nmc_wrong],bottom=[std_correct,mc_correct],color='#C8553D',label='Wrong')
    ax.set_xticks(xx,[f'Standard: {std_correct/n_test:.2%}',f'MC Dropout: {mc_correct/n_test:.2%}'])
    ax.set_ylim(0,n_test*1.12)
    for i,(co,wr) in enumerate([(std_correct,nstd_wrong),(mc_correct,nmc_wrong)]):
        ax.text(i,co/2,str(co),ha='center',va='center',color='white',fontweight='bold')
        ax.text(i,co+wr+18,f'{wr} wrong',ha='center',va='bottom',color='#C8553D')
    ax.legend()
    save(fig,dst,'19_v9_standard_vs_mc_correct_wrong.png',index,
         '08_test/per_image_predictions.csv; 09_uncertainty/uncertainty.csv',
         'Accuracy on the same V9 test MRIs; does not describe uncertainty quality')

    both_correct=sum(a[f][1]==a[f][0] and b[f][1]==b[f][0] for f in ids)
    std_only_correct=sum(a[f][1]==a[f][0] and b[f][1]!=b[f][0] for f in ids)
    mc_only_correct=sum(a[f][1]!=a[f][0] and b[f][1]==b[f][0] for f in ids)
    both_wrong=n_test-both_correct-std_only_correct-mc_only_correct
    fig,ax=basic('V9 Paired Errors — Did MC Dropout Correct or Introduce Mistakes?',
                 'Outcome for the same test MRI','Number of MRI images')
    labels=['Both correct','Standard only\ncorrect','MC only\ncorrect','Both wrong']
    vals=[both_correct,std_only_correct,mc_only_correct,both_wrong]
    colors=['#269B80','#C8553D','#4D83B9','#9B7A92']
    xx=np.arange(4)
    bars=ax.bar(xx,vals,color=colors,width=.62)
    ax.set_xticks(xx,labels);ax.set_ylim(0,max(vals)*1.15)
    ax.bar_label(bars,padding=4)
    save(fig,dst,'20_v9_standard_mc_error_overlap.png',index,
         '08_test/per_image_predictions.csv; 09_uncertainty/uncertainty.csv',
         'Paired outcome counts; identifies how many predictions change in correctness, without retraining')
    summary={
      'test_count':len(ids),'classes':list(CLASSES),
      'single_pass_correct':std_correct,'single_pass_wrong':n_test-std_correct,
      'single_pass_accuracy':std_correct/n_test,
      'mc_dropout_correct':mc_correct,'mc_dropout_wrong':n_test-mc_correct,
      'mc_dropout_accuracy':mc_correct/n_test,
      'standard_confusion_matrix':cm_standard.tolist(),
      'mc_dropout_confusion_matrix':cm_mc.tolist(),
      'both_correct':both_correct,'standard_correct_mc_wrong':std_only_correct,
      'standard_wrong_mc_correct':mc_only_correct,'both_wrong':both_wrong,
      'source_files':[str(single_path.relative_to(results)),str(mc_path.relative_to(results))],
      'interpretation':'Retrospective, saved V9 test predictions only. No extra inference or patient-level validation.'
    }
    (dst/'CONFUSION_MATRIX_COMPARISON.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'[CHECK] standard={std_correct}/{n_test}, MC Dropout={mc_correct}/{n_test}; '
          f'both wrong={both_wrong}; standard-only-correct={std_only_correct}; MC-only-correct={mc_only_correct}')
    print('[SAVED]',dst/'CONFUSION_MATRIX_COMPARISON.json')

def main():
    import argparse
    p=argparse.ArgumentParser(description='Create Step 20 post-hoc V9 research figures; no training.')
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
    a=p.parse_args()
    root=a.project_root.expanduser().resolve();result=root/'results';dst=result/'20_more_thesis_figures'
    if not result.is_dir():raise FileNotFoundError(f'V9 results directory not found: {result}')
    dst.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.spines.top':False,'axes.spines.right':False})
    index=[]
    calibration_plots(result,dst,index)
    faithfulness_plots(result,dst,index)
    stability_plots(result,dst,index)
    uncertainty_plots(result,dst,index)
    integrated_plots(result,dst,index)
    roc_pr_by_class(result,dst,index)
    comparison_confusion_plots(result,dst,index)
    idx=dst/'FIGURE_INDEX.txt'
    header=['V9 ADDITIONAL RESEARCH FIGURES — STEP 20',
            'Only existing saved CSV/JSON used; no model inference, training, or data changes.',
            'WARNING: These figures refer to the V9 test experiment and small reliability pilots.',
            'V9 uses its own deduplicated and group-preserving test split. Do not mix with earlier V8/V5 experiments.',
            'Pilot explanation metrics: 16 matched test MRIs; not clinically validated.',
            '']
    idx.write_text('\n'.join(header+[f'{name}\n  Source: {source}\n  Note: {note}\n' for name,source,note in index]),encoding='utf-8')
    print('\nSTEP 20 FINISHED — figures created:',len(index))
    print('Folder:',dst)
    print('Index:',idx)
    if not index:
        print('No usable saved data. Inspect your project output paths; NOTHING was fabricated.')

if __name__=='__main__':main()
