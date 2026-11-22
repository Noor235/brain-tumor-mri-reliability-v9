"""NeuroScope V9 — five-metric research demonstration.
Run inside ORIGINAL Brain_Tumor_Reproduction_V9 folder:
    streamlit run v9_final_reliability_app.py
"""
import csv
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from src import config as C
from src.common import source_image
from src.v9_live_reliability import load_v9, analyze, explain_metrics, warning_message, overlay_pic
from src.step23_warning_validation import F_COLS, score_with_model
from src.step22_full_test_reliability import project_seed

st.set_page_config(page_title='NeuroScope V9 | Research Reliability',page_icon='🧠',layout='wide')
st.markdown('''<style>
.stApp{background:#0b1220;color:#e9eef8}
.block-container{max-width:1150px;padding-top:2rem}
[data-testid="stMetricValue"]{color:#e5f3ff}
div[data-testid="stAlert"]{border-radius:10px}
</style>''', unsafe_allow_html=True)

ROOT=C.ROOT
ART=C.OUTPUT/'23_reliability_warning'

@st.cache_resource(show_spinner='Loading original V9 checkpoint…')
def trained_model():
    return load_v9()

@st.cache_data(show_spinner=False)
def saved_data():
    p=C.OUTPUT/'gradcam_features4_full_test/earlier/per_image_reliability_profile.csv'
    cal=C.OUTPUT/'12_calibration/per_image_test_calibration.csv'
    if not p.is_file() or not cal.is_file():
        raise FileNotFoundError('Missing original Step 22 corrected results or calibration CSV')
    profile=pd.read_csv(p)
    cdf=pd.read_csv(cal)
    merged=profile.merge(cdf[['file']+[f'calibrated_prob_{c}' for c in C.CLASSES]],on='file',validate='one_to_one')
    if len(merged)!=1374:raise ValueError('Saved profile is not the full 1,374 MRI set')
    return merged

@st.cache_data(show_spinner=False)
def reference_data():
    q=json.loads((ART/'reference_quantiles.json').read_text())
    model=json.loads((ART/'warning_model.json').read_text())
    summary=json.loads((ART/'comparison_summary.json').read_text())
    return q,model,summary


def warning_scores(metrics, learned):
    X=np.array([[float(metrics[c]) for c in F_COLS]])
    return {name:float(score_with_model(X, learned[name])[0])
            for name in ['Confidence only','Confidence + MC uncertainty','All five measurements']}


def show_analysis(image,values,known_label, source_kind, reference, learned, summary):
    st.divider()
    st.caption('This is an experimental classifier, NOT a diagnostic device. MRI labels must be independently verified.')
    a,b,c=st.columns(3)
    a.metric('V9 predicted category',values['predicted_class'].replace('notumor','No tumour').title())
    b.metric('Calibrated confidence',f'{float(values["calibrated_confidence"]):.1%}')
    c.metric('MC predictive entropy',f'{float(values["mc_predictive_entropy"]):.4f}')
    title,message,concerns=warning_message(values,known_label,reference)
    if title=='Known label mismatch':st.error(f'**{title}** — {message}',icon='🚨')
    elif concerns:st.warning(f'**{title}** — {message}',icon='⚠️')
    else:st.info(f'**{title}** — {message}',icon='ℹ️')

    # Step 24: validation-only MC uncertainty cutoff (research demonstration).
    # Separate from the individual five metrics and from any known-label check.
    st.subheader('Step 24 — Uncertainty referral recommendation')
    try:
        refpath=C.OUTPUT/'24_uncertainty_selective'/'thresholds.json'
        refcfg=json.loads(refpath.read_text(encoding='utf-8'))
        threshold=float(refcfg['primary_threshold'])
        entropy=float(values['mc_predictive_entropy'])
        if not np.isfinite(entropy):
            raise ValueError('Predictive entropy is non-finite')
        flagged=entropy>threshold
        if flagged:
            st.warning('**Refer for independent review.** MC Dropout uncertainty exceeds the validation-selected threshold. This does not establish that the prediction is wrong.',icon='⚠️')
        else:
            st.info('**No uncertainty-based referral triggered.** This does not establish that the prediction is correct; some incorrect V9 test predictions were not flagged.',icon='ℹ️')
        st.caption(f'MC predictive entropy = {entropy:.6f}; threshold = {threshold:.6f} '
                   '(90th percentile of 1,099 validation images). '
                   'This retrospective, research-only threshold is not clinically validated.')
    except (OSError,KeyError,TypeError,ValueError) as exc:
        st.error(f'Uncertainty referral threshold unavailable ({exc}); do not infer reliability.')
        flagged=None

    st.subheader('Five reliability measurements')
    interpretation=explain_metrics(values,reference)
    for name,(status,detail) in interpretation.items():
        with st.container(border=True):
            left,right=st.columns([2,1])
            left.markdown(f'**{name}**')
            right.markdown(f'**{status}**')
            st.caption(detail)

    try:
        if any(not np.isfinite(float(values[k])) for k in F_COLS):
            raise ValueError('At least one required measurement is not finite')
        sc=warning_scores(values,learned)
    except (ValueError,KeyError) as exc:
        st.error(f'Cannot calculate combined research warning score: {exc}. Treat this as needing review.')
        sc={}
    st.subheader('Experimental error-warning comparison')
    st.caption('The following are relative ranking scores learned from the previously examined V9 test cohort. They are **not** probabilities of an incorrect diagnosis.')
    if sc:
        cols=st.columns(3)
        for i,key in enumerate(['Confidence only','Confidence + MC uncertainty','All five measurements']):
            cols[i].metric(key,f'{sc[key]:.3f}')
    vals=summary['comparisons']
    st.caption('Direct confidence ranking ROC-AUC '
               
               f'{vals["Direct confidence (no training)"]["roc_auc"]:.3f}; '
               
               'internal group-cross-fitted learned scores: confidence only '
               
               f'{vals["Confidence only"]["roc_auc"]:.3f}, confidence + MC '
               
               f'{vals["Confidence + MC uncertainty"]["roc_auc"]:.3f}, all five '
               
               f'{vals["All five measurements"]["roc_auc"]:.3f}. The all-five model did not outperform simpler alternatives.')

    st.subheader('MRI and corrected Grad-CAM')
    col1,col2=st.columns(2)
    with col1:
        st.image(image,caption='Input MRI (source image)',use_container_width=True)
    with col2:
        if values.get('overlay') is not None:
            st.image(values['overlay'],caption='Grad-CAM from features[-4] (calculated now)',use_container_width=True)
        else:
            st.info('This is a saved Step 22 profile. To produce a fresh heatmap for this image, select Live analysis below.')

    if values.get('probabilities'):
        st.subheader('Calibrated class probabilities')
        p=pd.DataFrame({'Class':[c.replace('notumor','No tumour').title() for c in C.CLASSES],
                        'Probability':[values['probabilities'][c] for c in C.CLASSES]})
        st.bar_chart(p.set_index('Class'))
    export={k:(v.item() if isinstance(v,np.generic) else v) for k,v in values.items()
            if k not in ('overlay','heatmap') and not isinstance(v,Image.Image)}
    export.update({'source':source_kind,'independently_supplied_label':known_label,
                   'research_assessment':title,'research_warnings':concerns,
                   'step24_uncertainty_referral': ('refer_for_review' if flagged else 'not_referred') if flagged is not None else 'unavailable',
                   'exploratory_relative_warning_scores':sc,
                   'disclaimer':'Not clinically validated; an apparently favourable set of metrics does not prove correctness.'})
    st.download_button('Download MRI research report (JSON)',data=json.dumps(export,indent=2),
                       file_name='V9_MRI_research_report.json',mime='application/json')

st.title('NeuroScope V9')
st.markdown('**Brain Tumour MRI Classification · Five-Dimension Reliability Assessment**')
st.warning('RESEARCH USE ONLY. Do not use the prediction or reliability measurements for clinical decisions or to rule out a tumour.',icon='⚠️')

try:
    reference,learned,comparison=reference_data()
except Exception as e:
    st.error(f'Install the supplied results/23_reliability_warning/ folder into the original V9 project. Details: {e}')
    st.stop()

mode=st.radio('Choose how to test the system',
    ['Random held-out test MRI (saved study results)','Upload a new MRI (live five-metric calculation)'],
    horizontal=False)

if mode.startswith('Random'):
    st.caption('This mode retrieves a real V9 test MRI and its previously computed five metrics. The dataset class label is available for checking correctness.')
    try:
        df=saved_data()
    except Exception as e:
        st.error(str(e));st.stop()
    if 'rand_file' not in st.session_state:
        st.session_state.rand_file=random.choice(df['file'].tolist())
    if st.button('Pick another random MRI',type='primary'):
        st.session_state.rand_file=random.choice(df['file'].tolist())
    row=df.loc[df['file']==st.session_state.rand_file].iloc[0].to_dict()
    try:
        img=Image.open(source_image(row['file'])).convert('RGB')
    except Exception as e:
        st.error(f'Original V9 MRI image is missing. Check the data/brain-tumor-mri folder. {e}');st.stop()
    metrics={k:(float(row[k]) if k in F_COLS or k in ('raw_confidence',) else row[k]) for k in row}
    metrics['predicted_class']=row['deterministic_predicted_class']
    metrics['probabilities']={c:float(row[f'calibrated_prob_{c}']) for c in C.CLASSES}
    st.caption(f'File: `{row["file"]}` · Assigned V9 test split · Reference class: {row["true_class"]}')
    show_analysis(img,metrics,row['true_class'],'saved V9 test record',reference,learned,comparison)
    with st.expander('Recalculate a fresh corrected Grad-CAM for this image'):
        st.caption('Calculates prediction and five metrics again using the actual checkpoint. This can differ slightly from the stored stochastic measurements.')
        if st.button('Run live analysis on selected test MRI'):
            with st.spinner('Computing 20 stochastic predictions and 20 stochastic Grad-CAM maps…'):
                try:
                    model,dev,temp=trained_model()
                    v=analyze(img,model,dev,temp,seed=project_seed(row['file'],'earlier'))
                    show_analysis(img,v,row['true_class'],'fresh inference on V9 test image',reference,learned,comparison)
                except Exception as e:
                    st.exception(e)
else:
    st.caption('Upload a single JPG/PNG MRI. A phone photograph of a printed scan can differ substantially from the MRI images used for training.')
    uploaded=st.file_uploader('Upload an MRI',type=['png','jpg','jpeg','bmp','tif','tiff','webp'])
    photo=st.camera_input('Or capture a photograph of an MRI (research demonstration only)')
    reference_label=st.selectbox('Independently known tumour label (optional)',
            ['Unknown / not independently verified']+C.CLASSES)
    if (uploaded is not None or photo is not None) and st.button('Analyze MRI',type='primary'):
        try:
            src=uploaded if uploaded is not None else photo
            img=Image.open(src).convert('RGB')
            if min(img.size)<64:raise ValueError('Input image is too small for this model')
            with st.spinner('Running V9 inference, 20 MC passes and corrected Grad-CAM reliability analysis…'):
                model,dev,temp=trained_model()
                metrics=analyze(img,model,dev,temp)
            label=reference_label if reference_label in C.CLASSES else None
            st.session_state['last_v9_result']=(img.copy(),metrics,label)
        except Exception as exc:st.exception(exc)
    if 'last_v9_result' in st.session_state:
        img,metrics,label=st.session_state['last_v9_result']
        show_analysis(img,metrics,label,'live uploaded MRI',reference,learned,comparison)

with st.expander('Study methodology and evaluation limitations'):
    st.write('Original V9 model: EfficientNet-B4, grayscale 224×224, temperature scaling on validation data. '
             'Grad-CAM layer: features[-4]. Faithfulness: high-vs-random 20% patch deletion; '
             'stability: noise, brightness, 2-pixel shift; explanation uncertainty: variation across 20 stochastic heatmaps. '
             'All-five warning model: exploratory cross-fitting on previously examined V9 test data (29 errors). '
             'The same test cohort also influenced earlier layer decisions; no untouched external evaluation exists. '
             'Neither five favourable metrics nor a low warning score establishes diagnosis correctness.')
    st.image(str(ART/'cross_fitted_roc.png'),caption='Internal cross-fitted error-warning model ROC curves')
