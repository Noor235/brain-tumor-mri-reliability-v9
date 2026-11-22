"""BrainLens | Calm white + baby blue MRI research reliability website.

Run from your original V9 project root with:
    python3 -m streamlit run brainlens_v9.py
The trained V9 model is loaded read-only; uploaded images are never saved.
"""
from __future__ import annotations

import html
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

from src import config as C
from src.common import source_image
from src.v9_web_hirescam import (
    MC_REFERRAL_ENTROPY, analyze_new_mri, light_heatmap, load_v9,
)

st.set_page_config(
    page_title="BrainLens | MRI Reliability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown('''
<style>
:root { color-scheme: light; }
.stApp { background: #F8FCFF; color: #243A50; }
[data-testid="stAppViewContainer"] { background: #F8FCFF; }
[data-testid="stHeader"] { background: #F8FCFF !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1080px; }
h1, h2, h3, p, label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp h1 { color: #1D3E58; letter-spacing: -.035em; font-size: 2.4rem; }
.stApp h2 { color: #244764; font-size: 1.3rem; }
.stApp h3 { color: #244764; font-size: 1rem; }
.stApp p { color: #506A80; }
div.stButton > button[kind="primary"] { background: #8AC7EA; border: 1px solid #8AC7EA;
  color:#153A56; font-weight:700; border-radius: 12px; padding: .6rem 1.35rem; }
div.stButton > button[kind="primary"]:hover { background: #72BAE5; border-color: #72BAE5; color:#153A56; }
div.stButton > button { border-radius: 12px; border:1px solid #CDE2F1; color:#24516E; background:white; }
[data-testid="stFileUploader"], [data-testid="stSelectbox"] { border-radius: 14px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #E0EDF7 !important; border-radius: 17px !important;
 background: #FFFFFF; }
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stExpander"] { background: white; border: 1px solid #E0EDF7; border-radius: 14px; }
[data-testid="stAlert"] { border-radius:14px; }
hr { border-color: #EAF2F8; }
.hero { border:1px solid #D9ECF8; border-radius: 23px; padding: 27px 33px;
 background: linear-gradient(110deg, #FFFFFF 5%, #F0F9FF 80%, #E6F4FD 100%); margin-bottom: 15px; }
.hero-title { color:#254A66; font-size:36px; letter-spacing:-.045em; font-weight:800; line-height:1.12; }
.hero-sub { color:#597990; font-size:14px; line-height:1.6; margin-top:11px; max-width:600px; }
.hero-pill { display:inline-block; border:1px solid #C8E5F4; background:#EAF8FF; border-radius:99px;
 color:#437D9E; font-weight:700; font-size:11px; padding:5px 12px; margin-bottom:14px; }
.small-label { font-size:11px; text-transform:uppercase; letter-spacing:.13em; color:#7E9DB2; font-weight:800; }
.result-name {font-size:29px; color:#274961; font-weight:800; letter-spacing:-.03em; margin:1px 0 8px;}
.result-confidence { font-size:27px; color:#4F9DC9; font-weight:800; }
.result-caption {font-size:12px;color:#7390A4; }
.mrow { border-bottom:1px solid #EAF2F8; padding:13px 0 14px; }
.mrow:last-child { border:0; padding-bottom:0; }
.mtitle { color:#2E4A61; font-weight:700; font-size:14px; }
.mnumber { color:#245B82; font-size:15px; font-weight:800; margin-top:4px; }
.mnote { color:#71889B; font-size:12px; line-height:1.52; margin-top:4px; }
.soft-tag { display:inline-block; font-size:11px; font-weight:700; background:#EFF8FE;
 color:#337CA5; border:1px solid #D9ECF9; border-radius:99px; padding:3px 9px; }
.soft-tag.alert { background:#FFF5E8; color:#A76A2A; border-color:#F4E2CA; }
.soft-tag.negative { background:#FFF0EF; color:#B25858; border-color:#F4D6D3; }
.pbar { width:100%;height:9px;background:#EBF4FA;border-radius:30px;overflow:hidden; }
.pbar > span { display:block;height:100%;background:#8AC7EA;border-radius:30px; }
.probrow { margin: 10px 0 14px; }
.problabel { display:flex;justify-content:space-between;font-size:12px;color:#547085;margin-bottom:5px; }
.footerline {color:#91AABC;text-align:center;font-size:11px;margin-top:35px;}
</style>
''', unsafe_allow_html=True)

# This is static decorative vector markup, not a generated patient image.
BRAIN_SVG = '''<svg width="151" height="145" viewBox="0 0 170 164" fill="none" xmlns="http://www.w3.org/2000/svg"
 aria-hidden="true"><circle cx="85" cy="82" r="76" fill="#E4F4FD"/>
<path d="M87 127V43c-6-15-23-19-33-7-14-3-28 7-26 22-14 5-16 22-7 33-5 16 7 29 20 28 7 18 31 22 46 8Z"
 stroke="#74B9DE" stroke-width="4" stroke-linejoin="round" fill="#F7FDFF"/>
<path d="M87 127V43c6-15 23-19 33-7 14-3 28 7 26 22 14 5 16 22 7 33 5 16-7 29-20 28-7 18-31 22-46 8Z"
 stroke="#74B9DE" stroke-width="4" stroke-linejoin="round" fill="#F7FDFF"/>
<path d="M54 36c-6 10-4 21 6 25M24 75c10-4 20 1 25 10m-8 35c-1-11 6-18 17-18m-4-40c-10 1-15 11-13 22m46-13c-10 2-14 10-14 20m55-55c6 10 4 21-6 25m36 14c-10-4-20 1-25 10m8 35c1-11-6-18-17-18m4-40c10 1 15 11 13 22M87 71c10 2 14 10 14 20"
 stroke="#B8DEF2" stroke-width="4" stroke-linecap="round"/>
<circle cx="34" cy="24" r="4" fill="#B5E1F7"/><circle cx="144" cy="138" r="5" fill="#B5E1F7"/></svg>'''

st.markdown(f'''<div class="hero"><div style="display:flex;align-items:center;justify-content:space-between;gap:18px;">
 <div><div class="hero-pill">V9 · Research demonstration</div>
 <div class="hero-title">BrainLens <span style="font-weight:400;color:#6BAED5">MRI</span></div>
 <div class="hero-sub">Explore a brain MRI prediction and understand five different reliability signals.
 Clear explanations, no misleading overall trust score.</div></div>
 <div style="flex:none">{BRAIN_SVG}</div></div></div>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_test_profiles():
    path = C.OUTPUT / '33_hirescam_full_test' / 'earlier' / 'per_image_reliability_profile.csv'
    cal_path = C.OUTPUT / '12_calibration' / 'per_image_test_calibration.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing final HiResCAM Step 33 profiles: {path}')
    if not cal_path.is_file():
        raise FileNotFoundError(f'Missing Step 12 per-image calibration results: {cal_path}')
    df = pd.read_csv(path)
    cal = pd.read_csv(cal_path)
    if len(df) != 1374 or df['file'].nunique() != 1374:
        raise RuntimeError('Step 33 results must have exactly 1,374 distinct V9 test MRI records.')
    required = ['file', 'calibrated_confidence', 'mc_predictive_entropy',
                'faithfulness_del_high_minus_random_20', 'faithfulness_ins_high_minus_random_20',
                'mean_stability_cosine', 'mean_stability_top20_iou',
                'explanation_shape_uncertainty', 'true_class',
                'deterministic_predicted_class', 'deterministic_correct']
    if not set(required).issubset(df.columns):
        raise ValueError('Step 33 profile missing expected reliability fields.')
    prob_cols = [f'calibrated_prob_{cls}' for cls in C.CLASSES]
    if not set(prob_cols).issubset(cal.columns):
        raise ValueError('Step 12 calibration CSV missing per-class calibrated probabilities.')
    if cal['file'].nunique() != len(cal):
        raise ValueError('Duplicate IDs in Step 12 calibration CSV.')
    merged = df.merge(cal[['file'] + prob_cols], on='file', how='inner', validate='one_to_one')
    if len(merged) != 1374 or merged[required].isna().any().any():
        raise RuntimeError('Step 33 and Step 12 test files are inconsistent or incomplete.')
    return merged


@st.cache_resource(show_spinner='Loading your V9 checkpoint…')
def get_model():
    return load_v9()


def titlecase(value):
    return {'glioma':'Glioma','meningioma':'Meningioma','notumor':'No tumour','pituitary':'Pituitary'}.get(value, str(value))


def row_html(title: str, number: str, note: str, status: str, alert=False):
    badge_class = 'alert' if alert else ''
    return (f'<div class="mrow"><div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">'
            f'<span class="mtitle">{html.escape(title)}</span>'
            f'<span class="soft-tag {badge_class}">{html.escape(status)}</span></div>'
            f'<div class="mnumber">{html.escape(number)}</div>'
            f'<div class="mnote">{html.escape(note)}</div></div>')


def metric_rows(m, frame):
    conf = float(m['calibrated_confidence'])
    ent = float(m['mc_predictive_entropy'])
    deletion = float(m['faithfulness_del_high_minus_random_20'])
    insertion = float(m['faithfulness_ins_high_minus_random_20'])
    stability = float(m['mean_stability_cosine'])
    overlap = float(m['mean_stability_top20_iou'])
    expstd = float(m['explanation_shape_uncertainty'])
    s_med = float(frame['mean_stability_cosine'].median())
    u_med = float(frame['explanation_shape_uncertainty'].median())
    u_p75 = float(frame['explanation_shape_uncertainty'].quantile(.75))

    parts = []
    parts.append(row_html('01 · Calibrated confidence', f'{conf:.2%}',
               'Temperature-scaled probability for the predicted class. It is not proof the class is correct.',
               'High confidence' if conf >= .95 else 'Reported confidence', False))
    parts.append(row_html('02 · Prediction uncertainty', f'Entropy {ent:.6f}',
               f'20 MC Dropout passes. Validation-selected referral threshold: {MC_REFERRAL_ENTROPY:.6f}. '
               + ('Above threshold: review suggested.' if ent > MC_REFERRAL_ENTROPY
                  else 'Below threshold: no MC referral; correctness remains unknown.'),
               'Refer for review' if ent > MC_REFERRAL_ENTROPY else 'No MC referral',
               ent > MC_REFERRAL_ENTROPY))
    faith_status = ('Positive in both tests' if deletion > 0 and insertion > 0
                    else 'Mixed evidence')
    parts.append(row_html('03 · Explanation faithfulness',
               f'Deletion {deletion:+.2f} · Insertion {insertion:+.2f}',
               'HiResCAM-highlighted regions versus random regions in fixed 20% perturbation tests. '
               'This checks model influence, not whether the highlight marks a tumour.',
               faith_status, not (deletion > 0 and insertion > 0)))
    parts.append(row_html('04 · Explanation stability', f'Cosine {stability:.4f} · Overlap {overlap:.3f}',
               f'Similarity after small MRI noise, brightness, and shift. Cohort median cosine: {s_med:.4f}. '
               'A stable map may still support an incorrect prediction.',
               'Above cohort median' if stability >= s_med else 'Below cohort median', False))
    parts.append(row_html('05 · Explanation uncertainty', f'Spatial variation {expstd:.6f}',
               f'Variation of 20 normalized HiResCAM maps; cohort median {u_med:.6f}. '
               'Lower variation means more consistent maps, not guaranteed accuracy.',
               'Higher than cohort 75th percentile' if expstd > u_p75
               else 'Within cohort range', expstd > u_p75))
    return ''.join(parts)


def show_summary(m, image, source, frame, known_label=None, heatmap=None):
    prediction = m['predicted_class']
    conf = float(m['calibrated_confidence'])
    ent = float(m['mc_predictive_entropy'])
    st.markdown('### MRI analysis')
    with st.container(border=True):
        a, b = st.columns([1, 1], gap='large')
        with a:
            st.markdown('<div class="small-label">Original MRI</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with b:
            st.markdown('<div class="small-label">HiResCAM explanation</div>', unsafe_allow_html=True)
            if heatmap is not None:
                st.image(heatmap, use_container_width=True)
            else:
                st.info('Heatmap unavailable. The saved measurements remain available.')
        st.caption('The highlighted area influenced the model under this explanation method; it does not prove tumour location.')
        st.divider()
        st.markdown(f'<div class="small-label">Predicted category</div><div class="result-name">'
                    f'{html.escape(titlecase(prediction))}</div><div class="result-confidence">{conf:.2%}'
                    f' <span class="result-caption">calibrated confidence</span></div>',
                    unsafe_allow_html=True)
        if known_label in C.CLASSES:
            if known_label == prediction:
                st.info(f'Matches supplied reference label: {titlecase(known_label)}. This is not independent clinical verification.')
            else:
                st.error(f'Known label mismatch: reference is {titlecase(known_label)}, '
                         f'but V9 predicted {titlecase(prediction)}. Do not rely on this classification.')
        elif ent > MC_REFERRAL_ENTROPY:
            st.warning('Uncertainty referral: the MC Dropout entropy exceeds the validation-selected threshold. '
                       'This is a research warning, not a confirmed error.')
        else:
            st.info('No MC uncertainty referral was triggered. The prediction is **not verified**; '
                    'low uncertainty cannot confirm the correct tumour class.')

    st.markdown('### The five reliability measurements')
    with st.container(border=True):
        st.markdown(metric_rows(m, frame), unsafe_allow_html=True)

    with st.expander('See all four class probabilities'):
        probs = m.get('probabilities') or {}
        for cls in C.CLASSES:
            v = float(probs.get(cls, 0.0))
            label = html.escape(titlecase(cls))
            pct = max(0, min(100, v * 100))
            st.markdown(f'<div class="probrow"><div class="problabel"><span>{label}</span>'
                        f'<span>{v:.2%}</span></div><div class="pbar"><span style="width:{pct:.2f}%">'
                        '</span></div></div>', unsafe_allow_html=True)

    with st.expander('How to interpret these measurements'):
        st.markdown('''**Confidence and MC entropy** describe the model’s predicted probabilities, and uncertainty can help flag cases for review. **Faithfulness** evaluates whether highlighted image regions influenced a prediction under perturbation. **Stability** evaluates the similarity of heatmaps under small image changes. **Explanation uncertainty** measures variation across repeated dropout-driven HiResCAM maps.\n\nThese are separate measurements, not five independent votes on whether a diagnosis is correct. There is **no validated combined trustworthiness percentage**. The full V9 dataset lacks verified patient-level independence and external clinical validation.''')
    st.caption(f'Source: {source} · Research use only · Not clinically validated')


st.markdown('#### Choose an MRI')
mode = st.radio('MRI source', ['🎲 Random MRI from V9 test set', '↑ Upload your own MRI'],
                horizontal=True, label_visibility='collapsed')

try:
    profiles = load_test_profiles()
except Exception as exc:
    st.error('Unable to load the final V9 study results. Extract the Step 33 results into your existing V9 project folder.')
    st.code(str(exc))
    st.stop()

if mode.startswith('🎲'):
    left, right = st.columns([2.4, 1], vertical_alignment='bottom')
    with left:
        selection = st.selectbox('Random selection',
            ['Any test MRI (1,374)', 'Known incorrect predictions (29)', 'Known correct predictions (1,345)'],
            help='The latter two options sample from those subgroups rather than the full test set.')
    with right:
        select_clicked = st.button('Choose random MRI', type='primary', use_container_width=True)
    st.caption('Test images were assigned by the V9 split manifest; an original folder named “Training” can still contain V9 test images.')
    if select_clicked:
        filtered = profiles
        if selection.startswith('Known incorrect'):
            filtered = profiles[profiles['deterministic_correct'] == 0]
        elif selection.startswith('Known correct'):
            filtered = profiles[profiles['deterministic_correct'] == 1]
        r = filtered.iloc[random.SystemRandom().randrange(len(filtered))].to_dict()
        try:
            with Image.open(source_image(r['file'])) as img:
                pic = ImageOps.exif_transpose(img).convert('L').copy()
            st.session_state['bl_random'] = (r, pic)
            st.session_state['bl_random_map'] = None
        except Exception as exc:
            st.error(f'Unable to open the selected MRI: {exc}')
    if 'bl_random' in st.session_state:
        r, pic = st.session_state['bl_random']
        st.caption(f'V9 test image: `{r["file"]}`')
        m = {
            'predicted_class': r['deterministic_predicted_class'],
            **{k: float(r[k]) for k in ['calibrated_confidence', 'mc_predictive_entropy',
              'faithfulness_del_high_minus_random_20', 'faithfulness_ins_high_minus_random_20',
              'mean_stability_cosine', 'mean_stability_top20_iou', 'explanation_shape_uncertainty']},
            'probabilities': {c: float(r[f'calibrated_prob_{c}']) for c in C.CLASSES},
        }
        if st.session_state.get('bl_random_map') is None:
            try:
                with st.spinner('Creating the final HiResCAM view…'):
                    model, dev, _ = get_model()
                    mapped, live_prediction = light_heatmap(pic, model, dev)
                if live_prediction != m['predicted_class']:
                    st.warning('The live model prediction differs from the saved profile. Verify the checkpoint version.')
                st.session_state['bl_random_map'] = mapped
            except Exception as exc:
                st.warning(f'Could not render the HiResCAM overlay: {exc}')
        show_summary(m, pic, 'Saved full-test Step 33 measurements (not recomputed in this view)',
                     profiles, known_label=r['true_class'], heatmap=st.session_state.get('bl_random_map'))

else:
    st.caption('Upload a JPG or PNG brain MRI. A phone photo, unrelated image, or image from another scanner may be outside the training distribution.')
    uploaded = st.file_uploader('Upload MRI', type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tif', 'tiff'])
    known = st.selectbox('Known reference label (optional)', ['Unknown / not provided'] + C.CLASSES,
          help='Only choose a label if it was independently established. This does not make the model clinically validated.')
    if st.button('Analyze uploaded MRI', type='primary', disabled=uploaded is None):
        try:
            raw = uploaded.getvalue()
            if len(raw) > 20 * 1024 * 1024:
                raise ValueError('Image file is larger than 20 MB.')
            with Image.open(io.BytesIO(raw)) as image:
                pic = ImageOps.exif_transpose(image).convert('L').copy()
            with st.spinner('Running V9 inference, five measurements, and 20 HiResCAM dropout passes…'):
                model, dev, temp = get_model()
                outcome = analyze_new_mri(pic, model, dev, temp)
            st.session_state['bl_uploaded'] = (pic, outcome, known, hash(raw))
        except Exception as exc:
            st.error(f'Could not analyze this MRI: {exc}')
    stored = st.session_state.get('bl_uploaded')
    if stored and uploaded is not None and stored[3] == hash(uploaded.getvalue()):
        pic, outcome, saved_label, _ = stored
        show_summary(outcome, pic, 'Live V9 checkpoint calculations on uploaded MRI', profiles,
                     known_label=saved_label if saved_label in C.CLASSES else None,
                     heatmap=outcome['overlay'])

st.markdown('<div class="footerline">BrainLens MRI · V9 reliability research · HiResCAM · '
            'Not for diagnosis or clinical decision-making</div>', unsafe_allow_html=True)
