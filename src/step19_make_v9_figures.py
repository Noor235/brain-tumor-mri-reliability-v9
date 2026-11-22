"""V9 Step 19 — Create thesis figures from existing CSV/JSON only.

Install: copy this file into Brain_Tumor_Reproduction_V9/src/
Run:     ./.venv/bin/python -m src.step19_make_v9_figures

No model loading. No training. No input file modifications.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CLASS_DISPLAY = {'glioma': 'Glioma', 'meningioma': 'Meningioma',
                 'notumor': 'No Tumor', 'pituitary': 'Pituitary'}


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def required_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'REQUIRED INPUT NOT FOUND: {path}\n'
                                'Do not retrain. Check the actual results directory and file name.')


def numeric(value, name):
    try:
        x = float(value)
        if not np.isfinite(x):
            raise ValueError('non-finite')
        return x
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid numeric value for {name}: {value!r}') from exc


def save(fig, out, name, generated):
    path = out / name
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    generated.append(name)
    print(' [SAVED]', path)


def setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 13,
        'axes.labelsize': 10, 'figure.figsize': (9, 5.4),
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.pad_inches': 0.20,
    })


def line_chart(out, epochs, lines, ylabel, title, file_name, generated,
               percent=False, best_epoch=None):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    color_seq = ['#167D81', '#C67835', '#4F75C5']
    for k, (label, values) in enumerate(lines):
        vals = np.asarray(values, dtype=float) * (100.0 if percent else 1.0)
        ax.plot(epochs, vals, marker='o', markersize=4.0, linewidth=2.15,
                label=label, color=color_seq[k % len(color_seq)])
    if best_epoch is not None:
        ax.axvline(best_epoch, color='#555555', alpha=.6, linestyle='--', linewidth=1.3,
                   label=f'Last saved best checkpoint: epoch {best_epoch}')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=.2)
    ax.set_xticks(epochs if len(epochs) <= 25 else epochs[::2])
    if percent:
        ax.set_ylim(0, 105)
    elif ylabel.lower() == 'cross-entropy loss':
        ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    save(fig, out, file_name, generated)


def confusion_chart(out, cm, labels, normalize, generated):
    values = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0 if normalize else cm
    fig, ax = plt.subplots(figsize=(8.1, 6.8))
    img = ax.imshow(values, cmap='Blues', vmin=0)
    cb = fig.colorbar(img, ax=ax, fraction=.045, pad=.045)
    cb.set_label('Percent of actual class (%)' if normalize else 'Number of MRI images')
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('Actual class')
    ax.set_title('V9 Test Confusion Matrix — ' + ('Row-normalized (%)' if normalize else 'Counts'))
    threshold = float(values.max()) / 2 if values.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            label = f'{values[i,j]:.1f}%' if normalize else str(int(values[i,j]))
            ax.text(j, i, label, ha='center', va='center',
                    color='white' if values[i,j] > threshold else '#1a202c',
                    fontsize=11, fontweight='semibold')
    fig.tight_layout()
    save(fig, out, '04_confusion_matrix_percent.png' if normalize else '03_confusion_matrix_counts.png', generated)


def class_metrics_chart(out, cm, labels, generated):
    tp = np.diag(cm).astype(float)
    recalls = tp / np.maximum(cm.sum(axis=1), 1)
    precisions = tp / np.maximum(cm.sum(axis=0), 1)
    f1s = np.divide(2 * precisions * recalls, precisions + recalls,
                    out=np.zeros_like(precisions), where=(precisions + recalls) > 0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    width = .25
    for offset, values, name, color in [(-width, precisions, 'Precision', '#167D81'),
                                        (0, recalls, 'Recall', '#4F75C5'),
                                        (width, f1s, 'F1-score', '#C67835')]:
        ax.bar(x + offset, 100 * values, width, label=name, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel('Score (%)')
    ax.set_title('V9 Test Performance by Tumor Class')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    save(fig, out, '05_precision_recall_f1_by_class.png', generated)
    return precisions, recalls, f1s


def optional_confidence_chart(src, out, generated):
    if not src.is_file():
        print(' [SKIP] Test confidence histogram (per-image predictions CSV missing)')
        return
    rows = read_csv(src)
    if not rows or not {'correct', 'confidence'}.issubset(rows[0]):
        print(' [SKIP] Test confidence histogram (CSV does not contain required columns)')
        return
    correct, wrong = [], []
    for r in rows:
        confidence = numeric(r['confidence'], 'confidence')
        if not 0 <= confidence <= 1:
            raise ValueError('Prediction confidence must be in [0,1]')
        (correct if str(r['correct']).strip().lower() in {'1', 'true', 'yes'} else wrong).append(confidence)
    fig, ax = plt.subplots(figsize=(9, 5.3))
    bins = np.linspace(0, 1, 26)
    ax.hist(correct, bins=bins, alpha=.62, label=f'Correct (n={len(correct)})', color='#167D81')
    ax.hist(wrong, bins=bins, alpha=.85, label=f'Incorrect (n={len(wrong)})', color='#C67835')
    ax.set_xlabel('Uncalibrated, single-pass maximum class probability')
    ax.set_ylabel('Number of images')
    ax.set_title('V9 Test Prediction Confidence — Correct vs Incorrect')
    ax.set_xlim(0, 1)
    ax.grid(axis='y', alpha=.18)
    ax.legend()
    fig.tight_layout()
    save(fig, out, '06_test_confidence_distribution.png', generated)


def main():
    ap = argparse.ArgumentParser(description='Generate V9 thesis figures from saved results; no training.')
    ap.add_argument('--project-root', type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help='Brain_Tumor_Reproduction_V9 folder (default: parent of src)')
    args = ap.parse_args()
    root = args.project_root.expanduser().resolve()
    results = root / 'results'
    history_file = results / '07_training' / 'train_history.csv'
    metrics_file = results / '08_test' / 'test_metrics.json'
    required_file(history_file)
    required_file(metrics_file)
    rows = read_csv(history_file)
    expected = {'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'}
    if not rows or not expected.issubset(rows[0]):
        raise ValueError(f'Training history does not have expected columns: {sorted(expected)}')
    epochs = np.array([int(r['epoch']) for r in rows])
    if len(np.unique(epochs)) != len(epochs):
        raise ValueError('Duplicate epoch numbers in training history.')
    best_epochs = [int(r['epoch']) for r in rows
                   if str(r.get('saved_best', '')).strip().lower() in {'1', 'true', 'yes'}]
    best_epoch = max(best_epochs) if best_epochs else None
    with metrics_file.open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data.get('confusion_matrix'), list):
        raise ValueError('test_metrics.json is missing confusion_matrix.')
    cm = np.array(data['confusion_matrix'], dtype=int)
    raw_classes = data.get('classes', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    labels = [CLASS_DISPLAY.get(str(c).lower(), str(c).replace('_', ' ').title()) for c in raw_classes]
    if cm.shape != (len(labels), len(labels)) or np.any(cm < 0):
        raise ValueError('Invalid confusion matrix shape or negative values.')
    if int(cm.sum()) != int(data['test_count']):
        raise ValueError(f"Confusion matrix sums to {cm.sum()}, test_count says {data['test_count']}")
    if int(np.trace(cm)) != int(data['correct']):
        raise ValueError('Confusion matrix diagonal disagrees with correct count.')
    out = results / '19_thesis_figures'
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    setup_style()
    line_chart(out, epochs,
               [('Training accuracy', [numeric(r['train_accuracy'], 'train_accuracy') for r in rows]),
                ('Validation accuracy', [numeric(r['val_accuracy'], 'val_accuracy') for r in rows])],
               'Accuracy (%)', 'V9 Training vs Validation Accuracy', '01_training_validation_accuracy.png',
               generated, percent=True, best_epoch=best_epoch)
    line_chart(out, epochs,
               [('Training loss', [numeric(r['train_loss'], 'train_loss') for r in rows]),
                ('Validation loss', [numeric(r['val_loss'], 'val_loss') for r in rows])],
               'Cross-entropy loss', 'V9 Training vs Validation Loss', '02_training_validation_loss.png',
               generated, best_epoch=best_epoch)
    confusion_chart(out, cm, labels, False, generated)
    confusion_chart(out, cm, labels, True, generated)
    precision, recall, f1 = class_metrics_chart(out, cm, labels, generated)
    optional_confidence_chart(results / '08_test' / 'per_image_predictions.csv', out, generated)
    if {'val_macro_f1', 'val_glioma_recall'}.issubset(rows[0]):
        line_chart(out, epochs,
                   [('Validation macro-F1', [numeric(r['val_macro_f1'], 'val_macro_f1') for r in rows]),
                    ('Validation glioma recall', [numeric(r['val_glioma_recall'], 'val_glioma_recall') for r in rows])],
                   'Metric (%)', 'V9 Validation Macro-F1 and Glioma Recall',
                   '07_validation_f1_glioma_recall.png', generated, percent=True)
    accuracy = float(np.trace(cm)) / int(cm.sum())
    summary = [
        'KAGGLE V9 — GENERATED FIGURE INDEX',
        'Figures generated ONLY from saved V9 train_history.csv, test_metrics.json,',
        'and (where available) per_image_predictions.csv. No model inference or retraining.',
        f'Project folder: {root}',
        f'Training epochs recorded: {len(epochs)}',
        f'Last best checkpoint epoch in history: {best_epoch if best_epoch is not None else "unknown"}',
        f'Test images: {cm.sum()}',
        f'Correct test predictions: {np.trace(cm)}',
        f'Test accuracy: {accuracy:.4%}',
        '', 'Per-class test results:',
    ]
    for name, p, r, f in zip(labels, precision, recall, f1):
        summary.append(f'  {name}: precision={p:.4%}, recall={r:.4%}, F1={f:.4f}')
    summary += ['', 'Created figures:', *[f'  {x}' for x in generated], '',
                'Note: These graphs are descriptive and do not establish clinical validation.',
                'Test performance from held-out split; patient-disjointness not established.']
    summary_path = out / 'FIGURE_INDEX.txt'
    summary_path.write_text('\n'.join(summary)+'\n', encoding='utf-8')
    print('\n'+'='*72)
    print('DONE: saved', len(generated), 'figures under', out)
    print('Test accuracy cross-check:', f'{accuracy:.4%}')
    print('Model and existing datasets/checkpoint were not loaded or changed.')


if __name__ == '__main__':
    main()
