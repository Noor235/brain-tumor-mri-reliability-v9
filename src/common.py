"""CSV + data integrity helpers shared by independent V9 stages."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from . import config as C

def output_dir(name):
    p = C.OUTPUT / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'  Saved CSV: {path}', flush=True)

def load_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing required earlier-stage output: {path}\nRun the preceding step first.')
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def save_json(path, payload):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  Saved summary: {p}', flush=True)

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for block in iter(lambda:fp.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def sha_manifest_rows(rows, columns):
    lines = ['\t'.join(str(row.get(k,'')) for k in columns) for row in rows]
    return hashlib.sha256(('\n'.join(lines)).encode('utf-8')).hexdigest()

def source_image(file):
    p = (C.SOURCE / file).resolve()
    if not p.is_relative_to(C.SOURCE) or not p.is_file():
        raise FileNotFoundError(f'Unsafe or missing source image: {file}')
    return p

def check_labels(rows):
    names = set(C.CLASSES)
    extra = set(r['class_name'] for r in rows) - names
    if extra: raise ValueError(f'Unknown class names: {sorted(extra)}')

def check_no_overwrite(path):
    if Path(path).exists():
        raise FileExistsError(f'Existing artifact found; refusing to overwrite:\n{path}\nKeep your completed experiment; use a fresh output folder for another run.')

def count_by(rows, field): return dict(Counter(r[field] for r in rows))
