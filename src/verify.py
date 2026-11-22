"""Reusable strict V9 validation gate before training and testing."""
from collections import Counter, defaultdict
from . import config as C
from .common import load_csv, sha_file, sha_manifest_rows,source_image

def verify(rehash=False, detail=True):
    rows=load_csv(C.SPLIT)
    if not rows: raise ValueError('Empty V9 split.')
    src=load_csv(C.EXACT); groups=load_csv(C.NEAR)
    kept={r['file'] for r in src if r['keep']=='1'}
    if len({r['file'] for r in groups})!=len(groups) or {r['file'] for r in groups}!=kept:
        raise ValueError('Near-group manifest must contain exactly all retained exact representatives.')
    group_labels=defaultdict(set)
    for r in groups:
        group_labels[r['near_group']].add(r['class_name'])
    mixed_groups={g for g,labels in group_labels.items() if len(labels)>1}
    expected_excluded={r['file'] for r in groups if r['near_group'] in mixed_groups}
    exclusion_csv=C.OUTPUT/'04_split'/'excluded_mixed_label_groups.csv'
    if not exclusion_csv.is_file():
        raise ValueError('Missing Step 04 mixed-label exclusion record.')
    excluded=load_csv(exclusion_csv)
    recorded={r['file'] for r in excluded}
    if len(recorded)!=len(excluded) or recorded!=expected_excluded:
        raise ValueError('Mixed-label exclusion CSV does not match exactly the ambiguous near-groups.')
    allowed=kept-expected_excluded
    if len(rows)!=len(allowed) or {r['file'] for r in rows}!=allowed:
        raise ValueError('Split does not contain exactly all allowed representatives after mixed-label exclusions.')
    if len({r['file'] for r in rows})!=len(rows): raise ValueError('Repeated file paths in split.')
    if any(r['near_group'] in mixed_groups for r in rows):
        raise ValueError('Mixed-label near-group unexpectedly included in split.')
    if any(r['class_name'] not in C.CLASSES for r in rows):raise ValueError('Unexpected class name.')
    if {r['split'] for r in rows}!={'train','val','test'}:raise ValueError('Missing/invalid split labels.')
    by_group=defaultdict(set)
    by_pix=defaultdict(set)
    for r in rows:
        by_group[r['near_group']].add(r['split'])
        by_pix[r['pixel_sha256']].add(r['split'])
    bad_groups={k:v for k,v in by_group.items() if len(v)>1}
    bad_pixels={k:v for k,v in by_pix.items() if len(v)>1}
    if bad_groups or bad_pixels:raise ValueError(f'LEAKAGE: {len(bad_groups)} near groups and {len(bad_pixels)} pixel hashes across splits!')
    if rehash:
        for i,r in enumerate(rows,1):
            file=source_image(r['file'])
            if sha_file(file)!=r['raw_sha256']:raise RuntimeError(f'Original dataset image changed since inventory: {file}')
            if detail and i%700==0:print(f'  Checked content hashes: {i}/{len(rows)}',flush=True)
    else:
        for r in rows:source_image(r['file'])
    cols=['file','class_name','original_split','raw_sha256','pixel_sha256','exact_group','near_group','group_size','split']
    fingerprint=sha_manifest_rows(rows,cols)
    if detail:
        print('VERIFIED: included unique representative files:',len(rows),flush=True)
        print(f'VERIFIED: excluded {len(expected_excluded)} images from {len(mixed_groups)} mixed-label near-groups.',flush=True)
        print('VERIFIED: exact duplicate overlap across splits: ZERO',flush=True)
        print('VERIFIED: flagged near-similarity group overlap across splits: ZERO',flush=True)
        for s in ('train','val','test'):
            print(f"  {s:<5} {sum(r['split']==s for r in rows):>5} | " + ', '.join(f"{c}: {sum(r['split']==s and r['class_name']==c for r in rows)}" for c in C.CLASSES),flush=True)
        print('Dataset fingerprint:',fingerprint,flush=True)
        print('LIMITATION: Patient identity unknown; undetected near duplicates or same-patient images may still cross splits.',flush=True)
    return rows, fingerprint
