"""STEP 04: stratified GROUP splitting AFTER duplicate removal & near grouping."""
from collections import Counter,defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from . import config as C
from .common import output_dir,load_csv,write_csv,save_json,sha_manifest_rows,check_no_overwrite

def choose_fold(rows, seed, label):
    y=np.array([C.CLASSES.index(r['class_name']) for r in rows],dtype=int)
    groups=np.array([r['near_group'] for r in rows])
    # Choose the pre-declared fold with best class balance; NEVER based on any model prediction.
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    options=[]
    for main,held in splitter.split(np.zeros(len(rows)),y,groups):
        frac=float(len(held)/len(rows));score=abs(frac-.20)*3
        for c in range(len(C.CLASSES)):
            denom=max(int(np.sum(y==c)),1)
            score+=abs(float(np.sum(y[held]==c)/denom)-.20)
        options.append((score,main,held))
    score,train_idx,hold_idx=min(options,key=lambda z:z[0])
    print(f'  {label}: selected the most class-balanced of five group-preserving folds (score={score:.4f}).',flush=True)
    return train_idx,hold_idx

def main():
    C.heading('STEP 04 / NEW TRAIN–VALIDATION–TEST SPLIT: group-preserving, stratified')
    check_no_overwrite(C.SPLIT)
    all_rows=load_csv(C.NEAR)
    if not all_rows:
        raise RuntimeError('No near-duplicate representative records. Complete Step 03 first.')
    group_labels=defaultdict(set)
    for r in all_rows:
        group_labels[r['near_group']].add(r['class_name'])
    mixed_groups={g for g,labels in group_labels.items() if len(labels)>1}
    excluded=sorted((r for r in all_rows if r['near_group'] in mixed_groups),key=lambda r:r['file'])
    rows=[r for r in all_rows if r['near_group'] not in mixed_groups]
    if not rows or any(not any(r['class_name']==c for r in rows) for c in C.CLASSES):
        raise RuntimeError('After mixed-label exclusions, one or more classes have no samples.')
    if len(excluded)>max(100,int(len(all_rows)*.05)):
        raise RuntimeError('More than 5% / 100 representative images are in mixed-label groups: inspect before excluding.')
    print(f'Original exact-unique images: {len(all_rows)}.',flush=True)
    print(f'EXCLUDED from this experiment: {len(excluded)} images in {len(mixed_groups)} mixed-label near-similarity groups.',flush=True)
    print('Exclusions affect only the split manifest; all original MRI files and Step 03 results remain unchanged.',flush=True)
    print(f'Using {len(rows)} remaining images; original Training/Testing folders do NOT define V9 splits.',flush=True)
    print('Target: 20% held-out test; 20% of remainder validation. Groups kept together.',flush=True)
    remain_idx,test_idx=choose_fold(rows,C.SEED,'TEST set')
    remain=[rows[int(i)] for i in remain_idx]
    train_idx,val_idx=choose_fold(remain,C.SEED+17,'VALIDATION set')
    subsets={}
    for index in train_idx:subsets[remain[int(index)]['file']]='train'
    for index in val_idx:subsets[remain[int(index)]['file']]='val'
    for index in test_idx:subsets[rows[int(index)]['file']]='test'
    outrows=[dict(**r,split=subsets[r['file']]) for r in rows]
    outrows.sort(key=lambda r:r['file'])
    columns=['file','class_name','original_split','raw_sha256','pixel_sha256','exact_group','near_group','group_size','split']
    out=output_dir('04_split')
    write_csv(C.SPLIT,outrows,columns)
    exclusion_columns=['file','class_name','original_split','raw_sha256','pixel_sha256','exact_group','near_group','group_size']
    write_csv(out/'excluded_mixed_label_groups.csv',excluded,exclusion_columns)
    counts={s:dict(Counter(r['class_name'] for r in outrows if r['split']==s)) for s in ('train','val','test')}
    for split in ('train','val','test'):
        cnt=sum(1 for r in outrows if r['split']==split)
        print(f'  {split:<5}: {cnt:>5} ({cnt/len(outrows):.2%}) | per class: {counts[split]}',flush=True)
    file_fingerprint=sha_manifest_rows(outrows,columns)
    save_json(out/'split_summary.json',{'source_dataset':str(C.SOURCE),'unique_exact_representatives_before_mixed_exclusion':len(all_rows),'images_after_mixed_exclusion':len(rows),'excluded_mixed_label_images':len(excluded),'excluded_mixed_label_groups':len(mixed_groups),'excluded_group_ids':sorted(mixed_groups),
      'counts':counts,'seed_test':C.SEED,'seed_validation':C.SEED+17,
      'split_method':'stratified-group 5-fold choice for test, then 5-fold choice for validation within remainder',
      'nominal_target':'80/20 train+validation/test, then 80/20 train/validation inside train+validation',
      'near_group_includes_all_flagged_candidates':True,'split_manifest_sha256':file_fingerprint,
      'note':'All exact duplicate extra copies and all mixed-label near-similarity groups are excluded from V9 splits. Near matches are heuristic: the mixed labels are not independently diagnosed. Actual proportions can deviate from target; originals untouched.'})
    print('STEP 04 PASS — new split frozen in split_manifest.csv. Next step independently verifies all boundaries.',flush=True)
if __name__=='__main__':main()
