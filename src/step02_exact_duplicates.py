"""STEP 02: exact decoded-pixel/byte duplicate groups, retain ONE representative."""
from collections import defaultdict,Counter
from . import config as C
from .common import output_dir, load_csv, write_csv,save_json,check_labels
from .image_utils import UnionFind

def main():
    C.heading('STEP 02 / EXACT DUPLICATES: identify and remove repeat images from V9 manifest')
    rows=load_csv(C.MANIFEST);check_labels(rows)
    ids=[r['file'] for r in rows];uf=UnionFind(ids)
    for col in ('raw_sha256','pixel_sha256'):
        by=defaultdict(list)
        for r in rows:by[r[col]].append(r['file'])
        for members in by.values():
            for f in members[1:]:uf.union(members[0],f)
    clusters=defaultdict(list)
    for row in rows:clusters[uf.find(row['file'])].append(row)
    mixed=[x for x in clusters.values() if len({r['class_name'] for r in x})>1]
    out=output_dir('02_exact_duplicates')
    conflict_rows=[{'group':i,'files':' | '.join(r['file']+':'+r['class_name'] for r in x)} for i,x in enumerate(mixed,1)]
    write_csv(out/'label_conflicts.csv',conflict_rows,['group','files'])
    if mixed: raise RuntimeError(f'{len(mixed)} exact duplicate clusters have CONFLICTING CLASS LABELS. Review before proceeding. No records removed.')
    output=[]; dup_groups=0
    for group_index,cluster in enumerate(sorted(clusters.values(),key=lambda g:min(x['file'] for x in g)),1):
        # Prefer predefined Training when selecting a representative; ORIGINAL split is NOT retained.
        group=sorted(cluster,key=lambda r:(r['original_split']!='Training',r['file']))
        rep=group[0]['file'];group_id=f'exact_{group_index:05d}'
        if len(group)>1:dup_groups+=1
        for r in group:
            output.append({**r,'exact_group':group_id,'representative':rep,'keep':int(r['file']==rep),'duplicates_in_group':len(group)})
    output.sort(key=lambda r:r['file'])
    fields=list(rows[0])+['exact_group','representative','keep','duplicates_in_group']
    write_csv(C.EXACT,output,fields)
    kept=sum(int(r['keep']) for r in output)
    summary={'original_images':len(rows),'exact_duplicate_groups_with_2_or_more_images':dup_groups,
      'representative_images_retained':kept,'exact_duplicate_extra_copies_excluded':len(rows)-kept,
      'conflicting_exact_labels':0,'method':'SHA256 file bytes OR SHA256 dimensions and decoded EXIF-corrected grayscale pixel array',
      'note':'No physical MRI image was deleted. Only manifest entries are excluded. Raw dataset is preserved.'}
    save_json(out/'exact_summary.json',summary)
    print(f'Original images: {len(rows)} | unique representatives: {kept} | copies excluded from V9: {len(rows)-kept}',flush=True)
    print(f'Duplicate groups: {dup_groups} | conflicting-label groups: 0',flush=True)
    print('STEP 02 PASS — keep ONE image per exact duplicate group (manifest only).',flush=True)
if __name__=='__main__':main()
