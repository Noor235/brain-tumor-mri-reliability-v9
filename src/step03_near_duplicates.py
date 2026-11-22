"""STEP 03: screen pHash+SSIM visually similar samples and group, NOT delete, them."""
from collections import defaultdict, Counter
from PIL import Image, ImageDraw
from . import config as C
from .common import output_dir, load_csv, write_csv, save_json,source_image
from .image_utils import UnionFind,BKTree,decoded_image,preview,ssim

def montage(pairs,out):
    if not pairs:return
    chosen=sorted(pairs,key=lambda x:-float(x['ssim']))[:20]
    cv=Image.new('RGB',(720,len(chosen)*143),(250,250,250));d=ImageDraw.Draw(cv)
    for n,r in enumerate(chosen):
        y=n*143
        for x,file in [(8,r['file_a']),(142,r['file_b'])]:
            im=decoded_image(source_image(file)).resize((125,125)).convert('RGB');cv.paste(im,(x,y+6))
        label=f"pHash {r['phash_distance']} | SSIM {r['ssim']}\n{r['class_a']} vs {r['class_b']}"
        d.multiline_text((285,y+10),label,fill=(20,20,20),spacing=6)
        d.text((285,y+60),r['file_a'][-60:],fill=(20,20,20))
        d.text((285,y+82),r['file_b'][-60:],fill=(20,20,20))
    cv.save(out)

def main():
    C.heading('STEP 03 / NEAR DUPLICATES: image similarity screening and conservative grouping')
    rows=[r for r in load_csv(C.EXACT) if r['keep']=='1'];rows.sort(key=lambda r:r['file'])
    by_hash=defaultdict(list);tree=BKTree();uf=UnionFind([r['file'] for r in rows])
    image_cache={};pairs=[];total_prelim=0
    print(f'Scanning {len(rows)} UNIQUE exact representatives (not all original files).',flush=True)
    print(f'pHash Hamming <= {C.NEAR_PHASH_MAX_DISTANCE}, SSIM >= {C.NEAR_SSIM_MIN} on 96x96 images.',flush=True)
    print('Pairs passing thresholds are POSSIBLE near duplicates: they are grouped to prevent cross-split overlap, NOT deleted.',flush=True)
    for i,r in enumerate(rows,1):
        key=int(r['phash'],16);mine=r['file']
        # Build pHash search index incrementally: each candidate pair is visited exactly once.
        for matched_hash, distance in tree.find(key,C.NEAR_PHASH_MAX_DISTANCE):
            for prior in by_hash[matched_hash]:
                total_prelim+=1
                if mine not in image_cache:image_cache[mine]=preview(decoded_image(source_image(mine)))
                old=prior['file']
                if old not in image_cache:image_cache[old]=preview(decoded_image(source_image(old)))
                similarity=ssim(image_cache[mine],image_cache[old])
                if similarity>=C.NEAR_SSIM_MIN:
                    pairs.append(dict(file_a=old,class_a=prior['class_name'],file_b=mine,class_b=r['class_name'],
                                      phash_distance=distance,ssim=f'{similarity:.6f}',same_class=int(prior['class_name']==r['class_name'])))
                    # Group even cross-label candidates to avoid possible overlap; never relabel either record.
                    uf.union(mine,old)
        if key not in by_hash:tree.add(key)
        by_hash[key].append(r)
        if i%700==0:print(f'  Inspected {i}/{len(rows)} | similarity candidates accepted: {len(pairs)}',flush=True)
    clusters=defaultdict(list)
    for r in rows:clusters[uf.find(r['file'])].append(r)
    stable_groups=sorted(clusters.values(),key=lambda group:min(r['file'] for r in group))
    output=[]
    for idx,group in enumerate(stable_groups,1):
        for r in group:output.append(dict(file=r['file'],class_name=r['class_name'],original_split=r['original_split'],
                                         raw_sha256=r['raw_sha256'],pixel_sha256=r['pixel_sha256'],
                                         exact_group=r['exact_group'],near_group=f'group_{idx:05d}',group_size=len(group)))
    output.sort(key=lambda r:r['file'])
    out=output_dir('03_near_duplicates')
    write_csv(out/'near_candidate_pairs.csv',pairs,['file_a','class_a','file_b','class_b','phash_distance','ssim','same_class'])
    write_csv(C.NEAR,output,['file','class_name','original_split','raw_sha256','pixel_sha256','exact_group','near_group','group_size'])
    montage(pairs,out/'similarity_montage.png')
    mixed=sum(len({r['class_name'] for r in group})>1 for group in stable_groups)
    sizes=sorted([len(group) for group in stable_groups],reverse=True)
    summary={'exact_unique_images':len(rows),'potential_near_duplicate_pairs':len(pairs),
       'cross_class_candidate_pairs':sum(p['same_class']==0 for p in pairs),
       'p_hash_threshold':C.NEAR_PHASH_MAX_DISTANCE,'ssim_threshold':C.NEAR_SSIM_MIN,
       'candidate_pairs_compared_with_ssim':total_prelim,'image_similarity_groups':len(stable_groups),
       'groups_with_multiple_images':sum(s>1 for s in sizes),'largest_group_size':sizes[0] if sizes else 0,
       'mixed_class_groups':mixed,
       'critical_note':'Near matches are HEURISTIC. Similar MRIs may be different patients. Kept images are grouped into one split, not treated as verified duplicates or deleted.',
       'limitations':'Other near duplicates and same-patient slices can be missed. No verified patient IDs.'}
    save_json(out/'near_summary.json',summary)
    print('Pairs flagged as similar:',len(pairs),'| cross-class candidates:',summary['cross_class_candidate_pairs'],flush=True)
    print('Grouping:',len(stable_groups),'groups | largest group:',summary['largest_group_size'],'images | mixed-class groups:',mixed,flush=True)
    print('Please inspect similarity_montage.png and near_candidate_pairs.csv for possible false-positive similarity matches.',flush=True)
    if len(rows)>100 and sizes and sizes[0]>int(0.1*len(rows)):
        raise RuntimeError('One near-similarity cluster contains >10% of the dataset; likely chain-merge problem. Inspect candidate pairs and thresholds before splitting.')
    print('STEP 03 PASS — candidate near-duplicates assigned to a SINGLE group for subsequent splitting.',flush=True)
if __name__=='__main__':main()
