"""STEP 05: full integrity re-hash and no detected cross-split duplicate groups."""
from . import config as C
from .common import output_dir, save_json
from .verify import verify

def main():
    C.heading('STEP 05 / INDEPENDENT SPLIT CHECK: verify no duplicate leakage')
    rows,fingerprint=verify(rehash=True,detail=True)
    counts={s:sum(r['split']==s for r in rows) for s in ('train','val','test')}
    save_json(output_dir('05_verification')/'verification.json',{
      'status':'PASS','exact_pixel_groups_spanning_splits':0,'detected_near_similarity_groups_spanning_splits':0,
      'source_images_rehashed':len(rows),'counts':counts,'split_fingerprint':fingerprint,
      'limitations':['No verified patient identifiers; NOT patient-independent evidence.',
                     'Perceptual hash + SSIM cannot detect all variants of near-duplicate images.',
                     'Similarity grouping may unite different patients; the method errs towards avoiding cross-split candidate matches.']})
    print('STEP 05 PASS — required checkpoint for training.',flush=True)
if __name__=='__main__':main()
