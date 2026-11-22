"""STEP 01: inspect the original dataset and fingerprint every valid image."""
from collections import Counter
from . import config as C
from .common import output_dir, write_csv, save_json, sha_file
from .image_utils import decoded_image, pixel_digest, phash

def main():
    C.heading('STEP 01 / INVENTORY: audit raw newer four-class dataset (READ-ONLY)')
    print('Source:',C.SOURCE,'\nAll files are read-only. Original Training/Testing layout is recorded but will NOT define V9 splits.',flush=True)
    rows=[]; bad=[]; extras=[]; counts=Counter()
    for old_split in ('Training','Testing'):
        for cls in C.CLASSES:
            folder=C.SOURCE/old_split/cls
            if not folder.is_dir(): raise FileNotFoundError(f'Expected class directory missing: {folder}')
            paths=sorted(p for p in folder.rglob('*') if p.is_file())
            for p in paths:
                if p.suffix.lower() not in C.VALID_EXT:
                    extras.append(str(p.relative_to(C.SOURCE)));continue
                name=p.relative_to(C.SOURCE).as_posix()
                try:
                    img=decoded_image(p)
                    rows.append(dict(file=name,class_name=cls,original_split=old_split,
                      width=img.width,height=img.height,raw_sha256=sha_file(p),
                      pixel_sha256=pixel_digest(img),phash=f'{phash(img):016x}'))
                    counts[cls]+=1
                except Exception as exc:bad.append({'file':name,'error':str(exc)})
            print(f'  {old_split:<8} {cls:<12} images {sum(r["original_split"]==old_split and r["class_name"]==cls for r in rows):>4}',flush=True)
    out=output_dir('01_inventory')
    fields=['file','class_name','original_split','width','height','raw_sha256','pixel_sha256','phash']
    write_csv(C.MANIFEST,rows,fields)
    write_csv(out/'invalid_images.csv',bad,['file','error'])
    summary={'input_source':str(C.SOURCE),'image_count':len(rows),'per_class':dict(counts),'ignored_nonimages':extras,'invalid_images':bad,
      'counts_match_expected':all(counts[k]==v for k,v in C.EXPECTED.items()) and len(rows)==sum(C.EXPECTED.values()),
      'note':'Source predefined split recorded for audit only. V9 uses a NEW split after duplicate grouping.'}
    save_json(out/'inventory_summary.json',summary)
    print('Total MRI images:',len(rows),'| ignored non-images:',len(extras),'| invalid images:',len(bad),flush=True)
    if bad:raise RuntimeError('Damaged images detected. Resolve them before continuing.')
    if not summary['counts_match_expected'] and __import__('os').environ.get('V9_ALLOW_NONSTANDARD_SOURCE')!='1':
        raise ValueError('Expected 7200 four-class V9 MRIs (1800/class), not V8 2021 data. Check the source folder and inventory_summary.json.')
    print('STEP 01 PASS — original images have not been changed.',flush=True)
if __name__=='__main__':main()
