"""STEP 06: visualize samples from each split, no image modification."""
import random
from PIL import Image,ImageDraw,ImageOps
from . import config as C
from .common import output_dir,source_image
from .verify import verify

def main():
    C.heading('STEP 06 / VISUAL QA: review image examples from new splits')
    rows,_=verify(rehash=False,detail=False)
    rng=random.Random(C.SEED);cells=[]
    for split in ('train','val','test'):
        for cls in C.CLASSES:
            group=[r for r in rows if r['split']==split and r['class_name']==cls]
            if not group:raise RuntimeError(f'Empty split/class: {split}/{cls}')
            example=rng.choice(group);cells.append((split,cls,example['file']))
    w,h=900,3*225+50;canvas=Image.new('RGB',(w,h),(250,250,250));d=ImageDraw.Draw(canvas)
    for idx,(split,cls,file) in enumerate(cells):
        r,c=divmod(idx,4);x=c*225;y=r*225
        with Image.open(source_image(file)) as im:
            im=ImageOps.exif_transpose(im).convert('L').resize((185,185))
        canvas.paste(im.convert('RGB'),(x+20,y+25));d.text((x+20,y+7),f'{split} / {cls}',(20,20,20))
    path=output_dir('06_preview')/'split_samples.png';canvas.save(path)
    print('Saved comparison image:',path,flush=True)
    print('Manual inspection: look for cropped brain anatomy, black background shortcuts, or label errors.',flush=True)
if __name__=='__main__':main()
