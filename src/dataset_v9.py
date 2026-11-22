"""CSV driven Dataset; works on original MRI files without copying or modifying them."""
from PIL import Image,ImageOps
from torch.utils.data import Dataset,DataLoader
from . import config as C
from .common import source_image
from .transforms_v9 import transform

class ManifestDataset(Dataset):
    def __init__(self,rows,training=False):
        self.rows=rows
        self.transform=transform(training)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(source_image(row['file'])) as im:
            im=ImageOps.exif_transpose(im).convert('L')
            x=self.transform(im)
        return x,C.CLASSES.index(row['class_name'])

def loader(rows,split,training=False):
    ds=ManifestDataset([r for r in rows if r['split']==split],training=training)
    if len(ds)==0:raise ValueError(f'Empty {split} dataset!')
    return DataLoader(ds,batch_size=C.BATCH_SIZE,shuffle=training,
                      num_workers=C.NUM_WORKERS,pin_memory=False)
