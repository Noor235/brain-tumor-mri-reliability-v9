"""V9 immutable experiment settings. Source MRI files are NEVER edited."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(os.environ.get('KAGGLE_V9_SOURCE', str(ROOT / 'data' / 'brain-tumor-mri'))).expanduser().resolve()
OUTPUT = Path(os.environ.get('KAGGLE_V9_OUTPUT', str(ROOT / 'results'))).expanduser().resolve()
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
EXPECTED = {'glioma': 1800, 'meningioma': 1800, 'notumor': 1800, 'pituitary': 1800}
VALID_EXT = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
SEED = 42
TEST_FRACTION = 0.20
VAL_FRACTION_OF_REMAINDER = 0.20
NEAR_PHASH_MAX_DISTANCE = 8
NEAR_SSIM_MIN = 0.985
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.4
MC_PASSES = 20
NUM_WORKERS = 0  # Apple MPS friendly
MANIFEST = OUTPUT / '01_inventory' / 'image_manifest.csv'
EXACT = OUTPUT / '02_exact_duplicates' / 'dedup_manifest.csv'
NEAR = OUTPUT / '03_near_duplicates' / 'near_groups.csv'
SPLIT = OUTPUT / '04_split' / 'split_manifest.csv'
CHECKPOINT = OUTPUT / '07_training' / 'best_v9.pt'

def heading(title):
    print('\n' + '='*72, flush=True)
    print(title, flush=True)
    print('='*72, flush=True)
