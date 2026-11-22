"""STEP 00: environment/source availability, library versions, no training."""
import importlib
import sys
from . import config as C

def main():
    C.heading('STEP 00 / SYSTEM CHECK: V9 independent project setup')
    print('Python:',sys.version.split()[0]);print('Project:',C.ROOT)
    print('Read-only V9 source:',C.SOURCE)
    print('Output directory:',C.OUTPUT)
    print('Source exists:',C.SOURCE.is_dir())
    print('Training directory:',(C.SOURCE/'Training').is_dir())
    print('Testing directory:',(C.SOURCE/'Testing').is_dir())
    for name in ('numpy','PIL','cv2','sklearn','torch','torchvision'):
        try:
            m=importlib.import_module(name);print(f'  {name:<12}: OK ({getattr(m,"__version__","version unavailable")})')
        except Exception as exc:print(f'  {name:<12}: MISSING / ERROR: {exc}')
    print('Existing results will be saved inside V9 only; V8 results remain untouched.')
    if not C.SOURCE.is_dir():
        raise FileNotFoundError('V9 source folder not found. Expected Brain_Tumor_Reproduction_V2/data/brain-tumor-mri; if elsewhere, set KAGGLE_V9_SOURCE to the folder containing Training/ and Testing/.')
    if not (C.SOURCE/'Training').is_dir() or not (C.SOURCE/'Testing').is_dir():
        raise FileNotFoundError('V9 source folder must contain Training/ and Testing/. Check KAGGLE_V9_SOURCE or project location.')
    print('SYSTEM CHECK COMPLETE. Next: src.step01_inventory')
if __name__=='__main__':main()
