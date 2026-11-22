"""Paper-aligned, exactly matched to V7 preprocessing; train-only augmentation."""
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from . import config as C

class AddGaussianNoise:
    def __init__(self,std=.02):self.std=std
    def __call__(self,x):return (x+torch.randn_like(x)*self.std).clamp(0,1)

def transform(training=False):
    steps=[transforms.Grayscale(num_output_channels=1),
           transforms.Resize((C.IMAGE_SIZE,C.IMAGE_SIZE),interpolation=InterpolationMode.BICUBIC)]
    if training:
        steps += [transforms.RandomHorizontalFlip(p=.5),
                  transforms.RandomRotation(15,interpolation=InterpolationMode.BICUBIC),
                  transforms.ColorJitter(brightness=.2,contrast=.2)]
    steps.append(transforms.ToTensor())
    if training:steps.append(AddGaussianNoise(.02))
    steps.append(transforms.Normalize(mean=[.5],std=[.5]))
    return transforms.Compose(steps)
