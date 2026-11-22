"""UA-EffNet-DA backbone copied from the provided Kaggle V7 implementation.

Same model layout and key names: EfficientNet-B4 with averaged RGB->1ch weights;
channel attention, then spatial attention, dropout .4, four-class head.
"""
import torch
import torch.nn as nn
from torchvision import models
from . import config as C

class ChannelAttention(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(channels,channels//reduction),nn.ReLU(),
                              nn.Linear(channels//reduction,channels),nn.Sigmoid())
    def forward(self,x):
        n,c,_,_=x.shape
        return x*self.fc(self.avg_pool(x).view(n,c)).view(n,c,1,1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=7,padding=3)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        a=x.mean(dim=1,keepdim=True)
        b=x.max(dim=1,keepdim=True).values
        return x*self.sigmoid(self.conv(torch.cat([a,b],dim=1)))

class BrainTumorModel(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        weight=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        network=models.efficientnet_b4(weights=weight)
        original=network.features[0][0]
        gray=nn.Conv2d(1,original.out_channels,kernel_size=original.kernel_size,
            stride=original.stride,padding=original.padding,dilation=original.dilation,
            groups=original.groups,bias=(original.bias is not None))
        with torch.no_grad():
            gray.weight.copy_(original.weight.mean(dim=1,keepdim=True))
            if original.bias is not None:gray.bias.copy_(original.bias)
        network.features[0][0]=gray
        self.features=network.features
        self.channel_attention=ChannelAttention(1792)
        self.spatial_attention=SpatialAttention()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.dropout=nn.Dropout(C.DROPOUT)
        self.classifier=nn.Linear(1792,len(C.CLASSES))
    def forward(self,x):
        x=self.features(x)
        x=self.channel_attention(x)
        x=self.spatial_attention(x)
        x=self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))

def build_model(pretrained=True):return BrainTumorModel(pretrained=pretrained)
