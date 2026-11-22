"""Image fingerprinting and BK-tree, identical to V7 audit's methodology."""
import hashlib
import numpy as np
import cv2
from PIL import Image, ImageOps

def decoded_image(path):
    with Image.open(path) as src:
        img = ImageOps.exif_transpose(src).convert('L')
        img.load()
    return img

def pixel_digest(img):
    return hashlib.sha256(f'{img.width}x{img.height}:'.encode() + img.tobytes()).hexdigest()

def phash(img):
    small = np.asarray(img.resize((32, 32), Image.Resampling.BICUBIC), dtype=np.float32)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct.ravel()[1:])
    return int.from_bytes(np.packbits((dct > med).astype(np.uint8).ravel()).tobytes(), 'big')

def preview(img):
    return np.asarray(img.resize((96,96), Image.Resampling.BICUBIC), dtype=np.uint8).copy()

def ssim(a,b):
    a,b = a.astype(np.float32),b.astype(np.float32)
    ax=cv2.GaussianBlur(a,(7,7),1.5); bx=cv2.GaussianBlur(b,(7,7),1.5)
    axx=cv2.GaussianBlur(a*a,(7,7),1.5)-ax*ax
    bxx=cv2.GaussianBlur(b*b,(7,7),1.5)-bx*bx
    ab=cv2.GaussianBlur(a*b,(7,7),1.5)-ax*bx
    c1,c2=6.5025,58.5225
    return float(np.mean((2*ax*bx+c1)*(2*ab+c2)/np.maximum((ax*ax+bx*bx+c1)*(axx+bxx+c2),1e-8)))

class BKTree:
    def __init__(self): self.root=None
    def add(self, value):
        if self.root is None: self.root=(value,{});return
        node=self.root
        while True:
            k, edges=node;d=(k^value).bit_count()
            if d==0:return
            if d not in edges:edges[d]=(value,{});return
            node=edges[d]
    def find(self,value,radius):
        if self.root is None:return []
        stack=[self.root];res=[]
        while stack:
            k,edges=stack.pop();d=(k^value).bit_count()
            if d<=radius:res.append((k,d))
            for dist,child in edges.items():
                if d-radius<=dist<=d+radius:stack.append(child)
        return res

class UnionFind:
    def __init__(self,items):self.parent={x:x for x in items};self.size={x:1 for x in items}
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        x,y=self.find(a),self.find(b)
        if x==y:return
        if self.size[x]<self.size[y]:x,y=y,x
        self.parent[y]=x;self.size[x]+=self.size[y]
