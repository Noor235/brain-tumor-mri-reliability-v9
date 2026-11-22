"""STEP 10: illustrative Grad-CAM only. Visual attention is NOT proof of faithfulness."""
import csv
import numpy as np
import cv2
import torch
from PIL import Image,ImageDraw
from . import config as C
from .common import output_dir,source_image
from .verify import verify
from .transforms_v9 import transform
from .model import build_model
from .train_utils import device

class GradCAM:
    def __init__(self,model,layer):
        self.activation=None;self.gradient=None
        self.hook=layer.register_forward_hook(self._forward)
        self.grad_hook=layer.register_full_backward_hook(self._backward)
    def _forward(self,module,inp,out):self.activation=out.detach()
    def _backward(self,module,g_in,g_out):self.gradient=g_out[0].detach()
    def close(self):self.hook.remove();self.grad_hook.remove()
    def compute(self,model,x,target):
        model.zero_grad(set_to_none=True)
        logits=model(x);logits[0,target].backward()
        if self.activation is None or self.gradient is None:raise RuntimeError('Grad-CAM feature or gradient hook failed')
        grads=self.gradient[0];features=self.activation[0]
        weights=grads.mean((1,2),keepdim=True)
        cam=torch.relu((weights*features).sum(dim=0)).cpu().numpy()
        cam=cv2.resize(cam,(C.IMAGE_SIZE,C.IMAGE_SIZE),interpolation=cv2.INTER_LINEAR)
        if float(cam.max())>float(cam.min()):cam=(cam-cam.min())/(cam.max()-cam.min())
        else:cam=np.zeros_like(cam)
        return cam

def main():
    C.heading('STEP 10 / QUALITATIVE EXPLANATIONS: Grad-CAM (test examples only)')
    rows,_=verify(rehash=False,detail=False)
    if not C.CHECKPOINT.is_file():raise FileNotFoundError('Train V9 (step07) first.')
    dev=device();model=build_model(pretrained=False)
    model.load_state_dict(torch.load(C.CHECKPOINT,map_location='cpu',weights_only=True),strict=True)
    model=model.to(dev).eval()
    # Higher-resolution intermediate block selected from V9 layer diagnostic.
    # Uses the exact existing trained checkpoint; no retraining or ROI masking.
    layer=model.features[-4];cam=GradCAM(model,layer)
    out=output_dir('10_gradcam_features_minus4');tx=transform(False)
    try:
        for cls in C.CLASSES:
            examples=sorted((r for r in rows if r['split']=='test' and r['class_name']==cls),key=lambda r:r['file'])[:3]
            for idx,row in enumerate(examples,1):
                with Image.open(source_image(row['file'])) as src: img=src.convert('L')
                inp=tx(img).unsqueeze(0).to(dev)
                with torch.no_grad():
                    p=torch.softmax(model(inp),dim=1)[0]
                    target=int(p.argmax());confidence=float(p[target])
                heat=cam.compute(model,inp,target)
                h,w=heat.shape
                corner_fraction=float(heat[:h//4,3*w//4:].sum()/max(float(heat.sum()),1e-12))
                print(f'  {cls}_{idx}: upper-right corner heatmap mass = {corner_fraction:.1%}',flush=True)
                original=np.asarray(img.resize((224,224),Image.Resampling.BICUBIC),dtype=np.uint8)
                colors=cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255),cv2.COLORMAP_JET)
                colors=cv2.cvtColor(colors,cv2.COLOR_BGR2RGB)
                base=np.stack([original]*3,axis=2)
                overlay=np.uint8(np.clip(0.6*base+0.4*colors,0,255))
                canvas=Image.new('RGB',(3*224,255),(255,255,255))
                for k,image in enumerate((base,colors,overlay)):canvas.paste(Image.fromarray(image),(k*224,25))
                drawer=ImageDraw.Draw(canvas)
                drawer.text((6,6),f'True {cls} | predicted {C.CLASSES[target]} ({confidence:.1%})',fill=(0,0,0))
                filepath=out/f'{cls}_{idx}_gradcam.png';canvas.save(filepath)
                print('  Saved:',filepath.name,flush=True)
    finally:cam.close()
    print('Done. Updated Grad-CAM from features[-4] saved to results/10_gradcam_features_minus4/',flush=True)
if __name__=='__main__':main()
