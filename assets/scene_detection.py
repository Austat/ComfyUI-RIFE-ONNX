import torch
import torch.nn.functional as F
class SceneDetector:
 @staticmethod
 def detect(frames,pixel_threshold,histogram_threshold,size=64,bins=32):
  sample=F.interpolate(frames,size=(size,size),mode="area").clamp(0,1); luma=sample[:,0:1]*.2126+sample[:,1:2]*.7152+sample[:,2:3]*.0722; pixels=torch.mean(torch.abs(luma[:-1]-luma[1:]),dim=(1,2,3)); result=[]
  for i,pixel in enumerate(pixels):
   a=torch.histc(luma[i],bins=bins,min=0,max=1); b=torch.histc(luma[i+1],bins=bins,min=0,max=1); a/=a.sum().clamp_min(1); b/=b.sum().clamp_min(1); hist=.5*torch.sum(torch.abs(a-b)); result.append(bool(pixel>=pixel_threshold and hist>=histogram_threshold))
  return result
