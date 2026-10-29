import threading,torch
class OutputWriter:
 def __init__(self,rgb,alpha,pairs,multiplier,alpha_policy):
  self.rgb=rgb; self.alpha=alpha; self.multiplier=multiplier; self.alpha_policy=alpha_policy; ch=4 if alpha is not None and alpha_policy!="drop" else 3; self.tensor=torch.empty((pairs*multiplier+1,rgb.shape[2],rgb.shape[3],ch),dtype=torch.float32); self.lock=threading.Lock()
 def write(self,pair,step,frame):
  idx=pair*self.multiplier+step; frame=frame[:,:self.rgb.shape[2],:self.rgb.shape[3]].detach().to("cpu").permute(1,2,0)
  with self.lock:
   self.tensor[idx,:,:,:3].copy_(frame)
   if self.tensor.shape[-1]==4:
    a,b=self.alpha[pair],self.alpha[pair+1]; alpha=a+(b-a)*(step/self.multiplier) if self.alpha_policy=="linear" else (a if step*2<self.multiplier else b); self.tensor[idx,:,:,3:4].copy_(alpha)
 def finish(self):
  self.tensor[-1,:,:,:3].copy_(self.rgb[-1].permute(1,2,0))
  if self.tensor.shape[-1]==4:self.tensor[-1,:,:,3:4].copy_(self.alpha[-1])
  return self.tensor
