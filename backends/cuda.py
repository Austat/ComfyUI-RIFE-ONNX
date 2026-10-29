from collections import OrderedDict
import os,threading,onnxruntime as ort,torch
from .base import Backend
class CudaBackend(Backend):
 provider="CUDAExecutionProvider"; ort_device_type="cuda"
 def __init__(self,device): super().__init__(device); self.local=threading.local(); self.cache_size=max(1,int(os.getenv("ORT_IO_BINDING_CACHE_SIZE","4")))
 @property
 def torch_device(self): return torch.device("cuda",self.device.device_id or 0)
 def provider_options(self): return {"device_id":self.device.device_id or 0,"arena_extend_strategy":"kSameAsRequested","cudnn_conv_algo_search":"DEFAULT","do_copy_in_default_stream":True}
 def create_session(self,model,contract,height,width,batch,precision,workspace_mb,diagnostics=False):
  o=ort.SessionOptions(); o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; o.enable_profiling=diagnostics
  return ort.InferenceSession(str(model),sess_options=o,providers=[(self.provider,self.provider_options()),"CPUExecutionProvider"])
 def infer(self,session,contract,img0,img1,timestep):
  if img0.device.type!="cuda" or img0.device!=img1.device or img0.shape!=img1.shape: raise ValueError("CUDA inputs must share device and shape")
  img0=img0.contiguous(); img1=img1.contiguous(); shape,dtype=self.output_spec(session,contract,img0); key=(id(session),img0.device.index,tuple(img0.shape),img0.dtype,shape,dtype)
  cache=getattr(self.local,"buffers",None)
  if cache is None: cache=OrderedDict(); self.local.buffers=cache
  buf=cache.pop(key,None) or {"t":torch.empty((img0.shape[0],),device=img0.device,dtype=torch.float32),"out":torch.empty(shape,device=img0.device,dtype=dtype),"binding":session.io_binding()}; cache[key]=buf
  while len(cache)>self.cache_size: cache.popitem(last=False)
  buf["t"].fill_(float(timestep)); b=buf["binding"]; b.clear_binding_inputs(); b.clear_binding_outputs(); did=self.device.device_id or 0
  for name,t in (("img0",img0),("img1",img1),("timestep",buf["t"])): b.bind_input(name,self.ort_device_type,did,self.torch_to_numpy_dtype(t.dtype),tuple(t.shape),t.data_ptr())
  out=buf["out"]; b.bind_output(contract.output_name,self.ort_device_type,did,self.torch_to_numpy_dtype(out.dtype),tuple(out.shape),out.data_ptr()); b.synchronize_inputs(); session.run_with_iobinding(b); b.synchronize_outputs(); torch.cuda.synchronize(did); return out
