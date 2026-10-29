import os,threading,numpy as np,onnxruntime as ort,torch
from .base import Backend
class CpuBackend(Backend):
 provider="CPUExecutionProvider"
 def __init__(self,device): super().__init__(device); self.local=threading.local()
 @property
 def torch_device(self): return torch.device("cpu")
 def create_session(self,model,contract,height,width,batch,precision,workspace_mb,diagnostics=False):
  o=ort.SessionOptions(); o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; o.intra_op_num_threads=max(0,int(os.getenv("ORT_CPU_INTRA_OP_THREADS","0"))); o.inter_op_num_threads=max(0,int(os.getenv("ORT_CPU_INTER_OP_THREADS","1"))); o.execution_mode=ort.ExecutionMode.ORT_PARALLEL if o.inter_op_num_threads>1 else ort.ExecutionMode.ORT_SEQUENTIAL; o.enable_profiling=diagnostics
  return ort.InferenceSession(str(model),sess_options=o,providers=[self.provider])
 def infer(self,session,contract,img0,img1,timestep):
  if img0.shape!=img1.shape: raise ValueError("img0 and img1 must have identical shapes")
  t=getattr(self.local,"t",None)
  if t is None or t.shape!=(img0.shape[0],): t=np.empty((img0.shape[0],),np.float32); self.local.t=t
  t.fill(float(timestep)); feed={"img0":np.ascontiguousarray(img0.numpy(),dtype=np.float32),"img1":np.ascontiguousarray(img1.numpy(),dtype=np.float32),"timestep":t}; out=session.run([contract.output_name],feed)
  return torch.from_numpy(np.asarray(out[0]))
