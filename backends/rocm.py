import onnxruntime as ort,torch
from .cuda import CudaBackend
class RocmBackend(CudaBackend):
 provider="ROCmExecutionProvider"; ort_device_type="cuda"
 def provider_options(self): return {"device_id":self.device.device_id or 0}
 def create_session(self,model,contract,height,width,batch,precision,workspace_mb,diagnostics=False):
  o=ort.SessionOptions(); o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; o.enable_profiling=diagnostics
  return ort.InferenceSession(str(model),sess_options=o,providers=[(self.provider,self.provider_options())])
