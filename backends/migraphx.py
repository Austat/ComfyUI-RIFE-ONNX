import onnxruntime as ort
from .rocm import RocmBackend
class MigraphXBackend(RocmBackend):
 provider="MIGraphXExecutionProvider"
 def create_session(self,model,contract,height,width,batch,precision,workspace_mb,diagnostics=False):
  o=ort.SessionOptions(); o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; o.enable_profiling=diagnostics
  return ort.InferenceSession(str(model),sess_options=o,providers=[(self.provider,{"device_id":self.device.device_id or 0,"migraphx_fp16_enable":int(precision in {"auto","fp16"})})])
