import os, onnxruntime as ort
from .cuda import CudaBackend
from ..assets.config import BASE_PATH
class TensorRTBackend(CudaBackend):
 provider="TensorrtExecutionProvider"
 def create_session(self,model,contract,height,width,batch,precision,workspace_mb,diagnostics=False):
  # Intentionally mirrors the known-good monolithic implementation. Do not generalize profiles.
  options=ort.SessionOptions(); options.intra_op_num_threads=max(1,min(os.cpu_count() or 8,16)); options.inter_op_num_threads=1; options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  if diagnostics: options.enable_profiling=True; options.log_severity_level=1
  did=0 if self.device.device_id is None else int(self.device.device_id)
  engine=BASE_PATH/"trt_engine_cache"/"rife"/f"device_{did}"; timing=BASE_PATH/"trt_timing_cache"/"rife"/f"device_{did}"; engine.mkdir(parents=True,exist_ok=True); timing.mkdir(parents=True,exist_ok=True)
  minimum=f"1x3x{height}x{width}"; optimum=f"{batch}x3x{height}x{width}"
  trt={"device_id":did,"trt_engine_cache_enable":True,"trt_engine_cache_path":str(engine),"trt_timing_cache_enable":True,"trt_timing_cache_path":str(timing),"trt_fp16_enable":True,"trt_int8_enable":False,"trt_dla_enable":False,"trt_max_workspace_size":6*1024*1024*1024,"trt_profile_min_shapes":f"img0:{minimum},img1:{minimum},timestep:1","trt_profile_opt_shapes":f"img0:{optimum},img1:{optimum},timestep:{batch}","trt_profile_max_shapes":f"img0:{optimum},img1:{optimum},timestep:{batch}"}
  return ort.InferenceSession(str(model),sess_options=options,providers=[(self.provider,trt),("CUDAExecutionProvider",self.provider_options()),"CPUExecutionProvider"])
