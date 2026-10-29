import time,torch
from tqdm import tqdm
from .contracts import InterpolationRequest
from .device_manager import DeviceManager
from .executor import Executor
from .metrics import emit
from .model_registry import ModelRegistry
from .scene_detection import SceneDetector
from .session_manager import SessionManager
class RifeOnnxInterpolationNode:
 RETURN_TYPES=("IMAGE",); RETURN_NAMES=("interpolated_images",); FUNCTION="interpolate"; CATEGORY="image/video/interpolation"
 @classmethod
 def INPUT_TYPES(cls):
  return {"required":{"images":("IMAGE",),"model_file":(ModelRegistry.list_models(),),"provider":(DeviceManager.provider_choices(),),"devices":(DeviceManager.device_choices(),),"multiplier":("INT",{"default":2,"min":2,"max":16}),"batch_size":("INT",{"default":1,"min":1,"max":16}),"fallback_policy":(["strict","same_vendor","allow_cpu"],),"precision":(["auto","fp32","fp16"],),"workspace_mb":("INT",{"default":2048,"min":256,"max":16384,"step":256}),"keep_model_loaded":("BOOLEAN",{"default":True}),"scene_cut_detection":("BOOLEAN",{"default":True}),"scene_cut_pixel_threshold":("FLOAT",{"default":.22,"min":0.,"max":1.,"step":.01}),"scene_cut_histogram_threshold":("FLOAT",{"default":.35,"min":0.,"max":1.,"step":.01}),"scene_cut_policy":(["nearest","previous","next","crossfade"],),"alpha_policy":(["drop","nearest","linear"],),"diagnostics":("BOOLEAN",{"default":False})},"optional":{"device_ids":("STRING",{"default":""})}}
 def interpolate(self,images,model_file,provider,devices,multiplier,batch_size,fallback_policy,precision,workspace_mb,keep_model_loaded,scene_cut_detection,scene_cut_pixel_threshold,scene_cut_histogram_threshold,scene_cut_policy,alpha_policy,diagnostics,device_ids=""):
  started=time.perf_counter()
  if images.dim()==3:images=images.unsqueeze(0)
  if images.dim()!=4 or images.shape[-1] not in (3,4):raise ValueError(f"Expected NHWC RGB/RGBA, got {tuple(images.shape)}")
  if not torch.isfinite(images).all():raise ValueError("Input contains NaN or Inf")
  source=images.detach().to("cpu",torch.float32).clamp(0,1); rgb=source[...,:3].permute(0,3,1,2).contiguous(); alpha=source[...,3:4] if source.shape[-1]==4 else None
  if source.shape[0]<2:return (source[...,:3] if alpha_policy=="drop" else source,)
  pairs=rgb.shape[0]-1; progress=tqdm(total=pairs*(int(multiplier)-1),desc="RIFE interpolation",unit="frame",dynamic_ncols=True)
  try:
   contract=ModelRegistry.inspect(ModelRegistry.resolve(model_file)); selected=DeviceManager.resolve(provider,devices,device_ids); request=InterpolationRequest(int(multiplier),int(batch_size),fallback_policy,precision,int(workspace_mb),bool(scene_cut_detection),float(scene_cut_pixel_threshold),float(scene_cut_histogram_threshold),scene_cut_policy,alpha_policy,bool(keep_model_loaded),bool(diagnostics)); cuts=SceneDetector.detect(rgb,request.scene_cut_pixel_threshold,request.scene_cut_histogram_threshold) if request.scene_cut_enabled else [False]*pairs; output,metrics=Executor().execute(rgb,alpha,contract,selected,cuts,request,progress)
   if not keep_model_loaded:SessionManager.clear()
   if diagnostics:emit(metrics,pairs,time.perf_counter()-started)
   return (output,)
  finally:progress.close()
