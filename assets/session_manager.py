from collections import OrderedDict
import logging,threading,onnxruntime as ort
from ..backends import BACKENDS
from .config import MAX_CACHED_SESSIONS,invalidate_runtime_cache
from .contracts import DeviceSpec,SessionKey
log=logging.getLogger("rife_onnx")
class SessionManager:
 _lock=threading.RLock(); _cache=OrderedDict()
 @staticmethod
 def fallback_chain(provider,policy):
  if policy=="strict":return [provider]
  same={"TensorrtExecutionProvider":["TensorrtExecutionProvider","CUDAExecutionProvider"],"CUDAExecutionProvider":["CUDAExecutionProvider"],"ROCmExecutionProvider":["ROCmExecutionProvider","MIGraphXExecutionProvider"],"MIGraphXExecutionProvider":["MIGraphXExecutionProvider","ROCmExecutionProvider"]}.get(provider,[provider]); return same+(["CPUExecutionProvider"] if policy=="allow_cpu" else [])
 @classmethod
 def acquire(cls,device,contract,height,width,batch,request):
  if invalidate_runtime_cache(): cls.clear()
  available=set(ort.get_available_providers()); last=None
  for provider in cls.fallback_chain(device.provider,request.fallback_policy):
   if provider not in available:continue
   effective=DeviceSpec(provider,None if provider=="CPUExecutionProvider" else device.device_id,"CPU" if provider=="CPUExecutionProvider" else device.name); key=SessionKey(contract.fingerprint,provider,effective.device_id,height,width,batch,request.precision,request.workspace_mb,request.diagnostics)
   with cls._lock:
    if key in cls._cache: value=cls._cache.pop(key); cls._cache[key]=value; return value
   try:
    backend=BACKENDS[provider](effective); session=backend.create_session(contract.path,contract,height,width,batch,request.precision,request.workspace_mb,request.diagnostics); active=session.get_providers()[0]; value=(backend,session,active)
    with cls._lock:
     cls._cache[key]=value
     while len(cls._cache)>MAX_CACHED_SESSIONS:cls._cache.popitem(last=False)
    return value
   except Exception as exc:last=exc; log.warning("Provider %s failed: %s",provider,exc)
  raise RuntimeError(f"No execution provider succeeded. Last error: {last}")
 @classmethod
 def clear(cls):
  with cls._lock:cls._cache.clear()
