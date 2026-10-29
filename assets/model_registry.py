import hashlib,threading
from pathlib import Path
import folder_paths,onnxruntime as ort
from .config import MODEL_DIR
from .contracts import ModelContract
class ModelRegistry:
 _lock=threading.RLock(); _contracts={}
 @staticmethod
 def list_models():
  found={}
  for d in folder_paths.get_folder_paths("Rife"):
   for p in Path(d).glob("*.onnx"): found.setdefault(p.name,p)
  result=sorted(found); preferred="rife47_ensemble_True_scale_1_sim.onnx"
  if preferred in result: result.remove(preferred); result.insert(0,preferred)
  return result or ["(no .onnx models found)"]
 @staticmethod
 def resolve(name):
  for d in folder_paths.get_folder_paths("Rife"):
   p=Path(d)/name
   if p.is_file(): return p.resolve()
  raise FileNotFoundError(f"RIFE model {name!r} not found in {MODEL_DIR}")
 @classmethod
 def inspect(cls,path):
  stat=path.stat(); key=(str(path),stat.st_mtime_ns,stat.st_size)
  with cls._lock:
   if key in cls._contracts:return cls._contracts[key]
  o=ort.SessionOptions(); o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL; o.intra_op_num_threads=1; o.inter_op_num_threads=1
  s=ort.InferenceSession(str(path),sess_options=o,providers=["CPUExecutionProvider"]); ins={x.name:x for x in s.get_inputs()}; outs=s.get_outputs(); missing=[x for x in ("img0","img1","timestep") if x not in ins]
  if missing: raise RuntimeError(f"Unsupported ONNX model: missing inputs {missing}")
  for n in ("img0","img1"):
   if len(ins[n].shape)!=4 or (isinstance(ins[n].shape[1],int) and ins[n].shape[1]!=3): raise RuntimeError(f"{n} must be NCHW with 3 channels")
  if not outs: raise RuntimeError("ONNX model has no outputs")
  h=hashlib.sha256();
  with path.open("rb") as f:
   for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
  c=ModelContract(path,h.hexdigest(),outs[0].name,tuple(outs[0].shape),outs[0].type,tuple(ins["timestep"].shape));
  with cls._lock: cls._contracts={key:c}
  return c
