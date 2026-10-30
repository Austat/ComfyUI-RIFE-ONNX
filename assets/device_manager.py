import onnxruntime as ort,torch
from .contracts import DeviceSpec
NVIDIA={"CUDAExecutionProvider","TensorrtExecutionProvider"}; ORDER=["TensorrtExecutionProvider","CUDAExecutionProvider","ROCmExecutionProvider","MIGraphXExecutionProvider","CPUExecutionProvider"]
class DeviceManager:
 @staticmethod
 def provider_choices():
  a=set(ort.get_available_providers()); return [p for p in ORDER if p in a] or ["CPUExecutionProvider"]
 @staticmethod
 def device_choices():
  c=["Auto","CPU"]
  if torch.cuda.is_available():
   g=[f"Device {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]; c[1:1]=g
   if len(g)>1:c.insert(1,"Device "+":".join(map(str,range(len(g))))+": All GPUs")
  return c
 @classmethod
 def resolve(cls,provider,selection,explicit=""):
  if provider=="CPUExecutionProvider":return [DeviceSpec(provider,None,"CPU")]
  text=(explicit or selection or "Auto").strip()
  if text.lower()=="auto": ids=list(range(torch.cuda.device_count())) if provider in NVIDIA else [0]
  else:
   if text.lower().startswith("device "): text=text[7:].split(" ",1)[0].rstrip(":")
   ids=[]
   for token in text.replace(",",":").replace(";",":").replace(" ","").split(":"):
    if token and int(token) not in ids: ids.append(int(token))
  if not ids: raise ValueError("No devices selected")
  return [DeviceSpec(provider,i,torch.cuda.get_device_name(i) if torch.cuda.is_available() and i<torch.cuda.device_count() else f"Device {i}") for i in ids]
