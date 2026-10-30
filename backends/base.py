from abc import ABC,abstractmethod
import numpy as np, torch
class Backend(ABC):
 provider:str
 def __init__(self,device): self.device=device
 @property
 @abstractmethod
 def torch_device(self):...
 @abstractmethod
 def create_session(self,model,contract,height,width,batch,precision,workspace_mb,diagnostics=False):...
 @abstractmethod
 def infer(self,session,contract,img0,img1,timestep):...
 @staticmethod
 def torch_to_numpy_dtype(dtype):
  m={torch.float16:np.float16,torch.float32:np.float32,torch.float64:np.float64}
  if dtype not in m: raise TypeError(f"Unsupported tensor dtype: {dtype}")
  return m[dtype]
 @staticmethod
 def output_spec(session,contract,reference):
  item=next((x for x in session.get_outputs() if x.name==contract.output_name),None)
  if item is None: raise ValueError(f"Output {contract.output_name!r} not present")
  shape=tuple(d if isinstance(d,int) and d>0 else int(reference.shape[i]) for i,d in enumerate(item.shape))
  dtype={"tensor(float16)":torch.float16,"tensor(float)":torch.float32,"tensor(double)":torch.float64}[item.type]
  return shape,dtype
