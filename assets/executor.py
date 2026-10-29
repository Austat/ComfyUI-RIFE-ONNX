import queue,threading,time,torch
import torch.nn.functional as F
from .scheduler import Scheduler
from .output_writer import OutputWriter
from .session_manager import SessionManager
from .contracts import RunMetrics
class Executor:
 def execute(self,rgb,alpha,contract,devices,cuts,request,progress=None):
  pairs=rgb.shape[0]-1; jobs=Scheduler.build(pairs,min(request.batch_size,pairs)); writer=OutputWriter(rgb,alpha,pairs,request.multiplier,request.alpha_policy); failures=[]; lock=threading.Lock(); metrics={}; workers=[]; h,w=rgb.shape[2:]; ph=(-h)%contract.alignment; pw=(-w)%contract.alignment
  for requested in devices:
   backend,session,active=SessionManager.acquire(requested,contract,h+ph,w+pw,min(request.batch_size,pairs),request); metric=RunMetrics(requested,active); metrics[(active,backend.device.device_id)]=metric
   def run(backend=backend,session=session,metric=metric):
    started=time.perf_counter()
    try:
     if backend.torch_device.type=="cuda":torch.cuda.set_device(backend.torch_device)
     with torch.inference_mode():
      while True:
       try:item=jobs.get_nowait()
       except queue.Empty:break
       try:
        ids=list(range(item.start_pair,item.end_pair)); metric.chunks+=1; metric.source_pairs+=len(ids); local_cuts=[cuts[i] for i in ids]; metric.scene_cuts+=sum(local_cuts); a=rgb[item.start_pair:item.end_pair].to(backend.torch_device); b=rgb[item.start_pair+1:item.end_pair+1].to(backend.torch_device)
        if ph or pw:a=F.pad(a,(0,pw,0,ph),mode="replicate"); b=F.pad(b,(0,pw,0,ph),mode="replicate")
        for local,pair in enumerate(ids):writer.write(pair,0,a[local])
        active_ids=[i for i,c in enumerate(local_cuts) if not c]; index=torch.tensor(active_ids,device=backend.torch_device) if active_ids else None; left=a.index_select(0,index).contiguous() if active_ids else None; right=b.index_select(0,index).contiguous() if active_ids else None; positions={v:i for i,v in enumerate(active_ids)}
        for step in range(1,request.multiplier):
         inferred=backend.infer(session,contract,left,right,step/request.multiplier) if active_ids else None
         if active_ids: metric.inference_evaluations+=len(active_ids)
         for local,pair in enumerate(ids):
          if local_cuts[local]:
           ratio=step/request.multiplier; frame=a[local] if request.scene_cut_policy=="previous" else b[local] if request.scene_cut_policy=="next" else a[local]*(1-ratio)+b[local]*ratio if request.scene_cut_policy=="crossfade" else (a[local] if step*2<request.multiplier else b[local])
          else: frame=inferred[positions[local]]
          writer.write(pair,step,frame); metric.generated_frames+=1
         if progress: progress.update(len(ids))
       finally:jobs.task_done()
    except Exception as exc:
     metric.errors.append(repr(exc))
     with lock:failures.append(exc)
    finally:metric.wall_seconds=time.perf_counter()-started
   workers.append(threading.Thread(target=run,daemon=False))
  for t in workers:t.start()
  for t in workers:t.join()
  if failures:raise RuntimeError(f"RIFE execution failed: {failures[0]}") from failures[0]
  return writer.finish(),metrics
