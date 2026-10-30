from dataclasses import dataclass,field
from pathlib import Path
@dataclass(frozen=True)
class ModelContract:
 path:Path; fingerprint:str; output_name:str; output_shape:tuple; output_type:str; timestep_shape:tuple; alignment:int=64
@dataclass(frozen=True)
class DeviceSpec: provider:str; device_id:int|None; name:str
@dataclass(frozen=True)
class SessionKey: model_fingerprint:str; provider:str; device_id:int|None; height:int; width:int; batch_size:int; precision:str; workspace_mb:int; diagnostics:bool
@dataclass(frozen=True)
class WorkItem: start_pair:int; end_pair:int
@dataclass
class RunMetrics:
 device:DeviceSpec; active_provider:str=""; source_pairs:int=0; scene_cuts:int=0; inference_evaluations:int=0; generated_frames:int=0; chunks:int=0; input_staging_seconds:float=0.; execution_seconds:float=0.; output_seconds:float=0.; wall_seconds:float=0.; errors:list[str]=field(default_factory=list)
@dataclass(frozen=True)
class InterpolationRequest:
 multiplier:int; batch_size:int; fallback_policy:str; precision:str; workspace_mb:int; scene_cut_enabled:bool; scene_cut_pixel_threshold:float; scene_cut_histogram_threshold:float; scene_cut_policy:str; alpha_policy:str; keep_model_loaded:bool; diagnostics:bool
