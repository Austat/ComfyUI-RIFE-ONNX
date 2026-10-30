import json, logging, shutil
from pathlib import Path
import folder_paths, onnxruntime as ort, torch
log=logging.getLogger("rife_onnx")
MODEL_DIR=Path(folder_paths.models_dir)/"Rife"; MODEL_DIR.mkdir(parents=True,exist_ok=True)
BASE_PATH=Path(getattr(folder_paths,"base_path",MODEL_DIR.parent))
CACHE_DIR=BASE_PATH/"cache"/"rife_onnx"; CACHE_DIR.mkdir(parents=True,exist_ok=True)
MAX_CACHED_SESSIONS=8
try: folder_paths.add_model_folder_path("Rife",str(MODEL_DIR))
except Exception as exc: log.warning("Could not register Rife model directory: %s",exc)
def invalidate_runtime_cache():
    meta=CACHE_DIR/"runtime_versions.json"
    current={"torch":torch.__version__,"cuda":torch.version.cuda,"onnxruntime":ort.__version__}
    try: previous=json.loads(meta.read_text("utf-8")) if meta.exists() else None
    except (OSError,ValueError,TypeError): previous=None
    if previous==current:return False
    # Preserve the original working script's TensorRT invalidation behavior.
    for path in (BASE_PATH/"trt_engine_cache"/"rife",BASE_PATH/"trt_timing_cache"/"rife"):
        if path.is_dir(): shutil.rmtree(path,ignore_errors=True)
    meta.write_text(json.dumps(current,indent=2),encoding="utf-8"); return True
