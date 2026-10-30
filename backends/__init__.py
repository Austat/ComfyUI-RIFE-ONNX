import onnxruntime as ort
from .base import Backend
from .cpu import CpuBackend
from .cuda import CudaBackend
from .tensorrt import TensorRTBackend
from .rocm import RocmBackend
from .migraphx import MigraphXBackend
BACKENDS={"CPUExecutionProvider":CpuBackend,"CUDAExecutionProvider":CudaBackend,"TensorrtExecutionProvider":TensorRTBackend,"ROCmExecutionProvider":RocmBackend,"MIGraphXExecutionProvider":MigraphXBackend}
def available_backends():
    available=set(ort.get_available_providers()); return {k:v for k,v in BACKENDS.items() if k in available}
