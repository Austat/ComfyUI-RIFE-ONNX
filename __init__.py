# file: ComfyUI/custom_nodes/ComfyUI-RIFE-ONNX/__init__.py

from .rife_onnx_node import RifeOnnxInterpolationNode
from .rife_onnx_node_low_vram import RifeOnnxInterpolationNodeLowVram

NODE_CLASS_MAPPINGS = {
    "RifeOnnxInterpolationNode": RifeOnnxInterpolationNode,
    "RifeOnnxInterpolationNodeLowVram": RifeOnnxInterpolationNodeLowVram
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RifeOnnxInterpolationNode": "RIFE Interpolation node (Multi-GPU ONNX)",
    "RifeOnnxInterpolationNodeLowVram": "RIFE Interpolation node LowVRAM (ONNX)",
}
