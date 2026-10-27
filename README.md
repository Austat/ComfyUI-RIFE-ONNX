# ComfyUI RIFE ONNX Frame Interpolation
<br>
<br>

*Ever had a need for higher and smoother video playback? Maybe your video had lagg or some stutters here and there. 
RIFE- interpolation is just for cases like that, it smooths the playback by inserting frames inbetween.*

<br>
<br>
<img width="649" height="520" alt="image" src="https://github.com/user-attachments/assets/1741cdb4-2f3f-47cb-9113-d30944146546" />

A memory-conscious custom node for ComfyUI that performs video frame interpolation with a RIFE ONNX- model. 
It's an ONNX- port of [Practical-RIFE](https://github.com/hzwer/Practical-RIFE).

The node inserts AI-generated intermediate frames between consecutive input images. 
Its main design goal is to reduce peak GPU-memory usage by keeping the full image sequence in system RAM and still use fast TensorRT. 
It's achieved by transferring only one frame pair to the accelerator at a time, and moving each generated frame back to the CPU immediately.

Can use following methods:

```text
→ TensorrtExecutionProvider
→ CUDAExecutionProvider
→ ROCmExecutionProvider
→ CPUExecutionProvider
```

Quick setup:

```text
       Navigate to your ComfyUI custom_nodes directory. 

       TensorRT based operations need nVidia NVIDIA TensorRT 10.x. You can get it from here:
       https://developer.nvidia.com/tensorrt/download/10x

       Run:

       git clone https://github.com/Austat/ComfyUI-RIFE-ONNX

       cd ComfyUI-RIFE-ONNX

       pip install -r requirements.txt

       Download needed onnx - files to your ComfyUI or custom models/Rife/ - folder.
       ie.

       - rife_v4.25_comfy.onnx
       - rife_v4.25_lite_comfy.onnx
       - rife_v4.26_comfy.onnx
       - rife_v4.26_heavy_comfy.onnx

       Restart ComfyUI.
```

> [!IMPORTANT]
> This repository contains the node implementation, and couple of RIFE ONNX models. You can also provide your own compatible `.onnx` models.
> This repo has "models" - folder that has couple of them for start.

## Features

- Integrates as a ComfyUI `IMAGE` to `IMAGE` node.
- Supports interpolation multipliers from **2x to 16x**.
- Keeps the complete input and output sequence in CPU RAM rather than VRAM.
- Processes only one adjacent frame pair at a time.
- Uses ONNX Runtime I/O binding for CUDA and TensorRT execution.
- Writes GPU inference output directly into a preallocated PyTorch CUDA tensor.
- Supports TensorRT, CUDA, ROCm, and CPU execution-provider selection with automatic fallback.
- Pads image dimensions to multiples of 64 and crops results back to the original size.
- Reuses compatible ONNX Runtime sessions when requested.
- Invalidates TensorRT caches when the PyTorch, CUDA, or ONNX Runtime version changes.
- Provides optional timing, provider, padding, and CUDA-memory diagnostics.
- Sanitizes `NaN` and infinite model output and clamps pixels to `[0, 1]`.

<details>
  <summary>Details about RIFE-method with ONNX-framework</summary>

## What the node does

Given an ordered batch of frames:

```text
F0, F1, F2, ...
```

and a multiplier `m`, the node evaluates the RIFE model at evenly spaced time values between every pair:

```text
1/m, 2/m, ... (m-1)/m
```

For example, a multiplier of `4` transforms:

```text
F0, F1
```

into:

```text
F0, I(0.25), I(0.50), I(0.75), F1
```

For `N` input frames, the output count is:

```text
(N - 1) * multiplier + 1
```

Examples:

| Input frames | Multiplier | Output frames |
|---:|---:|---:|
| 2 | 2 | 3 |
| 10 | 2 | 19 |
| 10 | 4 | 37 |
| 100 | 8 | 793 |

The node returns the original frames and generated transition frames as a single ComfyUI image batch in `NHWC` RGB format.

## Where it is good

This implementation is especially useful for:

- **GPUs with limited VRAM.** The sequence does not remain resident on the GPU, and completed outputs are copied back to system RAM immediately.
- **Long frame sequences.** VRAM use is driven mainly by the current padded frame pair, model/session memory, and one output tensor rather than by the total number of frames.
- **ComfyUI video workflows.** It accepts and returns standard ComfyUI image batches.
- **Repeated runs at one resolution.** Session reuse and TensorRT engine/timing caches can reduce repeated setup work.
- **Provider experimentation.** TensorRT, CUDA, ROCm, and CPU can be selected from the node UI, with fallback providers attempted if the requested provider fails.
- **Debugging deployment problems.** Diagnostics expose active provider, dimensions, padding, stage timings, output count, and CUDA-memory information.
- **Avoiding hidden full-frame copies.** The CUDA path binds ONNX Runtime directly to PyTorch buffers instead of converting a GPU output through NumPy.

## Where it is bad

This node makes deliberate trade-offs and is not ideal in every situation:

- **It can be slower than a full-GPU pipeline.** Every pair is copied from CPU to GPU, and every generated frame is copied back to CPU synchronously.
- **System RAM use can become large.** The full input batch, every generated float32 frame, and the final concatenated output are held in CPU memory. `torch.cat` also allocates the final contiguous batch, so peak RAM can temporarily exceed the stored generated-frame list.
- **High multipliers scale linearly.** Each extra intermediate frame requires another model inference and another stored float32 image.
- **TensorRT is resolution-specific here.** The session key and optimization profile include padded height and width. Changing resolution can rebuild a session or TensorRT engine.
- **TensorRT startup may be expensive.** Initial engine construction can take substantially longer than later cached runs.
- **The GPU path is NVIDIA-oriented.** Direct CUDA I/O binding is used only when the active provider is TensorRT or CUDA. The ROCm and CPU paths use NumPy feeds.
- **Only the first three channels are retained.** Alpha and any channels after RGB are discarded.
- **Only batch size 1 is supported by the model call.** ComfyUI batches are processed as adjacent frame pairs, one pair at a time.
- **There is no scene-cut detection.** The model is asked to interpolate every adjacent pair, including unrelated shots, which can create severe transition artifacts.
- **There is no recursive interpolation.** All intermediate timestamps are inferred directly from the same two endpoint frames.
- **There is no audio handling.** The node operates only on image batches.
- **There is no tiling.** Very large individual frames may still exceed VRAM even though the sequence is streamed pair by pair.
- **The model contract is strict.** Input names, layout, channels, output shape, and float32 assumptions must match the implementation.
- **A single class-level session is shared.** A new model, provider, resolution, or diagnostics setting replaces the previous cached session.
- **GPU synchronization favors correctness over throughput.** Explicit synchronizations simplify buffer safety but reduce opportunities to overlap copies and inference.

## Memory behavior

### VRAM

The full video is never copied to the GPU. For each adjacent pair, the node holds approximately:

- two padded float32 input frames;
- one padded float32 output frame;
- model weights and execution-provider allocations;
- TensorRT or CUDA workspace and runtime overhead.

A raw RGB float32 tensor for one frame uses approximately:

```text
width * height * 3 * 4 bytes
```

At 1920x1080, one unpadded RGB float32 frame is about 23.7 MiB. Three such tensors are about 71.2 MiB before model weights, intermediate activations, padding, allocator reservations, and provider workspace.

### System RAM

CPU memory grows with the number of input and output frames. A useful lower-bound estimate for the final output tensor is:

```text
output_frames * width * height * 3 * 4 bytes
```

A 100-frame 1920x1080 input with a `4x` multiplier produces 397 float32 RGB frames. The final output tensor alone is roughly 9.2 GiB. Peak usage can be higher because the generated-frame list still exists while the final concatenated tensor is allocated.

> [!WARNING]
> “Low VRAM” does not mean “low memory.” This implementation shifts sequence storage from GPU memory to system RAM.

## Installation

### 1. Create the custom-node directory

Clone the Repository 

```text
       Navigate to your ComfyUI custom_nodes directory. 

       TensorRT based operations need nVidia NVIDIA TensorRT 10.x. You can get it from here:
       https://developer.nvidia.com/tensorrt/download/10x

       Run:

       git clone https://github.com/Austat/ComfyUI-RIFE-ONNX

       cd ComfyUI-RIFE-ONNX

       pip install -r requirements.txt

       Download needed onnx - files to your ComfyUI or custom models/Rife/ - folder.

       Restart ComfyUI.
```

A minimal layout is:

```text
ComfyUI/
├── custom_nodes/
│   └── ComfyUI-RIFE-ONNX/
│       ├── __init__.py
│       └── rife_onnx_node.py
└── models/
    └── Rife/
        └── your_model.onnx
```

Choose the ONNX Runtime package that matches the intended provider. The base `onnxruntime` package is appropriate for CPU execution. GPU/TensorRT/ROCm availability depends on the installed ONNX Runtime build and compatible system libraries.

Confirm the providers visible to ONNX Runtime:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### 2. Add a compatible model

Place one or more `.onnx` files under:

```text
ComfyUI/models/Rife/
```

The node prefers this filename when present:

```text
*.onnx
```

The implementation validates that the model has:

- inputs named `img0`, `img1`, and `timestep`;
- four-dimensional `img0` and `img1` inputs;
- three image channels when the channel dimension is statically declared;
- at least one output.

The runtime additionally assumes:

- `img0` and `img1` are float32 tensors in `[1, 3, H, W]` layout;
- `timestep` accepts a float32 tensor shaped as `[1]`;
- the first output is float32 and has the same shape as an input image;
- image dimensions can be processed after padding to a multiple of 64;
- pixel values use the `[0, 1]` range.

### 5. Restart ComfyUI

After restart, find the node under:

```text
image/video/interpolation
```

Display name:

```text
RIFE ONNX Interpolation
```

## Node inputs

| Input | Type | Description |
|---|---|---|
| `images` | `IMAGE` | Ordered ComfyUI image batch. Expected as `NHWC`, with RGB or RGBA data. Only RGB is retained. |
| `model_file` | selection | An `.onnx` file discovered in registered `rife` model folders. |
| `provider` | selection | Preferred ONNX Runtime execution provider. |
| `multiplier` | integer, 2-16 | Number of output intervals created from each original frame interval. |
| `keep_model_loaded` | boolean | Keeps the current ONNX Runtime session cached for compatible later runs. |
| `diagnostics` | boolean | Enables profiling/logging and prints dimensions, timing, provider, frame count, and CUDA memory data. |

## Node output

| Output | Type | Description |
|---|---|---|
| `interpolated_images` | `IMAGE` | RGB float32 image batch in ComfyUI `NHWC` format. |

## Provider behavior

The selected provider is tried first. The fallback chain is:

```text
selected provider
→ TensorrtExecutionProvider
→ CUDAExecutionProvider
→ ROCmExecutionProvider
→ CPUExecutionProvider
```

Duplicate entries are removed while preserving order. A provider is skipped when it is not listed by `onnxruntime.get_available_providers()`. If session construction fails, the error is printed and the next candidate is attempted.

### TensorRT

The TensorRT configuration enables:

- FP16 execution;
- engine caching;
- timing caching;
- CUDA fallback;
- fixed min/opt/max profiles for the current padded resolution;
- a configured maximum workspace of **6 GiB**.

> [!NOTE]
> The TensorRT workspace is limited to 6 GiB, the actual `trt_max_workspace_size` value is `24 * 1024 * 1024 * 1024`.
> The executable configuration therefore uses 6 GiB.


TensorRT cache directories are relative to the ComfyUI process working directory:

```text
./trt_engine_cache/rife
./trt_timing_cache/rife
./trt_cache_metadata/rife_runtime_versions.json
```

### CUDA

The CUDA provider uses:

- device 0;
- `kSameAsRequested` arena growth;
- default cuDNN convolution algorithm search;
- copies on the default stream;
- CPU fallback.

For GPU inference, ONNX Runtime I/O binding points directly at contiguous PyTorch CUDA allocations. This avoids an ONNX Runtime output to NumPy conversion and an extra full-frame CPU allocation.

### ROCm and CPU

The non-CUDA path converts tensors to contiguous CPU NumPy arrays, calls `session.run`, and wraps the first NumPy output with `torch.from_numpy`.

ROCm can be selected and used as an ONNX Runtime provider, but this implementation does not use a ROCm/PyTorch device I/O-binding path. Inputs are supplied from CPU memory.

## How it works

### 1. Model discovery

At import time, the node creates `ComfyUI/models/rife` and attempts to register it with ComfyUI. `INPUT_TYPES()` scans all registered `rife` model directories for `.onnx` files and places the preferred RIFE- filename first.

### 2. Runtime cache validation

Before interpolation, the node records:

- `torch.__version__`;
- `torch.version.cuda`;
- `onnxruntime.__version__`.

If those values differ from the stored metadata, TensorRT engine and timing caches are deleted and the cached session/model metadata is reset. This reduces the chance of reusing incompatible TensorRT artifacts after a runtime upgrade.

### 3. ONNX model validation

A lightweight CPU session reads model metadata with graph optimization disabled. Metadata is cached by absolute model path. The node validates required names, image rank, statically declared channel count, and output existence.

### 4. Input normalization

The node accepts a single image or an image batch. A three-dimensional image is promoted to a batch of one. Inputs must have at least three channels.

The batch is then:

1. reduced to RGB;
2. detached from any existing graph;
3. converted to CPU float32;
4. permuted from `NHWC` to `NCHW`;
5. made contiguous.

If fewer than two frames are supplied, the original batch is returned without loading or running a model.

### 5. Padding

Height and width are padded on the bottom and right to the next multiple of 64. Border pixels are replicated. Generated results are cropped to the original dimensions before storage.

### 6. Session creation and reuse

The session cache key includes:

```text
absolute model path
provider requested by the user
padded height
padded width
diagnostics setting
```

A matching session is reused. Otherwise, the old session is released, PyTorch's CUDA cache is emptied when available, and providers are tried in fallback order.

### 7. Pair-by-pair interpolation

For each adjacent input pair:

1. Copy only the two current frames to the selected device.
2. Apply replicate padding if needed.
3. Copy the first original frame back to the CPU output list.
4. Run inference for each intermediate timestamp.
5. Crop and copy each result immediately to CPU RAM.
6. Delete temporary GPU tensors before advancing to the next pair.

After all pairs, the final original frame is appended from CPU storage.

### 8. GPU I/O binding

For TensorRT and CUDA:

1. Allocate a contiguous float32 PyTorch CUDA output with the same shape as `img0`.
2. Bind `img0`, `img1`, and `timestep` by CUDA buffer pointer.
3. Bind model output directly to the PyTorch output pointer.
4. Synchronize inputs.
5. Execute with `run_with_iobinding`.
6. Synchronize outputs and the CUDA device.

The result is already a PyTorch tensor and does not need DLPack or NumPy conversion.

### 9. Output sanitation and assembly

Each result is checked for `[1, C, H, W]` shape. `NaN`, positive infinity, and negative infinity are replaced with `0`, `1`, and `0`, respectively. Values are clamped in place to `[0, 1]`.

All CPU result tensors are concatenated, permuted back to `NHWC`, and returned to ComfyUI.

## Diagnostics

When `diagnostics` is enabled, the node prints:

- original input dimensions;
- right and bottom padding;
- padded model dimensions;
- initialized provider list and setup duration;
- input, output, and expected frame counts;
- selected multiplier;
- active provider;
- accumulated inference, transfer, and post-processing durations;
- CUDA allocated, reserved, free, and total memory after interpolation.

ONNX Runtime profiling is also enabled. The code does not explicitly call `session.end_profiling()`, so it does not print or manage the generated profile path.

## Performance guidance

- Prefer a CUDA or TensorRT-capable ONNX Runtime build when using an NVIDIA GPU.
- Keep `keep_model_loaded` enabled for repeated runs with the same model, provider, resolution, and diagnostics setting.
- Use TensorRT when repeated execution justifies engine build time and cache storage.
- Use lower interpolation multipliers when throughput or RAM is constrained.
- Avoid mixing resolutions in a batch. ComfyUI image tensors normally require a common shape, and each resolution change invalidates session reuse.
- Estimate CPU RAM before interpolating long, high-resolution sequences.
- Split very long videos into chunks if RAM is the limiting resource. Preserve one overlapping endpoint frame between chunks to avoid dropping an interval, then remove the duplicate endpoint when combining results.
- Add scene-cut detection before this node when processing edited video.

## Common problems

### No `.onnx` models found

Place a compatible model in:

```text
ComfyUI/models/rife/
```

Then restart ComfyUI so the node input list is refreshed.

### `Expected RIFE ONNX inputs ...`

The selected ONNX graph does not expose all required input names:

```text
img0, img1, timestep
```

Use a compatible export or modify the node to map the model's actual names.

### All execution providers failed

Check the providers exposed by the active ComfyUI Python environment:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Also review the ComfyUI console. The node prints a separate session-creation error for each attempted provider.

### CUDA out of memory

Possible causes include:

- the individual padded frame is too large;
- model activations or weights exceed available VRAM;
- the TensorRT workspace limit is too large for the installed GPU;
- another ComfyUI model remains resident;
- ONNX Runtime or PyTorch has reserved GPU memory.

Try CUDA instead of TensorRT, reduce resolution, unload other models, disable session retention, or lower `trt_max_workspace_size` in the source.

### System out of memory

Reduce multiplier, resolution, frame count per run, or split the workflow into chunks. Remember that the final float32 output and the pre-concatenation tensors coexist during assembly.

### Interpolation artifacts at cuts

The node has no scene-cut detection. Split the sequence at cuts or use a separate cut detector so RIFE only receives temporally related frame pairs.

### TensorRT rebuilds unexpectedly

A rebuild can occur when the model path, requested provider, padded resolution, diagnostics setting, PyTorch version, CUDA version, or ONNX Runtime version changes. Cache deletion can also force reconstruction.

### Alpha channel disappeared

The implementation explicitly selects `images[..., :3]`. Preserve and reconstruct alpha separately if your workflow requires it.

## Known implementation limitations

1. Device 0 is hard-coded for CUDA and TensorRT provider options.
2. Direct GPU I/O binding is limited to CUDA-compatible providers.
3. The assumed output dtype is float32.
4. The assumed output shape matches the input image tensor.
5. Only the first model output is used.
6. Dynamic or alternative timestep shapes are not detected.
7. Models requiring normalization outside `[0, 1]` are not supported automatically.
8. The entire generated sequence is accumulated before return.
9. `torch.cuda.synchronize()` is called for every inference, limiting asynchronous overlap.
10. TensorRT caches use relative paths and therefore depend on the launch working directory.
11. Cache invalidation tracks runtime versions but not GPU model, driver version, TensorRT version, model hash, or provider configuration.
12. Exceptions during interpolation close the progress bar, but partial outputs are not returned.
13. The helper `_ortvalue_to_torch()` is currently unused because the primary CUDA path writes directly into a PyTorch allocation.
14. The timing dictionary measures broad wall-clock sections rather than a benchmark-grade breakdown.
15. The error message from `_find_model()` points to the default model directory even if additional registered RIFE directories exist.

## Security and operational notes

- Treat ONNX models as software dependencies and obtain them from a source you trust.
- Cache directories are writable and recursively deleted when recorded runtime versions change. Run ComfyUI with an appropriate working directory and filesystem permissions.
- Diagnostics may create ONNX Runtime profiling data in the process environment. Review generated files before sharing logs or support bundles.
- The node catches broad exceptions when registering the model directory and when reading some metadata, which favors startup resilience but can hide configuration details unless the surrounding logs are inspected.

## Suggested improvements

Potential future enhancements include:

- configurable CUDA device selection;
- configurable TensorRT workspace size;
- tiled inference for oversized frames;
- bounded/chunked CPU output assembly;
- optional float16 staging where model compatibility allows it;
- scene-cut detection or a cut-mask input;
- alpha-channel preservation;
- ROCm I/O binding;
- stream-aware asynchronous transfers;
- pinned CPU memory and double buffering;
- model input/output name mapping;
- validation of input/output dtypes and timestep shape;
- cache keys that include model hashes and relevant runtime/provider versions;
- explicit ONNX Runtime profile finalization and profile-path reporting;
- automated tests for frame count, padding, provider fallback, cache invalidation, and invalid model contracts.

## Development notes

The ComfyUI registration keys are:

```python
NODE_CLASS_MAPPINGS = {
    "RifeOnnxInterpolation": RifeOnnxInterpolationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RifeOnnxInterpolation": "RIFE 4.7 ONNX Interpolation (Low VRAM)",
}
```

Recommended test coverage:

- one input frame returns unchanged;
- two frames with multipliers 2, 4, and 16 produce the expected count;
- dimensions divisible and not divisible by 64 preserve final size;
- RGBA input returns RGB;
- missing model inputs fail with a clear error;
- unavailable providers fall back in the documented order;
- session reuse occurs only for matching cache keys;
- runtime-version changes clear the TensorRT caches;
- `NaN` and infinite output are sanitized;
- `keep_model_loaded=False` releases the class-level session.
