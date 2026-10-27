# file: ComfyUI/custom_nodes/ComfyUI-RIFE-ONNX/rife_onnx_node.py

"""
Memory-efficient ONNX RIFE frame interpolation node for ComfyUI.

Main VRAM optimizations:
- Input image batch stays in CPU RAM.
- Only one frame pair is transferred to GPU at a time.
- Generated frames are moved back to CPU immediately.
- TensorRT workspace is limited to 2 GiB.
- ONNX Runtime CUDA memory arena uses kSameAsRequested.
"""

import json
import os
import shutil
import time
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from tqdm import tqdm

import folder_paths


RIFE_MODEL_DIR = os.path.join(folder_paths.models_dir, "Rife")
os.makedirs(RIFE_MODEL_DIR, exist_ok=True)

try:
    folder_paths.add_model_folder_path("Rife", RIFE_MODEL_DIR)
except Exception:
    pass


class RifeOnnxInterpolationNodeLowVram:
    """Memory-efficient RIFE ONNX interpolation with GPU I/O binding."""

    RUNTIME_VERSION_FILE = "./trt_cache_metadata/rife_runtime_versions.json"
    TRT_ENGINE_CACHE_PATH = "./trt_engine_cache/rife"
    TRT_TIMING_CACHE_PATH = "./trt_timing_cache/rife"

    _session = None
    _session_key = None
    _active_provider = None
    _metadata_cache: Dict[str, Tuple] = {}

    @classmethod
    def INPUT_TYPES(cls):
        files = []

        for directory in folder_paths.get_folder_paths("Rife"):
            if os.path.isdir(directory):
                files.extend(
                    name
                    for name in os.listdir(directory)
                    if name.lower().endswith(".onnx")
                    and name not in files
                )

        files.sort()

        preferred = "rife47_ensemble_True_scale_1_sim.onnx"

        if preferred in files:
            files.remove(preferred)
            files.insert(0, preferred)

        if not files:
            files = ["(no .onnx models found)"]

        return {
            "required": {
                "images": ("IMAGE",),
                "model_file": (files,),
                "provider": (
                    [
                        "TensorrtExecutionProvider",
                        "CUDAExecutionProvider",
                        "ROCmExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                ),
                "multiplier": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": 16,
                        "step": 1,
                    },
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "diagnostics": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("interpolated_images",)
    FUNCTION = "interpolate"
    CATEGORY = "image/video/interpolation"

    @staticmethod
    def _find_model(model_file):
        for directory in folder_paths.get_folder_paths("Rife"):
            path = os.path.join(directory, model_file)

            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f"RIFE model '{model_file}' not found. "
            f"Put it in: {RIFE_MODEL_DIR}"
        )

    @classmethod
    def _check_runtime_versions_and_invalidate_cache(cls):
        current = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "onnxruntime": ort.__version__,
        }

        directory = os.path.dirname(cls.RUNTIME_VERSION_FILE)

        if directory:
            os.makedirs(directory, exist_ok=True)

        previous = None

        if os.path.exists(cls.RUNTIME_VERSION_FILE):
            try:
                with open(
                    cls.RUNTIME_VERSION_FILE,
                    "r",
                    encoding="utf-8",
                ) as handle:
                    previous = json.load(handle)
            except (OSError, ValueError, TypeError):
                pass

        if previous == current:
            return

        print(
            "[RifeOnnx] Runtime versions changed; "
            "invalidating TensorRT caches."
        )

        for path in (
            cls.TRT_ENGINE_CACHE_PATH,
            cls.TRT_TIMING_CACHE_PATH,
        ):
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except OSError as exc:
                    print(
                        f"[RifeOnnx] Could not delete {path}: {exc}"
                    )

        try:
            with open(
                cls.RUNTIME_VERSION_FILE,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(current, handle, indent=2)

        except OSError as exc:
            print(
                f"[RifeOnnx] Could not write runtime metadata: {exc}"
            )

        cls._session = None
        cls._session_key = None
        cls._active_provider = None
        cls._metadata_cache.clear()

    @classmethod
    def _model_metadata(cls, model_path):
        key = os.path.abspath(model_path)

        if key in cls._metadata_cache:
            return cls._metadata_cache[key]

        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        inputs = tuple(
            (item.name, tuple(item.shape), item.type)
            for item in session.get_inputs()
        )

        outputs = tuple(
            (item.name, tuple(item.shape), item.type)
            for item in session.get_outputs()
        )

        metadata = (inputs, outputs)
        cls._metadata_cache[key] = metadata

        return metadata

    @classmethod
    def _validate_model(cls, model_path):
        inputs, outputs = cls._model_metadata(model_path)

        by_name = {
            name: (shape, dtype)
            for name, shape, dtype in inputs
        }

        required = ("img0", "img1", "timestep")

        missing = [
            name
            for name in required
            if name not in by_name
        ]

        if missing:
            raise RuntimeError(
                f"Expected RIFE ONNX inputs {required}, "
                f"found {tuple(by_name)}; missing {missing}."
            )

        for name in ("img0", "img1"):
            shape, _ = by_name[name]

            if len(shape) != 4:
                raise RuntimeError(
                    f"Expected {name} as NCHW tensor, got {shape!r}."
                )

            if isinstance(shape[1], int) and shape[1] != 3:
                raise RuntimeError(
                    f"Expected {name} to have 3 channels, "
                    f"got {shape[1]}."
                )

        if not outputs:
            raise RuntimeError("The ONNX model has no outputs.")

        return outputs[0][0]

    @staticmethod
    def _cuda_stream_pointer() -> Optional[int]:
        if not torch.cuda.is_available():
            return None

        try:
            return int(
                torch.cuda.current_stream().cuda_stream
            )
        except Exception:
            return None

    @classmethod
    def _try_create_session(
        cls,
        model_path,
        provider,
        height,
        width,
        diagnostics,
    ):
        if provider not in ort.get_available_providers():
            print(
                f"[RifeOnnx] Provider unavailable: {provider}"
            )
            return None

        options = ort.SessionOptions()

        options.intra_op_num_threads = max(
            1,
            min(os.cpu_count() or 8, 16),
        )

        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if diagnostics:
            options.enable_profiling = True
            options.log_severity_level = 1

        stream_ptr = cls._cuda_stream_pointer()

        cuda_options = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
        }

        if stream_ptr is not None:
            cuda_options["user_compute_stream"] = str(stream_ptr)

        if provider == "TensorrtExecutionProvider":
            os.makedirs(
                cls.TRT_ENGINE_CACHE_PATH,
                exist_ok=True,
            )

            os.makedirs(
                cls.TRT_TIMING_CACHE_PATH,
                exist_ok=True,
            )

            image_shape = f"1x3x{height}x{width}"

            profile = (
                f"img0:{image_shape},"
                f"img1:{image_shape},"
                "timestep:1"
            )

            trt_options = {
                "device_id": 0,

                "trt_engine_cache_enable": True,
                "trt_engine_cache_path":
                    cls.TRT_ENGINE_CACHE_PATH,

                "trt_timing_cache_enable": True,
                "trt_timing_cache_path":
                    cls.TRT_TIMING_CACHE_PATH,

                "trt_fp16_enable": True,
                "trt_int8_enable": False,
                "trt_dla_enable": False,

                # Keep TRT workspace reasonable for an 8 GiB GPU.
                "trt_max_workspace_size":
                    6 * 1024 * 1024 * 1024,

                "trt_profile_min_shapes": profile,
                "trt_profile_opt_shapes": profile,
                "trt_profile_max_shapes": profile,
            }

            if stream_ptr is not None:
                trt_options["user_compute_stream"] = str(
                    stream_ptr
                )

            providers = [
                (
                    "TensorrtExecutionProvider",
                    trt_options,
                ),
                (
                    "CUDAExecutionProvider",
                    cuda_options,
                ),
                "CPUExecutionProvider",
            ]

        elif provider == "CUDAExecutionProvider":
            providers = [
                (
                    "CUDAExecutionProvider",
                    cuda_options,
                ),
                "CPUExecutionProvider",
            ]

        else:
            providers = [provider]

        started = time.perf_counter()

        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=providers,
            )

            print(
                f"[RifeOnnx] Initialized "
                f"{session.get_providers()} in "
                f"{time.perf_counter() - started:.2f}s"
            )

            return session

        except Exception as exc:
            print(
                f"[RifeOnnx] Provider {provider} failed: {exc}"
            )
            return None

    @classmethod
    def _load_session(
        cls,
        model_path,
        provider,
        height,
        width,
        diagnostics,
    ):
        key = (
            os.path.abspath(model_path),
            provider,
            height,
            width,
            bool(diagnostics),
        )

        if (
            cls._session is not None
            and cls._session_key == key
        ):
            return cls._session

        # Release an old session before building one for a
        # different resolution/provider.
        if cls._session is not None:
            cls._session = None
            cls._session_key = None
            cls._active_provider = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        chain = [
            provider,
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "CPUExecutionProvider",
        ]

        for candidate in dict.fromkeys(chain):
            session = cls._try_create_session(
                model_path,
                candidate,
                height,
                width,
                diagnostics,
            )

            if session is not None:
                cls._session = session
                cls._session_key = key
                cls._active_provider = (
                    session.get_providers()[0]
                )

                return session

        raise RuntimeError(
            "All ONNX Runtime execution providers failed."
        )

    @staticmethod
    def _ortvalue_to_torch(value):
        """
        Convert ONNX Runtime OrtValue to torch.Tensor.

        Prefer zero-copy DLPack. Do not silently create a potentially
        large CPU numpy copy when GPU DLPack conversion is unavailable.
        """
        try:
            if hasattr(value, "__dlpack__"):
                return torch.utils.dlpack.from_dlpack(value)

            if hasattr(value, "to_dlpack"):
                return torch.utils.dlpack.from_dlpack(
                    value.to_dlpack()
                )

        except Exception as exc:
            raise RuntimeError(
                "ONNX Runtime GPU output -> PyTorch DLPack conversion failed. "
                "Refusing numpy() fallback because it copies the complete "
                "GPU tensor into system RAM. "
                f"Original error: {exc}"
            ) from exc

        raise RuntimeError(
            "This ONNX Runtime OrtValue does not expose DLPack conversion. "
            "A numpy() fallback has intentionally been disabled because it "
            "causes an additional full-frame allocation in system RAM."
        )

    @classmethod
    def _run_gpu_iobinding(
        cls,
        session,
        output_name,
        img0,
        img1,
        timestep,
    ):
        """
        Run ONNX Runtime directly into a preallocated PyTorch CUDA tensor.

        This avoids:
        - OrtValue -> NumPy conversion
        - DLPack dependency
        - an additional full-frame CPU allocation
        - cloning the ONNX Runtime output
        """
        device_id = (
            img0.device.index
            if img0.device.index is not None
            else 0
        )

        # RIFE output has the same NCHW dimensions and dtype as img0.
        output = torch.empty_like(
            img0,
            device=img0.device,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )

        if not output.is_contiguous():
            output = output.contiguous()

        binding = session.io_binding()

        for name, tensor in (
            ("img0", img0),
            ("img1", img1),
            ("timestep", timestep),
        ):
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        # ONNX Runtime writes directly into the PyTorch CUDA allocation.
        binding.bind_output(
            name=output_name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )

        session.run_with_iobinding(binding)

        # Ensure ONNX Runtime has finished writing before the tensor is used.
        binding.synchronize_outputs()

        return output

    @staticmethod
    def _run_numpy(
        session,
        img0,
        img1,
        timestep,
    ):
        feed = {
            "img0": np.ascontiguousarray(
                img0.detach().cpu().numpy(),
                dtype=np.float32,
            ),
            "img1": np.ascontiguousarray(
                img1.detach().cpu().numpy(),
                dtype=np.float32,
            ),
            "timestep": np.ascontiguousarray(
                timestep.detach().cpu().numpy(),
                dtype=np.float32,
            ),
        }

        outputs = session.run(None, feed)

        if not outputs:
            raise RuntimeError(
                "The RIFE model returned no output."
            )

        return torch.from_numpy(
            np.asarray(outputs[0])
        )

    @classmethod
    def _infer(
        cls,
        session,
        output_name,
        img0,
        img1,
        t,
        use_cuda,
        diagnostics,
    ):
        device = img0.device

        timestep = torch.tensor(
            [float(t)],
            device=device,
            dtype=torch.float32,
        ).contiguous()

        if use_cuda:
            output = cls._run_gpu_iobinding(
                session,
                output_name,
                img0,
                img1,
                timestep,
            )

            if diagnostics:
                torch.cuda.synchronize(device)

        else:
            output = cls._run_numpy(
                session,
                img0,
                img1,
                timestep,
            )

        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError(
                "Expected RIFE output shape [1,C,H,W], "
                f"got {tuple(output.shape)}."
            )

        return torch.nan_to_num(
            output,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp_(0.0, 1.0)

    def interpolate(
        self,
        images,
        model_file,
        provider,
        multiplier,
        keep_model_loaded,
        diagnostics=False,
    ):
        if images.dim() == 3:
            images = images.unsqueeze(0)

        if images.dim() != 4 or images.shape[-1] < 3:
            raise ValueError(
                "Expected ComfyUI IMAGE in NHWC RGB(A) "
                f"format, got {tuple(images.shape)}."
            )

        if images.shape[0] < 2:
            return (images,)

        model_path = self._find_model(model_file)

        self._check_runtime_versions_and_invalidate_cache()

        output_name = self._validate_model(model_path)

        total, height, width, _ = images.shape

        pad_h = (32 - height % 32) % 32
        pad_w = (32 - width % 32) % 32

        padded_h = height + pad_h
        padded_w = width + pad_w

        session = self._load_session(
            model_path,
            provider,
            padded_h,
            padded_w,
            diagnostics,
        )

        use_cuda = (
            torch.cuda.is_available()
            and self._active_provider
            in (
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
            )
        )

        device = (
            torch.device("cuda", 0)
            if use_cuda
            else torch.device("cpu")
        )

        # IMPORTANT:
        # Keep the complete video batch in CPU RAM.
        frames = (
            images[..., :3]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        multiplier = int(multiplier)

        generated = []

        count = (total - 1) * (multiplier - 1)

        timings = {
            "inference": 0.0,
            "transfer": 0.0,
            "post": 0.0,
        }

        pbar = tqdm(
            total=count,
            desc="RIFE interpolation",
            unit="frame",
            dynamic_ncols=True,
        )

        try:
            with torch.inference_mode():
                for index in range(total - 1):
                    transfer_started = time.perf_counter()

                    # Transfer ONLY the current pair to GPU.
                    img0 = frames[
                        index:index + 1
                    ].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )

                    img1 = frames[
                        index + 1:index + 2
                    ].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )

                    if pad_h or pad_w:
                        img0 = F.pad(
                            img0,
                            (0, pad_w, 0, pad_h),
                            mode="replicate",
                        )

                        img1 = F.pad(
                            img1,
                            (0, pad_w, 0, pad_h),
                            mode="replicate",
                        )

                    timings["transfer"] += (
                        time.perf_counter()
                        - transfer_started
                    )

                    # Keep completed frame on CPU.
                    generated.append(
                        img0[
                            :,
                            :,
                            :height,
                            :width,
                        ]
                        .to("cpu")
                        .contiguous()
                    )

                    for step in range(
                        1,
                        multiplier,
                    ):
                        started = time.perf_counter()

                        result = self._infer(
                            session,
                            output_name,
                            img0,
                            img1,
                            step / float(multiplier),
                            use_cuda,
                            diagnostics,
                        )

                        timings["inference"] += (
                            time.perf_counter()
                            - started
                        )

                        transfer_started = (
                            time.perf_counter()
                        )

                        # Critical VRAM fix:
                        # immediately move result back to CPU.
                        result_cpu = (
                            result[
                                :,
                                :,
                                :height,
                                :width,
                            ]
                            .to("cpu")
                            .contiguous()
                        )

                        generated.append(result_cpu)

                        timings["transfer"] += (
                            time.perf_counter()
                            - transfer_started
                        )

                        del result
                        del result_cpu

                        pbar.update(1)

                    del img0
                    del img1

        finally:
            pbar.close()

        # Last original frame is already on CPU.
        generated.append(
            frames[
                -1:,
                :,
                :height,
                :width,
            ].contiguous()
        )

        started = time.perf_counter()

        # torch.cat now happens in system RAM instead of VRAM.
        output = torch.cat(
            generated,
            dim=0,
        )

        output = (
            output
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        timings["post"] += (
            time.perf_counter() - started
        )

        expected_frames = (
            (total - 1) * multiplier + 1
        )

        if diagnostics:
            print(
                "[RifeOnnx] "
                f"input={total}, "
                f"output={output.shape[0]}, "
                f"expected={expected_frames}, "
                f"multiplier={multiplier}, "
                f"provider={self._active_provider}, "
                f"padded={padded_w}x{padded_h}, "
                f"inference={timings['inference']:.3f}s, "
                f"transfer={timings['transfer']:.3f}s, "
                f"post={timings['post']:.3f}s"
            )

            if use_cuda:
                allocated = (
                    torch.cuda.memory_allocated(0)
                    / 1024**3
                )

                reserved = (
                    torch.cuda.memory_reserved(0)
                    / 1024**3
                )

                free, gpu_total = (
                    torch.cuda.mem_get_info(0)
                )

                print(
                    "[RifeOnnx] CUDA after interpolation: "
                    f"allocated={allocated:.2f} GiB, "
                    f"reserved={reserved:.2f} GiB, "
                    f"free={free / 1024**3:.2f} GiB, "
                    f"total={gpu_total / 1024**3:.2f} GiB"
                )

        del generated
        del frames

        if not keep_model_loaded:
            self.__class__._session = None
            self.__class__._session_key = None
            self.__class__._active_provider = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (output,)
