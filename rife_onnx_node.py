# file: ComfyUI/custom_nodes/ComfyUI-RIFE-ONNX/rife_onnx_node.py
"""RIFE ONNX interpolation node for ComfyUI with dynamic multi-GPU scheduling."""

import json
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
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


@dataclass
class GPUStats:
    device_id: int
    device_name: str
    provider: str = ""
    assigned_pairs: int = 0
    inferred_pairs: int = 0
    scene_cut_pairs: int = 0
    generated_frames: int = 0
    inference_seconds: float = 0.0
    transfer_to_gpu_seconds: float = 0.0
    transfer_to_cpu_seconds: float = 0.0
    worker_seconds: float = 0.0
    chunks: int = 0
    errors: int = 0


class RifeOnnxInterpolationNode:
    """Memory-efficient RIFE ONNX interpolation with N-GPU scheduling."""

    RUNTIME_VERSION_FILE = "./trt_cache_metadata/rife_runtime_versions.json"
    TRT_ENGINE_CACHE_PATH = "./trt_engine_cache/rife"
    TRT_TIMING_CACHE_PATH = "./trt_timing_cache/rife"

    _sessions: Dict[Tuple, ort.InferenceSession] = {}
    _active_providers: Dict[Tuple, str] = {}
    _metadata_cache: Dict[str, Tuple] = {}
    _cache_lock = threading.Lock()

    @staticmethod
    def _device_choices():
        choices = ["Auto"]
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for index in range(count):
                choices.append(f"Device {index}: {torch.cuda.get_device_name(index)}")
            if count > 1:
                joined = ":".join(str(i) for i in range(count))
                choices.append(f"Device {joined}: All GPUs")
        else:
            choices.append("CPU")
        return choices

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        for directory in folder_paths.get_folder_paths("Rife"):
            if os.path.isdir(directory):
                files.extend(
                    name for name in os.listdir(directory)
                    if name.lower().endswith(".onnx") and name not in files
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
                "provider": ([
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "ROCmExecutionProvider",
                    "CPUExecutionProvider",
                ],),
                "devices": (cls._device_choices(),),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "scene_cut_detection": ("BOOLEAN", {"default": True}),
                "scene_cut_pixel_threshold": (
                    "FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "scene_cut_histogram_threshold": (
                    "FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "diagnostics": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Allows manual values such as 0, 1, 0:1 or 0:1:2:3.
                # When connected, this overrides the devices dropdown.
                "device_ids": ("STRING", {"default": ""}),
            },
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
        raise FileNotFoundError(f"RIFE model '{model_file}' not found. Put it in: {RIFE_MODEL_DIR}")

    @classmethod
    def _clear_sessions(cls):
        with cls._cache_lock:
            cls._sessions.clear()
            cls._active_providers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                with open(cls.RUNTIME_VERSION_FILE, "r", encoding="utf-8") as handle:
                    previous = json.load(handle)
            except (OSError, ValueError, TypeError):
                pass
        if previous == current:
            return

        print("[RifeOnnx] Runtime versions changed; invalidating TensorRT caches.")
        for path in (cls.TRT_ENGINE_CACHE_PATH, cls.TRT_TIMING_CACHE_PATH):
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except OSError as exc:
                    print(f"[RifeOnnx] Could not delete {path}: {exc}")
        try:
            with open(cls.RUNTIME_VERSION_FILE, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
        except OSError as exc:
            print(f"[RifeOnnx] Could not write runtime metadata: {exc}")
        cls._clear_sessions()
        cls._metadata_cache.clear()

    @classmethod
    def _model_metadata(cls, model_path):
        key = os.path.abspath(model_path)
        if key in cls._metadata_cache:
            return cls._metadata_cache[key]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = tuple((item.name, tuple(item.shape), item.type) for item in session.get_inputs())
        outputs = tuple((item.name, tuple(item.shape), item.type) for item in session.get_outputs())
        metadata = (inputs, outputs)
        cls._metadata_cache[key] = metadata
        return metadata

    @classmethod
    def _validate_model(cls, model_path):
        inputs, outputs = cls._model_metadata(model_path)
        by_name = {name: (shape, dtype) for name, shape, dtype in inputs}
        required = ("img0", "img1", "timestep")
        missing = [name for name in required if name not in by_name]
        if missing:
            raise RuntimeError(f"Expected RIFE ONNX inputs {required}; missing {missing}.")
        for name in ("img0", "img1"):
            shape, _ = by_name[name]
            if len(shape) != 4:
                raise RuntimeError(f"Expected {name} as NCHW tensor, got {shape!r}.")
            if isinstance(shape[1], int) and shape[1] != 3:
                raise RuntimeError(f"Expected {name} to have 3 channels, got {shape[1]}.")
        if not outputs:
            raise RuntimeError("The ONNX model has no outputs.")
        return outputs[0][0]

    @staticmethod
    def _parse_devices(devices, device_ids, provider):
        if provider in ("CPUExecutionProvider", "ROCmExecutionProvider"):
            return [None]
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                raise RuntimeError("CUDA provider selected, but PyTorch reports no CUDA devices.")
            return [None]

        text = (device_ids or "").strip()
        if not text:
            text = str(devices).strip()
            if text.lower() == "auto":
                return list(range(torch.cuda.device_count()))
            if text.lower() == "cpu":
                return [None]
            if text.lower().startswith("device "):
                text = text[7:].split(" ", 1)[0].rstrip(":")

        normalized = text.replace(",", ":").replace(";", ":").replace(" ", "")
        result = []
        for token in normalized.split(":"):
            if not token:
                continue
            try:
                device_id = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid device selection '{text}'. Use Auto, 0, 1 or 0:1:2."
                ) from exc
            if device_id < 0 or device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {device_id} does not exist; available range is "
                    f"0..{torch.cuda.device_count() - 1}."
                )
            if device_id not in result:
                result.append(device_id)
        if not result:
            raise ValueError("No CUDA devices selected.")
        return result

    @classmethod
    def _try_create_session(
        cls, model_path, provider, height, width, batch_size, diagnostics, device_id
    ):
        if provider not in ort.get_available_providers():
            print(f"[RifeOnnx] Provider unavailable: {provider}")
            return None

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(os.cpu_count() or 8, 16))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if diagnostics:
            options.enable_profiling = True
            options.log_severity_level = 1

        did = 0 if device_id is None else int(device_id)
        cuda_options = {
            "device_id": did,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
        }

        if provider == "TensorrtExecutionProvider":
            engine_path = os.path.join(cls.TRT_ENGINE_CACHE_PATH, f"device_{did}")
            timing_path = os.path.join(cls.TRT_TIMING_CACHE_PATH, f"device_{did}")
            os.makedirs(engine_path, exist_ok=True)
            os.makedirs(timing_path, exist_ok=True)
            min_shape = f"1x3x{height}x{width}"
            opt_shape = f"{batch_size}x3x{height}x{width}"
            profile_min = f"img0:{min_shape},img1:{min_shape},timestep:1"
            profile_opt = f"img0:{opt_shape},img1:{opt_shape},timestep:{batch_size}"
            trt_options = {
                "device_id": did,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": engine_path,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": timing_path,
                "trt_fp16_enable": True,
                "trt_int8_enable": False,
                "trt_dla_enable": False,
                "trt_max_workspace_size": 6 * 1024 * 1024 * 1024,
                "trt_profile_min_shapes": profile_min,
                "trt_profile_opt_shapes": profile_opt,
                "trt_profile_max_shapes": profile_opt,
            }
            providers = [
                ("TensorrtExecutionProvider", trt_options),
                ("CUDAExecutionProvider", cuda_options),
                "CPUExecutionProvider",
            ]
        elif provider == "CUDAExecutionProvider":
            providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        else:
            providers = [provider]

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(
                model_path, sess_options=options, providers=providers
            )
            label = "CPU" if device_id is None else f"GPU {device_id}"
            print(
                f"[RifeOnnx] Initialized {label} {session.get_providers()} "
                f"in {time.perf_counter() - started:.2f}s"
            )
            return session
        except Exception as exc:
            label = "CPU" if device_id is None else f"GPU {device_id}"
            print(f"[RifeOnnx] {label} provider {provider} failed: {exc}")
            return None

    @classmethod
    def _load_session(
        cls, model_path, provider, height, width, batch_size, diagnostics, device_id
    ):
        key = (
            os.path.abspath(model_path), provider, device_id, height, width,
            int(batch_size), bool(diagnostics)
        )
        with cls._cache_lock:
            cached = cls._sessions.get(key)
        if cached is not None:
            return cached, cls._active_providers[key]

        chain = [provider]
        if device_id is not None:
            chain.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            chain.append("CPUExecutionProvider")

        for candidate in dict.fromkeys(chain):
            session = cls._try_create_session(
                model_path, candidate, height, width, batch_size, diagnostics, device_id
            )
            if session is not None:
                active = session.get_providers()[0]
                with cls._cache_lock:
                    cls._sessions[key] = session
                    cls._active_providers[key] = active
                return session, active
        raise RuntimeError(f"All ONNX Runtime providers failed for device {device_id}.")

    @classmethod
    def _run_gpu_iobinding(cls, session, output_name, img0, img1, timestep):
        device_id = img0.device.index if img0.device.index is not None else 0
        output = torch.empty_like(
            img0, device=img0.device, dtype=torch.float32,
            memory_format=torch.contiguous_format
        )
        binding = session.io_binding()
        for name, tensor in (("img0", img0), ("img1", img1), ("timestep", timestep)):
            tensor = tensor.contiguous()
            binding.bind_input(
                name=name, device_type="cuda", device_id=device_id,
                element_type=np.float32, shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr()
            )
        binding.bind_output(
            name=output_name, device_type="cuda", device_id=device_id,
            element_type=np.float32, shape=tuple(output.shape),
            buffer_ptr=output.data_ptr()
        )
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        torch.cuda.synchronize(device_id)
        return output

    @staticmethod
    def _run_numpy(session, img0, img1, timestep):
        feed = {
            "img0": np.ascontiguousarray(img0.detach().cpu().numpy(), dtype=np.float32),
            "img1": np.ascontiguousarray(img1.detach().cpu().numpy(), dtype=np.float32),
            "timestep": np.ascontiguousarray(
                timestep.detach().cpu().numpy(), dtype=np.float32
            ),
        }
        outputs = session.run(None, feed)
        if not outputs:
            raise RuntimeError("The RIFE model returned no output.")
        return torch.from_numpy(np.asarray(outputs[0]))

    @classmethod
    def _infer(cls, session, output_name, img0, img1, t, use_cuda):
        timestep = torch.full(
            (img0.shape[0],), float(t), device=img0.device, dtype=torch.float32
        ).contiguous()
        if use_cuda:
            output = cls._run_gpu_iobinding(
                session, output_name, img0, img1, timestep
            )
        else:
            output = cls._run_numpy(session, img0, img1, timestep)
        if output.ndim != 4 or output.shape[0] != img0.shape[0]:
            raise RuntimeError(
                f"RIFE batch mismatch: input={tuple(img0.shape)}, "
                f"output={tuple(output.shape)}."
            )
        return torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

    @staticmethod
    def _scene_cut_scores(frame0, frame1, size=64, bins=32):
        pair = torch.cat((frame0, frame1), dim=0).to(device="cpu", dtype=torch.float32)
        pair = F.interpolate(pair, size=(size, size), mode="area").clamp(0.0, 1.0)
        luma = pair[:, 0:1] * 0.2126 + pair[:, 1:2] * 0.7152 + pair[:, 2:3] * 0.0722
        pixel_score = float(torch.mean(torch.abs(luma[0] - luma[1])))
        hist0 = torch.histc(luma[0], bins=bins, min=0.0, max=1.0)
        hist1 = torch.histc(luma[1], bins=bins, min=0.0, max=1.0)
        hist0 = hist0 / hist0.sum().clamp_min(1.0)
        hist1 = hist1 / hist1.sum().clamp_min(1.0)
        histogram_score = float(0.5 * torch.sum(torch.abs(hist0 - hist1)))
        return pixel_score, histogram_score

    @classmethod
    def _is_scene_cut(cls, frame0, frame1, pixel_threshold, histogram_threshold):
        pixel_score, histogram_score = cls._scene_cut_scores(frame0, frame1)
        detected = (
            pixel_score >= float(pixel_threshold)
            and histogram_score >= float(histogram_threshold)
        )
        return detected, pixel_score, histogram_score

    @staticmethod
    def _print_summary(stats_by_device, pair_count, wall_seconds, diagnostics):
        assigned_total = sum(s.assigned_pairs for s in stats_by_device.values())
        inferred_total = sum(s.inferred_pairs for s in stats_by_device.values())
        cut_total = sum(s.scene_cut_pairs for s in stats_by_device.values())
        print("")
        print("[RifeOnnx] Multi-device processing summary")
        for key in sorted(stats_by_device, key=lambda x: (-1 if x is None else x)):
            s = stats_by_device[key]
            share = 100.0 * s.assigned_pairs / pair_count if pair_count else 0.0
            inferred_share = 100.0 * s.inferred_pairs / inferred_total if inferred_total else 0.0
            throughput = s.inferred_pairs / s.inference_seconds if s.inference_seconds > 0 else 0.0
            avg_ms = 1000.0 * s.inference_seconds / s.inferred_pairs if s.inferred_pairs else 0.0
            label = "CPU" if s.device_id < 0 else f"GPU {s.device_id} {s.device_name}"
            print(
                f"[RifeOnnx] {label}: {s.assigned_pairs}/{pair_count} pairs "
                f"({share:.2f} % of total), inferred={s.inferred_pairs} "
                f"({inferred_share:.2f} % of inference), time={s.worker_seconds:.3f}s, "
                f"inference={s.inference_seconds:.3f}s, throughput={throughput:.2f} pairs/s"
            )
            if diagnostics:
                print(
                    f"[RifeOnnx]   provider={s.provider}, chunks={s.chunks}, "
                    f"scene_cuts={s.scene_cut_pairs}, generated={s.generated_frames}, "
                    f"to_gpu={s.transfer_to_gpu_seconds:.3f}s, "
                    f"to_cpu={s.transfer_to_cpu_seconds:.3f}s, "
                    f"average={avg_ms:.2f}ms/inferred-pair, errors={s.errors}"
                )
        status = "OK" if assigned_total == pair_count else "ERROR"
        print(
            f"[RifeOnnx] Total: pairs={assigned_total}/{pair_count}, "
            f"inferred={inferred_total}, scene_cuts={cut_total}, "
            f"wall={wall_seconds:.3f}s, accounting={status}"
        )

    def interpolate(
        self, images, model_file, provider, devices, multiplier, batch_size,
        keep_model_loaded, scene_cut_detection=True,
        scene_cut_pixel_threshold=0.22, scene_cut_histogram_threshold=0.35,
        diagnostics=False, device_ids=""
    ):
        wall_started = time.perf_counter()
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4 or images.shape[-1] < 3:
            raise ValueError(
                f"Expected ComfyUI IMAGE in NHWC RGB(A), got {tuple(images.shape)}."
            )
        if images.shape[0] < 2:
            return (images,)

        multiplier = max(2, int(multiplier))
        requested_batch = max(1, int(batch_size))
        total, height, width, _ = images.shape
        pair_count = total - 1
        chunk_size = min(requested_batch, pair_count)

        model_path = self._find_model(model_file)
        self._check_runtime_versions_and_invalidate_cache()
        output_name = self._validate_model(model_path)
        selected = self._parse_devices(devices, device_ids, provider)

        alignment = 64
        pad_h = (-height) % alignment
        pad_w = (-width) % alignment
        padded_h = height + pad_h
        padded_w = width + pad_w

        frames = (
            images[..., :3].detach().to(device="cpu", dtype=torch.float32)
            .permute(0, 3, 1, 2).contiguous()
        )

        cuts = [False] * pair_count
        if scene_cut_detection:
            for index in range(pair_count):
                cut, pixel_score, histogram_score = self._is_scene_cut(
                    frames[index:index + 1], frames[index + 1:index + 2],
                    scene_cut_pixel_threshold, scene_cut_histogram_threshold
                )
                cuts[index] = cut
                if cut and diagnostics:
                    print(
                        f"[RifeOnnx] Scene cut {index}->{index + 1}: "
                        f"pixel={pixel_score:.4f}, histogram={histogram_score:.4f}"
                    )

        sessions = {}
        stats_by_device = {}
        for device_id in selected:
            session, active_provider = self._load_session(
                model_path, provider, padded_h, padded_w, chunk_size,
                diagnostics, device_id
            )
            sessions[device_id] = (session, active_provider)
            name = "CPU" if device_id is None else torch.cuda.get_device_name(device_id)
            stats_by_device[device_id] = GPUStats(
                device_id=-1 if device_id is None else device_id,
                device_name=name,
                provider=active_provider,
            )

        print(
            "[RifeOnnx] Selected devices: "
            + ", ".join(
                "CPU" if d is None else f"GPU {d} {torch.cuda.get_device_name(d)}"
                for d in selected
            )
        )

        jobs = queue.Queue()
        for start in range(0, pair_count, chunk_size):
            jobs.put((start, min(start + chunk_size, pair_count)))

        pair_outputs = [None] * pair_count
        output_lock = threading.Lock()
        error_lock = threading.Lock()
        errors = []
        pbar = tqdm(
            total=pair_count * (multiplier - 1), desc="RIFE interpolation",
            unit="frame", dynamic_ncols=True
        )

        def worker(device_id):
            session, active_provider = sessions[device_id]
            stats = stats_by_device[device_id]
            use_cuda = (
                device_id is not None
                and active_provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
            )
            device = torch.device("cpu") if not use_cuda else torch.device("cuda", device_id)
            worker_started = time.perf_counter()
            try:
                with torch.inference_mode():
                    if use_cuda:
                        torch.cuda.set_device(device_id)
                    while True:
                        try:
                            chunk_start, chunk_end = jobs.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            indices = list(range(chunk_start, chunk_end))
                            local_cuts = [cuts[i] for i in indices]
                            stats.chunks += 1
                            stats.assigned_pairs += len(indices)
                            stats.scene_cut_pairs += sum(local_cuts)
                            stats.inferred_pairs += len(indices) - sum(local_cuts)
                            stats.generated_frames += len(indices) * (multiplier - 1)

                            transfer_started = time.perf_counter()
                            img0 = frames[chunk_start:chunk_end].to(
                                device=device, dtype=torch.float32, non_blocking=False
                            )
                            img1 = frames[chunk_start + 1:chunk_end + 1].to(
                                device=device, dtype=torch.float32, non_blocking=False
                            )
                            if pad_h or pad_w:
                                img0 = F.pad(img0, (0, pad_w, 0, pad_h), mode="replicate")
                                img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
                            if use_cuda:
                                torch.cuda.synchronize(device_id)
                            stats.transfer_to_gpu_seconds += time.perf_counter() - transfer_started

                            local_outputs = [
                                [img0[i:i + 1, :, :height, :width].to("cpu").contiguous()]
                                for i in range(len(indices))
                            ]
                            active = [i for i, cut in enumerate(local_cuts) if not cut]
                            active_positions = {local: pos for pos, local in enumerate(active)}

                            for step in range(1, multiplier):
                                inferred = None
                                if active:
                                    active_tensor = torch.tensor(active, device=device)
                                    infer_started = time.perf_counter()
                                    inferred = self._infer(
                                        session, output_name,
                                        img0.index_select(0, active_tensor).contiguous(),
                                        img1.index_select(0, active_tensor).contiguous(),
                                        step / float(multiplier), use_cuda
                                    )
                                    if use_cuda:
                                        torch.cuda.synchronize(device_id)
                                    stats.inference_seconds += time.perf_counter() - infer_started

                                transfer_started = time.perf_counter()
                                for local_index, cut in enumerate(local_cuts):
                                    if cut:
                                        source = (
                                            img0[local_index:local_index + 1]
                                            if step * 2 < multiplier
                                            else img1[local_index:local_index + 1]
                                        )
                                    else:
                                        pos = active_positions[local_index]
                                        source = inferred[pos:pos + 1]
                                    local_outputs[local_index].append(
                                        source[:, :, :height, :width].to("cpu").contiguous()
                                    )
                                if use_cuda:
                                    torch.cuda.synchronize(device_id)
                                stats.transfer_to_cpu_seconds += time.perf_counter() - transfer_started
                                pbar.update(len(indices))
                                if inferred is not None:
                                    del inferred

                            with output_lock:
                                for local_index, pair_index in enumerate(indices):
                                    pair_outputs[pair_index] = local_outputs[local_index]
                            del img0, img1, local_outputs
                        finally:
                            jobs.task_done()
            except Exception as exc:
                stats.errors += 1
                with error_lock:
                    errors.append((device_id, exc))
            finally:
                stats.worker_seconds = time.perf_counter() - worker_started

        threads = [
            threading.Thread(target=worker, args=(device_id,), daemon=False)
            for device_id in selected
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            pbar.close()

        if errors:
            device_id, exc = errors[0]
            label = "CPU" if device_id is None else f"GPU {device_id}"
            raise RuntimeError(f"RIFE worker failed on {label}: {exc}") from exc
        missing = [i for i, value in enumerate(pair_outputs) if value is None]
        if missing:
            raise RuntimeError(f"No output produced for pair indices: {missing[:20]}")

        expected_frames = pair_count * multiplier + 1
        output = torch.empty(
            (expected_frames, height, width, 3), dtype=torch.float32, device="cpu"
        )
        output_index = 0
        for pair_index in range(pair_count):
            for frame in pair_outputs[pair_index]:
                output[output_index].copy_(frame[0, :, :height, :width].permute(1, 2, 0))
                output_index += 1
            pair_outputs[pair_index] = None
        output[output_index].copy_(frames[-1, :, :height, :width].permute(1, 2, 0))
        output_index += 1
        if output_index != expected_frames:
            raise RuntimeError(f"Output frame count {output_index} != expected {expected_frames}.")

        wall_seconds = time.perf_counter() - wall_started
        self._print_summary(stats_by_device, pair_count, wall_seconds, diagnostics)

        del pair_outputs, frames
        if not keep_model_loaded:
            self.__class__._clear_sessions()
        return (output,)
