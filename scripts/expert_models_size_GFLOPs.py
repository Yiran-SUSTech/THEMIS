#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agentic_eval.config import Settings
from src.agentic_eval.expert_models import (
    BackgroundExpert,
    BaseExpert,
    CLIPExpert,
    EXPERT_CLASS_MAP,
    ExpertModelError,
    IQAExpert,
    ImageNetExpert,
    Places365Expert,
    QwenVLExpert,
    TextEmbeddingExpert,
    YOLODetectExpert,
    YOLOPoseExpert,
    create_expert,
)

QINSIGHT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
QINSIGHT_DISTORTION_PROMPT = (
    'Analyze the given image and determine if it contains any of the following distortions: "noise", '
    '"compression", "blur", or "darken". If a distortion is present, classify its severity as '
    '"slight", "moderate", "obvious", "serious", or "catastrophic". Return the result in JSON '
    'format with the following keys: "distortion_class": The detected distortion (or "null" if none). '
    'and "severity": The severity level (or "null" if none).'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure expert-model GFLOPs and model size, then write gflops/model_size_mb back to configs/expert_config.yaml"
    )
    parser.add_argument("--config", default="configs/expert_config.yaml", help="Path to expert_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Do not write YAML; only print/save the report.")
    parser.add_argument("--force-cpu", action="store_true", help="Override expert devices to cpu before loading models.")
    parser.add_argument("--report-json", default=None, help="Optional path to save the profiling report as JSON.")
    parser.add_argument("--expert", action="append", default=None, help="Profile only specific expert key(s).")
    parser.add_argument("--groundingdino-swint-config", default=os.getenv("GROUNDINGDINO_SWINT_CONFIG", ""), help="Optional path to GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--groundingdino-swinb-config", default=os.getenv("GROUNDINGDINO_SWINB_CONFIG", ""), help="Optional path to GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--sam2-config-dir", default=os.getenv("SAM2_CONFIG_DIR", ""), help="Optional repo/config root containing configs/sam2.1/*.yaml")
    parser.add_argument("--unipose-repo-root", default=os.getenv("UNIPOSE_REPO_ROOT", ""), help="Optional UniPose repo root containing llava/ posegpt/ configs")
    parser.add_argument("--unipose-config", default=os.getenv("UNIPOSE_CONFIG", ""), help="Optional UniPose config path, e.g. configs/inference.py")
    parser.add_argument("--unipose-vision-tower", default=os.getenv("UNIPOSE_VISION_TOWER", ""), help="Optional absolute path to UniPose CLIP vision tower")
    return parser.parse_args()


def _iter_expert_nodes(raw_config: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    expert_models = raw_config.get("expert_models") or {}
    for group_name, group in expert_models.items():
        if not isinstance(group, dict):
            continue
        for expert_key, expert_node in group.items():
            if isinstance(expert_node, dict):
                yield group_name, expert_key, expert_node


def _is_places365_node(node: dict[str, Any]) -> bool:
    return (
        str(node.get("weights", "") or "").lower() == "places365"
        or "places365" in str(node.get("name", "") or "").lower()
        or (node.get("num_classes") == 365 and node.get("model") in {"resnet18", "resnet50"})
    )


SUPPORTED_MODEL_TYPES = {
    "classification",
    "eva_classification",
    "clip",
    "clip_score",
    "text_embedding",
    "yolo_pose",
    "detection",
    "iqa",
    "segmentation",
    "vqa",
    "mllm_scoring",
    "rtmpose",
    "mllm_pose",
    "open_vocabulary_detection",
}


MODEL_TYPE_ALIASES = {
    "eva_classification": "classification",
    "open_vocabulary_detection": "detection",
}


def _should_profile(node: dict[str, Any], requested_experts: set[str] | None, expert_key: str) -> bool:
    if requested_experts and expert_key not in requested_experts:
        return False
    if not bool(node.get("downloaded", False)):
        return False
    model_type = str(node.get("model_type", "") or "").strip().lower()
    if model_type in SUPPORTED_MODEL_TYPES:
        return True
    if model_type in EXPERT_CLASS_MAP:
        return True
    if model_type == "classification" and _is_places365_node(node):
        return True
    return False


def _effective_device(config: Any, force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    device = str(getattr(config, "device", "cpu") or "cpu").strip()
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
            parts = device.split(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                index = int(parts[1])
                if index >= torch.cuda.device_count():
                    return "cpu"
        except Exception:
            return "cpu"
    return device or "cpu"


def _make_dummy_image(path: Path, size: tuple[int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image = Image.new("RGB", size, color=(0, 0, 0))
        image.save(path)
    return path


def _resolve_dummy_size(config: Any, expert: BaseExpert | None = None) -> tuple[int, int]:
    input_size = getattr(config, "input_size", None)
    if isinstance(input_size, tuple) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])
    if isinstance(input_size, list) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    if model_type in {"vqa", "mllm_scoring"} or isinstance(expert, QwenVLExpert):
        return 512, 512
    if isinstance(expert, (YOLODetectExpert, YOLOPoseExpert)):
        return 640, 640
    if isinstance(expert, BackgroundExpert):
        return 1024, 1024
    return 224, 224


def _unique_tensor_bytes(tensors: Iterable[Any]) -> int:
    seen_ptrs: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if tensor is None:
            continue
        try:
            ptr = int(tensor.data_ptr())
            numel = int(tensor.numel())
            elem_size = int(tensor.element_size())
        except Exception:
            continue
        key = (ptr, numel)
        if key in seen_ptrs:
            continue
        seen_ptrs.add(key)
        total += numel * elem_size
    return total


def _module_size_bytes(module: Any) -> int:
    try:
        import torch.nn as nn
    except Exception:
        return 0
    if isinstance(module, nn.Module):
        return _unique_tensor_bytes(list(module.parameters()) + list(module.buffers()))
    return 0


def _unwrap_model_object(model_obj: Any) -> list[Any]:
    if model_obj is None:
        return []
    if isinstance(model_obj, tuple):
        return [item for item in model_obj if item is not None]
    if isinstance(model_obj, dict):
        return [item for item in model_obj.values() if item is not None]
    if hasattr(model_obj, "model"):
        inner = getattr(model_obj, "model")
        if inner is not None:
            return [inner]
    return [model_obj]


def _estimate_loaded_model_size_mb(model_obj: Any) -> float | None:
    components = _unwrap_model_object(model_obj)
    total_bytes = sum(_module_size_bytes(component) for component in components)
    if total_bytes <= 0 and isinstance(model_obj, tuple):
        total_bytes = sum(_module_size_bytes(component) for component in model_obj if component is not None)
    if total_bytes <= 0:
        return None
    return round(total_bytes / (1024 * 1024), 4)


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _maybe_get_model_device(model: Any) -> str:
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    try:
        first_param = next(model.parameters())
        return str(first_param.device)
    except Exception:
        return "cpu"


def _safe_open_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _resolve_existing_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_path = Path(str(candidate)).expanduser()
        if candidate_path.exists():
            return candidate_path
    return None


def _profile_flops(run_callable, use_cuda: bool) -> float:
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            run_callable()
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_flops = 0
    for event in prof.key_averages():
        total_flops += int(getattr(event, "flops", 0) or 0)
    return round(total_flops / 1_000_000_000, 6)


def _dummy_clip_texts(config: Any) -> list[str]:
    key = str(getattr(config, "name", "") or "").lower()
    if "dog" in key:
        class_label = "dog"
    elif "bird" in key:
        class_label = "bird"
    else:
        class_label = "test object"
    return [
        f"a photo of {class_label}",
        "a low quality image",
        "an unrealistic image",
        "a distorted image",
        "a cartoon image",
        "a painting",
    ]


def _build_qinsight_message(image_path: str | None, user_prompt: str) -> list[dict[str, Any]]:
    message = [
        {"role": "system", "content": [{"type": "text", "text": QINSIGHT_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    if image_path:
        message[1]["content"].append({"type": "image", "image": f"file://{image_path}"})
    return message


def _message_contains_image(message: list[dict[str, Any]]) -> bool:
    for turn in message:
        for item in turn.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                return True
    return False


def _prepare_qinsight_inputs(processor: Any, message: list[dict[str, Any]], image_path: str, process_vision_info_fn: Any | None) -> Any:
    text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
    processor_kwargs: dict[str, Any] = {
        "text": text,
        "padding": True,
        "return_tensors": "pt",
    }
    if _message_contains_image(message):
        if process_vision_info_fn is not None:
            image_inputs, video_inputs = process_vision_info_fn([message])
            processor_kwargs["images"] = image_inputs
            processor_kwargs["videos"] = video_inputs
        else:
            processor_kwargs["images"] = [_safe_open_image(Path(image_path))]
    return processor(**processor_kwargs)


def _normalized_model_type(config: Any) -> str:
    raw_model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    return MODEL_TYPE_ALIASES.get(raw_model_type, raw_model_type)


def _build_qwen_vl_load_attempts(torch_module: Any, device: str) -> list[tuple[str, dict[str, Any]]]:
    attempts: list[tuple[str, dict[str, Any]]] = []
    normalized_device = (device or "cpu").strip()
    if torch_module.cuda.is_available() and normalized_device.startswith("cuda"):
        attempts.append((
            "cuda_eager",
            {
                "device_map": {"": normalized_device},
                "attn_implementation": "eager",
                "torch_dtype": torch_module.float16,
            },
        ))
        attempts.append((
            "cuda_sdpa",
            {
                "device_map": {"": normalized_device},
                "attn_implementation": "sdpa",
                "torch_dtype": torch_module.float16,
            },
        ))
    attempts.append((
        "cpu_eager",
        {
            "device_map": "cpu",
            "attn_implementation": "eager",
            "torch_dtype": torch_module.float32,
        },
    ))
    return attempts


def _load_qinsight_bundle(config: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = str(getattr(config, "local_path", None) or getattr(config, "model", "") or "").strip()
    if not model_path:
        raise ExpertModelError("Q-Insight model path is empty")
    root = Path(model_path)
    if not root.exists():
        raise ExpertModelError(f"Q-Insight path not found: {root}")

    processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=True)
    load_errors: list[str] = []
    for attempt_name, load_kwargs in _build_qwen_vl_load_attempts(torch, str(getattr(config, "device", "cpu") or "cpu")):
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=True,
                **load_kwargs,
            )
            model.eval()
            return model, processor
        except Exception as exc:
            load_errors.append(f"{attempt_name}: {exc}")
    raise ExpertModelError(" | ".join(load_errors))


def _load_generic_qwen_vl_bundle(config: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = str(getattr(config, "local_path", None) or getattr(config, "model", "") or "").strip()
    if not model_path:
        raise ExpertModelError("Qwen-VL model path is empty")
    local_only = Path(model_path).exists()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only, trust_remote_code=True)
    load_errors: list[str] = []
    for attempt_name, load_kwargs in _build_qwen_vl_load_attempts(torch, str(getattr(config, "device", "cpu") or "cpu")):
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=local_only,
                trust_remote_code=True,
                **load_kwargs,
            )
            model.eval()
            return model, processor
        except Exception as exc:
            load_errors.append(f"{attempt_name}: {exc}")
    raise ExpertModelError(" | ".join(load_errors))


def _load_eva_bundle(config: Any) -> Any:
    import timm
    import torch

    model_name = str(getattr(config, "model", "") or "").strip()
    local_path = str(getattr(config, "local_path", None) or "").strip()
    num_classes = getattr(config, "num_classes", None) or 1000

    if local_path and Path(local_path).exists():
        try:
            model = timm.create_model(local_path, pretrained=False, num_classes=num_classes)
        except Exception:
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, checkpoint_path=local_path)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    device = str(getattr(config, "device", "cpu") or "cpu")
    model = model.to(device)
    model.eval()
    return model


def _load_groundingdino_bundle(config: Any, args: argparse.Namespace) -> Any:
    model_name = str(getattr(config, "model", "") or "").strip().lower()
    config_path = ""
    if "swint" in model_name:
        config_path = args.groundingdino_swint_config
    else:
        config_path = args.groundingdino_swinb_config
    if not config_path:
        raise ExpertModelError("GroundingDINO config path not provided; use --groundingdino-swint-config or --groundingdino-swinb-config")
    config_file = Path(config_path)
    if not config_file.exists():
        raise ExpertModelError(f"GroundingDINO config not found: {config_file}")
    weights_path = _resolve_existing_path(getattr(config, "weights", None), getattr(config, "local_path", None), getattr(config, "model", None))
    if weights_path is None:
        raise ExpertModelError("GroundingDINO checkpoint not found")
    try:
        from groundingdino.util.inference import load_model  # type: ignore
    except Exception as exc:
        raise ExpertModelError(f"groundingdino import failed: {exc}")
    model = load_model(str(config_file), str(weights_path))
    if hasattr(model, "eval"):
        model.eval()
    return model


def _load_sam2_bundle(config: Any, args: argparse.Namespace) -> Any:
    config_dir = Path(args.sam2_config_dir) if args.sam2_config_dir.strip() else None
    if config_dir is None:
        raise ExpertModelError("SAM2 config dir not provided; use --sam2-config-dir")
    model_name = str(getattr(config, "model", "") or "").strip().lower()
    relative_cfg_map = {
        "hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    relative_cfg = relative_cfg_map.get(model_name)
    if not relative_cfg:
        raise ExpertModelError(f"Unsupported SAM2 model name: {model_name}")
    config_path = config_dir / relative_cfg
    if not config_path.exists():
        raise ExpertModelError(f"SAM2 config not found: {config_path}")
    weights_path = _resolve_existing_path(getattr(config, "weights", None), getattr(config, "local_path", None))
    if weights_path is None:
        raise ExpertModelError("SAM2 checkpoint not found")
    try:
        from sam2.build_sam import build_sam2  # type: ignore
    except Exception as exc:
        raise ExpertModelError(f"sam2 import failed: {exc}")
    model = build_sam2(str(config_path), str(weights_path), device="cpu")
    if hasattr(model, "eval"):
        model.eval()
    return model


def _load_rtmpose_bundle(config: Any) -> Any:
    import torch

    weights_path = _resolve_existing_path(getattr(config, "weights", None), getattr(config, "local_path", None))
    if weights_path is None:
        raise ExpertModelError("RTMPose checkpoint not found")
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    return checkpoint


def _load_unipose_bundle(config: Any) -> tuple[Any, Any]:
    model_path = str(getattr(config, "local_path", None) or getattr(config, "model", "") or "").strip()
    if not model_path:
        raise ExpertModelError("UniPose model path is empty")
    root = Path(model_path)
    if not root.exists():
        raise ExpertModelError(f"UniPose path not found: {root}")
    adapter_config_path = root / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ExpertModelError(f"UniPose adapter config not found: {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_hint = str(adapter_config.get("base_model_name_or_path", "") or "").strip()
    base_path = (root / base_hint).resolve() if base_hint.startswith(".") else Path(base_hint)
    if not base_path.exists():
        raise ExpertModelError(f"UniPose base model not found: {base_path}")
    return _load_generic_qwen_vl_bundle(type("Cfg", (), {
        "local_path": str(base_path),
        "model": str(base_path),
        "device": str(getattr(config, "device", "cpu") or "cpu"),
    })())


def _prepare_profile_callable(expert: BaseExpert, model_obj: Any, dummy_image: Path, dummy_size: tuple[int, int]) -> Any:
    import torch

    config = expert.config
    device = str(getattr(config, "device", "cpu") or "cpu")

    if isinstance(expert, (ImageNetExpert, Places365Expert)):
        model = model_obj
        tensor = torch.zeros((1, 3, dummy_size[1], dummy_size[0]), device=device)
        return lambda: model(tensor)

    if isinstance(expert, CLIPExpert):
        model, processor = model_obj
        image = _safe_open_image(dummy_image)
        texts = _dummy_clip_texts(config)
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = _move_inputs_to_device(inputs, _maybe_get_model_device(model))
        return lambda: model(**inputs)

    if isinstance(expert, TextEmbeddingExpert):
        model, tokenizer = model_obj
        texts = ["dog", "wolf", "cat", "animal"]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
        inputs = _move_inputs_to_device(inputs, _maybe_get_model_device(model))
        return lambda: model(**inputs)

    if isinstance(expert, YOLODetectExpert):
        wrapper = model_obj
        model = getattr(wrapper, "model", wrapper)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        tensor = torch.zeros((1, 3, dummy_size[1], dummy_size[0]), device=device)
        return lambda: model(tensor)

    if isinstance(expert, YOLOPoseExpert):
        wrapper = model_obj
        model = getattr(wrapper, "model", wrapper)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        tensor = torch.zeros((1, 3, dummy_size[1], dummy_size[0]), device=device)
        return lambda: model(tensor)

    if isinstance(expert, IQAExpert):
        metrics: dict[str, Any] = model_obj
        tensor = torch.zeros((1, 3, dummy_size[1], dummy_size[0]), device=device)

        def _run_all_metrics():
            for metric in metrics.values():
                metric(tensor)

        return _run_all_metrics

    if isinstance(expert, BackgroundExpert):
        model = model_obj
        tensor = torch.zeros((1, 3, dummy_size[1], dummy_size[0]), device=device)
        return lambda: model(tensor)

    if isinstance(expert, QwenVLExpert):
        model, processor = model_obj
        image = _safe_open_image(dummy_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, _maybe_get_model_device(model))
        return lambda: model.generate(**inputs, max_new_tokens=1, do_sample=False)

    raise ExpertModelError(f"Profiling callable not implemented for {expert.__class__.__name__}")


def _profile_qinsight(config: Any, dummy_image: Path) -> dict[str, float | None]:
    import torch

    model, processor = _load_qinsight_bundle(config)
    loaded_model_size_mb = _estimate_loaded_model_size_mb(model)

    process_vision_info_fn = None
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        process_vision_info_fn = process_vision_info
    except Exception:
        process_vision_info_fn = None

    message = _build_qinsight_message(str(dummy_image), QINSIGHT_DISTORTION_PROMPT)
    inputs = _prepare_qinsight_inputs(processor, message, str(dummy_image), process_vision_info_fn)
    model_device = _maybe_get_model_device(model)
    inputs = _move_inputs_to_device(inputs, model_device)
    use_cuda = str(model_device).startswith("cuda")
    gflops = _profile_flops(lambda: model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True), use_cuda=use_cuda)
    return {
        "gflops": gflops,
        "model_size_mb": loaded_model_size_mb,
    }


def _profile_text_embedding(config: Any, settings: Settings) -> dict[str, float | None]:
    expert = TextEmbeddingExpert(config, settings)
    model_obj = expert.load_model()
    model_size_mb = _estimate_loaded_model_size_mb(model_obj)
    run_callable = _prepare_profile_callable(expert, model_obj, Path("unused.png"), (224, 224))
    use_cuda = str(getattr(config, "device", "cpu") or "cpu").startswith("cuda")
    gflops = _profile_flops(run_callable, use_cuda=use_cuda)
    return {"gflops": gflops, "model_size_mb": model_size_mb}


def _profile_eva_classification(config: Any, dummy_size: tuple[int, int]) -> dict[str, float | None]:
    import torch

    model = _load_eva_bundle(config)
    model_size_mb = _estimate_loaded_model_size_mb(model)
    tensor = torch.zeros((1, 3, dummy_size[1], dummy_size[0]), device=str(getattr(config, "device", "cpu") or "cpu"))
    gflops = _profile_flops(lambda: model(tensor), use_cuda=str(getattr(config, "device", "cpu") or "cpu").startswith("cuda"))
    return {"gflops": gflops, "model_size_mb": model_size_mb}


def _profile_groundingdino(config: Any, dummy_image: Path, args: argparse.Namespace) -> dict[str, float | None]:
    import torch
    from groundingdino.util.inference import load_image, predict  # type: ignore

    model = _load_groundingdino_bundle(config, args)
    model_size_mb = _estimate_loaded_model_size_mb(model)
    _, image = load_image(str(dummy_image))
    use_cuda = str(getattr(config, "device", "cpu") or "cpu").startswith("cuda")

    def _run():
        predict(
            model=model,
            image=image,
            caption="animal . object .",
            box_threshold=0.25,
            text_threshold=0.2,
        )

    gflops = _profile_flops(_run, use_cuda=use_cuda)
    return {"gflops": gflops, "model_size_mb": model_size_mb}


def _profile_sam2(config: Any, args: argparse.Namespace) -> dict[str, float | None]:
    model = _load_sam2_bundle(config, args)
    model_size_mb = _estimate_loaded_model_size_mb(model)
    return {"gflops": None, "model_size_mb": model_size_mb}


def _profile_rtmpose(config: Any) -> dict[str, float | None]:
    checkpoint = _load_rtmpose_bundle(config)
    size_mb = None
    try:
        import torch

        if isinstance(checkpoint, dict):
            tensors = [value for value in checkpoint.values() if hasattr(value, "numel") and hasattr(value, "element_size")]
            total_bytes = _unique_tensor_bytes(tensors)
            if total_bytes > 0:
                size_mb = round(total_bytes / (1024 * 1024), 4)
    except Exception:
        size_mb = None
    return {"gflops": None, "model_size_mb": size_mb}


def _profile_unipose(config: Any, dummy_image: Path) -> dict[str, float | None]:
    model, processor = _load_unipose_bundle(config)
    model_size_mb = _estimate_loaded_model_size_mb(model)
    image = _safe_open_image(dummy_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the visible pose or body structure in one short sentence."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    model_device = _maybe_get_model_device(model)
    inputs = _move_inputs_to_device(inputs, model_device)
    gflops = _profile_flops(lambda: model.generate(**inputs, max_new_tokens=1, do_sample=False), use_cuda=str(model_device).startswith("cuda"))
    return {"gflops": gflops, "model_size_mb": model_size_mb}


def _profile_one_expert(expert_key: str, config: Any, settings: Settings, dump_dir: Path, args: argparse.Namespace) -> dict[str, float | None]:
    model_type = _normalized_model_type(config)
    dummy_size = _resolve_dummy_size(config)
    dummy_name = f"{expert_key}_{dummy_size[0]}x{dummy_size[1]}.png"
    dummy_path = _make_dummy_image(dump_dir / dummy_name, dummy_size)

    if model_type == "mllm_scoring":
        return _profile_qinsight(config, dummy_path)
    if model_type == "text_embedding":
        return _profile_text_embedding(config, settings)
    if model_type == "classification" and str(getattr(config, "model_type", "") or "").strip().lower() == "eva_classification":
        return _profile_eva_classification(config, dummy_size)
    if model_type == "detection" and str(getattr(config, "model_type", "") or "").strip().lower() == "open_vocabulary_detection":
        return _profile_groundingdino(config, dummy_path, args)
    if model_type == "segmentation" and str(getattr(config, "model", "") or "").strip().lower().startswith("hiera_"):
        return _profile_sam2(config, args)
    if str(getattr(config, "model_type", "") or "").strip().lower() == "rtmpose":
        return _profile_rtmpose(config)
    if str(getattr(config, "model_type", "") or "").strip().lower() == "mllm_pose":
        return _profile_unipose(config, dummy_path)

    expert = create_expert(config, settings)
    dummy_size = _resolve_dummy_size(config, expert)
    if dummy_size != tuple(_safe_open_image(dummy_path).size):
        dummy_path = _make_dummy_image(dump_dir / dummy_name, dummy_size)

    model_obj = expert.load_model()
    model_size_mb = _estimate_loaded_model_size_mb(model_obj)
    run_callable = _prepare_profile_callable(expert, model_obj, dummy_path, dummy_size)
    use_cuda = str(getattr(config, "device", "cpu") or "cpu").startswith("cuda")
    gflops = _profile_flops(run_callable, use_cuda=use_cuda)
    return {
        "gflops": gflops,
        "model_size_mb": model_size_mb,
    }


def _write_yaml(path: Path, raw_config: dict[str, Any]) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    try:
        from ruamel.yaml import YAML  # type: ignore

        yaml_rt = YAML()
        yaml_rt.preserve_quotes = True
        yaml_rt.width = 4096
        with path.open("r", encoding="utf-8") as handle:
            current = yaml_rt.load(handle)
        current["expert_models"] = raw_config["expert_models"]
        with path.open("w", encoding="utf-8") as handle:
            yaml_rt.dump(current, handle)
        return
    except Exception:
        pass

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, allow_unicode=True, sort_keys=False)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    settings = Settings.from_yaml(str(config_path))
    requested_experts = set(args.expert) if args.expert else None

    model_dir_env = os.getenv("MODEL_DIR", "").strip()
    if model_dir_env:
        settings.model_dir = model_dir_env

    if args.force_cpu:
        for expert_config in settings.expert_configs.values():
            expert_config.device = "cpu"

    report: dict[str, Any] = {
        "config": str(config_path),
        "model_dir": settings.model_dir,
        "results": {},
        "skipped": {},
        "errors": {},
    }

    updated = 0
    with tempfile.TemporaryDirectory(prefix="themis_profile_inputs_") as temp_dir:
        dump_dir = Path(temp_dir)
        for group_name, expert_key, node in _iter_expert_nodes(raw_config):
            if not _should_profile(node, requested_experts, expert_key):
                continue
            config = settings.get_expert_config(expert_key)
            if config is None:
                report["skipped"][expert_key] = "missing runtime config"
                continue
            try:
                config.device = _effective_device(config, args.force_cpu)
                measured = _profile_one_expert(expert_key, config, settings, dump_dir, args)
                node.pop("profiling", None)
                node["gflops"] = measured["gflops"]
                node["model_size_mb"] = measured["model_size_mb"]
                report["results"][expert_key] = measured
                updated += 1
                print(f"[MEASURED] {expert_key}: {measured['gflops']} GFLOPs, {measured['model_size_mb']} MB")
            except Exception as exc:
                report["errors"][expert_key] = {
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "group": group_name,
                }
                print(f"[SKIPPED] {expert_key}: {exc.__class__.__name__}: {exc}")

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.dry_run:
        _write_yaml(config_path, raw_config)
        print(f"Updated YAML entries: {updated}")
        print(f"Backup written to: {config_path}.bak")
    else:
        print(f"Dry run only. Measured entries: {updated}")

    print(json.dumps({
        "measured": len(report["results"]),
        "errors": len(report["errors"]),
        "skipped": len(report["skipped"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
