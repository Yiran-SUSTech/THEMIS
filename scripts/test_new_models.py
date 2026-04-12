#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

os.environ["HF_HOME"] = "/mnt/afs/zhengmingkai/zyr/THEMIS/models"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/afs/zhengmingkai/zyr/THEMIS/models"

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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
QINSIGHT_TEXT_ONLY_PROMPT = "Reply with exactly <answer>OK</answer>."
UNIPOSE_PROMPT = "Describe the visible pose or body structure of the main animal or subject in one short sentence."


@dataclass
class TestResult:
    name: str
    status: str
    load_ok: bool = False
    run_ok: bool = False
    path: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    traceback: str = ""
    elapsed_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test newly added local models")
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", str(ROOT / "models")), help="Model root directory")
    parser.add_argument("--image", default=None, help="Optional image for inference")
    parser.add_argument("--device", default="cuda:0", help="Device for runnable tests, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Torch dtype for transformer models")
    parser.add_argument("--max-new-tokens", type=int, default=24, help="Generation length for Q-Insight smoke test")
    parser.add_argument("--groundingdino-swint-config", default="", help="Optional path to GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--groundingdino-swinb-config", default="", help="Optional path to GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--sam2-config-dir", default="", help="Optional repo/config root containing configs/sam2.1/*.yaml")
    parser.add_argument("--unipose-repo-root", default=os.getenv("UNIPOSE_REPO_ROOT", ""), help="Optional UniPose repo root containing llava/ posegpt/ configs")
    parser.add_argument("--unipose-config", default=os.getenv("UNIPOSE_CONFIG", ""), help="Optional UniPose config path, e.g. configs/inference.py")
    parser.add_argument("--unipose-vision-tower", default=os.getenv("UNIPOSE_VISION_TOWER", ""), help="Optional absolute path to UniPose CLIP vision tower, e.g. .../clip-vit-large-patch14-336")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "new_model_smoke_test.json"), help="JSON report path")
    return parser.parse_args()


def build_test_image() -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "themis_new_model_smoke"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_path = tmp_dir / "new_model_test_input.png"
    if image_path.exists():
        return str(image_path)

    image = Image.new("RGB", (512, 512), color=(236, 239, 244))
    draw = ImageDraw.Draw(image)
    draw.ellipse((130, 90, 380, 340), fill=(138, 104, 77), outline=(30, 30, 30), width=4)
    draw.ellipse((185, 145, 225, 185), fill=(20, 20, 20))
    draw.ellipse((285, 145, 325, 185), fill=(20, 20, 20))
    draw.polygon([(255, 195), (235, 238), (275, 238)], fill=(215, 184, 132), outline=(45, 45, 45))
    draw.rectangle((195, 340, 220, 455), fill=(93, 74, 60))
    draw.rectangle((290, 340, 315, 455), fill=(93, 74, 60))
    draw.line((365, 250, 445, 305), fill=(80, 65, 50), width=8)
    draw.text((20, 20), "THEMIS new-model smoke test", fill=(10, 10, 10))
    image.save(image_path)
    return str(image_path)


def maybe_import(module_name: str):
    try:
        module = __import__(module_name, fromlist=["*"])
        return module, ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def resolve_torch_dtype(torch_module: Any, dtype_name: str):
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if torch_module.cuda.is_available():
        return torch_module.bfloat16
    return torch_module.float32


def is_accelerator_device_available(torch_module: Any, device: str) -> bool:
    normalized = (device or "").strip().lower()
    return normalized != "cpu" and normalized.startswith("cuda") and torch_module.cuda.is_available()


def is_metax_device(torch_module: Any, device: str) -> bool:
    if not is_accelerator_device_available(torch_module, device):
        return False
    try:
        device_index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        device_name = str(torch_module.cuda.get_device_name(device_index)).lower()
    except Exception:  # noqa: BLE001
        return False
    return "metax" in device_name or "c500" in device_name or "maca" in device_name


def build_qwen_vl_load_attempts(torch_module: Any, device: str, dtype_name: str) -> list[tuple[str, dict[str, Any]]]:
    target_device = device if is_accelerator_device_available(torch_module, device) else "cpu"
    metax_backend = is_metax_device(torch_module, target_device)
    attempts: list[tuple[str, dict[str, Any]]] = []

    if target_device != "cpu":
        preferred_dtype = torch_module.float16 if metax_backend and dtype_name == "auto" else resolve_torch_dtype(torch_module, dtype_name)
        eager_kwargs = {
            "device_map": {"": target_device},
            "attn_implementation": "eager",
            "torch_dtype": preferred_dtype,
        }
        attempts.append(("cuda_eager", eager_kwargs))
        if not metax_backend:
            sdpa_kwargs = {
                "device_map": {"": target_device},
                "attn_implementation": "sdpa",
                "torch_dtype": preferred_dtype,
            }
            attempts.append(("cuda_sdpa", sdpa_kwargs))

    attempts.append(("cpu_eager", {"device_map": "cpu", "attn_implementation": "eager", "torch_dtype": torch_module.float32}))
    return attempts


def describe_tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def describe_tensor_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def build_qinsight_message(image_path: str | None, user_prompt: str) -> list[dict[str, Any]]:
    message = [
        {"role": "system", "content": [{"type": "text", "text": QINSIGHT_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    if image_path:
        message[1]["content"].append({"type": "image", "image": f"file://{image_path}"})
    return message


def message_contains_image(message: list[dict[str, Any]]) -> bool:
    for turn in message:
        for item in turn.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                return True
    return False


def prepare_qinsight_inputs(processor: Any, message: list[dict[str, Any]], image_path: str, process_vision_info_fn: Any | None) -> Any:
    text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
    processor_kwargs: dict[str, Any] = {
        "text": text,
        "padding": True,
        "return_tensors": "pt",
    }
    if message_contains_image(message):
        if process_vision_info_fn is not None:
            image_inputs, video_inputs = process_vision_info_fn([message])
            processor_kwargs["images"] = image_inputs
            processor_kwargs["videos"] = video_inputs
        else:
            processor_kwargs["images"] = [Image.open(image_path).convert("RGB")]
    return processor(**processor_kwargs)


def torch_load_file(file_path: Path) -> tuple[bool, str]:
    torch_module, import_error = maybe_import("torch")
    if torch_module is None:
        return False, f"torch import failed: {import_error}"
    try:
        torch_module.load(str(file_path), map_location="cpu")
        return True, "checkpoint readable via torch.load"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def make_result(name: str, path: Path) -> TestResult:
    return TestResult(name=name, status="failed", path=str(path))


@contextmanager
def pushd(path: Path | None):
    previous = Path.cwd()
    try:
        if path is not None:
            os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def test_deeplabcut(model_dir: Path) -> TestResult:
    root = model_dir / "DeepLabCut"
    result = make_result("DeepLabCut SuperAnimal Quadruped RTMPose", root)
    started = time.time()
    try:
        checkpoints = {
            "s": root / "superanimal_quadruped_rtmpose_s.pt",
            "m": root / "superanimal_quadruped_rtmpose_m.pt",
            "x": root / "superanimal_quadruped_rtmpose_x.pt",
        }
        missing = [name for name, path in checkpoints.items() if not path.exists()]
        result.details["files"] = {name: str(path) for name, path in checkpoints.items()}
        if missing:
            result.status = "failed"
            result.error = f"missing checkpoints: {missing}"
            return result

        result.load_ok = True
        result.details["checkpoint_readability"] = {}
        for name, path in checkpoints.items():
            ok, message = torch_load_file(path)
            result.details["checkpoint_readability"][name] = {"ok": ok, "message": message}
            result.load_ok = result.load_ok and ok

        deeplabcut_module, deeplabcut_error = maybe_import("deeplabcut")
        result.details["deeplabcut_importable"] = deeplabcut_module is not None
        if deeplabcut_module is None:
            result.status = "blocked"
            result.error = f"deeplabcut not importable: {deeplabcut_error}"
            result.details["note"] = "These RTMPose files are not enough for end-to-end SuperAnimal inference by themselves; DLC package and matching detector/zoo assets are also needed."
            return result

        detector_candidates = list(root.glob("*detector*.pt")) + list(root.glob("*superanimal_quadruped*.yaml"))
        result.details["companion_assets_found"] = [str(path) for path in detector_candidates]
        if not detector_candidates:
            result.status = "blocked"
            result.error = "DLC package is installed, but matching detector/config assets for SuperAnimal inference were not found in the model directory"
            result.details["note"] = "The downloaded RTMPose checkpoints look intact, but full DeepLabCut SuperAnimal inference typically also needs detector/config assets."
            return result

        result.status = "partial"
        result.details["note"] = "Checkpoint files are readable and DeepLabCut imports, but this script does not force a full DLC project/inference pipeline because the local zoo layout may differ."
        return result
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.error = str(exc)
        result.traceback = traceback.format_exc()
        return result
    finally:
        result.elapsed_seconds = round(time.time() - started, 3)


def test_unipose(model_dir: Path, image_path: str, device: str, dtype_name: str, unipose_repo_root: str, unipose_config: str, unipose_vision_tower: str) -> TestResult:
    root = model_dir / "unipose"
    result = make_result("UniPose adapter bundle", root)
    started = time.time()
    try:
        adapter_config_path = root / "adapter_config.json"
        adapter_weights = root / "adapter_model.safetensors"
        non_lora = root / "non_lora_trainables.bin"
        required = [adapter_config_path, adapter_weights]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            result.status = "failed"
            result.error = f"missing required files: {missing}"
            return result

        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_hint = str(adapter_config.get("base_model_name_or_path", "")).strip()
        base_path = (root / base_hint).resolve() if base_hint.startswith(".") else Path(base_hint)
        result.load_ok = True
        result.details["peft_type"] = adapter_config.get("peft_type")
        result.details["base_model_name_or_path"] = base_hint
        result.details["resolved_base_path"] = str(base_path)
        result.details["base_exists"] = base_path.exists()
        result.details["has_non_lora_trainables"] = non_lora.exists()

        peft_module, peft_error = maybe_import("peft")
        result.details["peft_importable"] = peft_module is not None
        if peft_module is None:
            result.status = "blocked"
            result.error = f"peft not importable: {peft_error}"
            return result

        from peft import PeftConfig  # type: ignore

        peft_config = PeftConfig.from_pretrained(str(root), local_files_only=True)
        result.details["peft_base_model_from_library"] = getattr(peft_config, "base_model_name_or_path", "")

        if not base_path.exists():
            result.status = "blocked"
            result.error = "UniPose directory is an adapter/LoRA bundle, but the referenced base model is not present locally"
            result.details["note"] = "This folder cannot run standalone inference until the base multimodal model is also available."
            return result

        transformers_module, transformers_error = maybe_import("transformers")
        result.details["transformers_importable"] = transformers_module is not None
        if transformers_module is None:
            result.status = "blocked"
            result.error = f"transformers not importable: {transformers_error}"
            return result

        torch_module, torch_error = maybe_import("torch")
        result.details["torch_importable"] = torch_module is not None
        if torch_module is None:
            result.status = "blocked"
            result.error = f"torch not importable: {torch_error}"
            return result

        resolved_repo_root = Path(unipose_repo_root).expanduser() if unipose_repo_root.strip() else None
        resolved_config_path = Path(unipose_config).expanduser() if unipose_config.strip() else None
        resolved_unipose_vision_tower = Path(unipose_vision_tower).expanduser() if unipose_vision_tower.strip() else None
        if resolved_config_path is None and resolved_repo_root is not None:
            candidate_config = resolved_repo_root / "configs" / "inference.py"
            if candidate_config.exists():
                resolved_config_path = candidate_config

        result.details["unipose_repo_root"] = str(resolved_repo_root) if resolved_repo_root else ""
        result.details["unipose_config"] = str(resolved_config_path) if resolved_config_path else ""
        result.details["unipose_vision_tower"] = str(resolved_unipose_vision_tower) if resolved_unipose_vision_tower else ""

        if resolved_repo_root is not None:
            for candidate in (resolved_repo_root, resolved_repo_root / "src"):
                candidate_str = str(candidate)
                if candidate.exists() and candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)

        target_device = device if is_accelerator_device_available(torch_module, device) else "cpu"
        device_index = int(target_device.split(":", 1)[1]) if target_device.startswith("cuda:") else 0
        metax_backend = is_metax_device(torch_module, target_device)
        if dtype_name == "auto":
            if target_device == "cpu":
                torch_dtype = torch_module.float32
            elif metax_backend:
                torch_dtype = torch_module.float16
            else:
                torch_dtype = torch_module.bfloat16
        else:
            torch_dtype = resolve_torch_dtype(torch_module, dtype_name)
        result.details["device_used"] = target_device
        result.details["is_metax_device"] = metax_backend
        result.details["attempted_inference"] = True
        result.details["inference_diagnostics"] = {
            "torch_dtype": str(torch_dtype),
            "device_map": {"": device_index} if target_device.startswith("cuda") else "cpu",
        }

        if resolved_config_path is None or not resolved_config_path.exists():
            result.status = "partial"
            result.details["note"] = "Adapter metadata and base model path are valid, but UniPose official inference also needs its config file. Pass --unipose-repo-root or --unipose-config."
            result.details["inference_diagnostics"]["error"] = "UniPose config file not found"
            return result

        llava_module, llava_error = maybe_import("llava")
        posegpt_module, posegpt_error = maybe_import("posegpt")
        result.details["llava_importable"] = llava_module is not None
        result.details["posegpt_importable"] = posegpt_module is not None
        if llava_module is None or posegpt_module is None:
            result.status = "partial"
            result.details["note"] = "Adapter metadata and base model path are valid, but UniPose official inference modules are not importable in this environment."
            result.details["inference_diagnostics"]["error"] = f"llava: {llava_error or 'ok'} | posegpt: {posegpt_error or 'ok'}"
            return result

        try:
            import numpy as np  # type: ignore
            from llava import conversation as conversation_lib  # type: ignore
            from llava.model.language_model.llava_mistral import LlavaMistralConfig  # type: ignore
            from posegpt.utils import Config  # type: ignore
            from posegpt.models.posegpt_full_mask import PoseGPTFullMask  # type: ignore
            from posegpt.constants import IMAGE_TOKEN  # type: ignore
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize  # type: ignore
            try:
                from torchvision.transforms import InterpolationMode  # type: ignore
                bicubic = InterpolationMode.BICUBIC
            except Exception:  # noqa: BLE001
                bicubic = Image.BICUBIC
            from transformers import AutoTokenizer  # type: ignore
            from peft import PeftModel  # type: ignore

            def hmr_transform(n_px: int = 256):
                def _convert_image_to_rgb(image: Image.Image) -> Image.Image:
                    return image.convert("RGB")

                return Compose([
                    Resize(n_px, interpolation=bicubic),
                    CenterCrop(n_px),
                    _convert_image_to_rgb,
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

            def load_unipose_model(config: Any):
                tokenizer = AutoTokenizer.from_pretrained(str(root), use_fast=False, local_files_only=True)
                tokenizer_length = int(len(tokenizer))
                base_vocab_size = None
                adapter_vocab_size = None
                base_config_path = base_path / "config.json"
                adapter_model_config_path = root / "config.json"
                if base_config_path.exists():
                    try:
                        base_vocab_size = int(json.loads(base_config_path.read_text(encoding="utf-8")).get("vocab_size"))
                    except Exception:  # noqa: BLE001
                        base_vocab_size = None
                if adapter_model_config_path.exists():
                    try:
                        adapter_vocab_size = int(json.loads(adapter_model_config_path.read_text(encoding="utf-8")).get("vocab_size"))
                    except Exception:  # noqa: BLE001
                        adapter_vocab_size = None
                target_vocab_size = max(tokenizer_length, adapter_vocab_size or 0, 34132)
                result.details["inference_diagnostics"]["debug_script_version"] = "unipose_embedding_fix_v5"
                result.details["inference_diagnostics"]["tokenizer_length"] = tokenizer_length
                result.details["inference_diagnostics"]["base_vocab_size"] = base_vocab_size
                result.details["inference_diagnostics"]["adapter_vocab_size"] = adapter_vocab_size
                result.details["inference_diagnostics"]["target_vocab_size"] = target_vocab_size
                lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(str(root), local_files_only=True)
                if base_vocab_size is not None:
                    lora_cfg_pretrained.vocab_size = base_vocab_size
                    text_config = getattr(lora_cfg_pretrained, "text_config", None)
                    if text_config is not None and hasattr(text_config, "vocab_size"):
                        text_config.vocab_size = base_vocab_size
                vision_tower_value = getattr(lora_cfg_pretrained, "mm_vision_tower", getattr(lora_cfg_pretrained, "vision_tower", None))
                if isinstance(vision_tower_value, str) and vision_tower_value:
                    vision_tower_candidates = []
                    if resolved_unipose_vision_tower is not None:
                        vision_tower_candidates.append(resolved_unipose_vision_tower)
                    raw_vision_path = Path(vision_tower_value)
                    if raw_vision_path.is_absolute():
                        vision_tower_candidates.append(raw_vision_path)
                    else:
                        vision_tower_candidates.extend([
                            (resolved_repo_root / vision_tower_value) if resolved_repo_root else None,
                            (resolved_repo_root / "cache" / vision_tower_value) if resolved_repo_root else None,
                            (resolved_repo_root / "cache" / raw_vision_path.name) if resolved_repo_root else None,
                            root / vision_tower_value,
                            root / raw_vision_path.name,
                            base_path / vision_tower_value,
                            base_path / raw_vision_path.name,
                            base_path.parent / vision_tower_value,
                            base_path.parent / raw_vision_path.name,
                        ])
                    for candidate in vision_tower_candidates:
                        if candidate is not None and candidate.exists():
                            resolved_vision_tower = str(candidate.resolve())
                            if hasattr(lora_cfg_pretrained, "mm_vision_tower"):
                                lora_cfg_pretrained.mm_vision_tower = resolved_vision_tower
                            if hasattr(lora_cfg_pretrained, "vision_tower"):
                                lora_cfg_pretrained.vision_tower = resolved_vision_tower
                            result.details["inference_diagnostics"]["resolved_vision_tower"] = resolved_vision_tower
                            break
                    result.details["inference_diagnostics"]["requested_vision_tower"] = vision_tower_value
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model = PoseGPTFullMask.from_pretrained(
                        str(base_path),
                        low_cpu_mem_usage=True,
                        attn_implementation=None,
                        torch_dtype=torch_dtype,
                        config=lora_cfg_pretrained,
                        tokenizer=tokenizer,
                        device_map={"": device_index} if target_device.startswith("cuda") else "cpu",
                        pose_vqvae_codebook_size=config.pose_vqvae_config.params.quantizer.params.nb_code,
                        evaluate_task=None,
                    )
                model.config.eos_token_id = tokenizer.eos_token_id
                model.config.bos_token_id = tokenizer.bos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
                model.generation_config.pad_token_id = tokenizer.pad_token_id
                model.generation_config.eos_token_id = tokenizer.eos_token_id
                token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
                    model.lm_head.weight = torch_module.nn.Parameter(torch_module.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
                    model.model.embed_tokens.weight = torch_module.nn.Parameter(torch_module.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
                model.model.mm_projector[0].weight = torch_module.nn.Parameter(torch_module.empty(4096, 2304, device=model.device, dtype=model.dtype))
                model.get_model().load_hmr_vit_backbone(**config)
                non_lora_trainables = torch_module.load(root / "non_lora_trainables.bin", map_location="cpu")
                non_lora_trainables = {
                    (key[len("base_model.model."):] if key.startswith("base_model.model.") else key): value
                    for key, value in non_lora_trainables.items()
                }
                model.resize_token_embeddings(target_vocab_size)
                current_state = model.state_dict()
                current_state_keys = list(current_state.keys())
                adjusted_state: dict[str, Any] = {}
                embedding_adjustments: list[dict[str, Any]] = []
                skipped_mismatched_keys: list[dict[str, Any]] = []
                remapped_checkpoint_keys: list[dict[str, str]] = []
                ignored_checkpoint_keys: list[str] = []
                result.details["inference_diagnostics"]["non_lora_key_sample"] = list(non_lora_trainables.keys())[:12]

                def resolve_state_key(source_key: str) -> str | None:
                    if source_key in current_state:
                        return source_key
                    suffix_matches = [state_key for state_key in current_state_keys if state_key.endswith(source_key)]
                    if len(suffix_matches) == 1:
                        return suffix_matches[0]
                    normalized_key = source_key.replace(".default", "")
                    if normalized_key != source_key:
                        if normalized_key in current_state:
                            return normalized_key
                        normalized_matches = [state_key for state_key in current_state_keys if state_key.endswith(normalized_key)]
                        if len(normalized_matches) == 1:
                            return normalized_matches[0]
                    return None

                def is_embedding_like(state_key: str) -> bool:
                    return state_key.endswith("embed_tokens.weight") or state_key.endswith("lm_head.weight")

                for key, value in non_lora_trainables.items():
                    resolved_key = resolve_state_key(key)
                    if resolved_key is None:
                        ignored_checkpoint_keys.append(key)
                        continue
                    if resolved_key != key:
                        remapped_checkpoint_keys.append({"source": key, "target": resolved_key})
                    target_value = current_state[resolved_key]
                    if tuple(value.shape) == tuple(target_value.shape):
                        adjusted_state[resolved_key] = value
                        continue
                    if value.ndim == 2 and target_value.ndim == 2 and value.shape[1] == target_value.shape[1] and is_embedding_like(resolved_key):
                        patched_value = target_value.detach().cpu().clone()
                        copy_rows = min(int(value.shape[0]), int(target_value.shape[0]))
                        patched_value[:copy_rows] = value[:copy_rows]
                        adjusted_state[resolved_key] = patched_value
                        embedding_adjustments.append({
                            "source_key": key,
                            "target_key": resolved_key,
                            "checkpoint_shape": [int(item) for item in value.shape],
                            "model_shape": [int(item) for item in target_value.shape],
                            "copied_rows": copy_rows,
                        })
                        continue
                    skipped_mismatched_keys.append({
                        "source_key": key,
                        "target_key": resolved_key,
                        "checkpoint_shape": [int(item) for item in value.shape],
                        "model_shape": [int(item) for item in target_value.shape],
                    })
                if remapped_checkpoint_keys:
                    result.details["inference_diagnostics"]["remapped_checkpoint_keys"] = remapped_checkpoint_keys[:20]
                if ignored_checkpoint_keys:
                    result.details["inference_diagnostics"]["ignored_checkpoint_keys"] = ignored_checkpoint_keys[:20]
                if embedding_adjustments:
                    result.details["inference_diagnostics"]["embedding_adjustments"] = embedding_adjustments
                if skipped_mismatched_keys:
                    result.details["inference_diagnostics"]["skipped_mismatched_keys"] = skipped_mismatched_keys
                incompatible_keys = model.load_state_dict(adjusted_state, strict=False)
                result.details["inference_diagnostics"]["load_state_dict_missing_keys"] = list(getattr(incompatible_keys, "missing_keys", []))[:20]
                result.details["inference_diagnostics"]["load_state_dict_unexpected_keys"] = list(getattr(incompatible_keys, "unexpected_keys", []))[:20]
                model = PeftModel.from_pretrained(
                    model,
                    str(root),
                    local_files_only=True,
                    ignore_mismatched_sizes=True,
                )
                model = model.merge_and_unload()
                model.get_model().load_pose_vqvae(**config)
                vision_tower = model.get_vision_tower()
                if not vision_tower.is_loaded:
                    raise RuntimeError("UniPose vision tower is not loaded")
                image_processor = vision_tower.image_processor
                model.get_pose_vqvae().to(model.device).to(torch_dtype)
                model.get_hmr_vit_backbone().to(model.device).to(torch_dtype)
                return model, image_processor

            config = Config.fromfile(str(resolved_config_path))
            config_dir = resolved_config_path.parent
            repo_root_for_assets = resolved_repo_root or config_dir.parent
            for asset_key in ("pose_vqvae_ckp_path", "hmr_vit_ckp_path"):
                asset_value = getattr(config, asset_key, None)
                if isinstance(asset_value, str) and asset_value and not Path(asset_value).is_absolute():
                    setattr(config, asset_key, str((repo_root_for_assets / asset_value).resolve()))
            result.details["inference_diagnostics"]["resolved_pose_vqvae_ckp_path"] = getattr(config, "pose_vqvae_ckp_path", "")
            result.details["inference_diagnostics"]["resolved_hmr_vit_ckp_path"] = getattr(config, "hmr_vit_ckp_path", "")
            missing_unipose_assets = []
            for asset_key in ("pose_vqvae_ckp_path", "hmr_vit_ckp_path"):
                asset_value = getattr(config, asset_key, None)
                if isinstance(asset_value, str) and asset_value and not Path(asset_value).exists():
                    missing_unipose_assets.append({"key": asset_key, "path": asset_value})
            if missing_unipose_assets:
                result.status = "partial"
                result.details["note"] = "UniPose official code is importable, but required auxiliary checkpoints are missing or not located where the official config expects."
                result.details["inference_diagnostics"]["missing_assets"] = missing_unipose_assets
                result.details["inference_diagnostics"]["error"] = "UniPose auxiliary checkpoints not found"
                return result
            conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]
            result.details["inference_diagnostics"]["cwd_before_unipose"] = str(Path.cwd())
            result.details["inference_diagnostics"]["cwd_for_unipose"] = str((resolved_repo_root or config_dir.parent).resolve())
            with pushd((resolved_repo_root or config_dir.parent).resolve()):
                model, image_processor = load_unipose_model(config)
                model.eval()
            hmr_image_processor = hmr_transform(256)
            result.load_ok = True
            result.details["inference_diagnostics"]["model_class"] = model.__class__.__name__
            result.details["inference_diagnostics"]["image_processor_class"] = image_processor.__class__.__name__

            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            processed_image = image_processor.preprocess(image_array, return_tensors="pt")["pixel_values"][0]
            processed_hmr_image = hmr_image_processor(image)
            zero_pose = torch_module.zeros((22, 3, 3), dtype=torch_dtype, device=target_device).unsqueeze(0)
            batch = {
                "body_poseA_rotmat": zero_pose,
                "body_poseB_rotmat": zero_pose.clone(),
                "images": torch_module.stack([processed_image, torch_module.zeros_like(processed_image)], dim=0).to(torch_dtype).to(target_device),
                "hmr_images": torch_module.stack([processed_hmr_image, torch_module.zeros_like(processed_hmr_image)], dim=0).to(torch_dtype).to(target_device),
                "tasks": [{"input": f"Generate pose of the image {IMAGE_TOKEN}."}],
                "caption": [""],
            }
            result.details["inference_diagnostics"]["input_shapes"] = {
                "body_poseA_rotmat": describe_tensor_shape(batch["body_poseA_rotmat"]),
                "body_poseB_rotmat": describe_tensor_shape(batch["body_poseB_rotmat"]),
                "images": describe_tensor_shape(batch["images"]),
                "hmr_images": describe_tensor_shape(batch["hmr_images"]),
            }

            with torch_module.no_grad():
                output = model.evaluate(**batch)
            result.run_ok = True
            result.status = "passed"
            result.details["inference_diagnostics"]["output_keys"] = list(output.keys()) if isinstance(output, dict) else []
            body_pose = output.get("body_pose") if isinstance(output, dict) else None
            text = output.get("text") if isinstance(output, dict) else None
            result.details["inference_diagnostics"]["has_body_pose"] = body_pose is not None
            result.details["inference_diagnostics"]["has_text"] = text is not None
            if body_pose is not None:
                result.details["inference_diagnostics"]["body_pose_shape"] = describe_tensor_shape(body_pose)
            if text is not None and len(text) > 0:
                result.details["generation_preview"] = str(text[0])[:200]
            result.details["note"] = "Official UniPose inference path completed successfully."
            return result
        except Exception as inference_exc:  # noqa: BLE001
            result.status = "partial"
            result.error = ""
            result.details["inference_diagnostics"]["error"] = str(inference_exc)
            result.details["note"] = "Adapter metadata and base model path are valid, but the UniPose official inference path did not complete in this environment."
            return result
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.error = str(exc)
        result.traceback = traceback.format_exc()
        return result
    finally:
        result.elapsed_seconds = round(time.time() - started, 3)


def test_groundingdino(model_dir: Path, image_path: str, swint_config: str, swinb_config: str) -> TestResult:
    root = model_dir / "grounded-dino"
    result = make_result("GroundingDINO", root)
    started = time.time()
    try:
        checkpoints = {
            "swint_ogc": {
                "path": root / "groundingdino_swint_ogc.pth",
                "config": swint_config.strip(),
            },
            "swinb_cogcoor": {
                "path": root / "groundingdino_swinb_cogcoor.pth",
                "config": swinb_config.strip(),
            },
        }
        result.details["variants"] = {}
        overall_load_ok = True
        overall_run_ok = True

        for key, item in checkpoints.items():
            ckpt_path = item["path"]
            variant_info: dict[str, Any] = {"path": str(ckpt_path), "config": item["config"]}
            if not ckpt_path.exists():
                variant_info["ok"] = False
                variant_info["error"] = "checkpoint missing"
                result.details["variants"][key] = variant_info
                overall_load_ok = False
                overall_run_ok = False
                continue

            ok, message = torch_load_file(ckpt_path)
            variant_info["checkpoint_readable"] = ok
            variant_info["checkpoint_message"] = message
            overall_load_ok = overall_load_ok and ok

            if not item["config"]:
                variant_info["run_status"] = "blocked"
                variant_info["run_error"] = "config path not provided"
                overall_run_ok = False
                result.details["variants"][key] = variant_info
                continue

            groundingdino_module, groundingdino_error = maybe_import("groundingdino.util.inference")
            if groundingdino_module is None:
                variant_info["run_status"] = "blocked"
                variant_info["run_error"] = f"groundingdino package not importable: {groundingdino_error}"
                overall_run_ok = False
                result.details["variants"][key] = variant_info
                continue

            config_path = Path(item["config"])
            if not config_path.exists():
                variant_info["run_status"] = "blocked"
                variant_info["run_error"] = f"config not found: {config_path}"
                overall_run_ok = False
                result.details["variants"][key] = variant_info
                continue

            try:
                from groundingdino.util.inference import load_image, load_model, predict  # type: ignore

                model = load_model(str(config_path), str(ckpt_path))
                image_source, image = load_image(image_path)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption="animal . object .",
                    box_threshold=0.25,
                    text_threshold=0.2,
                )
                variant_info["run_status"] = "passed"
                variant_info["detections"] = len(boxes) if boxes is not None else 0
                variant_info["phrases"] = [str(item) for item in (phrases or [])[:5]]
            except Exception as exc:  # noqa: BLE001
                variant_info["run_status"] = "failed"
                variant_info["run_error"] = str(exc)
                overall_run_ok = False

            result.details["variants"][key] = variant_info

        result.load_ok = overall_load_ok
        result.run_ok = overall_run_ok
        if overall_load_ok and overall_run_ok:
            result.status = "passed"
        elif overall_load_ok:
            result.status = "partial"
        else:
            result.status = "failed"
        return result
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.error = str(exc)
        result.traceback = traceback.format_exc()
        return result
    finally:
        result.elapsed_seconds = round(time.time() - started, 3)


def test_sam2(model_dir: Path, sam2_config_dir: str) -> TestResult:
    root = model_dir / "sam2"
    result = make_result("SAM2.1", root)
    started = time.time()
    try:
        config_dir = Path(sam2_config_dir) if sam2_config_dir.strip() else None
        variants = {
            "tiny": (root / "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
            "small": (root / "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
            "base_plus": (root / "sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
            "large": (root / "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        }
        result.details["variants"] = {}
        overall_load_ok = True
        overall_run_ok = True

        for key, (ckpt_path, relative_cfg) in variants.items():
            variant_info: dict[str, Any] = {"path": str(ckpt_path), "config_relative": relative_cfg}
            if not ckpt_path.exists():
                variant_info["ok"] = False
                variant_info["error"] = "checkpoint missing"
                result.details["variants"][key] = variant_info
                overall_load_ok = False
                overall_run_ok = False
                continue

            ok, message = torch_load_file(ckpt_path)
            variant_info["checkpoint_readable"] = ok
            variant_info["checkpoint_message"] = message
            overall_load_ok = overall_load_ok and ok

            sam2_module, sam2_error = maybe_import("sam2.build_sam")
            if sam2_module is None:
                variant_info["run_status"] = "blocked"
                variant_info["run_error"] = f"sam2 package not importable: {sam2_error}"
                overall_run_ok = False
                result.details["variants"][key] = variant_info
                continue

            if config_dir is None:
                variant_info["run_status"] = "blocked"
                variant_info["run_error"] = "--sam2-config-dir not provided"
                overall_run_ok = False
                result.details["variants"][key] = variant_info
                continue

            config_path = config_dir / relative_cfg
            variant_info["resolved_config"] = str(config_path)
            if not config_path.exists():
                variant_info["run_status"] = "blocked"
                variant_info["run_error"] = f"config not found: {config_path}"
                overall_run_ok = False
                result.details["variants"][key] = variant_info
                continue

            try:
                from sam2.build_sam import build_sam2  # type: ignore
                model = build_sam2(str(config_path), str(ckpt_path), device="cpu")
                variant_info["run_status"] = "passed"
                variant_info["model_class"] = model.__class__.__name__
            except Exception as exc:  # noqa: BLE001
                variant_info["run_status"] = "failed"
                variant_info["run_error"] = str(exc)
                overall_run_ok = False

            result.details["variants"][key] = variant_info

        result.load_ok = overall_load_ok
        result.run_ok = overall_run_ok
        if overall_load_ok and overall_run_ok:
            result.status = "passed"
        elif overall_load_ok:
            result.status = "partial"
        else:
            result.status = "failed"
        return result
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.error = str(exc)
        result.traceback = traceback.format_exc()
        return result
    finally:
        result.elapsed_seconds = round(time.time() - started, 3)


def test_qinsight(model_dir: Path, image_path: str, device: str, dtype_name: str, max_new_tokens: int) -> TestResult:
    root = model_dir / "Q-Insight" / "score_degradation"
    result = make_result("Q-Insight score_degradation", root)
    started = time.time()
    try:
        required = [
            root / "config.json",
            root / "preprocessor_config.json",
            root / "tokenizer_config.json",
            root / "model.safetensors.index.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        shards = sorted(root.glob("model-*.safetensors"))
        result.details["shard_count"] = len(shards)
        if missing:
            result.status = "failed"
            result.error = f"missing required files: {missing}"
            return result
        if not shards:
            result.status = "failed"
            result.error = "no safetensor shards found"
            return result

        torch_module, torch_error = maybe_import("torch")
        if torch_module is None:
            result.status = "blocked"
            result.error = f"torch not importable: {torch_error}"
            return result

        transformers_module, transformers_error = maybe_import("transformers")
        if transformers_module is None:
            result.status = "blocked"
            result.error = f"transformers not importable: {transformers_error}"
            return result

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig  # type: ignore

        qwen_vl_utils_fn = None
        qwen_vl_utils_module, qwen_vl_utils_error = maybe_import("qwen_vl_utils")
        result.details["qwen_vl_utils_importable"] = qwen_vl_utils_module is not None
        if qwen_vl_utils_module is not None:
            qwen_vl_utils_fn = getattr(qwen_vl_utils_module, "process_vision_info", None)
        else:
            result.details["qwen_vl_utils_error"] = qwen_vl_utils_error

        processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=True)
        load_attempt_errors: list[str] = []
        generation_attempt_errors: list[str] = []
        output_text = ""
        final_device_used = ""
        selected_load_strategy = ""
        selected_generation_strategy = ""
        attempted_backend = "metax" if is_metax_device(torch_module, device) else "generic"

        text_only_message = build_qinsight_message(None, QINSIGHT_TEXT_ONLY_PROMPT)
        vision_message = build_qinsight_message(image_path, QINSIGHT_DISTORTION_PROMPT)
        load_attempts = build_qwen_vl_load_attempts(torch_module, device, dtype_name)
        result.details["attempted_backend"] = attempted_backend
        result.details["is_metax_device"] = is_metax_device(torch_module, device)
        result.details["requested_device"] = device
        result.details["processor_class"] = processor.__class__.__name__
        result.details["attempt_diagnostics"] = {}

        for attempt_name, current_kwargs in load_attempts:
            model = None
            attempt_diagnostics: dict[str, Any] = {
                "load_kwargs": {key: str(value) for key, value in current_kwargs.items()},
                "text_only": {},
                "vision": {},
            }
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(root),
                    local_files_only=True,
                    trust_remote_code=True,
                    **current_kwargs,
                )
                result.load_ok = True
                selected_load_strategy = attempt_name
                model_device = next(model.parameters()).device
                final_device_used = str(model_device)
                attempt_diagnostics["model_class"] = model.__class__.__name__
                attempt_diagnostics["model_device"] = final_device_used
                result.details["model_class"] = model.__class__.__name__

                text_only_inputs = prepare_qinsight_inputs(processor, text_only_message, image_path, qwen_vl_utils_fn)
                text_only_inputs = {
                    key: value.to(model_device) if hasattr(value, "to") else value
                    for key, value in text_only_inputs.items()
                }
                attempt_diagnostics["text_only"] = {
                    "input_ids_shape": describe_tensor_shape(text_only_inputs.get("input_ids")),
                    "attention_mask_shape": describe_tensor_shape(text_only_inputs.get("attention_mask")),
                    "pixel_values_shape": describe_tensor_shape(text_only_inputs.get("pixel_values")),
                    "image_grid_thw_shape": describe_tensor_shape(text_only_inputs.get("image_grid_thw")),
                    "image_grid_thw": describe_tensor_value(text_only_inputs.get("image_grid_thw")),
                }
                with torch_module.no_grad():
                    text_only_generated_ids = model.generate(
                        **text_only_inputs,
                        generation_config=GenerationConfig(
                            do_sample=True,
                            temperature=1.0,
                            top_k=50,
                            top_p=0.95,
                            max_new_tokens=min(max_new_tokens, 32),
                        ),
                        use_cache=True,
                    )
                text_trimmed_ids = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(text_only_inputs["input_ids"], text_only_generated_ids)
                ]
                text_only_output = processor.batch_decode(
                    text_trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                attempt_diagnostics["text_only"]["status"] = "passed"
                attempt_diagnostics["text_only"]["preview"] = text_only_output[:120]

                current_inputs = prepare_qinsight_inputs(processor, vision_message, image_path, qwen_vl_utils_fn)
                current_inputs = {
                    key: value.to(model_device) if hasattr(value, "to") else value
                    for key, value in current_inputs.items()
                }
                attempt_diagnostics["vision"] = {
                    "input_ids_shape": describe_tensor_shape(current_inputs.get("input_ids")),
                    "attention_mask_shape": describe_tensor_shape(current_inputs.get("attention_mask")),
                    "pixel_values_shape": describe_tensor_shape(current_inputs.get("pixel_values")),
                    "image_grid_thw_shape": describe_tensor_shape(current_inputs.get("image_grid_thw")),
                    "image_grid_thw": describe_tensor_value(current_inputs.get("image_grid_thw")),
                }

                with torch_module.no_grad():
                    generated_ids = model.generate(
                        **current_inputs,
                        generation_config=GenerationConfig(
                            do_sample=True,
                            temperature=1.0,
                            top_k=50,
                            top_p=0.95,
                            max_new_tokens=max_new_tokens,
                        ),
                        use_cache=True,
                    )
                trimmed_ids = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(current_inputs["input_ids"], generated_ids)
                ]
                output_text = processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                attempt_diagnostics["vision"]["status"] = "passed"
                attempt_diagnostics["vision"]["preview"] = output_text[:300]
                selected_generation_strategy = attempt_name
                result.details["attempt_diagnostics"][attempt_name] = attempt_diagnostics
                break
            except Exception as exc:  # noqa: BLE001
                if "status" not in attempt_diagnostics.get("text_only", {}):
                    attempt_diagnostics.setdefault("text_only", {})["status"] = "failed"
                    attempt_diagnostics["text_only"]["error"] = str(exc)
                else:
                    attempt_diagnostics.setdefault("vision", {})["status"] = "failed"
                    attempt_diagnostics["vision"]["error"] = str(exc)
                result.details["attempt_diagnostics"][attempt_name] = attempt_diagnostics
                error_text = f"{attempt_name}: {exc}"
                if not result.load_ok:
                    load_attempt_errors.append(error_text)
                else:
                    generation_attempt_errors.append(error_text)
                    result.load_ok = True
                if model is not None:
                    del model
                    if torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()

        result.details["load_attempt_errors"] = load_attempt_errors
        result.details["generation_attempt_errors"] = generation_attempt_errors
        result.details["load_strategy"] = selected_load_strategy
        result.details["generation_strategy"] = selected_generation_strategy

        if not output_text:
            if generation_attempt_errors:
                raise RuntimeError(" | ".join(generation_attempt_errors))
            raise RuntimeError(" | ".join(load_attempt_errors))

        result.run_ok = True
        result.status = "passed"
        result.details["generation_preview"] = output_text[:300]
        result.details["device_used"] = final_device_used
        return result
    except Exception as exc:  # noqa: BLE001
        result.status = "failed"
        result.error = str(exc)
        result.traceback = traceback.format_exc()
        return result
    finally:
        result.elapsed_seconds = round(time.time() - started, 3)


def summarize(results: list[TestResult]) -> dict[str, Any]:
    counts = {"passed": 0, "partial": 0, "blocked": 0, "failed": 0}
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    image_path = args.image or build_test_image()

    results = [
        test_deeplabcut(model_dir),
        test_unipose(model_dir, image_path, args.device, args.dtype, args.unipose_repo_root, args.unipose_config, args.unipose_vision_tower),
        test_groundingdino(model_dir, image_path, args.groundingdino_swint_config, args.groundingdino_swinb_config),
        test_sam2(model_dir, args.sam2_config_dir),
        test_qinsight(model_dir, image_path, args.device, args.dtype, args.max_new_tokens),
    ]

    report = {
        "model_dir": str(model_dir),
        "image": image_path,
        "device": args.device,
        "dtype": args.dtype,
        "summary": summarize(results),
        "results": [asdict(item) for item in results],
        "notes": [
            "DeepLabCut SuperAnimal RTMPose and some vision model checkpoints are not always runnable standalone from checkpoint files alone; companion code/config assets may also be required.",
            "UniPose in the provided directory looks like a PEFT/LoRA adapter bundle, not a standalone full model.",
            "Q-Insight score_degradation looks like the only fully self-contained new model among the provided downloads.",
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report saved to: {output_path}")

    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
