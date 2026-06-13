"""
THEMIS C2I Common - Shared utilities, constants, and configurations.

All dispatch modes (sync, async, batch) import from here.
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
C2I_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(C2I_DIR) not in sys.path:
    sys.path.insert(0, str(C2I_DIR))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Path Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMAGE_DIR = PROJECT_ROOT / "test_images"
CLASS_IDS_TXT = IMAGE_DIR / "class_ids.txt"
IMAGENET_CLASSES_JSON = PROJECT_ROOT / "imagenet_classes.json"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"

PLAN_DIR = C2I_DIR / "output" / "plans"
APPROVED_DIR = C2I_DIR / "output" / "approved_plans"
JUDGE_FEEDBACK_DIR = C2I_DIR / "output" / "judge_feedback"
EXPERT_RESULTS_DIR = C2I_DIR / "output" / "expert_results"
FINAL_REPORTS_DIR = C2I_DIR / "output" / "final_reports"
BATCH_DIR = C2I_DIR / "output" / "batch"
GPU_PRESETS_DIR = PROJECT_ROOT / "gpu_configs"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GPU Group Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_gpu_group_configs(num_groups: int, base_config: dict) -> list[dict]:
    """Split GPUs into N groups, each with a full set of experts."""
    if num_groups == 1:
        return [base_config]

    group_configs = []
    for g in range(num_groups):
        offset = g * 4
        maca_offset = g * 2
        config = {
            "perceptual_quality_auditor": {"device": f"cuda:{offset}", "num_gpus": 2},
            "animal_pose_auditor": {"device": f"cuda:{offset + 2}", "num_gpus": 1},
            "geometric_depth_auditor": {"device": f"maca:{maca_offset}", "num_gpus": 1},
            "fine_grained_classifier": {"device": f"maca:{maca_offset + 1}", "num_gpus": 1},
            "open_vocabulary_detector": {"device": "cpu", "num_gpus": 0},
            "topology_boundary_auditor": {"device": "cpu", "num_gpus": 0},
            "image_text_auditor": {"device": "cpu", "num_gpus": 0},
        }
        group_configs.append(config)

    return group_configs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Loading Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_class_ids(txt_path: str) -> dict[str, int]:
    img_to_class = {}
    if not os.path.exists(txt_path):
        print(f"[WARN] class_ids.txt not found: {txt_path}")
        return img_to_class
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                try:
                    img_to_class[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return img_to_class


def load_imagenet_classes(json_path: str) -> dict[int, str]:
    class_map = {}
    if not os.path.exists(json_path):
        print(f"[WARN] imagenet_classes.json not found: {json_path}")
        return class_map
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        try:
            class_map[int(k)] = v
        except (ValueError, TypeError):
            continue
    return class_map


def resolve_image_path(image_dir: Path, img_id: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = image_dir / f"{img_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def compute_router_scores(plan: dict) -> dict:
    """Compute alignment and artifact scores from Router's checkpoint verdicts.

    Added by Yiran for Router-level scoring before Reflector.
    """
    checkpoint_verdicts = plan.get("checkpoint_verdicts", [])
    artifact_observations = plan.get("artifact_observations", [])

    testable = [cv for cv in checkpoint_verdicts if cv.get("is_testable", False)]
    present = [cv for cv in testable if cv.get("is_present", False)]
    untestable = [cv for cv in checkpoint_verdicts if not cv.get("is_testable", False)]
    alignment_score = 5.0 * len(present) / len(testable) if testable else 0.0

    if untestable:
        untestable_penalty = 0.5 * len(untestable) / len(checkpoint_verdicts) * alignment_score
        alignment_score -= untestable_penalty

    if artifact_observations:
        max_severity = max(ao.get("severity", 0.0) for ao in artifact_observations)
        severe_count = sum(1 for ao in artifact_observations if ao.get("severity", 0.0) >= 2.0)
        minor_count = len(artifact_observations) - severe_count
        artifact_score = 5.0 - max_severity - 0.3 * severe_count - 0.15 * minor_count
        artifact_score = max(0.0, artifact_score)
    else:
        artifact_score = 5.0

    return {
        "router_alignment_score": round(alignment_score, 2),
        "router_artifact_score": round(artifact_score, 2),
        "checkpoint_summary": {
            "total": len(checkpoint_verdicts),
            "testable": len(testable),
            "present": len(present),
            "absent": len(testable) - len(present),
            "untestable": len(untestable),
        },
        "artifact_summary": {
            "count": len(artifact_observations),
            "max_severity": max((ao.get("severity", 0.0) for ao in artifact_observations), default=0.0),
            "severe_count": sum(1 for ao in artifact_observations if ao.get("severity", 0.0) >= 2.0),
            "minor_count": sum(1 for ao in artifact_observations if 0 < ao.get("severity", 0.0) < 2.0),
        },
    }


def save_judge_feedback(
    judge_feedback_dir: Path,
    img_id: str,
    iteration: int,
    judge_result: dict,
    plan: dict,
    class_label: str,
) -> None:
    feedback_record = {
        "image_id": img_id,
        "class_label": class_label,
        "iteration": iteration,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_verdict": {
            "is_approved": judge_result.get("is_approved", False),
            "reasons_for_rejection": judge_result.get("reasons_for_rejection", ""),
            "suggestions": judge_result.get("suggestions", []),
        },
        "audited_plan": plan,
    }
    feedback_path = judge_feedback_dir / f"judge_feedback_{img_id}_iter{iteration}.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback_record, f, indent=4, ensure_ascii=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Image List Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_image_list(
    image_dir: Path,
    class_ids_txt: Path,
    image_id_filter: str = "",
    limit: int = 0,
) -> list[tuple[str, str, int, str]]:
    """Build list of (img_name, img_id, class_id, class_label) tuples.

    Returns filtered and limited image list ready for processing.
    """
    img_to_class = parse_class_ids(str(class_ids_txt))
    imagenet_classes = load_imagenet_classes(str(IMAGENET_CLASSES_JSON))

    if not img_to_class:
        print("[ERROR] No image-class mappings loaded. Check class_ids.txt path.")
        return []

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    valid_images = []
    for img_name in image_files:
        img_id = os.path.splitext(img_name)[0]
        if img_id not in img_to_class:
            continue
        class_id = img_to_class[img_id]
        class_label = imagenet_classes.get(class_id, f"class_{class_id}")
        valid_images.append((img_name, img_id, class_id, class_label))

    if image_id_filter:
        valid_images = [
            (n, iid, cid, cl) for n, iid, cid, cl in valid_images
            if iid == image_id_filter
        ]

    if limit > 0:
        valid_images = valid_images[:limit]

    return valid_images


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Expert Manager Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preload_expert_managers(
    num_groups: int,
    gpu_config_path: str | None,
    required_ids: list[str] | None = None,
    gpu_preset: str | None = None,
) -> list:
    """Pre-load ExpertManager instances for GPU execution.

    Priority: gpu_config_path > gpu_preset > num_groups (auto-split)

    Preset format supports two styles:
      - Multi-group: {"groups": [config0, config1, ...]}  → one ExpertManager per group
      - Single-group: {expert_id: {...}, ...}              → one ExpertManager total
    """
    from step3_execute import ExpertManager, DEFAULT_GPU_CONFIG, EXPERT_MODULE_MAP

    if gpu_config_path:
        with open(gpu_config_path, "r") as f:
            custom_config = json.load(f)
        # Support both formats
        if "groups" in custom_config:
            group_configs = custom_config["groups"]
        else:
            custom_config.pop("_description", None)
            group_configs = [custom_config]
    elif gpu_preset:
        preset_path = GPU_PRESETS_DIR / f"{gpu_preset}.json"
        if not preset_path.exists():
            print(f"[ERROR] GPU preset not found: {preset_path}")
            print(f"  Available presets: {list(GPU_PRESETS_DIR.glob('*.json'))}")
            sys.exit(1)
        with open(preset_path, "r") as f:
            preset_config = json.load(f)
        # Support both formats
        if "groups" in preset_config:
            group_configs = preset_config["groups"]
        else:
            preset_config.pop("_description", None)
            group_configs = [preset_config]
    else:
        group_configs = build_gpu_group_configs(num_groups, DEFAULT_GPU_CONFIG)

    if required_ids is None:
        required_ids = list(EXPERT_MODULE_MAP.keys())

    print(f"\n{'='*60}")
    print(f"  Pre-loading {len(group_configs)} GPU group(s)")
    print(f"{'='*60}")

    expert_managers = []
    for g, cfg in enumerate(group_configs):
        print(f"\n  --- GPU Group {g} ---")
        em = ExpertManager(gpu_config=cfg)
        em.load_all(required_ids)
        expert_managers.append(em)

    total_loaded = sum(len(em.loaded_experts) for em in expert_managers)
    print(f"\n  Total expert instances loaded: {total_loaded}")
    print(f"{'='*60}")

    return expert_managers
