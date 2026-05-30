"""
Step 3: Execute Approved Plans - Local Expert Model Evaluation Pipeline

This module handles:
1. Loading expert models onto designated GPUs at startup (via ExpertManager)
2. Parsing approved plans from c2i/output/approved_plans/
3. Running selected experts with dependency resolution (DINO -> SAM)
4. Parallel execution of independent experts via ThreadPoolExecutor
5. Collecting results into standardized Expert Testimony Bundles
6. Fault tolerance: individual expert failures are isolated and logged
"""

import os
import sys
import json
import time
import cv2
import traceback
import importlib
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
C2I_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERT_RESULTS_DIR = C2I_DIR / "output" / "expert_results"
APPROVED_PLANS_DIR = C2I_DIR / "output" / "approved_plans"

EXPERT_MODULE_MAP = {
    "animal_pose_auditor": {
        "module": "experts.expert_pose",
        "class_name": "AnimalPoseEstimator",
        "accepts_device": True,
        "internal_expert_id": "animal_pose_estimator",
    },
    "geometric_depth_auditor": {
        "module": "experts.expert_depth",
        "class_name": "MonocularDepthEstimator",
        "accepts_device": False,
        "internal_expert_id": "monocular_depth_estimator",
    },
    "topology_boundary_auditor": {
        "module": "experts.expert_sam",
        "class_name": "SegmentAnythingExpert",
        "accepts_device": False,
        "internal_expert_id": "sam_segmentor",
    },
    "open_vocabulary_detector": {
        "module": "experts.expert_detector",
        "class_name": "OpenVocabularyDetector",
        "accepts_device": False,
        "internal_expert_id": "open_vocabulary_detector",
    },
    "fine_grained_classifier": {
        "module": "experts.expert_classifier",
        "class_name": "FineGrainedClassifier",
        "accepts_device": False,
        "internal_expert_id": "fine_grained_classifier",
    },
    "perceptual_quality_auditor": {
        "module": "experts.expert_qinsight",
        "class_name": "QInsightDistortionAnalyzer",
        "accepts_device": True,
        "internal_expert_id": "qinsight_distortion_analyzer",
    },
    "image_text_auditor": {
        "module": "experts.expert_ocr",
        "class_name": "ImageTextAuditor",
        "accepts_device": False,
        "internal_expert_id": "image_text_auditor",
    },
}

DEFAULT_GPU_CONFIG = {
    "perceptual_quality_auditor": {"device": "cuda", "num_gpus": 2},
    "animal_pose_auditor": {"device": "cuda:0", "num_gpus": 1},
    "geometric_depth_auditor": {"device": "maca:0", "num_gpus": 1},
    "fine_grained_classifier": {"device": "maca:0", "num_gpus": 1},
    "open_vocabulary_detector": {"device": "cpu", "num_gpus": 0},
    "topology_boundary_auditor": {"device": "cpu", "num_gpus": 0},
    "image_text_auditor": {"device": "cpu", "num_gpus": 0},
}

EXPERT_DEPENDENCIES = {
    "topology_boundary_auditor": ["open_vocabulary_detector"],
}


class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ExpertManager:
    """Manages loading, GPU allocation, and lifecycle of expert models.

    At system startup, loads each expert model onto its designated GPU.
    Experts that are too large can occupy multiple GPUs (configured via gpu_config).
    """

    def __init__(self, gpu_config: dict | None = None):
        self.gpu_config = gpu_config or dict(DEFAULT_GPU_CONFIG)
        self.loaded_experts: dict = {}
        self.load_errors: dict = {}
        self._load_times: dict = {}

    def load_all(self, expert_ids: list[str] | None = None) -> None:
        ids_to_load = expert_ids or list(EXPERT_MODULE_MAP.keys())

        print(f"\n{'=' * 60}")
        print(f"  ExpertManager: Loading {len(ids_to_load)} expert models")
        print(f"{'=' * 60}")

        for eid in ids_to_load:
            self.load_expert(eid)

        success = len(self.loaded_experts)
        failed = len(self.load_errors)
        print(f"\n{'=' * 60}")
        print(f"  Expert Loading Complete: {success} OK, {failed} FAILED")
        if failed > 0:
            for eid, err in self.load_errors.items():
                print(f"    - {eid}: {err}")
        print(f"{'=' * 60}")

    def load_expert(self, expert_id: str) -> bool:
        if expert_id not in EXPERT_MODULE_MAP:
            msg = f"Unknown expert_id: {expert_id}"
            print(f"  [{expert_id}] ERROR: {msg}")
            self.load_errors[expert_id] = msg
            return False

        if expert_id in self.loaded_experts:
            print(f"  [{expert_id}] Already loaded, skipping.")
            return True

        config = EXPERT_MODULE_MAP[expert_id]
        gpu_cfg = self.gpu_config.get(expert_id, {})
        device = gpu_cfg.get("device", "cpu")
        num_gpus = gpu_cfg.get("num_gpus", 0)

        gpu_desc = f"{device}" + (f" ({num_gpus} GPUs)" if num_gpus > 1 else "")
        print(f"  [{expert_id}] Loading -> {gpu_desc} ...", end=" ", flush=True)

        start = time.time()
        try:
            module = importlib.import_module(config["module"])
            cls = getattr(module, config["class_name"])

            kwargs = self._build_init_kwargs(expert_id, gpu_cfg)
            instance = cls(**kwargs)

            self.loaded_experts[expert_id] = instance
            elapsed = time.time() - start
            self._load_times[expert_id] = elapsed
            print(f"OK ({elapsed:.2f}s)")
            return True

        except Exception as e:
            elapsed = time.time() - start
            self._load_times[expert_id] = elapsed
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.load_errors[expert_id] = error_msg
            print(f"FAILED ({elapsed:.2f}s)")
            print(f"           {error_msg}")
            return False

    def _build_init_kwargs(self, expert_id: str, gpu_cfg: dict) -> dict:
        kwargs = {}
        device = gpu_cfg.get("device", "cpu")

        # Use the accepts_device flag from EXPERT_MODULE_MAP for data-driven dispatch
        config = EXPERT_MODULE_MAP.get(expert_id, {})
        if config.get("accepts_device", False):
            # For CUDA-based experts, ensure a valid CUDA device; fallback to "cuda"
            if device.startswith("cuda"):
                kwargs["device"] = device
            else:
                kwargs["device"] = "cuda"

        return kwargs

    def get_expert(self, expert_id: str):
        return self.loaded_experts.get(expert_id)

    def is_loaded(self, expert_id: str) -> bool:
        return expert_id in self.loaded_experts

    def get_loaded_ids(self) -> list[str]:
        return list(self.loaded_experts.keys())

    def get_status(self) -> dict:
        return {
            "loaded": list(self.loaded_experts.keys()),
            "failed": dict(self.load_errors),
            "load_times": {k: round(v, 2) for k, v in self._load_times.items()},
        }

    def cleanup(self) -> None:
        self.loaded_experts.clear()
        self.load_errors.clear()
        self._load_times.clear()
        print("[ExpertManager] All expert models released from memory.")


def _invoke_expert_audit(
    expert_instance,
    expert_id: str,
    img_bgr,
    image_path: str,
    class_label: str,
    target_subject: str,
    hint_box: list | None = None,
) -> dict:
    """Invoke a single expert's audit() method with full exception isolation.

    Every call is wrapped in try/except so that a single expert crash
    (OOM, corrupt image, CUDA error, etc.) never takes down the entire pipeline.
    """
    start_time = time.time()

    try:
        if expert_id == "image_text_auditor":
            result = expert_instance.audit(img_bgr)
        elif expert_id == "open_vocabulary_detector":
            result = expert_instance.audit(img_bgr, query_text=target_subject, threshold=0.3)
        elif expert_id == "fine_grained_classifier":
            result = expert_instance.audit(img_bgr)
        elif expert_id == "animal_pose_auditor":
            result = expert_instance.audit(img_bgr)
        elif expert_id == "geometric_depth_auditor":
            result = expert_instance.audit(img_bgr, original_image_path=image_path)
        elif expert_id == "topology_boundary_auditor":
            result = expert_instance.audit(img_bgr, original_image_path=image_path, hint_box=hint_box)
        elif expert_id == "perceptual_quality_auditor":
            result = expert_instance.audit(image_path=image_path)
        else:
            raise ValueError(f"Unknown expert_id for execution: {expert_id}")

        elapsed_ms = (time.time() - start_time) * 1000

        if result.get("status") is None:
            result["status"] = "success"
        result["execution_time_ms"] = round(elapsed_ms, 2)

        return result

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        tb_str = traceback.format_exc()
        print(f"    [{expert_id}] EXCEPTION: {type(e).__name__}: {e}")
        return {
            "expert_id": expert_id,
            "status": "failed",
            "execution_time_ms": round(elapsed_ms, 2),
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": tb_str,
        }


def _wrap_testimony(expert_id: str, plan_entry: dict, result: dict) -> dict:
    """Wrap an expert result dict with plan-level metadata."""
    internal_id = result.get("expert_id", EXPERT_MODULE_MAP.get(expert_id, {}).get("internal_expert_id", expert_id))
    return {
        "expert_id": expert_id,
        "internal_expert_id": internal_id,
        "model_name": result.get("model_name", "unknown"),
        "target_subject": plan_entry.get("target_subject", ""),
        "weight": plan_entry.get("weight", 0.0),
        "reason": plan_entry.get("reason", ""),
        "status": result.get("status", "unknown"),
        "execution_time_ms": result.get("execution_time_ms", 0.0),
        "evidence": result.get("evidence", {}),
        "raw_metrics": result.get("raw_metrics", {}),
        "error": result.get("error"),
        "traceback": result.get("traceback"),
    }


def _make_failed_testimony(expert_id: str, plan_entry: dict, error_msg: str) -> dict:
    """Create a failed testimony entry when an expert cannot run."""
    internal_id = EXPERT_MODULE_MAP.get(expert_id, {}).get("internal_expert_id", expert_id)
    return {
        "expert_id": expert_id,
        "internal_expert_id": internal_id,
        "model_name": "N/A",
        "target_subject": plan_entry.get("target_subject", ""),
        "weight": plan_entry.get("weight", 0.0),
        "reason": plan_entry.get("reason", ""),
        "status": "failed",
        "execution_time_ms": 0.0,
        "evidence": {},
        "raw_metrics": {},
        "error": error_msg,
        "traceback": None,
    }


def execute_plan(
    approved_plan: dict,
    expert_manager: ExpertManager,
    image_path: str,
    class_label: str,
) -> dict:
    """Execute an approved plan by orchestrating the selected experts.

    Execution strategy:
      Phase 1 - Run dependency experts first (open_vocabulary_detector)
                to obtain bounding boxes for downstream experts (SAM).
      Phase 2 - Run all remaining independent experts in parallel
                using ThreadPoolExecutor.

    Returns:
        Expert Testimony Bundle dict ready for serialization.
    """
    pipeline_start = time.time()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        image_id = Path(image_path).stem if image_path else "unknown"
        return {
            "bundle_id": f"expert_results_{image_id}",
            "image_id": image_id,
            "status": "failed",
            "error": f"Cannot read image: {image_path}",
            "expert_testimonies": [],
            "execution_summary": {
                "total_experts_planned": 0,
                "successful_experts": 0,
                "failed_experts": 0,
                "total_execution_time_ms": 0.0,
                "failed_expert_ids": [],
            },
        }

    selected_experts = approved_plan.get("selected_experts", [])
    metadata = approved_plan.get("metadata", {})
    image_id = Path(image_path).stem

    print(f"  Image resolution: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    print(f"  Experts to run: {[e['expert_name'] for e in selected_experts]}")

    # ── Phase 1: Run dependency experts (detector) ──────────────────────
    detector_result = None
    hint_box = None
    detector_entry = None

    for entry in selected_experts:
        if entry["expert_name"] == "open_vocabulary_detector":
            detector_entry = entry
            break

    testimonies: list[dict] = []

    if detector_entry:
        eid = "open_vocabulary_detector"
        instance = expert_manager.get_expert(eid)

        if instance is not None:
            print(f"  [Phase 1] Running {eid} (query: '{detector_entry['target_subject']}')...")
            detector_result = _invoke_expert_audit(
                instance, eid, img_bgr, image_path, class_label,
                detector_entry["target_subject"],
            )

            if detector_result.get("status") != "failed":
                detected_objects = detector_result.get("evidence", {}).get("detected_objects", [])
                if detected_objects:
                    hint_box = detected_objects[0]["bounding_box"]
                    print(f"    Got hint_box from detector: {hint_box}")
                else:
                    print(f"    Detector found 0 objects, SAM will use center-point fallback.")

            testimonies.append(_wrap_testimony(eid, detector_entry, detector_result))
        else:
            print(f"  [Phase 1] {eid} not loaded, skipping.")
            testimonies.append(_make_failed_testimony(eid, detector_entry, "Expert model not loaded"))

    # ── Phase 2: Run remaining experts in parallel ──────────────────────
    remaining_entries = [
        e for e in selected_experts if e["expert_name"] != "open_vocabulary_detector"
    ]

    if remaining_entries:
        expert_names = [e["expert_name"] for e in remaining_entries]
        print(f"  [Phase 2] Running in parallel: {expert_names}")

        def _run_entry(entry: dict) -> dict:
            eid = entry["expert_name"]
            instance = expert_manager.get_expert(eid)

            if instance is None:
                return _make_failed_testimony(eid, entry, "Expert model not loaded")

            box = hint_box if eid == "topology_boundary_auditor" else None
            result = _invoke_expert_audit(
                instance, eid, img_bgr, image_path, class_label,
                entry["target_subject"], hint_box=box,
            )
            return _wrap_testimony(eid, entry, result)

        max_workers = min(len(remaining_entries), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_entry, entry) for entry in remaining_entries]
            for future in as_completed(futures):
                try:
                    testimony = future.result()
                    status_icon = "OK" if testimony["status"] == "success" else "FAIL"
                    print(f"    [{testimony['expert_id']}] {status_icon} "
                          f"({testimony['execution_time_ms']:.0f}ms)")
                    testimonies.append(testimony)
                except Exception as e:
                    print(f"    [FUTURE ERROR] {type(e).__name__}: {e}")

    # ── Assemble Expert Testimony Bundle ────────────────────────────────
    pipeline_elapsed = (time.time() - pipeline_start) * 1000

    successful = [t for t in testimonies if t.get("status") == "success"]
    failed = [t for t in testimonies if t.get("status") == "failed"]

    bundle = {
        "bundle_id": f"expert_results_{image_id}",
        "image_id": image_id,
        "image_path": str(image_path),
        "image_resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
        "class_id": metadata.get("class_id"),
        "class_label": class_label,
        "plan_reference": f"approved_plan_{image_id}",
        "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "focus_areas": approved_plan.get("focus_areas", []),
        "custom_prompts_for_reflector": approved_plan.get("custom_prompts_for_reflector", ""),
        "expert_testimonies": testimonies,
        "execution_summary": {
            "total_experts_planned": len(selected_experts),
            "successful_experts": len(successful),
            "failed_experts": len(failed),
            "total_execution_time_ms": round(pipeline_elapsed, 2),
            "failed_expert_ids": [t["expert_id"] for t in failed],
        },
    }

    return bundle


def save_testimony_bundle(bundle: dict, output_dir: str | Path | None = None) -> str:
    """Save the Expert Testimony Bundle to disk as expert_results_[image_id].json."""
    if output_dir is None:
        output_dir = EXPERT_RESULTS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id = bundle.get("image_id", "unknown")
    filename = f"expert_results_{image_id}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=4, ensure_ascii=False, cls=NumpySafeEncoder)

    print(f"  [SAVED] {filename}")
    return str(filepath)


def load_approved_plans(approved_dir: str | Path | None = None) -> list[dict]:
    """Load all approved_plan_*.json files from the directory."""
    if approved_dir is None:
        approved_dir = APPROVED_PLANS_DIR

    approved_dir = Path(approved_dir)
    if not approved_dir.exists():
        print(f"[WARN] Approved plans directory not found: {approved_dir}")
        return []

    plans = []
    for f in sorted(approved_dir.glob("approved_plan_*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                plan = json.load(fp)
            plan["_source_file"] = f.name
            plans.append(plan)
        except Exception as e:
            print(f"[WARN] Failed to load plan {f.name}: {e}")

    return plans


def resolve_image_path(image_path: str) -> str | None:
    """Resolve an image path that may be relative to the c2i dir or project root.

    Plan metadata stores paths like '../test_images/000000.png' which are
    relative to the c2i directory.  We try multiple base directories so that
    the resolution works regardless of where the script is invoked from.
    """
    if not image_path:
        return None

    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path

    candidates = [
        (C2I_DIR / image_path).resolve(),
        (PROJECT_ROOT / image_path).resolve(),
        Path(image_path).resolve(),
    ]

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)

    return None


def collect_required_expert_ids(plans: list[dict]) -> list[str]:
    """Collect the union of all expert IDs needed across all plans."""
    expert_ids = set()
    for plan in plans:
        for entry in plan.get("selected_experts", []):
            expert_ids.add(entry["expert_name"])
    return sorted(expert_ids)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="THEMIS Step 3: Execute Approved Plans with Local Expert Models"
    )
    parser.add_argument(
        "--approved-dir", type=str, default=None,
        help="Directory containing approved_plan_*.json files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save expert_results_*.json files",
    )
    parser.add_argument(
        "--image-id", type=str, default="",
        help="Process a single image by its ID (e.g., 000000)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of plans to process (0 = all)",
    )
    parser.add_argument(
        "--gpu-config", type=str, default=None,
        help="Path to a custom GPU allocation JSON file",
    )
    args = parser.parse_args()

    gpu_config = DEFAULT_GPU_CONFIG
    if args.gpu_config:
        try:
            with open(args.gpu_config, "r", encoding="utf-8") as f:
                gpu_config = json.load(f)
            print(f"[INFO] Loaded custom GPU config from: {args.gpu_config}")
        except Exception as e:
            print(f"[WARN] Failed to load GPU config, using defaults: {e}")

    plans = load_approved_plans(args.approved_dir)

    if args.image_id:
        plans = [p for p in plans if args.image_id in p.get("_source_file", "")]
        if not plans:
            print(f"[ERROR] No approved plan found for image ID: {args.image_id}")
            return

    if args.limit > 0:
        plans = plans[:args.limit]

    if not plans:
        print("[ERROR] No approved plans found. Run Step 1+2 first.")
        return

    required_ids = collect_required_expert_ids(plans)
    print(f"\n[INFO] Plans to execute: {len(plans)}")
    print(f"[INFO] Required experts: {required_ids}")

    expert_manager = ExpertManager(gpu_config=gpu_config)
    expert_manager.load_all(required_ids)

    if not expert_manager.loaded_experts:
        print("[ERROR] No experts loaded successfully. Cannot proceed.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Step 3: Executing {len(plans)} Approved Plans")
    print(f"  Loaded experts: {expert_manager.get_loaded_ids()}")
    print(f"{'=' * 60}")

    success_count = 0
    failed_count = 0
    total_start = time.time()

    for idx, plan in enumerate(plans, 1):
        metadata = plan.get("metadata", {})
        image_path_raw = metadata.get("original_image", "")
        class_label = metadata.get("class_label", "")
        source_file = plan.get("_source_file", "unknown")

        image_path = resolve_image_path(image_path_raw)
        if image_path is None:
            print(f"\n[{idx}/{len(plans)}] Image not found: {image_path_raw} (Plan: {source_file})")
            failed_count += 1
            continue

        print(f"\n[{idx}/{len(plans)}] {os.path.basename(image_path)} (Plan: {source_file})")

        try:
            bundle = execute_plan(plan, expert_manager, image_path, class_label)
            save_testimony_bundle(bundle, args.output_dir)
            success_count += 1
        except Exception as e:
            print(f"  [FATAL] Pipeline crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed_count += 1

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  Step 3 Execution Summary")
    print(f"{'=' * 60}")
    print(f"  Total plans:     {len(plans)}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {failed_count}")
    print(f"  Total elapsed:   {total_elapsed:.2f}s")
    print(f"{'=' * 60}")

    expert_manager.cleanup()


if __name__ == "__main__":
    main()
