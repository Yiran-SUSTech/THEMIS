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
import threading
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
        "accepts_device": True,
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
        "accepts_device": True,
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
    "perceptual_quality_auditor": {"device": "cuda", "num_gpus": 2},    # Q-Insight → GPU 0+1 (needs ~50GB)
    "animal_pose_auditor":        {"device": "cuda:2", "num_gpus": 1},  # ViTPose → GPU 2
    "geometric_depth_auditor":    {"device": "maca:1", "num_gpus": 1},  # Depth → MACA 1
    "fine_grained_classifier":    {"device": "maca:1", "num_gpus": 1},  # EVA-02 → MACA 1
    "open_vocabulary_detector": {"device": "cpu", "num_gpus": 0},
    "topology_boundary_auditor": {"device": "cpu", "num_gpus": 0},
    "image_text_auditor": {"device": "cpu", "num_gpus": 0},
}

CPU_EXPERT_IDS = {
    "open_vocabulary_detector",
    "topology_boundary_auditor",
    "image_text_auditor",
}

EXPERT_DEPENDENCIES = {
    "topology_boundary_auditor": ["open_vocabulary_detector"],
}

CPU_EXPERT_IDS = {
    "open_vocabulary_detector",
    "topology_boundary_auditor",
    "image_text_auditor",
    "animal_pose_auditor",
}


def _limit_onnx_threads(num_threads: int = 2) -> None:
    """Set ONNX Runtime global thread pool size to avoid CPU contention.

    When multiple ONNX sessions run concurrently (e.g. in multi-group mode),
    each session defaults to using ALL CPU cores, causing severe contention.
    Call this once at startup to cap the per-session thread count.
    """
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.inter_op_num_threads = 1
        ort.set_default_session_options(sess_opts)
        print(f"  [ONNX] Thread limit set: intra_op={num_threads}, inter_op=1")
    except Exception as e:
        print(f"  [ONNX] Failed to set thread limit: {e}")


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

    If a shared_cpu_manager is set, CPU experts not loaded in this manager
    will be resolved from the shared pool instead.
    """

    def __init__(self, gpu_config: dict | None = None, expert_output_dirs: dict | None = None,
                 shared_cpu_manager: 'ExpertManager | None' = None):
        self.gpu_config = gpu_config or dict(DEFAULT_GPU_CONFIG)
        self.expert_output_dirs = expert_output_dirs or {}
        self.loaded_experts: dict = {}
        self.load_errors: dict = {}
        self._load_times: dict = {}
        self.shared_cpu_manager = shared_cpu_manager

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
        num_gpus = gpu_cfg.get("num_gpus", 0)

        config = EXPERT_MODULE_MAP.get(expert_id, {})

        if config.get("accepts_device", False):
            kwargs["device"] = device

        if expert_id == "perceptual_quality_auditor":
            kwargs["num_gpus"] = num_gpus
            kwargs["model_path"] = str(PROJECT_ROOT / "models" / "Q-Insight" / "score_degradation")
            if "max_memory" in gpu_cfg:
                kwargs["max_memory"] = gpu_cfg["max_memory"]

        if expert_id == "open_vocabulary_detector":
            kwargs["model_path"] = str(PROJECT_ROOT / "new_models" / "groundingdino_sim.onnx")
            kwargs["config_path"] = str(PROJECT_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")

        if expert_id == "topology_boundary_auditor":
            kwargs["model_dir"] = str(PROJECT_ROOT / "new_models" / "sam1_onnx" / "machine_learning_models")
            if expert_id in self.expert_output_dirs:
                kwargs["output_dir"] = str(self.expert_output_dirs[expert_id])
            else:
                kwargs["output_dir"] = str(C2I_DIR / "output" / "sam_masks")
        elif expert_id == "geometric_depth_auditor":
            if expert_id in self.expert_output_dirs:
                kwargs["output_dir"] = str(self.expert_output_dirs[expert_id])
            else:
                kwargs["output_dir"] = str(C2I_DIR / "output" / "depth_maps")
        elif expert_id == "image_text_auditor":
            kwargs["det_model_path"] = str(PROJECT_ROOT / "models" / "Multilingual_PP-OCRv3_det_infer.onnx")

        return kwargs

    def get_expert(self, expert_id: str):
        instance = self.loaded_experts.get(expert_id)
        if instance is not None:
            return instance
        if self.shared_cpu_manager is not None and expert_id in CPU_EXPERT_IDS:
            return self.shared_cpu_manager.get_expert(expert_id)
        return None

    def is_loaded(self, expert_id: str) -> bool:
        if expert_id in self.loaded_experts:
            return True
        if self.shared_cpu_manager is not None and expert_id in CPU_EXPERT_IDS:
            return self.shared_cpu_manager.is_loaded(expert_id)
        return False

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
    save_pose_viz: bool = False,
    pose_viz_path: str | None = None,
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
            result = expert_instance.audit(
                img_bgr,
                save_viz=save_pose_viz,
                viz_output_path=pose_viz_path,
            )
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
    save_pose_viz: bool = False,
    cpu_semaphore: threading.Semaphore | None = None,
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
    # Support multiple open_vocabulary_detector entries with different targets.
    # Each detector run produces a hint_box keyed by its target_subject.
    hint_box_map: dict[str, list] = {}
    detector_entries = [
        e for e in selected_experts if e["expert_name"] == "open_vocabulary_detector"
    ]

    testimonies: list[dict] = []

    for det_idx, detector_entry in enumerate(detector_entries):
        eid = "open_vocabulary_detector"
        instance = expert_manager.get_expert(eid)
        target = detector_entry.get("target_subject", "")

        if instance is not None:
            if cpu_semaphore is not None:
                cpu_semaphore.acquire()
            try:
                print(f"  [Phase 1] Running {eid} #{det_idx+1} (query: '{target}')...")
                detector_result = _invoke_expert_audit(
                    instance, eid, img_bgr, image_path, class_label,
                    target,
                )
            finally:
                if cpu_semaphore is not None:
                    cpu_semaphore.release()

            if detector_result.get("status") != "failed":
                detected_objects = detector_result.get("evidence", {}).get("detected_objects", [])
                if detected_objects:
                    box = detected_objects[0]["bounding_box"]
                    hint_box_map[target] = box
                    print(f"    Got hint_box for '{target}': {box}")
                else:
                    print(f"    Detector found 0 objects for '{target}', SAM will use center-point fallback.")

            testimonies.append(_wrap_testimony(eid, detector_entry, detector_result))
        else:
            print(f"  [Phase 1] {eid} not loaded, skipping.")
            testimonies.append(_make_failed_testimony(eid, detector_entry, "Expert model not loaded"))

    # ── Phase 2: Run remaining experts in parallel ──────────────────────
    remaining_entries = [
        e for e in selected_experts if e["expert_name"] != "open_vocabulary_detector"
    ]

    if remaining_entries:
        expert_names = [f"{e['expert_name']}->{e.get('target_subject','')}" for e in remaining_entries]
        print(f"  [Phase 2] Running in parallel: {expert_names}")

        pose_viz_count: dict[str, int] = {}
        entry_pose_viz: dict[int, str | None] = {}
        for i, entry in enumerate(remaining_entries):
            eid = entry["expert_name"]
            pose_viz = None
            if save_pose_viz and eid == "animal_pose_auditor":
                pose_viz_dir = C2I_DIR / "output" / "pose_visualizations"
                count = pose_viz_count.get(eid, 0)
                pose_viz_count[eid] = count + 1
                target_slug = entry.get("target_subject", "subject").replace(" ", "_")[:20]
                suffix = f"_{target_slug}" if count > 0 or pose_viz_count.get(eid, 0) > 1 else ""
                pose_viz = str(pose_viz_dir / f"{image_id}_pose_viz{suffix}.png")
            entry_pose_viz[i] = pose_viz

        def _run_entry(entry: dict, idx: int) -> dict:
            eid = entry["expert_name"]
            is_cpu_expert = eid in CPU_EXPERT_IDS
            sem_ctx = cpu_semaphore if (is_cpu_expert and cpu_semaphore is not None) else None

            if sem_ctx is not None:
                sem_ctx.acquire()
            try:
                instance = expert_manager.get_expert(eid)

                if instance is None:
                    return _make_failed_testimony(eid, entry, "Expert model not loaded")

                box = None
                if eid == "topology_boundary_auditor":
                    target = entry.get("target_subject", "")
                    box = hint_box_map.get(target)
                    if box is None and hint_box_map:
                        box = next(iter(hint_box_map.values()))

                pose_viz = entry_pose_viz.get(idx)

                result = _invoke_expert_audit(
                    instance, eid, img_bgr, image_path, class_label,
                    entry["target_subject"], hint_box=box,
                    save_pose_viz=save_pose_viz, pose_viz_path=pose_viz,
                )
                return _wrap_testimony(eid, entry, result)
            finally:
                if sem_ctx is not None:
                    sem_ctx.release()

        max_workers = min(len(remaining_entries), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_entry, entry, i) for i, entry in enumerate(remaining_entries)]
            for future in as_completed(futures):
                try:
                    testimony = future.result()
                    status_icon = "OK" if testimony["status"] == "success" else "FAIL"
                    target_info = f"->{testimony.get('target_subject','')}" if testimony.get('target_subject') else ""
                    print(f"    [{testimony['expert_id']}{target_info}] {status_icon} "
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
        "image_description": approved_plan.get("image_description", ""),
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
