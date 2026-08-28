"""
THEMIS T2I Common - Shared utilities, constants, and configurations.

Adapted from c2i_harness/common.py with T2I-specific changes:
- Uses GenEval2 JSONL for prompt metadata instead of class_ids.txt
- Reuses c2i_harness.step3_execute for expert execution
- Removes C2I-specific scoring (compute_router_scores, parse_class_ids, load_imagenet_classes)
- Adds T2I-specific data loading (load_geneval2_data, build_t2i_image_list)
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
T2I_DIR = Path(__file__).resolve().parent
C2I_DIR = PROJECT_ROOT / "c2i_harness"

# Insert in reverse priority order so T2I_DIR ends up first in sys.path.
# This ensures "import common" resolves to t2i_harness/common.py, not
# c2i_harness/common.py (both define a module named "common").
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(C2I_DIR) not in sys.path:
    sys.path.insert(0, str(C2I_DIR))
if str(T2I_DIR) not in sys.path:
    sys.path.insert(0, str(T2I_DIR))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Path Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMAGE_DIR = PROJECT_ROOT / "test_images"
GENEVAL2_DATA_JSONL = PROJECT_ROOT / "geneval2_data.jsonl"
OBJECT_TO_CLASSID_JSON = T2I_DIR / "object_to_classid.json"
EXPERTS_REGISTRY_JSON = PROJECT_ROOT / "expert_registry.json"

# Output root: override via T2I_OUTPUT_DIR_NAME env var (set by run.py --output-dir).
# All output subdirectories derive from this single source of truth.
OUTPUT_DIR = T2I_DIR / os.environ.get("T2I_OUTPUT_DIR_NAME", "output")
PLAN_DIR = OUTPUT_DIR / "plans"
APPROVED_DIR = OUTPUT_DIR / "approved_plans"
JUDGE_FEEDBACK_DIR = OUTPUT_DIR / "judge_feedback"
EXPERT_RESULTS_DIR = OUTPUT_DIR / "expert_results"
FINAL_REPORTS_DIR = OUTPUT_DIR / "final_reports"
WITHOUT_EXPERT_REPORTS_DIR = OUTPUT_DIR / "without_expert_reports"
BATCH_DIR = OUTPUT_DIR / "batch"
ATOMIZED_DIR = OUTPUT_DIR / "atomized"
GPU_PRESETS_DIR = PROJECT_ROOT / "gpu_configs"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Debug Logging Infrastructure
#  - install_run_log(): tee stdout to OUTPUT_DIR/run_<ts>.log (timestamped)
#  - dump_debug_raw(): save full raw LLM responses for offline inspection
#  - record_failure(): collect per-image failures into stats['failed_images']
#  - bump_progress(): periodic "done/total" progress lines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_LOG_LOCK = threading.Lock()
_LOG_STREAM = None
_LOG_FILE_PATH: Path | None = None


def install_run_log() -> Path | None:
    """Tee stdout AND stderr into OUTPUT_DIR/run_<timestamp>.log with per-line timestamps.

    Console output stays unchanged; only the file copy gets timestamps
    (stderr lines get a [ERR] marker so tracebacks are easy to grep).
    Capturing stderr is essential: unhandled exception tracebacks and
    library warnings go to stderr, not stdout.
    Thread-safe (prints may come from asyncio loop + executor threads).
    Call once at process start (run.py main).
    """
    global _LOG_STREAM, _LOG_FILE_PATH
    if _LOG_STREAM is not None:
        return _LOG_FILE_PATH
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _LOG_FILE_PATH = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _LOG_STREAM = open(_LOG_FILE_PATH, "a", encoding="utf-8", buffering=1)
    except Exception as e:
        print(f"[WARN] Failed to open run log file: {e}")
        return None

    import atexit
    atexit.register(_flush_run_log)

    class _TeeStream:
        def __init__(self, orig, err_marker: bool):
            self._orig = orig
            self._err_marker = err_marker
            self._pending = ""

        def write(self, data):
            self._orig.write(data)
            with _LOG_LOCK:
                self._pending += data
                while "\n" in self._pending:
                    line, self._pending = self._pending.split("\n", 1)
                    if line.strip():
                        ts = datetime.now().strftime("%H:%M:%S")
                        marker = "[ERR] " if self._err_marker else ""
                        _LOG_STREAM.write(f"[{ts}] {marker}{line}\n")

        def flush(self):
            self._orig.flush()
            with _LOG_LOCK:
                _LOG_STREAM.flush()

        def __getattr__(self, name):
            return getattr(self._orig, name)

    sys.stdout = _TeeStream(sys.stdout, err_marker=False)
    sys.stderr = _TeeStream(sys.stderr, err_marker=True)
    print(f"[LOG] Run log file: {_LOG_FILE_PATH}")
    return _LOG_FILE_PATH


def _flush_run_log() -> None:
    try:
        if _LOG_STREAM is not None:
            with _LOG_LOCK:
                _LOG_STREAM.flush()
    except Exception:
        pass


DEBUG_RAW_DIR = OUTPUT_DIR / "debug_raw"


def dump_debug_raw(stage: str, img_id: str, raw_text, note: str = "") -> str | None:
    """Save a raw LLM response under OUTPUT_DIR/debug_raw/ for offline inspection.

    Called when JSON parsing fails so the FULL response is preserved instead of
    the truncated console preview. Returns the saved path, or None on failure.
    """
    try:
        DEBUG_RAW_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        path = DEBUG_RAW_DIR / f"{stage}_{img_id}_{ts}.txt"
        with open(path, "w", encoding="utf-8") as f:
            if note:
                f.write(note.rstrip() + "\n\n")
            f.write(raw_text if raw_text is not None else "<None>")
        return str(path)
    except Exception:
        return None


_FAILURES_LOCK = threading.Lock()


def record_failure(stats: dict, img_id, stage: str, error) -> None:
    """Append an image-level failure entry to stats['failed_images'].

    stats is the shared pipeline stats dict; safe to call from any thread.
    run.py turns the collected list into failed_images.json at the end.
    """
    if stats is None:
        return
    entry = {
        "img_id": str(img_id),
        "stage": stage,
        "error": str(error)[:500],
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    with _FAILURES_LOCK:
        stats.setdefault("failed_images", []).append(entry)


_PROGRESS_LOCK = threading.Lock()
_PROGRESS_STATE: dict[str, int] = {}


def bump_progress(label: str, total: int, every: int = 25) -> None:
    """Increment and periodically report progress for a pipeline stage."""
    with _PROGRESS_LOCK:
        _PROGRESS_STATE[label] = _PROGRESS_STATE.get(label, 0) + 1
        done = _PROGRESS_STATE[label]
    if total > 0 and (done % every == 0 or done >= total):
        print(f"  [PROGRESS] {label}: {done}/{total}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API Retry Utility
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default retry count (can be overridden via --api-retry)
DEFAULT_API_RETRY = 0


def _is_retryable_error(e: Exception) -> bool:
    """Determine if an API error is worth retrying.

    Non-retryable errors (will always fail with same input):
    - HTTP 400 with content filter / data inspection failures
    - HTTP 401/403 (auth errors)
    - HTTP 422 (invalid request format)

    Retryable errors (transient, may succeed on retry):
    - HTTP 429 (rate limit)
    - HTTP 500/502/503 (server errors)
    - Network timeouts / connection errors
    """
    error_str = str(e).lower()
    status_code = getattr(getattr(e, "response", None), "status_code", None)
    if status_code is not None:
        if status_code in (400, 401, 403, 422):
            return False
    non_retryable_keywords = [
        "datainspectionfailed",
        "data inspection failed",
        "content_filter",
        "content filter",
        "inappropriate content",
        "invalid_api_key",
        "invalid x-api-key",
        "authentication",
        "invalid request",
    ]
    for kw in non_retryable_keywords:
        if kw in error_str:
            return False
    return True


def api_call_with_retry(func, *args, max_retries=0, retry_delay=2.0, label="API", **kwargs):
    """Call an API function with automatic retry on failure.

    Args:
        func: The API call function (e.g., client.chat.completions.create).
        *args: Positional arguments to pass to func.
        max_retries: Number of retries after the first failure (0 = no retry).
        retry_delay: Base delay in seconds between retries (doubles each retry).
        label: Label for log messages (e.g., "Router", "Judge", "Reflector").
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The return value of func on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not _is_retryable_error(e):
                print(f"  [SKIP-RETRY] {label} non-retryable error: {e}")
                raise
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                print(f"  [RETRY] {label} API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"  [RETRY] Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"  [RETRY] {label} API call failed after {max_retries + 1} attempt(s): {e}")
    raise last_exception


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

def resolve_image_path(image_dir: Path, img_id: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = image_dir / f"{img_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_geneval2_data(jsonl_path: Path | str | None = None) -> dict[str, dict]:
    """Load the GenEval2 JSONL file and return a dict mapping prompt_id to prompt data.

    The prompt_id is the 0-based line index (as a string), which matches the
    stem of generated image files (e.g., line 0 -> "0.png").

    Args:
        jsonl_path: Path to the geneval2_data.jsonl file. If None, uses the
            default GENEVAL2_DATA_JSONL constant.

    Returns:
        Dict mapping prompt_id (str) to the parsed JSON record (dict).
        Returns an empty dict if the file does not exist.
    """
    if jsonl_path is None:
        jsonl_path = GENEVAL2_DATA_JSONL
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"[WARN] geneval2_data.jsonl not found: {jsonl_path}")
        return {}
    result: dict[str, dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse JSONL line {index}, skipping.")
                continue
            prompt_id = str(index)
            record["prompt_id"] = prompt_id
            result[prompt_id] = record
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Judge Feedback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_judge_feedback(
    judge_feedback_dir: Path,
    img_id: str,
    iteration: int,
    judge_result: dict,
    plan: dict,
    prompt: str,
) -> None:
    feedback_record = {
        "image_id": img_id,
        "prompt": prompt,
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
#  T2I Image List Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_t2i_image_list(
    image_dir: Path,
    geneval2_jsonl_path: Path | str | None = None,
    image_id_filter: str = "",
    limit: int = 0,
) -> list[tuple[str, str, str]]:
    """Build list of (img_name, img_id, prompt) tuples for T2I evaluation.

    Reads the GenEval2 JSONL to get prompt_id -> prompt mapping, then matches
    image files in image_dir by filename stem matching prompt_id.

    Args:
        image_dir: Directory containing generated test images.
        geneval2_jsonl_path: Path to geneval2_data.jsonl. If None, uses the
            default GENEVAL2_DATA_JSONL constant.
        image_id_filter: If non-empty, only include the image with this stem.
        limit: If > 0, limit the number of images returned.

    Returns:
        List of (img_name, img_id, prompt) tuples.
    """
    prompt_data = load_geneval2_data(geneval2_jsonl_path)

    if not prompt_data:
        print("[ERROR] No GenEval2 prompt data loaded. Check geneval2_data.jsonl path.")
        return []

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    valid_images: list[tuple[str, str, str]] = []
    for img_name in image_files:
        img_id = os.path.splitext(img_name)[0]
        if img_id not in prompt_data:
            continue
        prompt = prompt_data[img_id].get("prompt", "")
        valid_images.append((img_name, img_id, prompt))

    if image_id_filter:
        valid_images = [
            (n, iid, p) for n, iid, p in valid_images
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
    onnx_threads: int = 2,
) -> tuple:
    """Pre-load ExpertManager instances for GPU execution.

    Imports ExpertManager from c2i_harness.step3_execute (reused expert execution module).

    Priority: gpu_config_path > gpu_preset > num_groups (auto-split)

    Returns:
        (expert_managers, shared_cpu_manager, cpu_semaphore)
        - expert_managers: list of ExpertManager for GPU groups
        - shared_cpu_manager: ExpertManager with CPU-only experts (shared across groups)
        - cpu_semaphore: threading.Semaphore to limit concurrent CPU expert calls
    """
    from c2i_harness.step3_execute import (
        ExpertManager,
        DEFAULT_GPU_CONFIG,
        EXPERT_MODULE_MAP,
        CPU_EXPERT_IDS,
        _limit_onnx_threads,
    )

    _limit_onnx_threads(num_threads=onnx_threads)

    if gpu_config_path:
        with open(gpu_config_path, "r") as f:
            custom_config = json.load(f)
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
        if "groups" in preset_config:
            group_configs = preset_config["groups"]
        else:
            preset_config.pop("_description", None)
            group_configs = [preset_config]
    else:
        group_configs = build_gpu_group_configs(num_groups, DEFAULT_GPU_CONFIG)

    if required_ids is None:
        required_ids = list(EXPERT_MODULE_MAP.keys())

    has_multiple_groups = len(group_configs) > 1

    shared_cpu_manager = None
    cpu_semaphore = None

    if has_multiple_groups:
        print(f"\n{'='*60}")
        print(f"  Loading SHARED CPU expert pool (1 copy for all groups)")
        print(f"{'='*60}")
        cpu_config = {
            "open_vocabulary_detector": {"device": "cpu", "num_gpus": 0},
            "topology_boundary_auditor": {"device": "cpu", "num_gpus": 0},
            "image_text_auditor": {"device": "cpu", "num_gpus": 0},
        }
        shared_cpu_manager = ExpertManager(gpu_config=cpu_config)
        cpu_ids_to_load = [eid for eid in required_ids if eid in CPU_EXPERT_IDS]
        if cpu_ids_to_load:
            shared_cpu_manager.load_all(cpu_ids_to_load)

        import threading
        cpu_concurrency = min(len(group_configs), 4) if has_multiple_groups else 2
        cpu_semaphore = threading.Semaphore(cpu_concurrency)
        print(f"  CPU expert concurrency limit: {cpu_concurrency}")

    HEAVY_EXPERT_IDS = {"perceptual_quality_auditor"}

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  [GPU] torch.cuda.empty_cache() called")
    except Exception:
        pass

    def _create_group_manager(g: int, cfg: dict) -> ExpertManager:
        print(f"\n  --- GPU Group {g} ---")

        if has_multiple_groups:
            gpu_only_ids = [eid for eid in required_ids if eid not in CPU_EXPERT_IDS]
            filtered_cfg = {k: v for k, v in cfg.items() if k not in CPU_EXPERT_IDS}
        else:
            gpu_only_ids = required_ids
            filtered_cfg = cfg

        em = ExpertManager(gpu_config=filtered_cfg, shared_cpu_manager=shared_cpu_manager)
        return em, gpu_only_ids

    expert_managers = [None] * len(group_configs)

    if has_multiple_groups and len(group_configs) > 1:
        heavy_ids = [eid for eid in required_ids if eid in HEAVY_EXPERT_IDS and eid not in CPU_EXPERT_IDS]
        light_ids = [eid for eid in required_ids if eid not in CPU_EXPERT_IDS and eid not in HEAVY_EXPERT_IDS]

        print(f"\n{'='*60}")
        print(f"  Pre-loading {len(group_configs)} GPU group(s)")
        print(f"  Strategy: SEQUENTIAL heavy models, then PARALLEL light models")
        print(f"{'='*60}")

        for g, cfg in enumerate(group_configs):
            em, gpu_only_ids = _create_group_manager(g, cfg)
            expert_managers[g] = em

        if heavy_ids:
            print(f"\n  [Phase 1] Loading heavy models SEQUENTIALLY: {heavy_ids}")
            for g, em in enumerate(expert_managers):
                em.load_all(heavy_ids)

        if light_ids:
            print(f"\n  [Phase 2] Loading light models in PARALLEL: {light_ids}")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=len(expert_managers)) as pool:
                futures = {pool.submit(em.load_all, light_ids): g for g, em in enumerate(expert_managers)}
                for future in as_completed(futures):
                    g = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  [Group {g}] Light model loading error: {e}")
    else:
        print(f"\n{'='*60}")
        print(f"  Pre-loading {len(group_configs)} GPU group(s) in PARALLEL")
        print(f"{'='*60}")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _load_group(g: int, cfg: dict) -> ExpertManager:
            em, gpu_only_ids = _create_group_manager(g, cfg)
            em.load_all(gpu_only_ids)
            return em

        with ThreadPoolExecutor(max_workers=len(group_configs)) as pool:
            futures = {
                pool.submit(_load_group, g, cfg): g
                for g, cfg in enumerate(group_configs)
            }
            for future in as_completed(futures):
                g = futures[future]
                expert_managers[g] = future.result()

    total_loaded = sum(len(em.loaded_experts) for em in expert_managers)
    if shared_cpu_manager:
        total_loaded += len(shared_cpu_manager.loaded_experts)
    print(f"\n  Total expert instances loaded: {total_loaded}")
    print(f"{'='*60}")

    return expert_managers, shared_cpu_manager, cpu_semaphore
