#!/usr/bin/env python3
"""
THEMIS T2I Unified Dispatcher Entry Point

Usage:
  python t2i_harness/run.py --mode sync   --step 123 --limit 10
  python t2i_harness/run.py --mode async  --step 1234 --limit 100 --api-concurrency 5

Modes:
  sync   - Sequential processing (best for debugging / single-image testing)
  async  - Pipeline-parallel: API calls overlap with GPU execution (default)
"""

import os
import sys
import time
import argparse
from pathlib import Path

T2I_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = T2I_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(T2I_DIR) not in sys.path:
    sys.path.insert(0, str(T2I_DIR))


# Pre-scan --output-dir from argv so common.py picks it up at import time
# (common.py derives all output subdirs from T2I_OUTPUT_DIR_NAME env var).
def _early_parse_output_dir():
    for idx, tok in enumerate(sys.argv):
        if tok == "--output-dir" and idx + 1 < len(sys.argv):
            os.environ["T2I_OUTPUT_DIR_NAME"] = sys.argv[idx + 1]
            return
        if tok.startswith("--output-dir="):
            os.environ["T2I_OUTPUT_DIR_NAME"] = tok.split("=", 1)[1]
            return

_early_parse_output_dir()

from common import (
    IMAGE_DIR, GENEVAL2_DATA_JSONL, EXPERTS_REGISTRY_JSON,
    PLAN_DIR, APPROVED_DIR, JUDGE_FEEDBACK_DIR, EXPERT_RESULTS_DIR,
    OUTPUT_DIR, WITHOUT_EXPERT_REPORTS_DIR,
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    build_t2i_image_list, preload_expert_managers,
)
from step1_router import load_experts_registry


def main():
    parser = argparse.ArgumentParser(
        description="THEMIS T2I Unified Dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serial mode, full pipeline, 10 images
  python t2i_harness/run.py --mode sync --step 123 --limit 10

  # Async mode, 5 API concurrency, full pipeline with Reflector
  python t2i_harness/run.py --mode async --step 1234 --limit 100 --api-concurrency 5

  # Test single image
  python t2i_harness/run.py --mode sync --step 1234 --image-id 0

  # Only run Reflector (load existing expert results from disk)
  python t2i_harness/run.py --mode async --step 4 --limit 100
""",
    )

    # ── Core parameters ────────────────────────────────────────
    parser.add_argument(
        "--mode", type=str, default="async",
        choices=["sync", "async"],
        help="Execution mode (default: async)",
    )
    parser.add_argument(
        "--step", type=str, default="123",
        choices=["1", "2", "12", "3", "4", "123", "1234"],
        help="Which steps to run (default: 123 = Atomize+Router+Judge+Expert). "
             "Step 0 (Atomize) runs automatically as part of Step 1+2.",
    )
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="Max Judge-Router iteration rounds (default: 2)")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR),
                        help=f"Directory containing generated test images (default: {IMAGE_DIR})")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory name (relative to t2i_harness/) or absolute path "
                             "(default: output). All plans/approved_plans/expert_results/"
                             "final_reports/atomized etc. are written under this directory.")
    parser.add_argument("--geneval2-jsonl", type=str, default=str(GENEVAL2_DATA_JSONL),
                        help=f"Path to geneval2_data.jsonl (default: {GENEVAL2_DATA_JSONL})")
    parser.add_argument("--plan-dir", type=str, default=str(PLAN_DIR))
    parser.add_argument("--approved-dir", type=str, default=str(APPROVED_DIR))
    parser.add_argument("--expert-results-dir", type=str, default=str(EXPERT_RESULTS_DIR))
    parser.add_argument("--save-feedback", action="store_true", default=False,
                        help="Save Judge feedback details")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images to process (0 = all)")
    parser.add_argument("--image-id", type=str, default="",
                        help="Process single image by prompt_id (matches image filename stem)")

    # ── GPU parameters ─────────────────────────────────────────
    parser.add_argument("--gpu-groups", type=int, default=1,
                        help="Number of parallel GPU groups (default: 1)")
    parser.add_argument("--gpu-config", type=str, default=None,
                        help="Path to custom GPU allocation JSON")
    parser.add_argument("--gpu-preset", type=str, default=None,
                        help="GPU preset name from gpu_configs/ (e.g. 2x_c500, 8x_c500)")

    # ── Async mode parameters ──────────────────────────────────
    parser.add_argument("--api-concurrency", type=int, default=5,
                        help="[async] Max concurrent API calls (default: 5)")
    parser.add_argument("--api-retry", type=int, default=0,
                        help="Number of API call retries on failure (default: 0, no retry)")
    parser.add_argument("--temp-router", type=float, default=0.0,
                        help="Temperature for Router API calls (default: 0.0)")
    parser.add_argument("--temp-judge", type=float, default=0.0,
                        help="Temperature for Judge API calls (default: 0.0)")
    parser.add_argument("--temp-reflector", type=float, default=0.5,
                        help="Temperature for Reflector API calls (default: 0.5)")
    parser.add_argument("--model-router", type=str, default="",
                        help="LLM model name for Router (default: qwen3.6-plus)")
    parser.add_argument("--model-judge", type=str, default="",
                        help="LLM model name for Judge (default: qwen3.6-plus)")
    parser.add_argument("--model-reflector", type=str, default="",
                        help="LLM model name for Reflector (default: qwen3.7-plus)")

    # ── Reflector reference image parameters ─────────────────────
    parser.add_argument("--ref-image-dir", type=str, default="",
                        help="Directory containing human-scored reference images "
                             "for Reflector calibration (default: not used)")
    parser.add_argument("--ref-annotations", type=str,
                        default="t2i_harness/t2i_ref_annotations.json",
                        help="Path to human reference annotations JSON")
    parser.add_argument("--enable-self-reflection", action="store_true", default=True,
                        help="Enable Reflector self-reflection (two-round API calls). "
                             "Default: enabled. Use --no-self-reflection to disable.")
    parser.add_argument("--no-self-reflection", action="store_false", dest="enable_self_reflection",
                        help="Disable Reflector self-reflection (single-round mode for faster processing)")

    args = parser.parse_args()

    # ── Override LLM model names if specified ───────────────────
    if args.model_router or args.model_judge or args.model_reflector:
        import step0_atomize
        import step1_router
        import step2_judge
        import step4_reflector
        if args.model_router:
            step0_atomize.ATOMIZE_MODEL = args.model_router
            step1_router.ROUTER_MODEL = args.model_router
            print(f"  [CONFIG] Router/Atomize model overridden to: {args.model_router}")
        if args.model_judge:
            step2_judge.JUDGE_MODEL = args.model_judge
            print(f"  [CONFIG] Judge model overridden to: {args.model_judge}")
        if args.model_reflector:
            step4_reflector.REFLECTOR_MODEL = args.model_reflector
            print(f"  [CONFIG] Reflector model overridden to: {args.model_reflector}")

    # ── Resolve paths ──────────────────────────────────────────
    image_dir = Path(args.image_dir)
    plan_dir = Path(args.plan_dir)
    approved_dir = Path(args.approved_dir)
    expert_results_dir = Path(args.expert_results_dir)

    plan_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)
    expert_results_dir.mkdir(parents=True, exist_ok=True)

    judge_feedback_dir = None
    if args.save_feedback:
        judge_feedback_dir = JUDGE_FEEDBACK_DIR
        judge_feedback_dir.mkdir(parents=True, exist_ok=True)

    step = args.step
    run_step12 = step in ("1", "2", "12", "123", "1234")
    run_step3 = step in ("3", "123", "1234")
    run_step4 = step in ("4", "1234")

    # ── Validate API key ───────────────────────────────────────
    if (run_step12 or run_step4) and not DASHSCOPE_API_KEY:
        print("[ERROR] DASHSCOPE_API_KEY not set. Required for Step 1+2 and Step 4.")
        print("  Export it: export DASHSCOPE_API_KEY='your-key-here'")
        sys.exit(1)

    # ── Build image list ───────────────────────────────────────
    valid_images = build_t2i_image_list(
        image_dir,
        geneval2_jsonl_path=args.geneval2_jsonl,
        image_id_filter=args.image_id,
        limit=args.limit,
    )

    if not valid_images and (run_step12 or run_step4):
        print(f"[ERROR] No valid images found in {image_dir}")
        print(f"  Ensure images are named by prompt_id (e.g., 0.png, 1.png, ...)")
        print(f"  and geneval2_data.jsonl exists at: {args.geneval2_jsonl}")
        sys.exit(1)

    # ── Pre-load expert managers if needed ─────────────────────
    expert_managers = []
    shared_cpu_manager = None
    cpu_semaphore = None
    if run_step3:
        from c2i_harness.step3_execute import DEFAULT_GPU_CONFIG, EXPERT_MODULE_MAP

        # Collect required expert IDs from existing approved plans (if any)
        required_ids = list(EXPERT_MODULE_MAP.keys())
        try:
            from c2i_harness.step3_execute import load_approved_plans, collect_required_expert_ids
            plans = load_approved_plans(approved_dir)
            if plans:
                required_ids = collect_required_expert_ids(plans)
                print(f"  [INFO] Collecting required experts from {len(plans)} approved plans: {required_ids}")
        except Exception:
            pass

        expert_managers, shared_cpu_manager, cpu_semaphore = preload_expert_managers(
            num_groups=args.gpu_groups,
            gpu_config_path=args.gpu_config,
            required_ids=required_ids,
            gpu_preset=args.gpu_preset,
        )

    # ── Load experts registry ──────────────────────────────────
    experts_registry_str = load_experts_registry(str(EXPERTS_REGISTRY_JSON))

    # ── Print banner ───────────────────────────────────────────
    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"  THEMIS T2I Dispatcher")
    print(f"  Mode:             {args.mode}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"  Step:             {step}")
    print(f"  Images:           {len(valid_images)}")
    print(f"  GenEval2 JSONL:   {args.geneval2_jsonl}")
    if args.mode == "async":
        print(f"  API concurrency:  {args.api_concurrency}")
    if run_step3:
        if args.gpu_preset:
            print(f"  GPU preset:       {args.gpu_preset}")
        elif args.gpu_config:
            print(f"  GPU config:       {args.gpu_config}")
        else:
            print(f"  GPU groups:       {args.gpu_groups}")
    print(f"{'='*60}\n")

    # ── Dispatch to mode ───────────────────────────────────────
    stats = {}

    if args.mode == "sync":
        from dispatch_sync import run_sync_pipeline
        final_reports_dir = OUTPUT_DIR / "final_reports" if run_step4 else None
        stats = run_sync_pipeline(
            valid_images=valid_images,
            image_dir=image_dir,
            experts_registry_str=experts_registry_str,
            max_iterations=args.max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            step=step,
            final_reports_dir=final_reports_dir,
            ref_image_dir=Path(args.ref_image_dir) if args.ref_image_dir else None,
            enable_self_reflection=args.enable_self_reflection,
            api_retry=args.api_retry,
            temp_router=args.temp_router,
            temp_judge=args.temp_judge,
            temp_reflector=args.temp_reflector,
        )

    elif args.mode == "async":
        from dispatch_async import run_async_pipeline
        final_reports_dir = OUTPUT_DIR / "final_reports" if run_step4 else None
        stats = run_async_pipeline(
            valid_images=valid_images,
            image_dir=image_dir,
            experts_registry_str=experts_registry_str,
            max_iterations=args.max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            expert_results_dir=expert_results_dir,
            expert_managers=expert_managers,
            api_concurrency=args.api_concurrency,
            step=step,
            final_reports_dir=final_reports_dir,
            ref_image_dir=Path(args.ref_image_dir) if args.ref_image_dir else None,
            enable_self_reflection=args.enable_self_reflection,
            temp_router=args.temp_router,
            temp_judge=args.temp_judge,
            temp_reflector=args.temp_reflector,
            cpu_semaphore=cpu_semaphore,
            api_retry=args.api_retry,
        )

    # ── Summary ────────────────────────────────────────────────
    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  Pipeline Summary ({args.mode} mode)")
    print(f"{'='*60}")
    print(f"  Total images:     {len(valid_images)}")
    for k, v in stats.items():
        print(f"  {k:20s}  {v}")
    print(f"  Total elapsed:    {total_elapsed:.2f}s")
    if len(valid_images) > 0 and total_elapsed > 0:
        throughput = len(valid_images) / total_elapsed
        print(f"  Throughput:       {throughput:.2f} img/s ({1/throughput:.1f} s/img)")

    if run_step3 and expert_managers:
        all_failed = set()
        for g, em in enumerate(expert_managers):
            if em.load_errors:
                all_failed.update(em.load_errors.keys())
        if shared_cpu_manager and shared_cpu_manager.load_errors:
            all_failed.update(shared_cpu_manager.load_errors.keys())
        if all_failed:
            print(f"\n  WARNING: {len(all_failed)} expert(s) FAILED to load: {sorted(all_failed)}")
            print(f"    Results for these experts will be missing from all images!")
        else:
            print(f"\n  All expert models loaded successfully")

    print(f"{'='*60}")

    # Cleanup
    for em in expert_managers:
        em.cleanup()
    if shared_cpu_manager is not None:
        shared_cpu_manager.cleanup()


if __name__ == "__main__":
    main()
