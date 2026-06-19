#!/usr/bin/env python3
"""
THEMIS C2I Unified Dispatcher Entry Point

Usage:
  python c2i/run.py --mode sync   --step 123 --limit 10
  python c2i/run.py --mode async  --step 123 --limit 100 --api-concurrency 5
  python c2i/run.py --mode batch  --step 123 --limit 1000

Modes:
  sync   - Sequential processing (original dispatcher.py behavior)
  async  - Pipeline-parallel: API calls overlap with GPU execution
  batch  - Batch API: submit all requests at once, poll for results (~50% cost)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

C2I_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = C2I_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(C2I_DIR) not in sys.path:
    sys.path.insert(0, str(C2I_DIR))

from common import (
    IMAGE_DIR, CLASS_IDS_TXT, EXPERTS_REGISTRY_JSON,
    PLAN_DIR, APPROVED_DIR, JUDGE_FEEDBACK_DIR, EXPERT_RESULTS_DIR, BATCH_DIR,
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    build_image_list, preload_expert_managers,
)
from step1_router import load_experts_registry
from step3_execute import DEFAULT_GPU_CONFIG, EXPERT_MODULE_MAP


def main():
    parser = argparse.ArgumentParser(
        description="THEMIS C2I Unified Dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serial mode, full pipeline, 10 images
  python c2i_faster/run.py --mode sync --step 123 --limit 10

  # Async mode, 5 API concurrency, GPU preset for 2-card setup
  python c2i_faster/run.py --mode async --step 1234 --limit 100 --api-concurrency 5 --gpu-preset 2x_c500

  # Async mode, 8-card preset, full pipeline
  python c2i_faster/run.py --mode async --step 1234 --limit 100 --api-concurrency 5 --gpu-preset 8x_c500

  # Batch mode, large scale, only Step 1+2
  python c2i_faster/run.py --mode batch --step 12 --limit 1000

  # Custom GPU config file
  python c2i_faster/run.py --mode async --step 123 --gpu-config my_config.json
""",
    )

    # ── Core parameters (shared across all modes) ──────────────
    parser.add_argument(
        "--mode", type=str, default="async",
        choices=["sync", "async", "batch"],
        help="Execution mode (default: async)",
    )
    parser.add_argument(
        "--step", type=str, default="123",
        choices=["1", "2", "12", "3", "4", "123", "1234"],
        help="Which steps to run (default: 123 = Router+Judge+Expert)",
    )
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="Max Judge-Router iteration rounds (default: 2)")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR))
    parser.add_argument("--class-ids", type=str, default=str(CLASS_IDS_TXT))
    parser.add_argument("--plan-dir", type=str, default=str(PLAN_DIR))
    parser.add_argument("--approved-dir", type=str, default=str(APPROVED_DIR))
    parser.add_argument("--expert-results-dir", type=str, default=str(EXPERT_RESULTS_DIR))
    parser.add_argument("--save-feedback", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--image-id", type=str, default="", help="Process single image by ID")
    parser.add_argument("--session", action="store_true", default=False,
                        help="Use conversation session mode (Router+Judge+Reflector share context)")
    parser.add_argument("--save-pose-viz", action="store_true", default=False,
                        help="Save pose visualization images")
    parser.add_argument("--ref-enable", action="store_true", default=False,
                        help="Enable 3 human-annotated reference images for Reflector anchoring (default: False)")
    parser.add_argument("--enable-checklist", action="store_true", default=False,
                        help="Enable checklist annotation output (fine_grained_details + veto_activated) matching human annotation format (default: False)")

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

    # ── Batch mode parameters ──────────────────────────────────
    parser.add_argument("--batch-dir", type=str, default=str(BATCH_DIR),
                        help="[batch] Directory for batch JSONL files")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="[batch] Seconds between status polls (default: 30)")

    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────
    image_dir = Path(args.image_dir)
    plan_dir = Path(args.plan_dir)
    approved_dir = Path(args.approved_dir)
    expert_results_dir = Path(args.expert_results_dir)
    batch_dir = Path(args.batch_dir)

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
        sys.exit(1)

    # ── Build image list ───────────────────────────────────────
    valid_images = build_image_list(
        image_dir, Path(args.class_ids),
        image_id_filter=args.image_id, limit=args.limit,
    )

    if not valid_images and run_step12:
        print("[ERROR] No valid images found.")
        sys.exit(1)

    # ── Pre-load expert managers if needed ─────────────────────
    expert_managers = []
    shared_cpu_manager = None
    cpu_semaphore = None
    if run_step3:
        expert_managers, shared_cpu_manager, cpu_semaphore = preload_expert_managers(
            num_groups=args.gpu_groups,
            gpu_config_path=args.gpu_config,
            gpu_preset=args.gpu_preset,
        )

    # ── Load experts registry ──────────────────────────────────
    experts_registry_str = load_experts_registry(str(EXPERTS_REGISTRY_JSON))

    # ── Print banner ───────────────────────────────────────────
    total_start = time.time()
    print(f"\n{'='*60}")
    print(f"  THEMIS C2I Dispatcher")
    print(f"  Mode:             {args.mode}")
    print(f"  Step:             {step}")
    print(f"  Images:           {len(valid_images)}")
    if args.mode == "async":
        print(f"  API concurrency:  {args.api_concurrency}")
    if args.session:
        print(f"  Session mode:     ON (shared conversation context)")
    if args.ref_enable:
        print(f"  Reflector refs:   ON (3 human-annotated reference images)")
    if args.enable_checklist:
        print(f"  Checklist output: ON (human-annotation-style fine_grained_details)")
    if args.mode == "batch":
        print(f"  Poll interval:    {args.poll_interval}s")
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
        final_reports_dir = C2I_DIR / "output" / "final_reports" if run_step4 else None
        checklist_dir = C2I_DIR / "output" / "checklist_annotations" if args.enable_checklist else None
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
            use_session=args.session,
            final_reports_dir=final_reports_dir,
            save_pose_viz=args.save_pose_viz,
            ref_enable=args.ref_enable,
            enable_checklist=args.enable_checklist,
            checklist_dir=checklist_dir,
            api_retry=args.api_retry,
        )

    elif args.mode == "async":
        from dispatch_async import run_async_pipeline
        final_reports_dir = C2I_DIR / "output" / "final_reports" if run_step4 else None
        checklist_dir = C2I_DIR / "output" / "checklist_annotations" if args.enable_checklist else None
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
            use_session=args.session,
            cpu_semaphore=cpu_semaphore,
            ref_enable=args.ref_enable,
            enable_checklist=args.enable_checklist,
            checklist_dir=checklist_dir,
            api_retry=args.api_retry,
        )

    elif args.mode == "batch":
        from dispatch_batch import run_batch_pipeline
        stats = run_batch_pipeline(
            valid_images=valid_images,
            image_dir=image_dir,
            experts_registry_str=experts_registry_str,
            max_iterations=args.max_iterations,
            plan_dir=plan_dir,
            approved_dir=approved_dir,
            judge_feedback_dir=judge_feedback_dir,
            expert_results_dir=expert_results_dir,
            batch_dir=batch_dir,
            expert_managers=expert_managers if expert_managers else None,
            poll_interval=args.poll_interval,
            step=step,
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
            print(f"\n  ⚠ WARNING: {len(all_failed)} expert(s) FAILED to load: {sorted(all_failed)}")
            print(f"    Results for these experts will be missing from all images!")
        else:
            print(f"\n  ✓ All expert models loaded successfully")

    print(f"{'='*60}")

    # Cleanup
    for em in expert_managers:
        em.cleanup()
    if shared_cpu_manager is not None:
        shared_cpu_manager.cleanup()


if __name__ == "__main__":
    main()
