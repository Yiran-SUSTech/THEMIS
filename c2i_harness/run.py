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

# Pre-scan --output-dir from argv so common.py picks it up at import time
# (common.py derives all output subdirs from C2I_OUTPUT_DIR_NAME env var).
def _early_parse_output_dir():
    for idx, tok in enumerate(sys.argv):
        if tok == "--output-dir" and idx + 1 < len(sys.argv):
            os.environ["C2I_OUTPUT_DIR_NAME"] = sys.argv[idx + 1]
            return
        if tok.startswith("--output-dir="):
            os.environ["C2I_OUTPUT_DIR_NAME"] = tok.split("=", 1)[1]
            return

_early_parse_output_dir()

from common import (
    IMAGE_DIR, CLASS_IDS_TXT, EXPERTS_REGISTRY_JSON,
    PLAN_DIR, APPROVED_DIR, JUDGE_FEEDBACK_DIR, EXPERT_RESULTS_DIR, BATCH_DIR,
    WITHOUT_EXPERT_REPORTS_DIR, OUTPUT_DIR,
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
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出目录名(相对 c2i_faster/)或绝对路径 (默认: output)。"
                             "所有 plans/approved_plans/expert_results/final_reports/"
                             "checklist_annotations/sam_masks/depth_maps 等均写入此目录下。")
    parser.add_argument("--class-ids", type=str, default=None,
                        help="Path to class_ids.txt. If None, defaults to <image-dir>/class_ids.txt")
    parser.add_argument("--plan-dir", type=str, default=str(PLAN_DIR))
    parser.add_argument("--approved-dir", type=str, default=str(APPROVED_DIR))
    parser.add_argument("--expert-results-dir", type=str, default=str(EXPERT_RESULTS_DIR))
    parser.add_argument("--save-feedback", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--image-id", type=str, default="", help="Process single image by ID")
    parser.add_argument("--save-pose-viz", action="store_true", default=False,
                        help="Save pose visualization images")
    parser.add_argument("--ref-enable", action="store_true", default=False,
                        help="Enable 3 human-annotated reference images for Reflector anchoring (default: False)")
    parser.add_argument("--enable-checklist", action="store_true", default=False,
                        help="Enable checklist annotation output (fine_grained_details + veto_activated) matching human annotation format (default: False)")
    parser.add_argument("--without-expert", action="store_true", default=False,
                        help="Ablation mode: router-only direct scoring, no experts/judge/reflector (default: False)")
    parser.add_argument("--pose-hard-cap", action="store_true", default=False,
                        help="Enable hard caps on artifact_score based on pose low-confidence analysis (disabled by default due to domain-shift concerns)")

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

    # ── Batch mode parameters ──────────────────────────────────
    parser.add_argument("--batch-dir", type=str, default=str(BATCH_DIR),
                        help="[batch] Directory for batch JSONL files")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="[batch] Seconds between status polls (default: 30)")

    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────
    image_dir = Path(args.image_dir)
    # If --class-ids not specified, fall back to <image-dir>/class_ids.txt
    if args.class_ids is None:
        args.class_ids = str(image_dir / "class_ids.txt")
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
    # In without-expert mode, only the router API is used (no judge/reflector)
    if args.without_expert:
        if not DASHSCOPE_API_KEY:
            print("[ERROR] DASHSCOPE_API_KEY not set. Required for Router direct scoring.")
            sys.exit(1)
    elif (run_step12 or run_step4) and not DASHSCOPE_API_KEY:
        print("[ERROR] DASHSCOPE_API_KEY not set. Required for Step 1+2 and Step 4.")
        sys.exit(1)

    # ── Build image list ───────────────────────────────────────
    valid_images = build_image_list(
        image_dir, Path(args.class_ids),
        image_id_filter=args.image_id, limit=args.limit,
    )

    if not valid_images and (run_step12 or args.without_expert):
        print("[ERROR] No valid images found.")
        sys.exit(1)

    # ── Pre-load expert managers if needed ─────────────────────
    # Skip expert loading entirely in without-expert mode
    expert_managers = []
    shared_cpu_manager = None
    cpu_semaphore = None
    if run_step3 and not args.without_expert:
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
    print(f"  Output dir:       {OUTPUT_DIR}")
    if args.without_expert:
        print(f"  Step:             without-expert (router-only direct scoring)")
    else:
        print(f"  Step:             {step}")
    print(f"  Images:           {len(valid_images)}")
    if args.mode == "async":
        print(f"  API concurrency:  {args.api_concurrency}")
    if args.ref_enable and not args.without_expert:
        print(f"  Reflector refs:   ON (3 human-annotated reference images)")
    if args.enable_checklist and not args.without_expert:
        print(f"  Checklist output: ON (human-annotation-style fine_grained_details)")
    if args.mode == "batch":
        print(f"  Poll interval:    {args.poll_interval}s")
    if run_step3 and not args.without_expert:
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
        checklist_dir = OUTPUT_DIR / "checklist_annotations" if args.enable_checklist else None
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
            save_pose_viz=args.save_pose_viz,
            ref_enable=args.ref_enable,
            enable_checklist=args.enable_checklist,
            checklist_dir=checklist_dir,
            api_retry=args.api_retry,
            temp_router=args.temp_router,
            temp_judge=args.temp_judge,
            temp_reflector=args.temp_reflector,
            without_expert=args.without_expert,
            without_expert_dir=WITHOUT_EXPERT_REPORTS_DIR,
            pose_hard_cap=args.pose_hard_cap,
        )

    elif args.mode == "async":
        from dispatch_async import run_async_pipeline
        final_reports_dir = OUTPUT_DIR / "final_reports" if run_step4 else None
        checklist_dir = OUTPUT_DIR / "checklist_annotations" if args.enable_checklist else None
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
            temp_router=args.temp_router,
            temp_judge=args.temp_judge,
            temp_reflector=args.temp_reflector,
            cpu_semaphore=cpu_semaphore,
            ref_enable=args.ref_enable,
            enable_checklist=args.enable_checklist,
            checklist_dir=checklist_dir,
            api_retry=args.api_retry,
            without_expert=args.without_expert,
            without_expert_dir=WITHOUT_EXPERT_REPORTS_DIR,
            pose_hard_cap=args.pose_hard_cap,
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
            pose_hard_cap=args.pose_hard_cap,
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
