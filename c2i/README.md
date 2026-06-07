# THEMIS C2I Evaluation System — Getting Started

## Overview

THEMIS C2I is an agentic image quality evaluation system that produces two scores for AI-generated images:
- **Alignment Score (0-5)** — Whether the image matches the target class
- **Artifact Score (0-5)** — Visual quality (artifacts, melting, structural collapse, etc.)

The evaluation pipeline consists of 4 steps:

```
Step 1 Router  →  Step 2 Judge  →  Step 3 Expert  →  Step 4 Reflector
 (VLM planning)   (VLM review)    (Local GPU inference)  (VLM final scoring)
```

## Environment Setup

### 1. Environment Variables (Required)

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
export MACA_PATH=/opt/maca   # Required on MetaX GPU servers
```

Add these to `~/.bashrc` for persistence.

### 2. Activate Python Environment

```bash
source /mnt/afs/zhengmingkai/miniconda3/envs/themis/bin/activate
# or
conda activate themis
```

### 3. Navigate to Project Directory

```bash
cd /mnt/afs/zhengmingkai/hhy/themis/THEMIS
```

## Unified Entry Point

All evaluation tasks are launched via `c2i/run.py`:

```bash
python c2i/run.py --mode <mode> --step <steps> [options...]
```

## Execution Modes

| Mode | Use Case | Characteristics |
|------|----------|-----------------|
| `sync` | Debugging / single-image testing | Sequential execution, clearest logs |
| `async` | Daily batch evaluation | Concurrent API + GPU pipeline overlap, **default mode** |
| `batch` | Large-scale evaluation (1000+ images) | Batch API submission, ~50% cost discount, minutes-to-hours latency |

## Quick Start

### Test Single Image (Verify Environment)

```bash
python c2i/run.py --mode sync --step 123 --image-id 000000
```

### Run 10 Images (Async Mode, 5 Concurrent API Calls)

```bash
python c2i/run.py --mode async --step 123 --limit 10 --api-concurrency 5
```

### Full Pipeline Including Reflector (Steps 1-4)

```bash
python c2i/run.py --mode sync --step 1234 --limit 5 --session
```

### GPU Inference Only (With Existing Approved Plans)

```bash
python c2i/run.py --mode async --step 3 --gpu-groups 1
```

### Large-Scale Batch Mode

```bash
python c2i/run.py --mode batch --step 12 --limit 1000 --poll-interval 30
# After batch completes, run GPU layer
python c2i/run.py --mode async --step 3 --gpu-groups 2
```

## Full Parameter Reference

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `async` | Execution mode: `sync` / `async` / `batch` |
| `--step` | `123` | Steps to run: `1` / `2` / `12` / `3` / `4` / `123` / `1234` |
| `--limit` | `0` (all) | Max number of images to process |
| `--image-id` | - | Process a single image by ID |
| `--max-iterations` | `2` | Max Router-Judge iteration rounds |
| `--image-dir` | `test_images/` | Input image directory |
| `--class-ids` | `test_images/class_ids.txt` | Image-to-class mapping file |
| `--save-feedback` | `false` | Save Judge feedback details |
| `--session` | `false` | Use conversation session for Reflector (Step 4) |
| `--save-pose-viz` | `false` | Save pose visualization images |

### GPU Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu-groups` | `1` | Number of parallel GPU groups (2 = two images processed simultaneously) |
| `--gpu-config` | - | Path to custom GPU allocation JSON file |

### Async Mode Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--api-concurrency` | `5` | Max concurrent API calls |

### Batch Mode Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-dir` | `c2i/output/batch/` | Directory for batch JSONL files |
| `--poll-interval` | `30` | Seconds between batch status polls |

## Output Directory Structure

```
c2i/output/
├── plans/              # Step 1 Router initial plans
├── approved_plans/     # Step 2 Judge approved plans
├── judge_feedback/     # Judge feedback details (requires --save-feedback)
├── expert_results/     # Step 3 expert inference results
├── final_reports/      # Step 4 Reflector final scoring reports
├── batch/              # Batch mode JSONL files
├── depth_maps/         # Depth map outputs
├── sam_masks/          # Segmentation mask outputs
└── pose_visualizations/ # Pose visualizations (requires --save-pose-viz)
```

## Input Data Format

### test_images/ Directory

```
test_images/
├── 000000.png
├── 000001.png
├── ...
└── class_ids.txt
```

### class_ids.txt Format

```
000000 0
000001 0
000002 1
...
```

Each line: `<image_id> <imagenet_class_id>`

## Performance Reference

| Configuration | Per-Image Latency | Throughput |
|---------------|-------------------|------------|
| sync, full pipeline | ~90s | 1 img/90s |
| async, 5 concurrency, 1 GPU group | ~43s/img | 10 img/7min |
| async, 10 concurrency, 2 GPU groups | ~20s/img (estimated) | - |
| batch, Step 1+2 | minutes (async) | Not QPS-limited |

## Code Structure

```
c2i/
├── run.py               # Unified entry point
├── common.py            # Shared utilities and constants
├── dispatch_sync.py     # Sequential mode
├── dispatch_async.py    # Async pipeline mode
├── dispatch_batch.py    # Batch API mode
├── step1_router.py      # Router Agent (VLM planning)
├── step2_judge.py       # Judge Agent (VLM review)
├── step3_execute.py     # Expert Manager (local GPU inference)
├── step4_reflector.py   # Reflector Agent (VLM final scoring)
└── conversation_session.py  # Session management
```

## FAQ

### Q: `TypeError: expected str, bytes or os.PathLike object, not NoneType`

Set `export MACA_PATH=/opt/maca`. This environment variable is required by the MetaX triton backend.

### Q: DINO takes a long time to load (100s+)

The first load downloads the bert-base-uncased tokenizer (~440MB). Subsequent runs use the cache.

### Q: How to re-run only the Reflector?

```bash
python c2i/run.py --mode sync --step 4 --session
```

### Q: How to check per-expert execution time for a single image?

```bash
cat c2i/output/expert_results/expert_results_000000.json | python -m json.tool | grep execution_time_ms
```
