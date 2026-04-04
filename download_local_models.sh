#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${ROOT_DIR}/models"
HF_HOME_DEFAULT="${MODEL_DIR}/hf-cache"
HF_ETAG_TIMEOUT_DEFAULT="60"
HF_DOWNLOAD_TIMEOUT_DEFAULT="120"
HF_MAX_WORKERS_DEFAULT="1"
HF_RETRIES_DEFAULT="6"
HF_RETRY_WAIT_DEFAULT="5"
HF_TRANSFER_DEFAULT="0"

QWEN_3B_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_3B_LOCAL_DIR="${MODEL_DIR}/Qwen2.5-VL-3B-Instruct"
QWEN_7B_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_7B_LOCAL_DIR="${MODEL_DIR}/Qwen2.5-VL-7B-Instruct"

usage() {
  cat <<'EOF'
Usage:
  bash download_local_models.sh [options]

Options:
  --all               Download all recommended manual assets (default)
  --qwen              Download both Qwen VL models
  --qwen-3b           Download Qwen2.5-VL-3B-Instruct only
  --qwen-7b           Download Qwen2.5-VL-7B-Instruct only
  --with-pyiqa-warmup Also warm up pyiqa model weights after Python deps are installed
  --model-dir PATH    Override local model directory (default: ./models)
  --hf-token TOKEN    Hugging Face token for gated/rate-limited downloads
  --help              Show this help message

Notes:
- This script uses huggingface-cli or hf if available, otherwise falls back to python -m huggingface_hub
- Downloads are forced into low-concurrency retry mode for unstable networks
- pyiqa weights are normally downloaded automatically on first use; --with-pyiqa-warmup just prefetches them
EOF
}

DOWNLOAD_QWEN_3B=false
DOWNLOAD_QWEN_7B=false
WARMUP_PYIQA=false
HF_TOKEN="${HF_TOKEN:-}"
CUSTOM_MODEL_DIR=""

if [ "$#" -eq 0 ]; then
  DOWNLOAD_QWEN_3B=true
  DOWNLOAD_QWEN_7B=true
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      DOWNLOAD_QWEN_3B=true
      DOWNLOAD_QWEN_7B=true
      ;;
    --qwen)
      DOWNLOAD_QWEN_3B=true
      DOWNLOAD_QWEN_7B=true
      ;;
    --qwen-3b)
      DOWNLOAD_QWEN_3B=true
      ;;
    --qwen-7b)
      DOWNLOAD_QWEN_7B=true
      ;;
    --with-pyiqa-warmup)
      WARMUP_PYIQA=true
      ;;
    --model-dir)
      shift
      CUSTOM_MODEL_DIR="${1:-}"
      ;;
    --hf-token)
      shift
      HF_TOKEN="${1:-}"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [ -n "$CUSTOM_MODEL_DIR" ]; then
  MODEL_DIR="$CUSTOM_MODEL_DIR"
  HF_HOME_DEFAULT="${MODEL_DIR}/hf-cache"
  QWEN_3B_LOCAL_DIR="${MODEL_DIR}/Qwen2.5-VL-3B-Instruct"
  QWEN_7B_LOCAL_DIR="${MODEL_DIR}/Qwen2.5-VL-7B-Instruct"
fi

mkdir -p "$MODEL_DIR"
mkdir -p "$HF_HOME_DEFAULT"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-$HF_ETAG_TIMEOUT_DEFAULT}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-$HF_DOWNLOAD_TIMEOUT_DEFAULT}"
export HF_MAX_WORKERS="${HF_MAX_WORKERS:-$HF_MAX_WORKERS_DEFAULT}"
export HF_DOWNLOAD_RETRIES="${HF_DOWNLOAD_RETRIES:-$HF_RETRIES_DEFAULT}"
export HF_RETRY_WAIT_SECONDS="${HF_RETRY_WAIT_SECONDS:-$HF_RETRY_WAIT_DEFAULT}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-$HF_TRANSFER_DEFAULT}"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_python_module() {
  local module_name="$1"
  python - <<PY
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("${module_name}") else 1)
PY
}

run_with_retries() {
  local attempt=1
  local max_attempts="$HF_DOWNLOAD_RETRIES"

  while [ "$attempt" -le "$max_attempts" ]; do
    echo "Download attempt ${attempt}/${max_attempts}"
    if "$@"; then
      return 0
    fi

    if [ "$attempt" -eq "$max_attempts" ]; then
      break
    fi

    echo "Attempt ${attempt} failed; waiting ${HF_RETRY_WAIT_SECONDS}s before retry..." >&2
    sleep "$HF_RETRY_WAIT_SECONDS"
    attempt=$((attempt + 1))
  done

  return 1
}

hf_download_repo() {
  local repo_id="$1"
  local local_dir="$2"

  if [ -n "$HF_TOKEN" ]; then
    HF_TOKEN="$HF_TOKEN" hf download "$repo_id" --local-dir "$local_dir" --token "$HF_TOKEN" --max-workers "$HF_MAX_WORKERS"
  else
    hf download "$repo_id" --local-dir "$local_dir" --max-workers "$HF_MAX_WORKERS"
  fi
}

huggingface_cli_download_repo() {
  local repo_id="$1"
  local local_dir="$2"

  if [ -n "$HF_TOKEN" ]; then
    HF_TOKEN="$HF_TOKEN" huggingface-cli download "$repo_id" --local-dir "$local_dir" --token "$HF_TOKEN"
  else
    huggingface-cli download "$repo_id" --local-dir "$local_dir"
  fi
}

python_module_download_repo() {
  local repo_id="$1"
  local local_dir="$2"

  if [ -n "$HF_TOKEN" ]; then
    python -m huggingface_hub download "$repo_id" --local-dir "$local_dir" --token "$HF_TOKEN"
  else
    python -m huggingface_hub download "$repo_id" --local-dir "$local_dir"
  fi
}

download_hf_repo() {
  local repo_id="$1"
  local local_dir="$2"

  echo "Downloading ${repo_id} -> ${local_dir}"
  echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT} HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT} HF_MAX_WORKERS=${HF_MAX_WORKERS} HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
  mkdir -p "$local_dir"

  if have_cmd hf; then
    run_with_retries hf_download_repo "$repo_id" "$local_dir" && return 0
    echo "hf download failed after retries." >&2
    echo "Tip: rerun the same command to resume, or try HF_ENDPOINT=https://hf-mirror.com" >&2
    return 1
  fi

  if have_cmd huggingface-cli; then
    run_with_retries huggingface_cli_download_repo "$repo_id" "$local_dir" && return 0
    echo "huggingface-cli download failed after retries." >&2
    echo "Tip: rerun the same command to resume, or try HF_ENDPOINT=https://hf-mirror.com" >&2
    return 1
  fi

  if ensure_python_module huggingface_hub; then
    run_with_retries python_module_download_repo "$repo_id" "$local_dir" && return 0
    echo "python -m huggingface_hub download failed after retries." >&2
    echo "Tip: rerun the same command to resume, or try HF_ENDPOINT=https://hf-mirror.com" >&2
    return 1
  fi

  echo "Missing downloader: install 'huggingface_hub[cli]' first." >&2
  echo "Example: pip install -U \"huggingface_hub[cli]\"" >&2
  exit 1
}

warmup_pyiqa() {
  if ! ensure_python_module pyiqa; then
    echo "Skipping pyiqa warmup because pyiqa is not installed in: $(python -c 'import sys; print(sys.executable)')" >&2
    echo "Install deps into this interpreter: $(python -c 'import sys; print(sys.executable)') -m pip install -r requirements-agentic-eval.txt" >&2
    return 0
  fi

  echo "Warming up pyiqa weights for maniqa, musiq, and clipiqa"
  python - <<'PY'
import pyiqa

for name in ("maniqa", "musiq", "clipiqa"):
    print(f"Creating metric: {name}")
    metric = pyiqa.create_metric(name, device="cpu")
    print(f"Loaded metric: {name}: {metric.__class__.__name__}")
PY
}

if $DOWNLOAD_QWEN_3B; then
  download_hf_repo "$QWEN_3B_MODEL_ID" "$QWEN_3B_LOCAL_DIR"
fi

if $DOWNLOAD_QWEN_7B; then
  download_hf_repo "$QWEN_7B_MODEL_ID" "$QWEN_7B_LOCAL_DIR"
fi

if $WARMUP_PYIQA; then
  warmup_pyiqa
fi

echo
echo "Done."
echo "Recommended env vars:"
echo "  SEMANTIC_LOCAL_FAST_MODEL=${QWEN_3B_LOCAL_DIR}"
echo "  SEMANTIC_LOCAL_STRONGER_MODEL=${QWEN_7B_LOCAL_DIR}"
echo "  LOCAL_SEMANTIC_MODEL=${QWEN_3B_LOCAL_DIR}"
echo "  LOCAL_ARTIFACT_METRICS=maniqa,musiq,clipiqa"
