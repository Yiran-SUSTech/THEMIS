#!/bin/bash
# =============================================================================
# THEMIS Quick Start Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}") && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "THEMIS Quick Start"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[1/4] Creating .env file..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
    exit 1
fi

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "[2/4] Virtual environment not found. Please run setup_env.sh first"
    exit 1
fi

# Activate venv
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "[3/4] Models not found. Downloading..."
    python scripts/download_models.py --model-dir ./models --qwen 3b 7b --imagenet --clip --yolo --iqa
else
    echo "[3/4] Models found. Skipping download."
fi

# Run test
echo "[4/4] Testing models..."
python scripts/test_models.py --model-dir ./models --output ./model_test_report.json

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run evaluation:"
echo "  python -m src.agentic_eval.run_single ./test_images/beacon.png --class-label 'Beacon' --output result.json"
echo ""
