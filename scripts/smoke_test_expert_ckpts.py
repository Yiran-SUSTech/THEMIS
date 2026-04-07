#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval.config import Settings
from agentic_eval.expert_models import run_expert_evaluation


DEFAULT_EXPERTS = [
    "imagenet_fast",
    "imagenet_strong",
    "imagenet_eva02_large",
    "imagenet_eva_giant_224",
    "bge_candidate_generator",
    "e5_candidate_generator",
    "clip",
    "animal_pose",
    "body_pose",
    "body_pose_strong",
    "hand_detection",
    "face_detection",
    "places365",
    "places365_strong",
    "background_removal",
    "iqa_fast",
    "iqa_default",
    "iqa_richer",
    "boundary_artifact",
    "vqa",
    "ocr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test local expert checkpoints through THEMIS loaders")
    parser.add_argument("--config", default=str(ROOT / "configs" / "expert_config.yaml"), help="Path to expert_config.yaml")
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", str(ROOT / "models")), help="Model directory")
    parser.add_argument("--image", default=None, help="Optional test image path")
    parser.add_argument("--experts", default="all", help="Comma-separated expert keys, or 'all'")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "expert_ckpt_smoke_test.json"), help="JSON report output path")
    return parser.parse_args()


def build_test_image() -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "themis_smoke_test"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_path = tmp_dir / "smoke_test_input.png"
    if image_path.exists():
        return str(image_path)

    image = Image.new("RGB", (448, 448), color=(235, 240, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((110, 80, 340, 320), fill=(130, 90, 60), outline=(40, 40, 40), width=4)
    draw.ellipse((155, 130, 205, 185), fill=(25, 25, 25))
    draw.ellipse((245, 130, 295, 185), fill=(25, 25, 25))
    draw.polygon([(220, 190), (200, 240), (240, 240)], fill=(210, 180, 120), outline=(50, 50, 50))
    draw.rectangle((170, 320, 190, 420), fill=(90, 70, 55))
    draw.rectangle((258, 320, 278, 420), fill=(90, 70, 55))
    draw.text((120, 20), "THEMIS smoke test image", fill=(20, 20, 20))
    image.save(image_path)
    return str(image_path)


def summarize_result(result: Any) -> dict[str, Any]:
    extra_info = getattr(result, "extra_info", None)
    evidence = getattr(result, "evidence", None)
    return {
        "expert": getattr(result, "expert", None),
        "summary": getattr(result, "summary", None),
        "findings_count": len(getattr(result, "findings", []) or []),
        "severity": getattr(result, "severity", None),
        "confidence": getattr(result, "confidence", None),
        "model": getattr(result, "model", None),
        "extra_info_keys": sorted(list(extra_info.keys())) if isinstance(extra_info, dict) else [],
        "evidence_keys": sorted(list(evidence.keys())) if isinstance(evidence, dict) else [],
    }


def build_kwargs(expert_name: str, image_path: str) -> dict[str, Any]:
    common = {
        "class_label": "patas monkey",
        "prompt": "a realistic patas monkey standing outdoors",
    }
    if expert_name in {"bge_candidate_generator", "e5_candidate_generator"}:
        common["candidate_pool"] = ["monkey", "baboon", "macaque", "patas monkey", "dog"]
        common["top_k"] = 8
    if expert_name in {"vqa", "ocr"}:
        common["question"] = "Return the visible subject, its notable features, and any visible uncertainty as structured evidence."
    return common


def resolve_experts(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return list(DEFAULT_EXPERTS)
    return [item.strip() for item in arg.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    os.environ["MODEL_DIR"] = args.model_dir

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    settings = Settings.from_env()
    yaml_settings = Settings.from_yaml(str(config_path))
    settings.expert_configs = yaml_settings.expert_configs
    settings.model_dir = args.model_dir

    image_path = args.image or build_test_image()
    experts = resolve_experts(args.experts)

    report: dict[str, Any] = {
        "config": str(config_path.resolve()),
        "model_dir": args.model_dir,
        "image": image_path,
        "experts": {},
    }

    passed = 0
    failed = 0
    for expert_name in experts:
        print(f"[TEST] {expert_name}", flush=True)
        try:
            result = run_expert_evaluation(
                expert_name,
                image_path,
                settings,
                **build_kwargs(expert_name, image_path),
            )
            report["experts"][expert_name] = {
                "success": True,
                "result": summarize_result(result),
            }
            passed += 1
            print(f"[PASS] {expert_name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            report["experts"][expert_name] = {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failed += 1
            print(f"[FAIL] {expert_name}: {exc}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(json.dumps({
        "passed": passed,
        "failed": failed,
        "output": str(output_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
