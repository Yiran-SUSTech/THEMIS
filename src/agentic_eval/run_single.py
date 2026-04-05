from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .config import Settings
from .graph import build_graph
from .schemas import ImageInput



def _sanitize_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"



def _planner_dir_name(planner_model: str) -> str:
    model_name = Path(planner_model).name if planner_model.startswith("/") else planner_model
    return _sanitize_path_part(model_name)



def _label_dir_name(class_label: str | None, prompt: str | None) -> str:
    return _sanitize_path_part(class_label or prompt or "unlabeled")



def _resolve_output_paths(
    output_arg: str,
    planner_model: str,
    class_label: str | None,
    prompt: str | None,
    planner_log_dir_arg: str | None,
) -> tuple[Path, Path]:
    output_path = Path(output_arg)
    planner_dir = _planner_dir_name(planner_model)
    label_dir = _label_dir_name(class_label, prompt)
    target_dir = output_path.parent / planner_dir / label_dir
    resolved_output_path = target_dir / output_path.name
    if planner_log_dir_arg:
        log_root = Path(planner_log_dir_arg)
        log_dir = log_root / planner_dir / label_dir
    else:
        log_dir = target_dir / f"{output_path.stem}_logs"
    return resolved_output_path, log_dir



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal single-image agentic evaluation")
    parser.add_argument("image", help="Path to the image")
    parser.add_argument("--prompt", default=None, help="Text prompt for t2i")
    parser.add_argument("--class-label", default=None, help="Class label for c2i")
    parser.add_argument("--output", default="single_eval_output.json", help="Path to save output JSON")
    parser.add_argument("--planner-model", default=None, 
                        help="Override planner model. Use a local path (e.g., /path/to/Qwen2.5-VL-3B) or HuggingFace ID (e.g., Qwen/Qwen2.5-VL-3B-Instruct).")
    parser.add_argument("--judge-model", default=None,
                        help="Override judge model. Use a local path or HuggingFace ID.")
    parser.add_argument("--reflector-model", default=None,
                        help="Override reflector model. Use a local path or HuggingFace ID.")
    parser.add_argument("--planner-log-dir", default=None, help="Directory to save planner/judge/reflector rejection logs.")
    return parser.parse_args()



def _apply_model_override(settings: Settings, model_type: str, model_value: str | None) -> None:
    if not model_value:
        return
    
    model_override = model_value.strip()
    is_local = model_override.startswith("/")
    
    if model_type == "planner":
        if is_local:
            settings.planner_local_enabled = True
            settings.planner_local_model = model_override
        else:
            settings.planner_local_enabled = False
            settings.planner_model = model_override
    elif model_type == "judge":
        if is_local:
            settings.judge_local_enabled = True
            settings.judge_local_model = model_override
        else:
            settings.judge_local_enabled = False
            settings.judge_model = model_override
    elif model_type == "reflector":
        if is_local:
            settings.reflector_local_enabled = True
            settings.reflector_local_model = model_override
        else:
            settings.reflector_local_enabled = False
            settings.reflector_model = model_override



def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    
    _apply_model_override(settings, "planner", args.planner_model)
    _apply_model_override(settings, "judge", args.judge_model)
    _apply_model_override(settings, "reflector", args.reflector_model)

    output_path, log_dir = _resolve_output_paths(
        output_arg=args.output,
        planner_model=settings.planner_local_model if settings.planner_local_enabled else settings.planner_model,
        class_label=args.class_label,
        prompt=args.prompt,
        planner_log_dir_arg=args.planner_log_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    app = build_graph(settings)
    state = {
        "input": ImageInput(image_path=str(Path(args.image).resolve()), prompt=args.prompt, class_label=args.class_label),
        "plan_revision_count": 0,
        "reflection_revision_count": 0,
        "log_dir": str(log_dir.resolve()),
    }
    result = app.invoke(state)
    final_result = result["final_result"].model_dump()

    output_path.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "alignment_score": final_result["report"]["alignment_score"],
        "artifact_score": final_result["report"]["artifact_score"],
        "final_score": final_result["final_score"],
        "hard_failure": final_result["report"]["hard_failure"],
        "output": str(output_path.resolve()),
        "logs": str(log_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
