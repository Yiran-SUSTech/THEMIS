import argparse
import json
import os
from pathlib import Path

import httpx
from anthropic import Anthropic
from dotenv import load_dotenv

from test_api import call_json_probe, call_text_probe, call_vision_probe


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe multiple models against an Anthropic-compatible gateway.")
    parser.add_argument("image", help="Path to a local test image")
    parser.add_argument("models", nargs="+", help="One or more model names to probe")
    parser.add_argument(
        "--base-url",
        default=os.getenv("ANTHROPIC_BASE_URL"),
        help="Anthropic-compatible base URL, e.g. https://api.example.com/anthropic",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"),
        help="API key/token",
    )
    parser.add_argument(
        "--output",
        default="model_probe_matrix.json",
        help="Where to save the detailed probe matrix JSON",
    )
    parser.add_argument(
        "--table-output",
        default=None,
        help="Optional path to save the markdown compatibility table",
    )
    parser.add_argument(
        "--ignore-env-proxy",
        action="store_true",
        default=True,
        help="Ignore HTTP(S)_PROXY / ALL_PROXY from the shell environment",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Respect proxy variables from the shell environment",
    )
    return parser.parse_args()


def run_probe(client: Anthropic, model: str, image_path: str) -> dict:
    result = {
        "model": model,
        "text_probe": None,
        "vision_probe": None,
        "json_probe": None,
    }

    text_probe_ok = False
    vision_probe_ok = False
    json_probe_ok = False

    try:
        text_probe = call_text_probe(client, model)
        result["text_probe"] = text_probe
        text_probe_ok = text_probe["raw_text"].strip() == "OK"
    except Exception as exc:  # noqa: BLE001
        result["text_probe"] = {"error": str(exc)}

    try:
        vision_probe = call_vision_probe(client, model, image_path)
        result["vision_probe"] = vision_probe
        vision_probe_ok = bool(vision_probe["raw_text"].strip())
    except Exception as exc:  # noqa: BLE001
        result["vision_probe"] = {"error": str(exc)}

    try:
        json_probe = call_json_probe(client, model, image_path)
        result["json_probe"] = json_probe
        json_probe_ok = json_probe["parsed_json"] is not None
    except Exception as exc:  # noqa: BLE001
        result["json_probe"] = {"error": str(exc)}

    result["summary"] = {
        "text_probe_ok": text_probe_ok,
        "vision_probe_ok": vision_probe_ok,
        "json_probe_ok": json_probe_ok,
        "suitable_for_v1": text_probe_ok and vision_probe_ok and json_probe_ok,
    }
    return result


def probe_status(label: str, payload: dict | None) -> str:
    if not payload:
        return "FAIL"
    if payload.get("error"):
        return f"FAIL ({payload['error']})"
    return f"OK"


def build_table(results: list[dict]) -> str:
    lines = [
        "| Model | Text | Vision | JSON | Suitable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in results:
        summary = item["summary"]
        lines.append(
            "| {model} | {text} | {vision} | {jsonp} | {suitable} |".format(
                model=item["model"],
                text="OK" if summary["text_probe_ok"] else "FAIL",
                vision="OK" if summary["vision_probe_ok"] else "FAIL",
                jsonp="OK" if summary["json_probe_ok"] else "FAIL",
                suitable="YES" if summary["suitable_for_v1"] else "NO",
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not args.base_url:
        raise SystemExit("Missing ANTHROPIC_BASE_URL or --base-url")
    if not args.api_key:
        raise SystemExit("Missing ANTHROPIC_AUTH_TOKEN/ANTHROPIC_API_KEY or --api-key")

    image_path = str(Path(args.image).resolve())
    use_env_proxy = args.use_env_proxy and not args.ignore_env_proxy
    http_client = httpx.Client(trust_env=use_env_proxy, timeout=120.0)
    client = Anthropic(api_key=args.api_key, base_url=args.base_url, http_client=http_client)

    try:
        results = [run_probe(client, model, image_path) for model in args.models]
    finally:
        http_client.close()

    payload = {
        "base_url": args.base_url,
        "image": image_path,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    table = build_table(results)
    if args.table_output:
        Path(args.table_output).write_text(table + "\n", encoding="utf-8")

    print(table)
    print(f"\nSaved detailed result to: {output_path.resolve()}")
    if args.table_output:
        print(f"Saved table to: {Path(args.table_output).resolve()}")


if __name__ == "__main__":
    main()
