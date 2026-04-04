import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic
import httpx


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an Anthropic-compatible multimodal API.")
    parser.add_argument("image", help="Path to a local test image")
    parser.add_argument(
        "--model",
        default=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        help="Model name exposed by the provider",
    )
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
        default="api_probe_output.json",
        help="Where to save the probe result JSON",
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


def image_to_payload(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": data,
        },
    }


def extract_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def parse_json_maybe(text: str):
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    return json.loads(candidate)


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def call_text_probe(client: Anthropic, model: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=128,
        temperature=0,
        system="You are a precise evaluation assistant.",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OK",
            }
        ],
    )
    return {
        "raw_text": extract_text(response),
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": to_jsonable(getattr(response, "usage", None)),
    }


def call_vision_probe(client: Anthropic, model: str, image_path: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=256,
        temperature=0,
        system="You evaluate image quality for generative model assessment.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Check this image for obvious distortion, melting, duplicated parts, "
                            "or broken structure. Return one short sentence only."
                        ),
                    },
                    image_to_payload(image_path),
                ],
            }
        ],
    )
    return {
        "raw_text": extract_text(response),
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": to_jsonable(getattr(response, "usage", None)),
    }


def call_json_probe(client: Anthropic, model: str, image_path: str) -> dict:
    prompt = (
        "Return valid JSON only. No markdown. No extra text.\\n"
        "Schema:\\n"
        "{\\n"
        '  "plan_ok": true,\\n'
        '  "alignment_score": 0.0,\\n'
        '  "artifact_score": 0.0,\\n'
        '  "hard_failure": false,\\n'
        '  "issues": ["..."],\\n'
        '  "summary": "..."\\n'
        "}\\n"
        "Rules:\\n"
        "- alignment_score and artifact_score must be numbers between 0 and 1\\n"
        "- hard_failure must be boolean\\n"
        "- issues must be an array of strings"
    )

    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        system="You must output strict JSON only.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_to_payload(image_path),
                ],
            }
        ],
    )
    raw_text = extract_text(response)

    parsed = None
    error = None
    try:
        parsed = parse_json_maybe(raw_text)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return {
        "raw_text": raw_text,
        "parsed_json": parsed,
        "json_error": error,
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": to_jsonable(getattr(response, "usage", None)),
    }


def main() -> None:
    args = parse_args()

    if not args.base_url:
        raise SystemExit("Missing ANTHROPIC_BASE_URL or --base-url")
    if not args.api_key:
        raise SystemExit("Missing ANTHROPIC_AUTH_TOKEN/ANTHROPIC_API_KEY or --api-key")

    use_env_proxy = args.use_env_proxy and not args.ignore_env_proxy
    http_client = httpx.Client(trust_env=use_env_proxy, timeout=120.0)
    client = Anthropic(api_key=args.api_key, base_url=args.base_url, http_client=http_client)

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "image": str(Path(args.image).resolve()),
    }

    text_probe_ok = False
    vision_probe_ok = False
    json_probe_ok = False

    try:
        text_probe = call_text_probe(client, args.model)
        result["text_probe"] = text_probe
        text_probe_ok = text_probe["raw_text"].strip() == "OK"
    except Exception as exc:  # noqa: BLE001
        result["text_probe"] = {"error": str(exc)}

    try:
        vision_probe = call_vision_probe(client, args.model, args.image)
        result["vision_probe"] = vision_probe
        vision_probe_ok = bool(vision_probe["raw_text"].strip())
    except Exception as exc:  # noqa: BLE001
        result["vision_probe"] = {"error": str(exc)}

    try:
        json_probe = call_json_probe(client, args.model, args.image)
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

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved detailed result to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
