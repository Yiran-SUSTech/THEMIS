from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from time import sleep
from typing import Any

import httpx
from anthropic import Anthropic

from .config import Settings


class ClaudeVisionClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http_client = httpx.Client(trust_env=False, timeout=180.0)
        self.client = Anthropic(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
            http_client=self.http_client,
        )

    def close(self) -> None:
        self.http_client.close()

    def image_block(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
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

    def _extract_text(self, response: Any) -> str:
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    def _extract_json(self, text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(candidate[start:end + 1])
            raise

    def invoke_json(
        self,
        *,
        system: str,
        user_text: str,
        image_path: str,
        schema: dict[str, Any],
        max_tokens: int = 1200,
        model: str | None = None,
        include_image: bool = True,
    ) -> dict[str, Any]:
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{user_text}\n\n"
                    "Return valid JSON only. No markdown. No extra text.\n"
                    "Follow this JSON schema exactly:\n"
                    f"{schema_text}"
                ),
            }
        ]
        if include_image:
            content.append(self.image_block(image_path))

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.messages.create(
                    model=model or self.settings.model,
                    max_tokens=max_tokens,
                    temperature=0,
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                )
                text = self._extract_text(response)
                return self._extract_json(text)
            except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt == 0:
                    sleep(1)
                    continue
                raise
            except Exception as exc:
                message = str(exc).lower()
                if attempt == 0 and ("timeout" in message or "timed out" in message):
                    last_error = exc
                    sleep(1)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("invoke_json failed without a captured exception")
