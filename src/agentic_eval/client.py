from __future__ import annotations

from typing import Any

from .config import Settings


class ClaudeVisionClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def close(self) -> None:
        return None

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
        raise RuntimeError("Remote invocation has been removed. This project now runs local models only.")
