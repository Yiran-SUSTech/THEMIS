import json
from openai import OpenAI

from step1_router import (
    _COMMON_ROUTER_INSTRUCTIONS,
    _ROUTER_DIRECT_SCORE_INSTRUCTIONS,
    extract_expert_ids,
    build_router_registry_summary,
    _build_context_block,
    parse_json_safely,
    get_structured_taxonomy_info,
)
from step4_reflector import _REFLECTOR_SYSTEM_TEMPLATE
from common import api_call_with_retry


class ConversationSession:
    def __init__(self, system_content, api_retry=0):
        if isinstance(system_content, list):
            self.messages = [{"role": "system", "content": system_content}]
        else:
            self.messages = [{"role": "system", "content": system_content}]
        self.api_retry = api_retry

    def add_user(self, content):
        self.messages.append({"role": "user", "content": content})

    def call_api(self, client, model, temperature=0, response_format=None, label="Session"):
        kwargs = {"model": model, "messages": self.messages, "temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format
        kwargs["extra_body"] = {"enable_thinking": False}
        try:
            completion = api_call_with_retry(
                client.chat.completions.create,
                max_retries=self.api_retry,
                label=label,
                **kwargs,
            )
        except Exception as e:
            print(f"  [ERROR] {label} API call failed: {type(e).__name__}: {e}")
            raise
        raw = completion.choices[0].message.content
        finish_reason = getattr(completion.choices[0], "finish_reason", "unknown")
        usage = getattr(completion, "usage", None)
        if raw is None or raw.strip() == "":
            reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
            usage_info = f"prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}" if usage else "no usage info"
            print(f"  [ERROR] {label} API returned empty content (content is {'None' if raw is None else 'empty string'}, finish_reason={finish_reason}, {usage_info})")
            if reasoning:
                print(f"  [WARN] {label} reasoning_content found ({len(reasoning)} chars), attempting to extract JSON from it")
                raw = reasoning
            else:
                msg = completion.choices[0].message
                print(f"  [DEBUG] {label} full message: content={repr(msg.content)}, role={getattr(msg, 'role', 'N/A')}, function_call={getattr(msg, 'function_call', None)}, tool_calls={getattr(msg, 'tool_calls', None)}, refusal={getattr(msg, 'refusal', None)}")
                self.messages.append({"role": "assistant", "content": raw or ""})
                return raw, completion
        self.messages.append({"role": "assistant", "content": raw})
        return raw, completion

    @property
    def turn_count(self):
        return sum(1 for m in self.messages if m["role"] == "assistant")


def build_combined_system_content(
    experts_registry_str: str,
    class_label: str,
    taxonomy_info: dict | None,
    structured_taxonomy_info: dict | None = None,
) -> list[dict]:
    _, expert_ids_str, registry_summary = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )
    router_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)

    judge_evaluation_dimensions = """1. **Checkpoint Verdicts:** Are is_testable/is_present judgments reasonable? Did the Router avoid evaluating too many checkpoints by marking them untestable?
2. **Artifact Observations:** Are severity ratings consistent? If no artifacts found, does the image suggest missed issues?
3. **Expert Selection:** Is each expert's target_subject compatible with its capabilities? Are critical experts missing?
4. **Weights:** Do weights reflect structural criticality (class subject > auxiliary)?
5. **Focus Areas:** Do focus areas cover critical diagnostic features?

Judge output format:
{
  "is_approved": true/false,
  "reasons_for_rejection": "",
  "suggestions": []
}"""

    combined_text = (
        "You are a multi-role AI image evaluation system for auditing AI-generated images. "
        "In this conversation, you will play three roles as instructed by the user's messages. "
        "Always output pure JSON (no markdown wrapping).\n\n"
        "## Role 1: Router Agent\n"
        "You are a highly logical Router Agent for image auditing. "
        "You must prioritize the provided Taxonomy Knowledge as the source of truth.\n\n"
        f"{router_instructions}\n\n"
        "## Role 2: Judge Agent\n"
        "You are a meta-cognitive Judge Agent. You rigorously analyze evaluation strategies "
        "for logical rigor, scientific validity, and completeness.\n\n"
        f"{judge_evaluation_dimensions}\n\n"
        "## Role 3: Reflector (Supreme Judge)\n"
        f"{_REFLECTOR_SYSTEM_TEMPLATE}\n\n"
        f"## Expert Registry (Available Tools)\n{registry_summary}"
    )

    return [
        {
            "type": "text",
            "text": combined_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_reflector_only_system_content(
    experts_registry_str: str,
    class_label: str,
    taxonomy_info: dict | None,
    structured_taxonomy_info: dict | None = None,
) -> list[dict]:
    """Build system content for an independent Reflector session (not shared with Router/Judge)."""
    _, _, registry_summary = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )

    combined_text = (
        f"{_REFLECTOR_SYSTEM_TEMPLATE}\n\n"
        f"## Expert Registry (Available Tools)\n{registry_summary}"
    )

    return [
        {
            "type": "text",
            "text": combined_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_direct_score_system_content(
    experts_registry_str: str,
    class_label: str,
    taxonomy_info: dict | None,
    structured_taxonomy_info: dict | None = None,
) -> list[dict]:
    """Build system content for the without-expert direct-scoring session.

    Uses the same Router role and registry context as the normal mode, but replaces
    the expert-selection instructions with direct-scoring instructions.
    """
    _, _, registry_summary = _build_context_block(
        class_label, taxonomy_info, experts_registry_str, structured_taxonomy_info,
    )

    combined_text = (
        "You are a highly logical Router Agent for image auditing. "
        "You must prioritize the provided Taxonomy Knowledge as the source of truth. "
        "Output JSON only.\n\n"
        f"{_ROUTER_DIRECT_SCORE_INSTRUCTIONS}\n\n"
        f"## Expert Registry (Reference Only — NOT used in this mode)\n{registry_summary}"
    )

    return [
        {
            "type": "text",
            "text": combined_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
