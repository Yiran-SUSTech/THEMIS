import json
from openai import OpenAI

from step1_router import (
    _COMMON_ROUTER_INSTRUCTIONS,
    extract_expert_ids,
    build_router_registry_summary,
    _build_context_block,
    parse_json_safely,
)
from step4_reflector import _REFLECTOR_SYSTEM_TEMPLATE


class ConversationSession:
    def __init__(self, system_content):
        if isinstance(system_content, list):
            self.messages = [{"role": "system", "content": system_content}]
        else:
            self.messages = [{"role": "system", "content": system_content}]

    def add_user(self, content):
        self.messages.append({"role": "user", "content": content})

    def call_api(self, client, model, temperature=0, response_format=None):
        kwargs = {"model": model, "messages": self.messages, "temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format
        completion = client.chat.completions.create(**kwargs)
        raw = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": raw})
        return raw, completion

    @property
    def turn_count(self):
        return sum(1 for m in self.messages if m["role"] == "assistant")


def build_combined_system_content(
    experts_registry_str: str,
    class_label: str,
    taxonomy_info: dict | None,
) -> list[dict]:
    _, expert_ids_str, registry_summary = _build_context_block(
        class_label, taxonomy_info, experts_registry_str,
    )
    router_instructions = _COMMON_ROUTER_INSTRUCTIONS.format(expert_ids_str=expert_ids_str)

    judge_evaluation_dimensions = """1. **Expert Necessity & Applicability:** For each selected expert, is it necessary AND is its `target_subject` compatible with its capabilities? Cross-reference `applicable_scenes`/`best_for`/`topology_map` with the actual subjects.
2. **Missing Experts:** Are critical experts missing for ANY visible subject? Base this on `applicable_scenes` and `best_for`, NOT blanket rules.
3. **Weight Rationality:** Do weights reflect structural criticality? Class subject's experts should generally have higher weights.
4. **Focus Areas Completeness:** Do focus areas cover critical diagnostic features for ALL visible subjects per Taxonomy?
5. **Custom Prompts Quality:** Are custom prompts specific and actionable?

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
