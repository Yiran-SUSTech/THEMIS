from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any

from .client import ClaudeVisionClient
from .expert_performance import (
    EXPERT_PERFORMANCE_TABLE,
    get_expert_performance,
    get_reliability_weight,
    detect_expert_conflicts,
    calculate_weighted_severity,
)
from .local_experts import LocalArtifactExpert, LocalExpertError, LocalPlanner, LocalSemanticExpert, LocalJudge, LocalReflector
from .prompts import (
    EXPERT_SYSTEM,
    JUDGE_SYSTEM,
    PLANNER_SYSTEM,
    REFLECTOR_SYSTEM,
    REPORT_SYSTEM,
    build_task_context,
    get_reflector_prompt_with_reliability,
    get_report_prompt_with_reliability,
)
from .schemas import (
    EvaluationPlan,
    EvaluationReport,
    ExpertResult,
    ExpertReliabilitySummary,
    ExpertConflictInfo,
    FinalResult,
    GraphState,
    PlanReview,
    ReflectionReview,
    RoleModels,
)


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer"},
                    "expert": {"type": "string", "enum": ["semantic", "structural", "artifact", "vqa"]},
                    "goal": {"type": "string"},
                    "model_profile": {
                        "type": "string",
                        "enum": [
                            "local_fast",
                            "local_default",
                            "local_richer",
                            "local_stronger",
                            "remote_mid",
                            "remote_strong",
                        ],
                    },
                    "prompt_focus": {"type": "string"},
                    "allow_escalation": {"type": "boolean"},
                },
                "required": ["step_id", "expert", "goal", "model_profile", "prompt_focus", "allow_escalation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "steps"],
    "additionalProperties": False,
}

PLAN_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "feedback": {"type": "string"},
        "missing_checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approved", "feedback", "missing_checks"],
    "additionalProperties": False,
}

EXPERT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expert": {"type": "string", "enum": ["semantic", "structural", "artifact", "vqa"]},
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "number"},
    },
    "required": ["expert", "summary", "findings", "severity"],
    "additionalProperties": False,
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "alignment_reasoning": {"type": "string"},
        "artifact_reasoning": {"type": "string"},
        "alignment_score": {"type": "number"},
        "artifact_score": {"type": "number"},
        "hard_failure": {"type": "boolean"},
        "confidence": {"type": "number"},
        "key_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "alignment_reasoning",
        "artifact_reasoning",
        "alignment_score",
        "artifact_score",
        "hard_failure",
        "confidence",
        "key_issues",
    ],
    "additionalProperties": False,
}

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "feedback": {"type": "string"},
        "suggested_fixes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approved", "feedback", "suggested_fixes"],
    "additionalProperties": False,
}


def _print_stage(message: str) -> None:
    print(f"[agentic_eval] {message}", flush=True)


def _append_log(state: GraphState, filename: str, payload: dict[str, Any]) -> None:
    log_dir = state.get("log_dir")
    if not log_dir:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / filename
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")



def _write_json(state: GraphState, filename: str, payload: dict[str, Any]) -> None:
    log_dir = state.get("log_dir")
    if not log_dir:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / filename
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stage_message(role: str, action: str, model: str, elapsed: float, status: str, revision: str = "") -> str:
    revision_part = f" revision={revision}" if revision else ""
    return f"[agentic_eval] {status} role={role} action={action}{revision_part} model={model} elapsed={elapsed:.1f}s"


def _write_stage_line(message: str, *, final: bool) -> None:
    suffix = "\n" if final else ""
    sys.stdout.write(f"\r\033[K{message}{suffix}")
    sys.stdout.flush()


def _heartbeat_stage(role: str, action: str, model: str, revision: str, started_at: float, stop_event: Event) -> None:
    while not stop_event.wait(1.0):
        elapsed = perf_counter() - started_at
        _write_stage_line(_stage_message(role, action, model, elapsed, "running", revision), final=False)


def _start_stage(role: str, action: str, model: str, revision: str = "") -> tuple[float, Event, Thread]:
    started_at = perf_counter()
    _write_stage_line(_stage_message(role, action, model, 0.0, "running", revision), final=False)
    stop_event = Event()
    heartbeat = Thread(target=_heartbeat_stage, args=(role, action, model, revision, started_at, stop_event), daemon=True)
    heartbeat.start()
    return started_at, stop_event, heartbeat


def _finish_stage(role: str, action: str, model: str, started_at: float, stop_event: Event, heartbeat: Thread, revision: str = "") -> float:
    elapsed = round(perf_counter() - started_at, 4)
    stop_event.set()
    heartbeat.join(timeout=0.1)
    _write_stage_line(_stage_message(role, action, model, elapsed, "done", revision), final=True)
    return elapsed


def _fail_stage(role: str, action: str, model: str, started_at: float, stop_event: Event, heartbeat: Thread, exc: Exception, revision: str = "") -> None:
    elapsed = round(perf_counter() - started_at, 4)
    stop_event.set()
    heartbeat.join(timeout=0.1)
    _write_stage_line(_stage_message(role, action, model, elapsed, f"failed error={exc.__class__.__name__}", revision), final=True)


def _artifact_local_model_name(model_profile: str) -> str:
    if model_profile == "local_fast":
        return "maniqa"
    if model_profile == "local_richer":
        return "maniqa+musiq+clipiqa"
    return "maniqa+musiq"


def _planned_step_model(client: ClaudeVisionClient, expert_name: str, model_profile: str) -> str:
    if expert_name == "semantic" and model_profile in {"local_fast", "local_stronger"} and client.settings.local_semantic_enabled:
        if model_profile == "local_fast":
            return client.settings.semantic_local_fast_model
        return client.settings.semantic_local_stronger_model
    if expert_name == "artifact" and model_profile in {"local_fast", "local_default", "local_richer"} and client.settings.local_artifact_enabled:
        return _artifact_local_model_name(model_profile)
    fallback_profile = model_profile if model_profile in {"remote_mid", "remote_strong"} else "remote_strong"
    return resolve_remote_model(client, expert_name, fallback_profile)


def planner_node(state: GraphState, client: ClaudeVisionClient) -> GraphState:
    image_input = state["input"]
    planner_run_number = state.get("planner_run_count", 0) + 1
    planner_revision = f"{state.get('plan_revision_count', 0) + 1}/{client.settings.max_plan_revisions + 1}"
    started_at, stop_event, heartbeat = _start_stage("planner", "build_plan", client.settings.planner_local_model if client.settings.planner_local_enabled else client.settings.planner_model, planner_revision)
    prior_feedback = ""
    feedback_text = state.get("planner_feedback")
    feedback_source = state.get("planner_feedback_source") or "reviewer"
    if feedback_text:
        prior_feedback = f"\nPrevious {feedback_source} feedback: {feedback_text}"
    elif state.get("plan_review") and not state["plan_review"].approved:
        prior_feedback = f"\nPrevious judge feedback: {state['plan_review'].feedback}"
    elif state.get("reflection") and not state["reflection"].approved:
        prior_feedback = f"\nPrevious reflector feedback: {state['reflection'].feedback}"

    user_text = (
        f"{build_task_context(image_input.prompt, image_input.class_label)}\n"
        "Create a concise evaluation plan using the available experts: semantic, structural, artifact, vqa. "
        "Plan jointly from the image itself and the prompt/class label together as one multimodal grounding task. "
        "The plan must cover semantic alignment, whole-subject global structure, and a separate local artifact pass. "
        "Unless the image is trivially simple, include at least one semantic step, one structural step, and one artifact step; include vqa only if earlier evidence leaves a material unresolved question. "
        "Step order should usually be: semantic alignment first, structural coherence second, local artifact severity third, then optional vqa. "
        "For each step, choose a model_profile. Strongly prefer the cheapest adequate route first: semantic should start with local_fast or local_stronger, artifact should start with local_default or local_richer, structural should avoid remote_strong unless remote_mid is clearly insufficient, and vqa should be omitted unless earlier evidence leaves a material unresolved question. "
        "For semantic steps, check whether the visible subject is plausibly the labeled class from image evidence alone, then note the strongest confirming or contradicting traits. "
        "For structural steps, first test whether the whole animal forms a coherent instance of the labeled subject, then target the most likely subject-specific confusion risks and part-integrity failures visible in this image. Use the class label to name distinguishing morphology, body proportions, pose plausibility, and scene compatibility that separate this subject from nearby lookalikes. "
        "For artifact steps, explicitly inspect localized generation failures such as malformed face or muzzle regions, duplicated or fused limbs, broken joints, wrong tail attachment, malformed hands or feet, asymmetric anatomy, fur or edge boundary corruption, and other visible synthesis artifacts when relevant. "
        "Do not frame any step as comparing against external reference photos or doing open-ended species research; inspect this image directly. "
        "Escalate to remote_strong only when the task is fine-grained, evidence is ambiguous, or prior judge/reflector feedback indicates the earlier route was insufficient. "
        "Set prompt_focus to the exact visual evidence the expert should inspect, using subject-specific failure modes instead of a generic checklist. Set allow_escalation to false for steps that should stay on the chosen route."
        f"{prior_feedback}"
    )
    try:
        if client.settings.planner_local_enabled:
            try:
                _print_stage(f"planner route=local model={client.settings.planner_local_model}")
                payload = LocalPlanner(client.settings).plan(
                    image_path=image_input.image_path,
                    prompt=image_input.prompt,
                    class_label=image_input.class_label,
                    prior_feedback=prior_feedback.strip(),
                )
                elapsed = _finish_stage("planner", "build_plan", client.settings.planner_local_model, started_at, stop_event, heartbeat, planner_revision)
                plan = EvaluationPlan.model_validate(payload)
                _write_json(state, f"plan_round_{planner_run_number}.json", {
                    "round": planner_run_number,
                    "revision": state.get("plan_revision_count", 0) + 1,
                    "planner_model": client.settings.planner_local_model,
                    "planner_route": "local",
                    "planner_feedback_source": state.get("planner_feedback_source"),
                    "planner_feedback": state.get("planner_feedback"),
                    "plan": plan.model_dump(mode="json"),
                })
                return {
                    "plan": plan,
                    "expert_results": [],
                    "planner_feedback": None,
                    "planner_feedback_source": None,
                    "planner_model_used": client.settings.planner_local_model,
                    "planner_elapsed_seconds": elapsed,
                    "planner_run_count": planner_run_number,
                }
            except (LocalExpertError, ValueError):
                _write_stage_line(_stage_message("planner", "build_plan", client.settings.planner_model, perf_counter() - started_at, "running", planner_revision), final=False)

        model_used = client.settings.planner_model
        payload = client.invoke_json(
            system=PLANNER_SYSTEM,
            user_text=user_text,
            image_path=image_input.image_path,
            schema=PLAN_SCHEMA,
            model=model_used,
        )
        elapsed = _finish_stage("planner", "build_plan", model_used, started_at, stop_event, heartbeat, planner_revision)
        plan = EvaluationPlan.model_validate(payload)
        _write_json(state, f"plan_round_{planner_run_number}.json", {
            "round": planner_run_number,
            "revision": state.get("plan_revision_count", 0) + 1,
            "planner_model": model_used,
            "planner_route": "remote",
            "planner_feedback_source": state.get("planner_feedback_source"),
            "planner_feedback": state.get("planner_feedback"),
            "plan": plan.model_dump(mode="json"),
        })
        return {
            "plan": plan,
            "expert_results": [],
            "planner_feedback": None,
            "planner_feedback_source": None,
            "planner_model_used": model_used,
            "planner_elapsed_seconds": elapsed,
            "planner_run_count": planner_run_number,
        }
    except Exception as exc:
        _fail_stage("planner", "build_plan", client.settings.planner_model, started_at, stop_event, heartbeat, exc, planner_revision)
        raise



def judge_node(state: GraphState, client: ClaudeVisionClient) -> GraphState:
    judge_run_number = state.get("judge_run_count", 0) + 1
    planner_revision = f"{state.get('plan_revision_count', 0) + 1}/{client.settings.max_plan_revisions + 1}"
    started_at, stop_event, heartbeat = _start_stage("judge", "review_plan", client.settings.judge_model, planner_revision)
    image_input = state["input"]
    plan = state["plan"]
    
    settings = client.settings
    
    if settings.judge_local_enabled and settings.judge_local_model:
        _print_stage(f"judge route=local model={settings.judge_local_model}")
        try:
            local_judge = LocalJudge(settings)
            payload = local_judge.evaluate(
                image_path=image_input.image_path,
                plan=plan,
                prompt=image_input.prompt,
                class_label=image_input.class_label,
            )
            review = PlanReview.model_validate(payload)
            model_used = settings.judge_local_model
        except LocalExpertError as exc:
            _fail_stage("judge", "review_plan", settings.judge_local_model, started_at, stop_event, heartbeat, exc, planner_revision)
            raise
    else:
        plan_json = plan.model_dump_json(indent=2)
        user_text = (
            f"{build_task_context(image_input.prompt, image_input.class_label)}\n"
            f"Plan:\n{plan_json}\n"
            "Approve only if the plan covers alignment, global structure, local artifacts, uses VQA only when necessary, and clearly reasons jointly from the image and the prompt/class label. "
            "For structural inspection, require the plan to target the likely subject-specific failure modes instead of using a rote generic checklist."
        )
        model_used = settings.judge_model
        payload = client.invoke_json(
            system=JUDGE_SYSTEM,
            user_text=user_text,
            image_path=image_input.image_path,
            schema=PLAN_REVIEW_SCHEMA,
            model=model_used,
        )
        review = PlanReview.model_validate(payload)
    
    elapsed = _finish_stage("judge", "review_plan", model_used, started_at, stop_event, heartbeat, planner_revision)
    result: GraphState = {
        "plan_review": review,
        "judge_model_used": model_used,
        "judge_elapsed_seconds": elapsed,
        "judge_run_count": judge_run_number,
    }
    _write_json(state, f"judge_round_{judge_run_number}.json", {
        "round": judge_run_number,
        "revision": state.get("plan_revision_count", 0) + 1,
        "approved": review.approved,
        "feedback": review.feedback,
        "missing_checks": review.missing_checks,
        "plan": plan.model_dump(mode="json"),
        "judge_model": model_used,
    })
    if review.approved:
        result["planner_feedback"] = None
        result["planner_feedback_source"] = None
    else:
        _print_stage(f"judge_rejected feedback={review.feedback}")
        _append_log(state, "judge_rejections.jsonl", {
            "revision": state.get("plan_revision_count", 0) + 1,
            "approved": review.approved,
            "feedback": review.feedback,
            "missing_checks": review.missing_checks,
            "plan": plan.model_dump(mode="json"),
        })
        result["planner_feedback"] = review.feedback
        result["planner_feedback_source"] = "judge"
    return result



def resolve_remote_model(client: ClaudeVisionClient, expert_name: str, model_profile: str) -> str:
    settings = client.settings
    if expert_name == "structural":
        if model_profile == "remote_mid":
            return settings.structural_remote_mid_model
        return settings.structural_remote_strong_model
    if expert_name == "vqa":
        if model_profile == "remote_mid":
            return settings.vqa_remote_mid_model
        return settings.vqa_remote_strong_model
    if expert_name == "artifact":
        return settings.artifact_remote_strong_model
    return settings.remote_expert_model



def run_remote_expert(
    state: GraphState,
    client: ClaudeVisionClient,
    expert_name: str,
    goal: str,
    model_profile: str,
    prompt_focus: str,
) -> ExpertResult:
    image_input = state["input"]
    remote_model = resolve_remote_model(client, expert_name, model_profile)
    expert_instruction = ""
    if expert_name == "structural":
        expert_instruction = (
            " For structural evaluation, inspect the most relevant anatomical and part-attachment failures for this image, "
            "such as facial melting, extra or fused fingers, extra limbs, wrong tail attachment, melted hands or feet, broken joints, "
            "or impossible boundaries. When evidence is ambiguous, treat the ambiguity as failure risk rather than assuming the structure is correct."
        )
    elif expert_name == "artifact":
        expert_instruction = (
            " For artifact evaluation, extra appendages, impossible limbs, malformed extremities, duplicated parts, broken joints, "
            "wrong tail attachment, impossible anatomy, and severe boundary corruption should be treated as severe failures, not minor issues, whenever visible evidence supports them."
        )
    user_text = (
        f"{build_task_context(image_input.prompt, image_input.class_label)}\n"
        f"Expert: {expert_name}\n"
        f"Goal: {goal}\n"
        f"Prompt focus: {prompt_focus or 'Use the goal and image evidence.'}\n"
        "Return grounded findings only. Severity is 0 for no issue and 1 for severe issue."
        f"{expert_instruction}"
    )
    payload = client.invoke_json(
        system=EXPERT_SYSTEM,
        user_text=user_text,
        image_path=image_input.image_path,
        schema=EXPERT_SCHEMA,
        model=remote_model,
    )
    payload.setdefault("confidence", 0.8)
    payload.setdefault("source", "remote")
    payload.setdefault("model", remote_model)
    return ExpertResult.model_validate(payload)



def run_expert(state: GraphState, client: ClaudeVisionClient, step) -> ExpertResult:
    image_input = state["input"]
    expert_name = step.expert
    goal = step.goal
    model_profile = step.model_profile
    prompt_focus = step.prompt_focus
    allow_escalation = step.allow_escalation
    is_fine_grained_class_only = bool(image_input.class_label and not image_input.prompt)

    if expert_name == "semantic" and is_fine_grained_class_only:
        return run_remote_expert(state, client, expert_name, goal, "remote_strong", prompt_focus)

    if expert_name == "semantic" and client.settings.local_semantic_enabled and model_profile in {"local_fast", "local_stronger"}:
        original_model = client.settings.local_semantic_model
        try:
            if model_profile == "local_fast":
                client.settings.local_semantic_model = client.settings.semantic_local_fast_model
            else:
                client.settings.local_semantic_model = client.settings.semantic_local_stronger_model
            local_payload = LocalSemanticExpert(client.settings).evaluate(
                image_path=image_input.image_path,
                prompt=image_input.prompt,
                class_label=image_input.class_label,
            )
            local_result = ExpertResult.model_validate(local_payload)
            if not allow_escalation or local_result.confidence >= client.settings.semantic_escalation_threshold:
                return local_result
        except LocalExpertError:
            pass
        finally:
            client.settings.local_semantic_model = original_model

    if expert_name == "artifact":
        return run_remote_expert(state, client, expert_name, goal, "remote_strong", prompt_focus)

    fallback_profile = model_profile
    if expert_name == "structural" and fallback_profile not in {"remote_mid", "remote_strong"}:
        fallback_profile = "remote_strong" if is_fine_grained_class_only else "remote_mid"
    elif fallback_profile not in {"remote_mid", "remote_strong"}:
        fallback_profile = "remote_strong"
    return run_remote_expert(state, client, expert_name, goal, fallback_profile, prompt_focus)



def execute_plan_node(state: GraphState, client: ClaudeVisionClient) -> GraphState:
    steps = state["plan"].steps
    _print_stage(f"plan_steps={len(steps)}")
    results: list[ExpertResult] = []
    expert_elapsed_seconds: dict[str, float] = {}
    for step in steps:
        action = f"expert_{step.expert}_step_{step.step_id}"
        model_name = _planned_step_model(client, step.expert, step.model_profile)
        started_at, stop_event, heartbeat = _start_stage(step.expert, action, model_name)
        try:
            result = run_expert(state, client, step)
            elapsed = _finish_stage(step.expert, action, model_name, started_at, stop_event, heartbeat)
        except Exception as exc:
            _fail_stage(step.expert, action, model_name, started_at, stop_event, heartbeat, exc)
            raise
        results.append(result)
        expert_elapsed_seconds[step.expert] = round(expert_elapsed_seconds.get(step.expert, 0.0) + elapsed, 4)
    return {"expert_results": results, "expert_elapsed_seconds": expert_elapsed_seconds}



def report_node(state: GraphState, client: ClaudeVisionClient) -> GraphState:
    started_at, stop_event, heartbeat = _start_stage("report", "synthesize_report", client.settings.report_model)
    image_input = state["input"]
    expert_results = state["expert_results"]
    
    expert_payload = []
    for item in expert_results:
        result_dict = item.model_dump()
        performance = get_expert_performance(item.expert)
        if performance:
            result_dict["reliability"] = performance.reliability.value
            result_dict["confidence_weight"] = performance.confidence_weight
            result_dict["benchmark"] = performance.benchmark
            result_dict["accuracy"] = performance.accuracy
        else:
            result_dict["reliability"] = "unknown"
            result_dict["confidence_weight"] = 0.5
        expert_payload.append(result_dict)
    
    conflicts = detect_expert_conflicts(expert_payload)
    weighted_severity = calculate_weighted_severity(expert_payload)
    
    reliability_summary = {
        "high_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "high"),
        "medium_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "medium"),
        "low_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "low"),
        "unknown_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "unknown"),
        "weighted_severity": weighted_severity,
        "conflicts": conflicts,
    }

    user_text = (
        f"{build_task_context(image_input.prompt, image_input.class_label)}\n"
        f"Expert outputs:\n{json.dumps(expert_payload, indent=2)}\n"
        f"Detected conflicts:\n{json.dumps(conflicts, indent=2) if conflicts else 'None'}\n"
        f"Weighted severity: {weighted_severity:.3f}\n"
        "Synthesize a report with separate alignment and artifact scores. "
        "Directly reinspect the image and do not defer blindly to the experts if they appear too optimistic. "
        "Consider expert reliability when weighing their conclusions:\n"
        "- HIGH reliability experts (weight 1.0): Trust unless directly contradicted by visual evidence\n"
        "- MEDIUM reliability experts (weight 0.7): Consider but seek confirmation\n"
        "- LOW reliability experts (weight 0.4): Use as hints only\n"
        "- When experts conflict, prioritize higher reliability\n"
        "Artifact score must be a quality score where 1 means minimal visible artifacts and 0 means severe visible artifacts. "
        "If the image shows likely species mismatch, extra appendages, impossible limbs, malformed extremities, duplicated parts, broken joints, wrong tail attachment, impossible anatomy, or severe boundary corruption, lower the scores even if some experts missed the issue. "
        "If visible evidence supports a severe anatomical or structural generation failure, set hard_failure to true. "
        "When evidence is ambiguous, prefer a conservative judgment rather than assuming the image is correct."
    )
    model_used = client.settings.report_model
    try:
        payload = client.invoke_json(
            system=REPORT_SYSTEM,
            user_text=user_text,
            image_path=image_input.image_path,
            schema=REPORT_SCHEMA,
            model=model_used,
        )
        elapsed = _finish_stage("report", "synthesize_report", model_used, started_at, stop_event, heartbeat)
        
        report = EvaluationReport.model_validate(payload)
        report.expert_reliability_summary = ExpertReliabilitySummary(
            high_reliability_count=reliability_summary["high_reliability_count"],
            medium_reliability_count=reliability_summary["medium_reliability_count"],
            low_reliability_count=reliability_summary["low_reliability_count"],
            unknown_reliability_count=reliability_summary["unknown_reliability_count"],
            weighted_severity=weighted_severity,
            conflicts=[ExpertConflictInfo(**c) for c in conflicts],
        )
        report.reliability_adjusted_scores = len(conflicts) > 0 or weighted_severity > 0.3
        
        return {
            "report": report,
            "report_model_used": model_used,
            "report_elapsed_seconds": elapsed,
        }
    except Exception as exc:
        _fail_stage("report", "synthesize_report", model_used, started_at, stop_event, heartbeat, exc)
        raise



def reflector_node(state: GraphState, client: ClaudeVisionClient) -> GraphState:
    reflection_revision = f"{state.get('reflection_revision_count', 0) + 1}/{client.settings.max_reflection_revisions + 1}"
    started_at, stop_event, heartbeat = _start_stage("reflector", "review_report", client.settings.reflector_model, reflection_revision)
    image_input = state["input"]
    expert_results = state["expert_results"]
    
    settings = client.settings
    
    expert_payload = []
    for item in expert_results:
        result_dict = item.model_dump()
        performance = get_expert_performance(item.expert)
        if performance:
            result_dict["reliability"] = performance.reliability.value
            result_dict["confidence_weight"] = performance.confidence_weight
            result_dict["benchmark"] = performance.benchmark
            result_dict["accuracy"] = performance.accuracy
        else:
            result_dict["reliability"] = "unknown"
            result_dict["confidence_weight"] = 0.5
        expert_payload.append(result_dict)
    
    conflicts = detect_expert_conflicts(expert_payload)
    weighted_severity = calculate_weighted_severity(expert_payload)
    
    reliability_summary = {
        "high_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "high"),
        "medium_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "medium"),
        "low_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "low"),
        "unknown_reliability_count": sum(1 for r in expert_payload if r.get("reliability") == "unknown"),
        "weighted_severity": weighted_severity,
        "overall_confidence": sum(r.get("confidence_weight", 0.5) for r in expert_payload) / len(expert_payload) if expert_payload else 0.5,
    }
    
    report_payload = state["report"].model_dump()
    
    if settings.reflector_local_enabled and settings.reflector_local_model:
        _print_stage(f"reflector route=local model={settings.reflector_local_model}")
        try:
            local_reflector = LocalReflector(settings)
            payload = local_reflector.evaluate(
                image_path=image_input.image_path,
                report=report_payload,
                expert_results=expert_payload,
                reliability_summary=reliability_summary,
                conflicts=conflicts,
                weighted_severity=weighted_severity,
                prompt=image_input.prompt,
                class_label=image_input.class_label,
            )
            review = ReflectionReview.model_validate(payload)
            model_used = settings.reflector_local_model
        except LocalExpertError as exc:
            _fail_stage("reflector", "review_report", settings.reflector_local_model, started_at, stop_event, heartbeat, exc, reflection_revision)
            raise
    else:
        user_text = (
            f"{build_task_context(image_input.prompt, image_input.class_label)}\n"
            f"Expert outputs with reliability:\n{json.dumps(expert_payload, indent=2)}\n"
            f"Detected conflicts:\n{json.dumps(conflicts, indent=2) if conflicts else 'None'}\n"
            f"Weighted severity: {weighted_severity:.3f}\n"
            f"Reliability summary: {json.dumps(reliability_summary, indent=2)}\n"
            f"Report:\n{json.dumps(report_payload, indent=2)}\n"
            "Act as a second-pass critic. Reinspect the image directly and look specifically for severe failures the experts or report may have missed. "
            "Consider expert reliability when evaluating the report:\n"
            "- HIGH reliability experts (weight 1.0): If they report issues, trust them strongly\n"
            "- MEDIUM reliability experts (weight 0.7): Consider their findings but verify with visual evidence\n"
            "- LOW reliability experts (weight 0.4): Use as hints only, require stronger confirmation\n"
            "- When experts conflict, prioritize the one with higher reliability\n"
            "- If report relies heavily on LOW reliability experts, flag for additional verification\n"
            "- If weighted severity significantly differs from report severity, note the discrepancy\n"
            "Reject the report if it is too optimistic about species match, anatomy, appendages, limbs, extremities, tail attachment, duplicated parts, impossible structure, or boundary corruption. "
            "If visible evidence is ambiguous, treat the ambiguity as failure risk rather than assuming the image is clean. "
            "If disapproving, explain which severe issue was missed and why replanning or re-evaluation is needed."
        )
        model_used = settings.reflector_model
        try:
            payload = client.invoke_json(
                system=REFLECTOR_SYSTEM,
                user_text=user_text,
                image_path=image_input.image_path,
                schema=REFLECTION_SCHEMA,
                model=model_used,
            )
            review = ReflectionReview.model_validate(payload)
        except Exception as exc:
            _fail_stage("reflector", "review_report", model_used, started_at, stop_event, heartbeat, exc, reflection_revision)
            raise
    
    elapsed = _finish_stage("reflector", "review_report", model_used, started_at, stop_event, heartbeat, reflection_revision)
    result: GraphState = {
        "reflection": review,
        "reflector_model_used": model_used,
        "reflector_elapsed_seconds": elapsed,
    }
    _write_json(state, f"reflector_round_{state.get('reflection_revision_count', 0) + 1}.json", {
        "revision": state.get('reflection_revision_count', 0) + 1,
        "approved": review.approved,
        "feedback": review.feedback,
        "suggested_fixes": review.suggested_fixes,
        "report": state["report"].model_dump(mode="json"),
        "expert_results": expert_payload,
        "conflicts": conflicts,
        "reliability_summary": reliability_summary,
        "reflector_model": model_used,
    })
    if review.approved:
        result["planner_feedback"] = None
        result["planner_feedback_source"] = None
        else:
            _print_stage(f"reflector_rejected feedback={review.feedback}")
            _append_log(state, "reflector_rejections.jsonl", {
                "revision": state.get('reflection_revision_count', 0) + 1,
                "approved": review.approved,
                "feedback": review.feedback,
                "suggested_fixes": review.suggested_fixes,
                "report": state["report"].model_dump(mode="json"),
                "expert_results": expert_payload,
                "conflicts": conflicts,
                "reliability_summary": reliability_summary,
            })
            result["planner_feedback"] = review.feedback
            result["planner_feedback_source"] = "reflector"
        return result
    except Exception as exc:
        _fail_stage("reflector", "review_report", model_used, started_at, stop_event, heartbeat, exc, reflection_revision)
        raise



def finalize_node(state: GraphState) -> GraphState:
    report = state["report"].model_copy(update={
        "role_timings": {
            "planner": state.get("planner_elapsed_seconds", 0.0),
            "judge": state.get("judge_elapsed_seconds", 0.0),
            "report": state.get("report_elapsed_seconds", 0.0),
            "reflector": state.get("reflector_elapsed_seconds", 0.0),
            "experts": state.get("expert_elapsed_seconds", {}),
        }
    })
    final_score = round(report.alignment_score * report.artifact_score, 4)
    _print_stage(
        "summary "
        f"planner={state.get('planner_elapsed_seconds', 0.0):.4f}s "
        f"judge={state.get('judge_elapsed_seconds', 0.0):.4f}s "
        f"report={state.get('report_elapsed_seconds', 0.0):.4f}s "
        f"reflector={state.get('reflector_elapsed_seconds', 0.0):.4f}s "
        f"experts={state.get('expert_elapsed_seconds', {})}"
    )
    expert_models = {
        result.expert: (result.model or "unknown")
        for result in state["expert_results"]
    }
    role_models = RoleModels(
        planner=state.get("planner_model_used") or "unknown",
        judge=state.get("judge_model_used") or "unknown",
        report=state.get("report_model_used") or "unknown",
        reflector=state.get("reflector_model_used") or "unknown",
        experts=expert_models,
    )
    final_result = FinalResult(
        plan=state["plan"],
        plan_review=state["plan_review"],
        expert_results=state["expert_results"],
        report=report,
        reflection=state["reflection"],
        role_models=role_models,
        final_score=final_score,
    )
    return {"final_result": final_result}
