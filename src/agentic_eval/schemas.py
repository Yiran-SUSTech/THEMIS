from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


ExpertName = Literal["semantic", "structural", "artifact", "vqa"]
ModelProfile = Literal[
    "local_fast",
    "local_default",
    "local_richer",
    "local_stronger",
    "remote_mid",
    "remote_strong",
]


class ImageInput(BaseModel):
    image_path: str
    prompt: str | None = None
    class_label: str | None = None


class PlanStep(BaseModel):
    step_id: int
    expert: ExpertName
    goal: str
    model_profile: ModelProfile
    prompt_focus: str = ""
    allow_escalation: bool = True


class EvaluationPlan(BaseModel):
    rationale: str
    steps: list[PlanStep]


class PlanReview(BaseModel):
    approved: bool
    feedback: str
    missing_checks: list[str] = Field(default_factory=list)


class ExpertResult(BaseModel):
    expert: ExpertName
    summary: str
    findings: list[str] = Field(default_factory=list)
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "remote"
    model: str | None = None


class RoleTimings(BaseModel):
    planner: float = 0.0
    judge: float = 0.0
    report: float = 0.0
    reflector: float = 0.0
    experts: dict[str, float] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    alignment_reasoning: str
    artifact_reasoning: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    artifact_score: float = Field(ge=0.0, le=1.0)
    hard_failure: bool
    confidence: float = Field(ge=0.0, le=1.0)
    key_issues: list[str] = Field(default_factory=list)
    role_timings: RoleTimings = Field(default_factory=RoleTimings)


class ReflectionReview(BaseModel):
    approved: bool
    feedback: str
    suggested_fixes: list[str] = Field(default_factory=list)


class RoleModels(BaseModel):
    planner: str
    judge: str
    report: str
    reflector: str
    experts: dict[str, str] = Field(default_factory=dict)


class FinalResult(BaseModel):
    plan: EvaluationPlan
    plan_review: PlanReview
    expert_results: list[ExpertResult]
    report: EvaluationReport
    reflection: ReflectionReview
    role_models: RoleModels
    final_score: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    input: ImageInput
    plan: EvaluationPlan
    plan_review: PlanReview
    expert_results: list[ExpertResult]
    report: EvaluationReport
    reflection: ReflectionReview
    final_result: FinalResult
    planner_feedback: str | None
    planner_feedback_source: str | None
    planner_model_used: str
    judge_model_used: str
    report_model_used: str
    reflector_model_used: str
    planner_elapsed_seconds: float
    judge_elapsed_seconds: float
    report_elapsed_seconds: float
    reflector_elapsed_seconds: float
    expert_elapsed_seconds: dict[str, float]
    log_dir: str
    plan_revision_count: int
    reflection_revision_count: int
    planner_run_count: int
    judge_run_count: int
