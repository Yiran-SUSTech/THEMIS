from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


ExpertName = Literal["semantic", "structural", "quality", "vqa", "clip_semantic", "yolo_pose", "places365", "iqa", "background", "imagenet", "vqa_expert", "clip", "clip_score", "imagenet_fast", "imagenet_strong", "imagenet_eva02_large", "imagenet_eva_giant_224", "bge_candidate_generator", "e5_candidate_generator", "animal_pose", "body_pose", "body_pose_strong", "hand_detection", "face_detection", "places365_strong", "building_expert", "background_removal", "complexity", "iqa_fast", "iqa_default", "iqa_richer", "q_insight", "boundary_artifact", "aigen_detection", "ocr", "dog_breed", "bird_expert"]
ModelProfile = Literal[
    "local_fast",
    "local_default",
    "local_richer",
    "local_stronger",
]
StepType = Literal[
    "semantic_check",
    "structural_check",
    "quality_check",
    "vqa_evidence",
    "candidate_generation",
    "label_space_check",
    "confusable_disambiguation",
]
ReplanActionType = Literal[
    "add_step",
    "retarget_step",
    "replace_model",
    "reorder_steps",
    "tighten_task_fit",
    "reweight_evidence",
    "rerun_with_stronger_model",
]
FeedbackSource = Literal["judge", "reflector"]


class ExpertReliabilityInfo(BaseModel):
    expert_name: str
    model_name: str
    reliability: str = "medium"
    confidence_weight: float = 0.7
    benchmark: Optional[str] = None
    accuracy: Optional[float] = None
    notes: str = ""


class ImageInput(BaseModel):
    image_path: str
    prompt: str | None = None
    class_label: str | None = None


class ReplanAction(BaseModel):
    action: ReplanActionType
    reason: str
    priority: Literal["low", "medium", "high"] = "medium"
    target_step_id: int | None = None
    step_type: StepType | None = None
    suggested_expert: ExpertName | None = None
    suggested_model: str = ""
    prompt_focus: str = ""
    expected_signal: str = ""


class PlannerFeedbackPayload(BaseModel):
    source: FeedbackSource
    summary: str
    blocking_issues: list[str] = Field(default_factory=list)
    missing_checks: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    task_fit_issues: list[str] = Field(default_factory=list)
    replan_actions: list[ReplanAction] = Field(default_factory=list)


class PlanStep(BaseModel):
    step_id: int
    expert: ExpertName
    step_type: StepType = "semantic_check"
    goal: str
    model_profile: ModelProfile
    planned_model: str = ""
    selection_reason: str = ""
    prompt_focus: str = ""
    depends_on: list[int] = Field(default_factory=list)
    expected_signal: str = ""
    use_previous_outputs: bool = False
    allow_escalation: bool = True


class EvaluationPlan(BaseModel):
    rationale: str
    steps: list[PlanStep]


class PlanReview(BaseModel):
    approved: bool
    feedback: str
    missing_checks: list[str] = Field(default_factory=list)
    task_fit_issues: list[str] = Field(default_factory=list)
    replan_actions: list[ReplanAction] = Field(default_factory=list)


class ExpertResult(BaseModel):
    expert: ExpertName
    summary: str
    findings: list[str] = Field(default_factory=list)
    severity: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "local"
    model: str | None = None
    reliability: str = "medium"
    confidence_weight: float = 0.7
    evidence: dict[str, Any] = Field(default_factory=dict)
    extra_info: dict[str, Any] | None = None
    task_fit: float = Field(default=1.0, ge=0.0, le=1.0)
    task_fit_reason: str = ""


class ExpertConflictInfo(BaseModel):
    experts: list[str]
    severity_difference: float
    recommended_trust: str
    reason: str


class ExpertReliabilitySummary(BaseModel):
    high_reliability_count: int = 0
    medium_reliability_count: int = 0
    low_reliability_count: int = 0
    unknown_reliability_count: int = 0
    conflicts: list[ExpertConflictInfo] = Field(default_factory=list)
    weighted_severity: float = 0.0
    task_fit_weighted_severity: float = 0.0
    semantic_weighted_severity: float = 0.0
    semantic_task_fit_weighted_severity: float = 0.0
    structural_weighted_severity: float = 0.0
    structural_task_fit_weighted_severity: float = 0.0
    artifact_weighted_severity: float = 0.0
    artifact_task_fit_weighted_severity: float = 0.0
    low_task_fit_experts: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.7
    task_fit_applied: bool = False


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
    expert_reliability_summary: ExpertReliabilitySummary = Field(default_factory=ExpertReliabilitySummary)
    reliability_adjusted_scores: bool = False


class ReflectionReview(BaseModel):
    approved: bool
    feedback: str
    suggested_fixes: list[str] = Field(default_factory=list)
    task_fit_issues: list[str] = Field(default_factory=list)
    replan_actions: list[ReplanAction] = Field(default_factory=list)


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
    planner_feedback_payload: PlannerFeedbackPayload | None
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
