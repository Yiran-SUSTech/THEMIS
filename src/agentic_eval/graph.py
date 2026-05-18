from __future__ import annotations

from langgraph.graph import END, StateGraph

from .client import ClaudeVisionClient
from .config import Settings
from .nodes import execute_plan_node, finalize_node, judge_node, planner_node, reflector_node, report_node
from .schemas import GraphState



def build_graph(settings: Settings):
    graph = StateGraph(GraphState)
    client = ClaudeVisionClient(settings)

    def planner(state: GraphState) -> GraphState:
        return planner_node(state, client)

    def judge(state: GraphState) -> GraphState:
        return judge_node(state, client)

    def execute_plan(state: GraphState) -> GraphState:
        return execute_plan_node(state, client)

    def report(state: GraphState) -> GraphState:
        return report_node(state, client)

    def reflect(state: GraphState) -> GraphState:
        return reflector_node(state, client)

    def increment_plan_revision(state: GraphState) -> GraphState:
        return {"plan_revision_count": state.get("plan_revision_count", 0) + 1}

    def increment_reflection_revision(state: GraphState) -> GraphState:
        return {"reflection_revision_count": state.get("reflection_revision_count", 0) + 1}

    graph.add_node("planner", planner)
    graph.add_node("judge", judge)
    graph.add_node("execute_plan", execute_plan)
    graph.add_node("report", report)
    graph.add_node("reflect", reflect)
    graph.add_node("increment_plan_revision", increment_plan_revision)
    graph.add_node("increment_reflection_revision", increment_reflection_revision)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "judge")

    def route_after_judge(state: GraphState) -> str:
        review = state["plan_review"]
        revision_count = state.get("plan_revision_count", 0)
        if review.approved or revision_count >= settings.max_plan_revisions:
            return "execute_plan"
        return "increment_plan_revision"

    def route_after_reflect(state: GraphState) -> str:
        review = state["reflection"]
        revision_count = state.get("reflection_revision_count", 0)
        if review.approved or revision_count >= settings.max_reflection_revisions:
            return "finalize"
        return "increment_reflection_revision"

    graph.add_conditional_edges("judge", route_after_judge, {"increment_plan_revision": "increment_plan_revision", "execute_plan": "execute_plan"})
    graph.add_edge("increment_plan_revision", "planner")
    graph.add_edge("execute_plan", "report")
    graph.add_edge("report", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, {"increment_reflection_revision": "increment_reflection_revision", "finalize": "finalize"})
    graph.add_edge("increment_reflection_revision", "planner")
    graph.add_edge("finalize", END)

    return graph.compile()
