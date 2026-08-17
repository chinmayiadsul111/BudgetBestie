"""
Graph wiring: Planner -> Retrieval -> Reasoning -> Critic -> (loop or Refiner).

    Planner
       |
    Retrieval
       |
    Reasoning  <---------+
       |                 |
    Critic ---rejected---+   (only while refine_loop_count < max_refine_loops)
       |
    approved / loops exhausted
       |
    Refiner
       |
      END

The Critic->Reasoning edge is what makes this "self-reflective" rather than a
straight-line RAG pipeline: a rejected draft goes back with feedback attached
instead of the graph just shipping whatever came out first.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.agents.planner import planner_node
from src.agents.retrieval import retrieval_node
from src.agents.reasoning import reasoning_node
from src.agents.critic import critic_node
from src.agents.refiner import refiner_node
from src.config_loader import get_config
from src.logging_setup import get_logger
from src.schemas import GraphState

logger = get_logger(__name__)


def _route_after_critic(state: GraphState) -> str:
    cfg = get_config()
    max_loops = cfg.get("reflection.max_refine_loops", 2)

    verdict = state.critic_verdict
    if verdict and verdict.approved:
        return "refiner"

    if state.refine_loop_count >= max_loops:
        logger.warning(
            "refine_loop_limit_reached",
            extra={"loops_used": state.refine_loop_count, "max_loops": max_loops},
        )
        return "refiner"  # refiner_node handles the honest-fallback path

    return "reasoning"  # loop back with critic feedback attached


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("critic", critic_node)
    builder.add_node("refiner", refiner_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retrieval")
    builder.add_edge("retrieval", "reasoning")
    builder.add_edge("reasoning", "critic")
    builder.add_conditional_edges(
        "critic", _route_after_critic, {"reasoning": "reasoning", "refiner": "refiner"}
    )
    builder.add_edge("refiner", END)

    return builder.compile()


_COMPILED_GRAPH = None


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
        logger.info("graph_compiled")
    return _COMPILED_GRAPH
