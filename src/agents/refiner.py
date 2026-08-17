"""
Refiner agent — last node in the graph.

If the Critic approved the draft, lightly polish tone/formatting for the
final response. If the Critic rejected it and refine_loop_count has hit the
configured max, produce an honest "I'm not confident enough to answer this
precisely" response instead of shipping a low-confidence guess — this is the
whole point of building a hallucination-aware pipeline instead of a plain
single-shot chatbot.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from src.llm_provider import get_chat_model
from src.logging_setup import get_logger
from src.schemas import GraphState

logger = get_logger(__name__)

POLISH_SYSTEM_PROMPT = """Lightly polish this financial assistant's answer for
tone and clarity. Keep it Gen-Z-friendly but not cringe, keep every fact and
number exactly as given, don't add new claims. Return only the final text."""


def refiner_node(state: GraphState) -> dict:
    verdict = state.critic_verdict

    if verdict and verdict.approved:
        llm = get_chat_model(temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=POLISH_SYSTEM_PROMPT),
            HumanMessage(content=state.draft_answer),
        ])
        logger.info("refiner_polished_approved_answer", extra={"confidence": verdict.confidence})
        return {"final_answer": response.content}

    # Rejected and out of refine attempts -> honest fallback, not a guess.
    logger.warning(
        "refiner_returning_honest_fallback",
        extra={
            "refine_loops_used": state.refine_loop_count,
            "issues": verdict.issues if verdict else ["unknown"],
        },
    )
    fallback = (
        "I don't have enough grounded information to answer that confidently "
        "right now, and I'd rather be upfront than guess with your money. "
        "Try rephrasing with more specifics, or ask me something I can check "
        "against your transaction history or the guidance docs I have access to."
    )
    return {"final_answer": fallback}
