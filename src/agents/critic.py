"""
Critic agent — the hallucination-aware self-reflection step.

Reviews the Reasoning agent's draft answer against the actual evidence it was
given (SQL rows, retrieved chunks) and scores confidence. This is the
component that stops the pipeline from confidently telling a 22-year-old to
do something financially wrong.

Rejections route back to Reasoning (up to reflection.max_refine_loops) with
the critic's specific complaint attached, rather than silently retrying blind.
"""
from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, HumanMessage

from src.config_loader import get_config
from src.llm_provider import get_chat_model
from src.logging_setup import get_logger
from src.schemas import GraphState, CriticVerdict

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a strict fact-checking Critic reviewing a financial
assistant's draft answer. Your job is to catch hallucination, not to rewrite
the answer.

Check specifically:
1. Does every numeric claim in the draft trace back to the provided SQL
   results? Flag any number that appears invented.
2. Does every piece of financial advice trace back to the provided reference
   material, OR is it uncontroversial/generic enough to not need a citation
   (e.g. "consider setting a budget")? Flag specific/strong advice that has
   no grounding.
3. Is the tone appropriate (not preachy, not reckless)?

Respond ONLY with JSON, no markdown fences:
{"approved": <bool>, "confidence": <float 0-1>, "issues": ["<issue>", ...], "reasoning": "<one paragraph>"}
"""


def critic_node(state: GraphState) -> dict:
    cfg = get_config()
    threshold = cfg.get("reflection.confidence_threshold", 0.7)
    llm = get_chat_model(temperature=0.0)

    evidence_summary = _summarize_evidence(state)
    prompt = (
        f"User question: {state.user_question}\n\n"
        f"Draft answer:\n{state.draft_answer}\n\n"
        f"Evidence available to the draft-writer:\n{evidence_summary}"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        parsed = json.loads(_strip_fences(response.content))
        verdict = CriticVerdict(**parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "critic_json_parse_failed_defaulting_to_reject",
            extra={"raw_response": response.content, "error": str(exc)},
        )
        # Fail safe: if we can't parse the verdict, don't blindly approve.
        verdict = CriticVerdict(
            approved=False, confidence=0.0,
            issues=["critic_output_unparseable"], reasoning="Fallback rejection: parse failure.",
        )

    # Confidence threshold is enforced here, not just left to the LLM's
    # self-reported "approved" boolean — belt and suspenders.
    if verdict.confidence < threshold:
        verdict.approved = False

    logger.info(
        "critic_verdict" if verdict.approved else "critic_rejected",
        extra={**verdict.model_dump(), "threshold": threshold},
    )

    return {"critic_verdict": verdict, "refine_loop_count": state.refine_loop_count + 1}


def _summarize_evidence(state: GraphState) -> str:
    parts = []
    if state.sql_result:
        if state.sql_result.error:
            parts.append(f"SQL FAILED: {state.sql_result.error}")
        else:
            parts.append(f"SQL rows: {state.sql_result.rows}")
    if state.retrieved_chunks:
        parts.append("Retrieved docs: " + "; ".join(c.source for c in state.retrieved_chunks))
    return "\n".join(parts) if parts else "No evidence was provided to the draft-writer."


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    return t.strip()
