"""
Planner agent — first node in the graph.

Decides, before spending any tokens on retrieval or SQL generation, whether
the user's question needs:
  - a SQL lookup against their transaction history (e.g. "how much did I
    spend on food delivery last month?"),
  - a retrieval lookup against financial guidance docs (e.g. "should I pay
    off my credit card or save first?"),
  - both (e.g. "based on my spending, am I following the 50/30/20 rule?"),
  - or neither (small talk / out-of-scope — handled gracefully, not forced
    through an SQL/RAG pipeline it doesn't need).
"""
from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, HumanMessage

from src.llm_provider import get_chat_model
from src.logging_setup import get_logger
from src.schemas import GraphState, PlanStep

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Planner in a personal-finance assistant pipeline.
Given a user's question, decide what information is needed to answer it well.

Respond ONLY with JSON in this exact shape, no markdown fences, no prose:
{"needs_sql": <bool>, "needs_retrieval": <bool>, "rationale": "<one sentence>"}

- needs_sql: true if answering requires looking at the user's actual transaction
  history (spending amounts, categories, dates, merchants).
- needs_retrieval: true if answering benefits from general financial guidance
  (budgeting rules, credit score mechanics, debt payoff strategy, emergency funds).
- Many questions need both. A pure greeting or out-of-scope question needs neither.
"""


def planner_node(state: GraphState) -> dict:
    llm = get_chat_model(temperature=0.0)  # deterministic classification, not creative writing

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state.user_question),
    ])

    try:
        parsed = json.loads(_strip_fences(response.content))
        plan = PlanStep(**parsed)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "planner_json_parse_failed_defaulting_to_both",
            extra={"raw_response": response.content, "error": str(exc)},
        )
        # Fail safe: if we can't parse the plan, do both lookups rather than none.
        plan = PlanStep(needs_sql=True, needs_retrieval=True, rationale="fallback: parse failure")

    logger.info("planner_decision", extra=plan.model_dump())
    return {"plan": plan}


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    return t.strip()
