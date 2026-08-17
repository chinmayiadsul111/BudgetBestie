"""
Reasoning agent — the workhorse node.

1. If the Planner flagged needs_sql, generates a SELECT query against the
   `transactions` table and executes it (read-only, enforced in src/db.py).
2. Synthesizes a draft natural-language answer using whichever of
   {SQL results, retrieved SOP chunks} are available.

This node is re-entered on refine loops: if the Critic rejects the draft, it
comes back here with the critic's feedback appended to the prompt so the
retry isn't a blind repeat of the same mistake.
"""
from __future__ import annotations

import re

from langchain_core.messages import SystemMessage, HumanMessage

from src.db import run_readonly_sql
from src.llm_provider import get_chat_model
from src.logging_setup import get_logger
from src.schemas import GraphState, SqlResult

logger = get_logger(__name__)

SCHEMA_DESCRIPTION = """
Table: transactions
Columns:
  id INTEGER
  date DATE            -- ISO format, e.g. '2026-05-14'
  category TEXT         -- one of: rent, groceries, dining, subscriptions,
                          -- transport, shopping, credit_card_payment, income
  merchant TEXT
  amount FLOAT           -- POSITIVE = money out (expense), NEGATIVE = money in (income)
  account TEXT           -- checking | credit_card
"""

SQL_SYSTEM_PROMPT = f"""You write a single SQLite SELECT statement to answer a
personal finance question, given this schema:
{SCHEMA_DESCRIPTION}

Rules:
- Output ONLY the SQL statement. No markdown fences, no explanation.
- SELECT statements only — never INSERT/UPDATE/DELETE/DROP.
- Remember amount sign convention: expenses are positive, income is negative.
  To sum actual spending, filter category != 'income' and sum(amount).
- Use date('now') style SQLite functions for relative date filters if needed.
"""

ANSWER_SYSTEM_PROMPT = """You are BudgetBestie: a warm, direct, slightly witty
personal finance assistant for a Gen Z audience. Answer the user's question
using ONLY the SQL results and/or reference material provided below — never
invent numbers or advice that isn't grounded in what's given.

If you're missing information to fully answer, say so plainly rather than
guessing. Keep the tone approachable, not preachy. No emojis in every
sentence — use them sparingly if at all.
"""


def reasoning_node(state: GraphState) -> dict:
    llm = get_chat_model()
    updates: dict = {}

    if state.plan and state.plan.needs_sql:
        sql_result = _generate_and_run_sql(llm, state.user_question)
        updates["sql_result"] = sql_result

    draft = _synthesize_answer(
        llm=llm,
        question=state.user_question,
        sql_result=updates.get("sql_result") or state.sql_result,
        retrieved_chunks=state.retrieved_chunks,
        prior_critic_feedback=state.critic_verdict.reasoning if state.critic_verdict else None,
    )
    updates["draft_answer"] = draft
    return updates


def _generate_and_run_sql(llm, question: str) -> SqlResult:
    response = llm.invoke([
        SystemMessage(content=SQL_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ])
    sql = _extract_sql(response.content)

    try:
        rows = run_readonly_sql(sql)
        return SqlResult(sql=sql, rows=rows)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sql_generation_or_execution_failed", extra={"sql": sql, "error": str(exc)})
        return SqlResult(sql=sql, rows=[], error=str(exc))


def _extract_sql(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def _synthesize_answer(
    llm, question: str, sql_result, retrieved_chunks, prior_critic_feedback: str | None
) -> str:
    context_parts = []

    if sql_result and not sql_result.error:
        context_parts.append(f"SQL query run: {sql_result.sql}\nResults: {sql_result.rows}")
    elif sql_result and sql_result.error:
        context_parts.append(f"SQL query FAILED: {sql_result.error}")

    if retrieved_chunks:
        joined = "\n---\n".join(f"[{c.source}] {c.content}" for c in retrieved_chunks)
        context_parts.append(f"Reference material:\n{joined}")

    if not context_parts:
        context_parts.append("No SQL results or reference material available.")

    prompt = f"User question: {question}\n\nAvailable context:\n" + "\n\n".join(context_parts)
    if prior_critic_feedback:
        prompt += (
            f"\n\nYour previous draft was rejected by a reviewer for this reason: "
            f"'{prior_critic_feedback}'. Address that specifically this time."
        )

    response = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content
