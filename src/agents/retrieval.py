"""
Retrieval agent — pulls relevant chunks from the financial SOP knowledge base
via FAISS, only when the Planner flagged needs_retrieval=True. Skipping this
entirely for SQL-only questions saves an embedding call and keeps the trace
clean in LangSmith.
"""
from __future__ import annotations

from src.logging_setup import get_logger
from src.schemas import GraphState, RetrievedChunk
from src.vectorstore import retrieve

logger = get_logger(__name__)


def retrieval_node(state: GraphState) -> dict:
    if not state.plan or not state.plan.needs_retrieval:
        logger.info("retrieval_skipped_not_needed")
        return {"retrieved_chunks": []}

    raw_results = retrieve(state.user_question)
    chunks = [RetrievedChunk(**r) for r in raw_results]

    if not chunks:
        logger.warning(
            "retrieval_returned_no_results_above_threshold",
            extra={"question": state.user_question},
        )

    return {"retrieved_chunks": chunks}
