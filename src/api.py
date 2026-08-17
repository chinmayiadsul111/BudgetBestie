"""
FastAPI service exposing the BudgetBestie multi-agent pipeline.

Run with: python run.py   (see COMMANDS.md — no CLI flags needed)
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config_loader import get_config
from src.db import ensure_schema_and_seed
from src.graph import get_graph
from src.logging_setup import get_logger
from src.observability import init_observability
from src.schemas import AskRequest, AskResponse, GraphState

logger = get_logger(__name__)

app = FastAPI(
    title="BudgetBestie",
    description="Self-reflective multi-agent finance assistant (LangGraph + FAISS + FastAPI)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_observability()
    ensure_schema_and_seed()
    get_graph()  # compile once at startup, not on first request
    logger.info("startup_complete")


@app.get("/health")
def health() -> dict:
    cfg = get_config()
    return {
        "status": "ok",
        "llm_provider": cfg.get("llm.provider"),
        "embeddings_provider": cfg.get("embeddings.provider"),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    trace_id = str(uuid.uuid4())
    start = time.time()
    logger.info("ask_received", extra={"trace_id": trace_id, "question": request.question})

    graph = get_graph()
    initial_state = GraphState(user_question=request.question, trace_id=trace_id)

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("graph_execution_failed", extra={"trace_id": trace_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Something went wrong processing that question.")

    elapsed_ms = round((time.time() - start) * 1000, 1)
    logger.info("ask_completed", extra={"trace_id": trace_id, "elapsed_ms": elapsed_ms})

    verdict = result.get("critic_verdict")
    plan = result.get("plan")

    return AskResponse(
        answer=result.get("final_answer", ""),
        confidence=verdict.confidence if verdict else 0.0,
        used_sql=bool(plan and plan.needs_sql),
        used_retrieval=bool(plan and plan.needs_retrieval),
        sources=[c.source for c in result.get("retrieved_chunks", [])],
        refine_loops=result.get("refine_loop_count", 0),
        trace_id=trace_id,
    )
