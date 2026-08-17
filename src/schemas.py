"""
Typed contracts shared across agents, the graph, and the API layer.

Using Pydantic (not raw dicts) means a malformed agent output fails loudly at
the boundary where it's produced, not three steps later inside a prompt.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float


class PlanStep(BaseModel):
    needs_sql: bool
    needs_retrieval: bool
    rationale: str


class SqlResult(BaseModel):
    sql: str
    rows: list[dict] = Field(default_factory=list)
    error: Optional[str] = None


class CriticVerdict(BaseModel):
    approved: bool
    confidence: float
    issues: list[str] = Field(default_factory=list)
    reasoning: str


class GraphState(BaseModel):
    """
    The single object that flows through every node of the LangGraph pipeline.
    Each agent reads what it needs and writes its own fields — nobody mutates
    another agent's fields, which keeps the graph debuggable.
    """
    user_question: str

    plan: Optional[PlanStep] = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    sql_result: Optional[SqlResult] = None
    draft_answer: Optional[str] = None
    critic_verdict: Optional[CriticVerdict] = None
    final_answer: Optional[str] = None

    refine_loop_count: int = 0
    trace_id: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    confidence: float
    used_sql: bool
    used_retrieval: bool
    sources: list[str] = Field(default_factory=list)
    refine_loops: int
    trace_id: Optional[str] = None
