"""
Interactive terminal demo — runs the agent graph directly, no server, no curl.

    python demo.py

Type a question, see the pipeline's answer plus a trace of what each agent
decided. Type 'exit' to quit.
"""
from src.db import ensure_schema_and_seed
from src.graph import get_graph
from src.observability import init_observability
from src.schemas import GraphState

SAMPLE_QUESTIONS = [
    "How much did I spend on food delivery this month?",
    "Am I following the 50/30/20 rule based on my spending?",
    "Should I pay off my credit card or build my emergency fund first?",
    "What's my current credit utilization based on my transactions?",
]


def main() -> None:
    init_observability()
    ensure_schema_and_seed()
    graph = get_graph()

    print("=" * 70)
    print("BudgetBestie — self-reflective finance assistant (terminal demo)")
    print("Sample questions you can try:")
    for q in SAMPLE_QUESTIONS:
        print(f"  - {q}")
    print("Type 'exit' to quit.")
    print("=" * 70)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        state = GraphState(user_question=question)
        result = graph.invoke(state)

        plan = result.get("plan")
        verdict = result.get("critic_verdict")

        print(f"\nBudgetBestie: {result.get('final_answer')}")
        print(
            f"[trace] needs_sql={plan.needs_sql if plan else None} "
            f"needs_retrieval={plan.needs_retrieval if plan else None} "
            f"confidence={verdict.confidence if verdict else None} "
            f"refine_loops={result.get('refine_loop_count', 0)}"
        )


if __name__ == "__main__":
    main()
