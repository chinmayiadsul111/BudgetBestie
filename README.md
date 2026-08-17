# 💸 BudgetBestie

**A self-reflective, hallucination-aware multi-agent finance assistant.**
Built with LangGraph, FastAPI, FAISS, and Ollama — swappable to Azure
OpenAI/OpenAI with a one-line config change.

> Gen Z gets a lot of financial advice from TikTok and none of it is
> fact-checked. BudgetBestie answers real money questions — grounded in your
> actual transaction history and vetted guidance docs — and it's built to
> **say "I'm not sure" instead of confidently guessing wrong.**

---

## What it actually does

Ask it things like:
- *"How much did I spend on food delivery this month?"*
- *"Am I following the 50/30/20 rule based on my spending?"*
- *"Should I pay off my credit card or build my emergency fund first?"*

Under the hood, a 5-agent LangGraph pipeline decides what evidence it needs,
gathers it, drafts an answer, **critiques its own draft against that
evidence**, and either ships the answer or loops back to try again — up to a
configured retry limit, after which it honestly says it isn't confident
rather than fabricating a number.

## Architecture

```
User question
     │
     ▼
 ┌─────────┐     ┌───────────┐     ┌───────────┐     ┌─────────┐     ┌─────────┐
 │ Planner │ ──▶ │ Retrieval │ ──▶ │ Reasoning │ ──▶ │ Critic  │ ──▶ │ Refiner │ ──▶ answer
 └─────────┘     └───────────┘     └───────────┘     └────┬────┘     └─────────┘
                                          ▲                │
                                          └── rejected ────┘
                                       (feedback attached, capped retries)
```

| Agent | Job |
|---|---|
| **Planner** | Decides if the question needs a SQL lookup, a docs lookup, both, or neither |
| **Retrieval** | FAISS similarity search over financial guidance docs (budgeting rules, credit basics, debt payoff strategy) |
| **Reasoning** | Generates & runs a read-only SQL query against the user's transaction history; drafts an answer from whatever evidence is available |
| **Critic** | Checks every number and every piece of advice in the draft against the actual evidence provided — rejects unsupported claims with a confidence score |
| **Refiner** | Polishes an approved answer, or returns an honest "not confident enough" response if retries are exhausted |

## Why this matters (the real engineering problem)

Financial advice is a domain where a fluent, confident, *wrong* answer is
actively harmful — not just embarrassing. The Critic/Refiner loop exists
specifically to make hallucination a **first-class failure mode the system
checks for**, not an assumed-away edge case. See
[`docs/ENGINEERING_DECISIONS.md`](docs/ENGINEERING_DECISIONS.md) for the
alternatives considered and why this shape won.

## Production-grade details

- **GPU/CPU auto-detection** (`src/device_utils.py`) — probes `torch.cuda`
  at startup, falls back to CPU cleanly if unavailable, fully overridable in
  config.
- **Structured JSON logging** (`src/logging_setup.py`) — every agent
  decision, SQL query, retrieval miss, and Critic rejection is logged as a
  JSON line to a rotating log file, plus human-readable console output.
- **LangSmith tracing** (`src/observability.py`) — full graph execution
  traces, degrades gracefully to "no tracing" if no API key is set rather
  than crashing.
- **Provider-agnostic LLM/embeddings layer** — swap Ollama → Azure OpenAI →
  OpenAI by changing one field in `config.yaml`. No code touches a specific
  provider SDK outside `llm_provider.py` / `embeddings_provider.py`.
- **SQL safety guard** — the Reasoning agent's generated SQL is only ever
  executed as a read-only `SELECT`; anything else is rejected before it
  touches the database.
- **Config/secrets separation** — `config.yaml` holds structure and
  defaults (git-tracked), `.env` holds secrets (git-ignored).

## Tech stack

`LangGraph` · `LangChain` · `FastAPI` · `FAISS` · `Ollama` / `Azure OpenAI` /
`OpenAI` · `SQLAlchemy` (SQLite, Postgres-ready) · `Pydantic` · `LangSmith`

## Quickstart

```
pip install -r requirements.txt
cp config/.env.example .env
ollama pull llama3.1:8b && ollama pull nomic-embed-text
python demo.py
```

Full command reference: [`COMMANDS.md`](COMMANDS.md)

## Project docs

- [`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md) — what this does and why, in plain language
- [`docs/ENGINEERING_DECISIONS.md`](docs/ENGINEERING_DECISIONS.md) — design choices and alternatives considered
- [`docs/LEARNINGS.md`](docs/LEARNINGS.md) — what building this taught me
- [`INTERVIEW_PREP.md`](INTERVIEW_PREP.md) — technical Q&A prep for this project

## Project structure

```
budgetbestie/
├── config/              # all settings — never hardcode config in src/
├── src/
│   ├── agents/           # the 5 graph nodes
│   ├── config_loader.py
│   ├── device_utils.py
│   ├── logging_setup.py
│   ├── observability.py
│   ├── llm_provider.py
│   ├── embeddings_provider.py
│   ├── vectorstore.py
│   ├── db.py
│   ├── graph.py          # LangGraph wiring
│   ├── api.py            # FastAPI app
│   └── schemas.py
├── data/sop_docs/        # financial guidance knowledge base (RAG source)
├── run.py                # start the API server
├── demo.py                # interactive terminal demo
└── docs/, INTERVIEW_PREP.md, COMMANDS.md
```

## Disclaimer

This is a portfolio/demo project. It is not financial advice software and
the sample data is synthetic.
