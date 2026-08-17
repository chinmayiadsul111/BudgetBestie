# Engineering Decisions

Each section: the decision, the alternatives, why this one won, and when
you'd revisit it.

---

## 1. LangGraph vs. a plain chain / manual orchestration

**Decision:** LangGraph `StateGraph` with a typed `GraphState`.

**Alternatives considered:**
- A linear LangChain `Chain`/`LCEL` pipeline (`prompt | llm | parser`).
- Hand-rolled orchestration: plain Python functions calling each other.
- A different agent framework (CrewAI, AutoGen).

**Why LangGraph won:** The self-correction loop (Critic → Reasoning) is a
*cycle*, not a straight line. LCEL chains are fundamentally DAGs — they don't
support "go back to an earlier step conditionally." LangGraph's conditional
edges (`add_conditional_edges`) are built exactly for this. CrewAI/AutoGen
are more oriented around autonomous agent-to-agent delegation with less
explicit control over state shape and routing logic; for a pipeline where I
want precise, auditable control over exactly when a loop happens and how many
times, LangGraph's explicit graph + typed state was the better fit.

**When to revisit:** If the pipeline grew to need dynamic agent selection
(not knowing in advance which agents will run, decided by an LLM at runtime)
rather than a fixed graph shape, a more autonomous framework might fit
better.

---

## 2. Separate Critic agent vs. self-critique inside the Reasoning prompt

**Decision:** Critic is a distinct graph node with its own LLM call and its
own prompt, not a "also double check yourself" instruction tacked onto the
Reasoning prompt.

**Alternatives considered:**
- Single prompt asking the model to answer AND self-rate confidence in one
  shot.
- Chain-of-thought with a "let's verify" section in the same completion.

**Why separate won:** A model asked to both answer and grade its own answer
in the same completion tends to rubber-stamp itself — there's no genuine
adversarial distance between generation and evaluation when they happen in
one autoregressive pass. Making the Critic a separate call, with a prompt
that only sees the *evidence* and the *draft* (not the reasoning that
produced it), forces an actual comparison rather than a confidence
restatement. This cost is one extra LLM call per request (latency +
inference cost) — a deliberate tradeoff of cost for reliability.

**When to revisit:** If latency becomes unacceptable for the product's use
case, a cheaper/faster model specifically for the Critic role (rather than
skipping the step) preserves the separation while cutting cost.

---

## 3. Confidence threshold enforced in code, not just trusted from the LLM

**Decision:** `critic.py` overrides `verdict.approved = False` if
`confidence < threshold`, even if the LLM's own JSON said `approved: true`.

**Why:** LLMs are inconsistent about honoring numeric thresholds inside their
own free-form judgment — asking a model to output both a boolean and a
number and expecting them to always agree with an external threshold is
asking for a bug. Enforcing the threshold as a plain `if` statement in Python
means the actual safety property doesn't depend on prompt compliance.

---

## 4. Provider abstraction (Ollama ↔ Azure OpenAI ↔ OpenAI) via config, not code branches in agents

**Decision:** `llm_provider.py` / `embeddings_provider.py` are the *only*
places that import a provider-specific SDK. Every agent calls
`get_chat_model()` and doesn't know or care which provider is behind it.

**Alternatives considered:**
- Just hardcode Ollama everywhere, switch later if needed.
- Duplicate agent files per provider.

**Why this won:** This is the standard adapter/strategy pattern applied to
LLM clients — it's the difference between a one-line config change and a
multi-file refactor when a client wants to move from local dev to a hosted
model in production (a completely realistic ask). The cost is one extra layer
of indirection, which is worth it here because "which model provider" is a
near-certain axis of change for any real deployment of this kind of system.

---

## 5. SQLite by default, Postgres-compatible by config

**Decision:** SQLAlchemy `create_engine(url)` where `url` is a plain config
string; SQLite path resolution happens in `db.py`, everything else is
dialect-agnostic SQL.

**Why:** Zero-infrastructure local demo (matches the resume's original stack,
which used PostgreSQL in production) without forcing anyone cloning the repo
to stand up a database server just to try it. The schema is deliberately
simple (one table) specifically so it behaves identically on both engines —
no SQLite-specific or Postgres-specific SQL features are used anywhere in the
generated queries.

---

## 6. Read-only SQL guard as a string check, not a sandboxed DB user

**Decision:** `run_readonly_sql()` rejects any statement not starting with
`select` (case-insensitive) before execution.

**Alternatives considered:**
- A dedicated read-only database role/user with revoked write grants
  (the "real" production answer).
- A full SQL parser (e.g. `sqlglot`) to validate statement type properly
  instead of a string prefix check.

**Why the simple version, here:** For a demo project on SQLite, DB-level
read-only roles aren't available the way they are in Postgres, and a full SQL
parser is more machinery than the actual risk surface justifies at this
scale. The honest limitation: a prefix check can be fooled by adversarial
input in ways a real parser or DB-level permission can't (e.g. multi-statement
strings). **In production, this would be backed by a database-level read-only
role as defense in depth, not just an application-level string check** —
worth saying explicitly if asked, rather than presenting the check as
bulletproof.

---

## 7. Structured JSON logging (custom formatter) vs. a logging-as-a-service SDK

**Decision:** A custom `JsonFormatter` + `RotatingFileHandler`, no external
logging service SDK.

**Why:** Keeps the project runnable with zero external accounts/API keys
beyond the LLM provider itself, while still producing genuinely structured,
machine-parseable logs (one JSON object per line — trivially ingestible by
Datadog/ELK/CloudWatch later without changing the emission code, only the
handler). LangSmith is used separately for LLM-specific tracing (prompts,
token counts, latencies per node) — the JSON logs are for application-level
events (SQL blocked, retrieval empty, critic rejected, device fallback).

---

## 8. GPU/CPU auto-detection with graceful fallback vs. failing fast

**Decision:** If `torch.cuda.is_available()` probing throws for any reason
(no torch installed, broken driver, etc.), fall back to CPU rather than
crashing — but log a `WARNING`, not silently.

**Why:** This project's actual GPU-bound work (embeddings) is small enough
that CPU is a perfectly usable fallback for a demo, so failing the whole
startup over a missing GPU would be the wrong tradeoff here. This mirrors the
original production system's real constraint differently — there, GPU
availability was a hard requirement (real-time inference SLA) and the correct
behavior would be the opposite: fail fast and loud rather than silently
degrading to CPU and missing a latency target.
