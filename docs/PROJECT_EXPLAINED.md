# Project Explained (for you, in plain language)

This doc is so YOU can explain this project cold, in your own words, without
re-reading code the night before an interview.

## The one-sentence pitch

"I built a multi-agent AI system that answers personal finance questions by
combining a text-to-SQL agent with a RAG pipeline, and added a self-critique
loop so it catches its own hallucinations before answering — instead of just
trusting whatever the first LLM call produces."

## Walk through a single request, step by step

Say the user asks: **"Am I following the 50/30/20 rule based on my spending?"**

1. **Planner** reads the question and decides: this needs BOTH a SQL lookup
   (to know the actual spending numbers) AND a retrieval lookup (to know what
   the 50/30/20 rule actually says). It outputs a small JSON object saying so.

2. **Retrieval** takes that "yes, retrieval needed" signal and does a FAISS
   similarity search over the markdown docs in `data/sop_docs/`. It finds the
   `50_30_20_rule.md` chunk and returns it with a similarity score. If nothing
   scores above the configured threshold, it returns an empty list — that's a
   meaningful signal, not a failure.

3. **Reasoning** does two things:
   - Writes a SQL query like `SELECT category, SUM(amount) ... GROUP BY
     category` against the `transactions` table, executes it (read-only), and
     gets back real numbers.
   - Combines those SQL results + the retrieved rule text into one prompt and
     asks the LLM to draft an answer.

4. **Critic** is a *separate* LLM call whose only job is to look at the draft
   answer and ask: "Does every number in this draft actually appear in the
   SQL results? Does every piece of advice trace back to the retrieved docs?"
   It outputs approved/rejected + a confidence score + specific issues.

5. **Routing after Critic**: if approved (and confidence is above the
   threshold), go to Refiner. If rejected, go BACK to Reasoning — but this
   time the prompt includes the Critic's specific complaint, so the retry
   isn't blind. This can happen up to `max_refine_loops` times (default 2)
   before the system gives up gracefully.

6. **Refiner** either polishes the tone of an approved answer, or — if all
   retries were rejected — returns an honest "I'm not confident enough to
   answer that precisely" message instead of a wrong guess.

## Why a graph and not just one big prompt?

A single "answer this finance question" prompt can't fact-check itself — the
same model call that made a mistake has no independent way to catch it.
Splitting into separate agents means the Critic step has a genuinely
different vantage point: it's given the evidence and the draft as two
separate things and asked to compare them, which is a much easier and more
reliable task than "be right the first time."

## Why SQLite instead of a real Postgres instance?

The `database.url` config is a full connection string — swapping to Postgres
in production means changing one line, SQLAlchemy handles the rest. SQLite
keeps the demo runnable by anyone who clones the repo with zero infrastructure
setup, which matters a lot for a portfolio project you want people to
actually run.

## Why Ollama as the default instead of an API key?

Same reasoning — a recruiter or interviewer cloning your repo shouldn't need
your OpenAI billing key to try it. The provider abstraction in
`llm_provider.py` means the *architecture* is identical either way; only the
config changes.

## The honest gaps (say these proactively if asked)

- The SQL-generation step trusts the LLM to write correct SQL against a small,
  fixed schema. At larger schema complexity you'd want schema-aware
  constrained generation or a validation layer, not just a read-only guard.
- The Critic uses the same underlying model family as the Reasoning agent by
  default — a stronger setup would use a different model (or at least a
  different, more adversarial prompt posture) for the Critic to reduce
  correlated blind spots.
- There's no user-specific memory across requests — each `/ask` call is
  stateless. A production version would need a memory/session layer.
