# What Building This Taught Me

## On multi-agent design

Splitting a task into agents is only worth the added latency and complexity
when the agents genuinely need different *vantage points*, not just different
prompts. The Planner/Retrieval/Reasoning split is mostly about efficiency
(don't do retrieval you don't need) — but the Reasoning/Critic split is about
something more fundamental: a model can't reliably grade its own single-pass
output, because generation and evaluation share the same context and the same
blind spots. Making the Critic a genuinely separate call, fed only the
evidence and the draft (not the reasoning trace that produced it), was the
detail that actually made the self-correction loop catch real issues instead
of just adding latency for nothing.

## On hallucination-aware systems specifically

The instinct is to try to prevent hallucination entirely with a clever
prompt. That doesn't work reliably. What worked better was accepting that the
first draft *will* sometimes be wrong, and building a pipeline that expects
that and has a defined recovery path — retry with specific feedback, then
fail honestly if retries are exhausted. "The system can say 'I don't know'"
turned out to be a more important design goal than "the system is always
right."

## On config-driven architecture

Building the LLM/embeddings provider switch as an abstraction from day one
(rather than hardcoding Ollama and refactoring later) felt like overhead
early on, but it meant testing against a hosted model was a one-line config
change, not a code change. The general lesson: anything that's a near-certain
axis of change (which model provider, which environment, which database)
deserves a config seam even before you need the second option — it's cheap to
build in early and expensive to retrofit.

## On SQL-generation safety

Trusting an LLM to write SQL and just running it is a real risk, not a
theoretical one — it's easy to forget the model can output anything, including
syntactically valid destructive statements. Building the read-only guard
forced me to think about the difference between "the model was well-behaved
in my testing" and "the system is actually safe regardless of what the model
outputs" — those are different claims, and only the second one is worth
making about a production system.

## On observability

Adding structured JSON logging and LangSmith tracing felt like polish at
first, but debugging the refine-loop behavior (why did this question take 2
retries?) was genuinely difficult without being able to see, per node: what
the Planner decided, what got retrieved, what SQL ran, and what the Critic's
specific complaint was. Observability wasn't optional once the pipeline had
more than one possible path through it.

## What I'd do differently next time

- Use a distinct (and ideally cheaper/faster) model specifically for the
  Critic role from the start, instead of defaulting to the same model as
  Reasoning — would reduce both cost and correlated blind spots.
- Add a proper SQL statement parser instead of a prefix-string check, even
  for the demo — it's not much more work and it's a more honest safety claim.
- Instrument token usage and per-node latency as first-class structured log
  fields from the beginning, not added after the fact.
