# Commands

Every operation in this project is a plain `python <file>.py` call — no CLI
flags, no argparse. All configuration lives in `config/config.yaml` and `.env`.

## 1. Setup (run once)

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config/.env.example .env
```

Then open `.env` and fill in whichever provider you're using (see
`config/config.yaml` → `llm.provider`).

## 2. Pull local models (only if using Ollama — the default)

Ollama itself isn't a Python command, but it's the one external step needed:

```
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Ollama must be running (`ollama serve`, or the desktop app) before you start
the app.

## 3. Seed the demo database

Creates and populates `data/transactions.db` with ~4 months of realistic
sample spending data. This also runs automatically on first API startup, but
you can run it standalone:

```
python -m src.db
```

## 4. Run the interactive terminal demo (no server, fastest way to try it)

```
python demo.py
```

Type a question, see the answer plus a trace line showing which agents fired
and the Critic's confidence score. Try:
- "How much did I spend on food delivery this month?"
- "Am I following the 50/30/20 rule based on my spending?"
- "Should I pay off my credit card or build my emergency fund first?"

## 5. Run the API server

```
python run.py
```

Starts FastAPI + Uvicorn on the host/port set in `config/config.yaml`
(`server.host`, `server.port` — default `0.0.0.0:8000`).

## 6. Call the API (from a second terminal)

Still no CLI tool needed — a tiny Python snippet instead of curl:

```
python -c "
import requests
r = requests.post('http://localhost:8000/ask', json={'question': 'How much did I spend on subscriptions this month?'})
print(r.json())
"
```

Or visit `http://localhost:8000/docs` for the interactive Swagger UI.

## 7. Rebuild the FAISS index from scratch

Normally the index is built once and cached at `data/faiss_index/`. To force
a rebuild after editing the SOP docs in `data/sop_docs/`:

```
python -c "from src.vectorstore import build_or_load_index; build_or_load_index(force_rebuild=True)"
```

## 8. Switch from local (Ollama) to a hosted API model

No code changes. Edit `config/config.yaml`:

```yaml
llm:
  provider: "azure_openai"   # was "ollama"
```

Fill in the matching credentials in `.env` (see `config/.env.example`), then
re-run `python run.py` or `python demo.py`.

## 9. View structured logs

Every run appends JSON log lines to `logs/budgetbestie.log.json`:

```
python -c "
import json
with open('logs/budgetbestie.log.json') as f:
    for line in f.readlines()[-10:]:
        print(json.dumps(json.loads(line), indent=2))
"
```

## 10. Enable LangSmith tracing

Set in `.env`:
```
LANGCHAIN_API_KEY=your-key-here
```
`observability.langsmith.enabled: true` is already the default in
`config.yaml`. Traces appear at https://smith.langchain.com under the
`budgetbestie` project.
