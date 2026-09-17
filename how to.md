# Lewi 1 — training + agent test harness

Three pieces, in the order you'd actually use them:

1. **`train.py`** — QLoRA SFT (and optional DPO) on your cleaned datasets.
   Needs a GPU (written for a 24GB card doing QLoRA on a ~7-8B model;
   workable on 12GB with smaller batch/seq settings). Reads
   `configs/train.yaml` and your `datasets/` folder — those aren't
   included in this delivery since I don't have your repo state; drop
   `train.py` into the project you already have and it'll pick them up.

2. **`api.py`** — a FastAPI server that runs the actual tool-using agent
   loop, feature-flagged so you can test each tool independently:
   - `web_search` — Anthropic-hosted web search
   - `code_exec` — sandboxed local Python (cwd-locked, timeout-capped, no network)
   - `file_io` — read/write, locked to `./workspace/`
   - `memory` — write/search durable lessons in `learning_curve/memory/memories.jsonl`

   It calls the Anthropic API directly (`ANTHROPIC_API_KEY` env var) so
   you have a working agent loop today, independent of whether your
   fine-tune has finished. Once you have a merged checkpoint you're
   serving locally (e.g. via vLLM with an OpenAI-compatible endpoint),
   swap the `client.messages.create(...)` call for that endpoint — the
   rest of the loop (tool dispatch, tracing, logging) doesn't change.

3. **`index.html`** — a local test console: chat on the left, a live
   tool-call trace on the right so you can see exactly what the agent
   did on every turn, not just its final answer.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python api.py          # starts on http://localhost:8000
```

Then open `index.html` directly in a browser (double-click it, or
`open index.html`). It talks to `localhost:8000` automatically.

## `learning_curve/`

Flat, append-only JSONL for now, as you asked — swap in a real database
whenever you're ready:

- `training_runs/` — written by `train.py`: per-step loss (`<run_id>.jsonl`),
  a final summary (`<run_id>_summary.json`), and held-out eval generations
  (`<run_id>_eval_outputs.jsonl`) for you to review manually.
- `interactions/` — written by `api.py`: one file per day, one JSON line
  per chat turn, including the full tool-call trace. This is the raw
  material for a future continual-learning pass (e.g. distilling good
  interactions back into SFT data) — nothing here does that automatically yet.
- `memory/memories.jsonl` — durable lessons the agent chose to store via
  the `memory_write` tool, readable via `memory_search` or the `/memory`
  endpoint.

## What "reliable agentic tools" means here, honestly

The four tools above are the standard set that makes an agent loop
actually do things instead of just talk: look things up, compute things,
persist work across turns, and remember lessons across sessions. That's
what makes it *agentic*. It doesn't make it AGI — the model is still
whatever you point `MODEL` at (Claude by default, your fine-tuned
Qwen/Mistral/Llama once it's trained and served). Worth keeping that
distinction sharp as you iterate, so "the agent didn't use a tool it
should have" and "the underlying model gave a wrong answer" don't get
diagnosed as the same problem.
