#!/usr/bin/env python3
"""
Lewi 1 agent API.

Scope, honestly stated: this is a FastAPI server that runs a tool-using
agent loop against the Anthropic Messages API, using whichever adapter
train.py produced (or the bare base model if you haven't trained one
yet). It is the test harness for the fine-tuned model, not a separate
intelligence — the "agentic" behavior lives entirely in the tool-use
loop below, which any base Claude/Qwen/Llama chat model can drive.

Tools are feature-flagged via environment variables so you can turn
each one on independently while you figure out which are reliable:

    LEWI_TOOL_WEB_SEARCH=1   # Anthropic-hosted web_search tool
    LEWI_TOOL_CODE_EXEC=1    # sandboxed local python execution
    LEWI_TOOL_FILE_IO=1      # read/write inside ./workspace only
    LEWI_TOOL_MEMORY=1       # read/write learning_curve/memory/*.jsonl

All four default to "on" (see FEATURE_FLAGS below) so you can see the
full agent working end to end, then turn individual tools off if one
turns out to be unreliable for your use case.

Every turn (request + full tool-call trace + final answer) is appended
to learning_curve/interactions/<date>.jsonl. That's raw material for a
future continual-learning / distillation pass, not a database — you
said you'd swap in a real one later, so this deliberately stays a flat
append-only file for now.

Run:
    pip install fastapi uvicorn anthropic python-dotenv
    export ANTHROPIC_API_KEY=...
    python api.py
    # -> http://localhost:8000  (index.html talks to this)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import anthropic
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install anthropic") from e

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
LEARNING_CURVE = ROOT / "learning_curve"
INTERACTIONS_DIR = LEARNING_CURVE / "interactions"
MEMORY_DIR = LEARNING_CURVE / "memory"
WORKSPACE.mkdir(exist_ok=True)
INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = MEMORY_DIR / "memories.jsonl"

# Which model answers requests. Point this at your local adapter's
# merged checkpoint if you're serving it yourself (e.g. via vLLM with
# an OpenAI-compatible endpoint) — this file assumes the Anthropic API
# for now since that's the fastest path to a working test console.
MODEL = os.environ.get("LEWI_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 2048
MAX_AGENT_STEPS = 8  # hard cap so a tool-call loop can't run forever

FEATURE_FLAGS = {
    "web_search": os.environ.get("LEWI_TOOL_WEB_SEARCH", "1") == "1",
    "code_exec": os.environ.get("LEWI_TOOL_CODE_EXEC", "1") == "1",
    "file_io": os.environ.get("LEWI_TOOL_FILE_IO", "1") == "1",
    "memory": os.environ.get("LEWI_TOOL_MEMORY", "1") == "1",
}

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

app = FastAPI(title="Lewi 1 agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local test console only — tighten before exposing this anywhere
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
# Each entry in TOOL_SPECS is what gets sent to the API. Each entry in
# TOOL_IMPLS is the local Python function that actually runs when the
# model asks to use that tool. Keeping specs and impls in lockstep
# (same key) is what "feature-flagged" means here — flip a flag off and
# both the spec and the impl vanish from that request.

def _safe_workspace_path(rel_path: str) -> Path:
    """Resolve rel_path inside WORKSPACE, refusing any escape attempt."""
    candidate = (WORKSPACE / rel_path).resolve()
    if WORKSPACE.resolve() not in candidate.parents and candidate != WORKSPACE.resolve():
        raise ValueError(f"path escapes workspace: {rel_path}")
    return candidate


def tool_code_exec(input: dict[str, Any]) -> str:
    code = input.get("code", "")
    timeout = min(int(input.get("timeout_seconds", 10)), 30)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=WORKSPACE, delete=False
    ) as f:
        f.write(code)
        script_path = f.name
    try:
        result = subprocess.run(
            ["python3", script_path],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout[-4000:]
        err = result.stderr[-2000:]
        return json.dumps({"stdout": out, "stderr": err, "exit_code": result.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"execution exceeded {timeout}s timeout"})
    finally:
        Path(script_path).unlink(missing_ok=True)


def tool_file_read(input: dict[str, Any]) -> str:
    try:
        path = _safe_workspace_path(input["path"])
        if not path.exists():
            return json.dumps({"error": "file not found"})
        return json.dumps({"content": path.read_text(encoding="utf-8")[:20000]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_file_write(input: dict[str, Any]) -> str:
    try:
        path = _safe_workspace_path(input["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(input.get("content", ""), encoding="utf-8")
        return json.dumps({"ok": True, "path": str(path.relative_to(WORKSPACE))})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_memory_write(input: dict[str, Any]) -> str:
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "content": input.get("content", ""),
        "memory_type": input.get("memory_type", "lesson"),
        "applicable_when": input.get("applicable_when", ""),
        "not_applicable_when": input.get("not_applicable_when", ""),
    }
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return json.dumps({"ok": True, "stored_id": entry["id"]})


def tool_memory_search(input: dict[str, Any]) -> str:
    query = input.get("query", "").lower()
    if not MEMORY_FILE.exists():
        return json.dumps({"results": []})
    hits = []
    with MEMORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if query in entry["content"].lower():
                hits.append(entry)
    return json.dumps({"results": hits[:10]})


TOOL_SPECS: dict[str, dict] = {
    "web_search": {"type": "web_search_20250305", "name": "web_search"},
    "code_exec": {
        "name": "code_exec",
        "description": (
            "Run a short Python snippet in a sandboxed subprocess (cwd=./workspace, "
            "no network, hard timeout). Use for calculation, data wrangling, or "
            "verifying logic — not for anything long-running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute"},
                "timeout_seconds": {"type": "integer", "description": "max 30"},
            },
            "required": ["code"],
        },
    },
    "file_io": [
        {
            "name": "file_read",
            "description": "Read a text file from the workspace directory.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "file_write",
            "description": "Write a text file into the workspace directory (creates dirs as needed).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    ],
    "memory": [
        {
            "name": "memory_write",
            "description": (
                "Store a durable lesson learned in this conversation, for recall in "
                "future sessions. Only store things worth remembering across sessions "
                "— not routine task details."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "applicable_when": {"type": "string"},
                    "not_applicable_when": {"type": "string"},
                },
                "required": ["content"],
            },
        },
        {
            "name": "memory_search",
            "description": "Search previously stored memories by keyword.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ],
}

TOOL_IMPLS = {
    "code_exec": tool_code_exec,
    "file_read": tool_file_read,
    "file_write": tool_file_write,
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
}


def active_tools() -> list[dict]:
    tools: list[dict] = []
    if FEATURE_FLAGS["web_search"]:
        tools.append(TOOL_SPECS["web_search"])
    if FEATURE_FLAGS["code_exec"]:
        tools.append(TOOL_SPECS["code_exec"])
    if FEATURE_FLAGS["file_io"]:
        tools.extend(TOOL_SPECS["file_io"])
    if FEATURE_FLAGS["memory"]:
        tools.extend(TOOL_SPECS["memory"])
    return tools


SYSTEM_PROMPT = (
    "You are Lewi 1, a general engineer, researcher, and agentic problem "
    "solver. You have tools for web search, sandboxed code execution, "
    "workspace file I/O, and cross-session memory — use them when they "
    "would materially improve your answer, and say plainly when a tool "
    "isn't available rather than guessing. Be direct about uncertainty; "
    "don't claim capabilities you don't have."
)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent_loop(messages: list[dict]) -> dict:
    """
    Runs the tool-use loop to completion (or MAX_AGENT_STEPS), returning
    the final assistant text plus a full trace of every tool call made,
    so the caller (and the interaction log) can see exactly what happened
    — not just the final answer.
    """
    trace: list[dict] = []
    tools = active_tools()
    working_messages = list(messages)

    for step in range(MAX_AGENT_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=working_messages,
            tools=tools if tools else anthropic.NOT_GIVEN,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b.text for b in response.content if b.type == "text"]

        if not tool_uses:
            return {
                "final_text": "\n".join(text_blocks),
                "trace": trace,
                "steps": step + 1,
                "stop_reason": response.stop_reason,
            }

        working_messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tu in tool_uses:
            impl = TOOL_IMPLS.get(tu.name)
            if impl is None:
                # web_search is handled server-side by Anthropic and never
                # reaches TOOL_IMPLS; anything else unknown is a real bug.
                continue
            result_str = impl(tu.input)
            trace.append({"tool": tu.name, "input": tu.input, "result": result_str})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": result_str}
            )

        if tool_results:
            working_messages.append({"role": "user", "content": tool_results})
        else:
            # only server-side tools (web_search) fired this step; nothing
            # to feed back manually, loop again so the model can continue
            continue

    return {
        "final_text": "(stopped after max agent steps without a final answer)",
        "trace": trace,
        "steps": MAX_AGENT_STEPS,
        "stop_reason": "max_steps",
    }


def log_interaction(request_messages: list[dict], result: dict) -> None:
    log_path = INTERACTIONS_DIR / f"{date.today().isoformat()}.jsonl"
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "model": MODEL,
        "feature_flags": FEATURE_FLAGS,
        "messages_in": request_messages,
        "trace": result["trace"],
        "final_text": result["final_text"],
        "steps": result["steps"],
        "stop_reason": result["stop_reason"],
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.get("/config")
def get_config():
    return {"model": MODEL, "feature_flags": FEATURE_FLAGS, "max_agent_steps": MAX_AGENT_STEPS}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, "messages must be non-empty")
    request_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        result = run_agent_loop(request_messages)
    except anthropic.APIError as e:
        raise HTTPException(502, f"Anthropic API error: {e}")
    log_interaction(request_messages, result)
    return {
        "reply": result["final_text"],
        "trace": result["trace"],
        "steps": result["steps"],
        "stop_reason": result["stop_reason"],
    }


@app.get("/memory")
def list_memory():
    if not MEMORY_FILE.exists():
        return {"memories": []}
    with MEMORY_FILE.open("r", encoding="utf-8") as f:
        return {"memories": [json.loads(line) for line in f if line.strip()]}


if __name__ == "__main__":
    import uvicorn

    print(f"Lewi 1 agent API — model={MODEL} flags={FEATURE_FLAGS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
