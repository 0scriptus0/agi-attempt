#!/usr/bin/env python3
"""Lewi 1 agent API.

A small, explicit cognitive loop around the Anthropic Messages API:

    context -> memory -> model -> tools -> observations -> model -> answer

The API intentionally keeps model weights, episodic interaction logs, and
persistent memories separate.  That makes the agent easier to debug and gives
train.py clean material for later reflection/distillation.

Run locally:
    pip install fastapi uvicorn anthropic python-dotenv
    export ANTHROPIC_API_KEY=...
    python api.py
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
from pydantic import BaseModel, Field

try:
    import anthropic
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pip install anthropic") from exc

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
LEARNING_CURVE = ROOT / "learning_curve"
INTERACTIONS_DIR = LEARNING_CURVE / "interactions"
MEMORY_DIR = LEARNING_CURVE / "memory"
MEMORY_FILE = MEMORY_DIR / "memories.jsonl"

for directory in (WORKSPACE, INTERACTIONS_DIR, MEMORY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("LEWI_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("LEWI_MAX_TOKENS", "4096"))
MAX_AGENT_STEPS = int(os.environ.get("LEWI_MAX_AGENT_STEPS", "8"))
MEMORY_RESULTS = int(os.environ.get("LEWI_MEMORY_RESULTS", "8"))

FEATURE_FLAGS = {
    "web_search": os.environ.get("LEWI_TOOL_WEB_SEARCH", "1") == "1",
    "code_exec": os.environ.get("LEWI_TOOL_CODE_EXEC", "1") == "1",
    "file_io": os.environ.get("LEWI_TOOL_FILE_IO", "1") == "1",
    "memory": os.environ.get("LEWI_TOOL_MEMORY", "1") == "1",
}

client = anthropic.Anthropic()
app = FastAPI(title="Lewi 1 agent API", version="2.0")

# The frontend is local by default. Set LEWI_ALLOWED_ORIGIN to the exact
# frontend origin before exposing the API outside localhost.
allowed_origin = os.environ.get("LEWI_ALLOWED_ORIGIN", "http://localhost:8000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


def _safe_workspace_path(rel_path: str) -> Path:
    """Resolve a user/model path while preventing workspace escape."""
    candidate = (WORKSPACE / rel_path).resolve()
    workspace = WORKSPACE.resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"path escapes workspace: {rel_path}")
    return candidate


def tool_code_exec(input: dict[str, Any]) -> str:
    """Run short Python in the project workspace with a hard timeout.

    This is an execution convenience, not a security sandbox. Do not expose
    the endpoint to untrusted users without putting the subprocess in a real
    OS/container sandbox.
    """
    code = str(input.get("code", ""))
    if not code.strip():
        return json.dumps({"error": "code must be non-empty"})
    try:
        timeout = max(1, min(int(input.get("timeout_seconds", 10)), 30))
    except (TypeError, ValueError):
        timeout = 10

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=WORKSPACE, delete=False, encoding="utf-8"
        ) as handle:
            handle.write(code)
            script_path = Path(handle.name)

        result = subprocess.run(
            ["python3", str(script_path)],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.dumps(
            {
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
                "exit_code": result.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"execution exceeded {timeout}s timeout"})
    except Exception as exc:
        return json.dumps({"error": f"execution failed: {exc}"})
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)


def tool_file_read(input: dict[str, Any]) -> str:
    try:
        path = _safe_workspace_path(str(input["path"]))
        if not path.is_file():
            return json.dumps({"error": "file not found"})
        return json.dumps({"path": str(path.relative_to(WORKSPACE)), "content": path.read_text(encoding="utf-8")[:20000]})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def tool_file_write(input: dict[str, Any]) -> str:
    try:
        path = _safe_workspace_path(str(input["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(input.get("content", "")), encoding="utf-8")
        return json.dumps({"ok": True, "path": str(path.relative_to(WORKSPACE))})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def tool_memory_write(input: dict[str, Any]) -> str:
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "content": str(input.get("content", "")).strip(),
        "memory_type": str(input.get("memory_type", "lesson")),
        "applicable_when": str(input.get("applicable_when", "")),
        "not_applicable_when": str(input.get("not_applicable_when", "")),
    }
    if not entry["content"]:
        return json.dumps({"error": "memory content must be non-empty"})
    with MEMORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return json.dumps({"ok": True, "stored_id": entry["id"]})


def tool_memory_search(input: dict[str, Any]) -> str:
    query = str(input.get("query", "")).strip().lower()
    if not query or not MEMORY_FILE.exists():
        return json.dumps({"results": []})

    terms = [term for term in query.split() if term]
    hits: list[tuple[int, dict[str, Any]]] = []
    with MEMORY_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            haystack = " ".join(
                str(entry.get(field, ""))
                for field in ("content", "memory_type", "applicable_when", "not_applicable_when")
            ).lower()
            score = sum(haystack.count(term) for term in terms)
            if score:
                hits.append((score, entry))

    hits.sort(key=lambda item: (item[0], item[1].get("timestamp", 0)), reverse=True)
    return json.dumps({"results": [entry for _, entry in hits[:MEMORY_RESULTS]]}, ensure_ascii=False)


TOOL_SPECS: dict[str, Any] = {
    "web_search": {"type": "web_search_20250305", "name": "web_search"},
    "code_exec": {
        "name": "code_exec",
        "description": "Run short Python for calculations, data work, and verification. This is not a security sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["code"],
        },
    },
    "file_io": [
        {
            "name": "file_read",
            "description": "Read a UTF-8 text file inside the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "file_write",
            "description": "Write a UTF-8 text file inside the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    ],
    "memory": [
        {
            "name": "memory_write",
            "description": "Store a durable lesson that may be useful in future sessions.",
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
            "description": "Retrieve relevant stored memories using simple term scoring.",
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


def active_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if FEATURE_FLAGS["web_search"]:
        tools.append(TOOL_SPECS["web_search"])
    if FEATURE_FLAGS["code_exec"]:
        tools.append(TOOL_SPECS["code_exec"])
    if FEATURE_FLAGS["file_io"]:
        tools.extend(TOOL_SPECS["file_io"])
    if FEATURE_FLAGS["memory"]:
        tools.extend(TOOL_SPECS["memory"])
    return tools


SYSTEM_PROMPT = """You are Lewi 1, a general engineer, researcher, and agentic problem solver.

Use the loop deliberately:
1. Understand the user's goal and constraints.
2. Retrieve memory when prior experience may matter.
3. Form a concrete plan before expensive or multi-step work.
4. Use tools when they materially improve reliability.
5. Inspect tool results rather than assuming success.
6. Recover from failures by changing the approach.
7. Give a concise final answer with uncertainty stated honestly.

Do not claim that a tool succeeded unless its result supports that claim.
Treat stored memories as fallible experience: apply them only when they fit.
"""


def retrieve_memory_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrieve a small memory set before planning, without forcing memory on every turn."""
    if not FEATURE_FLAGS["memory"] or not MEMORY_FILE.exists():
        return []
    user_text = " ".join(
        str(message.get("content", ""))
        for message in messages[-3:]
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ).strip()
    if not user_text:
        return []
    result = json.loads(tool_memory_search({"query": user_text}))
    return result.get("results", [])


def build_system_prompt(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return SYSTEM_PROMPT
    memory_text = "\n".join(
        f"- [{m.get('memory_type', 'lesson')}] {m.get('content', '')} "
        f"(applies: {m.get('applicable_when', 'unspecified')})"
        for m in memories
    )
    return SYSTEM_PROMPT + "\nRelevant prior experience:\n" + memory_text


def run_agent_loop(messages: list[dict[str, Any]]) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    working_messages = list(messages)
    memories = retrieve_memory_context(working_messages)
    system_prompt = build_system_prompt(memories)

    for step in range(1, MAX_AGENT_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=working_messages,
            tools=active_tools() or anthropic.NOT_GIVEN,
        )

        # Server-side tools such as web_search are executed by Anthropic and
        # return their result in response.content. Only client-side tool_use
        # blocks need a locally generated tool_result message.
        local_tool_uses = [
            block for block in response.content
            if block.type == "tool_use" and block.name in TOOL_IMPLS
        ]

        if not local_tool_uses:
            text = "\n".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            return {
                "final_text": text,
                "trace": trace,
                "steps": step,
                "stop_reason": response.stop_reason,
                "memories_used": memories,
            }

        working_messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool_use in local_tool_uses:
            started = time.time()
            try:
                result = TOOL_IMPLS[tool_use.name](tool_use.input)
            except Exception as exc:  # keep one bad tool from killing the loop
                result = json.dumps({"error": f"tool exception: {exc}"})
            trace.append(
                {
                    "step": step,
                    "tool": tool_use.name,
                    "input": tool_use.input,
                    "result": result,
                    "duration_ms": round((time.time() - started) * 1000, 2),
                }
            )
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": result}
            )
        working_messages.append({"role": "user", "content": tool_results})

    return {
        "final_text": "I reached the agent step limit before producing a final answer.",
        "trace": trace,
        "steps": MAX_AGENT_STEPS,
        "stop_reason": "max_steps",
        "memories_used": memories,
    }


def log_interaction(request_messages: list[dict[str, Any]], result: dict[str, Any]) -> None:
    log_path = INTERACTIONS_DIR / f"{date.today().isoformat()}.jsonl"
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "model": MODEL,
        "feature_flags": FEATURE_FLAGS,
        "messages_in": request_messages,
        "memories_used": result.get("memories_used", []),
        "trace": result["trace"],
        "final_text": result["final_text"],
        "steps": result["steps"],
        "stop_reason": result["stop_reason"],
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.get("/config")
def get_config() -> dict[str, Any]:
    return {
        "model": MODEL,
        "feature_flags": FEATURE_FLAGS,
        "max_agent_steps": MAX_AGENT_STEPS,
        "memory_results": MEMORY_RESULTS,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    request_messages = [message.model_dump() for message in req.messages]
    try:
        result = run_agent_loop(request_messages)
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc
    log_interaction(request_messages, result)
    return {
        "reply": result["final_text"],
        "trace": result["trace"],
        "steps": result["steps"],
        "stop_reason": result["stop_reason"],
        "memories_used": result.get("memories_used", []),
    }


@app.get("/memory")
def list_memory() -> dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {"memories": []}
    memories = []
    with MEMORY_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {"memories": memories}


if __name__ == "__main__":
    import uvicorn

    print(f"Lewi 1 API — model={MODEL} flags={FEATURE_FLAGS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
