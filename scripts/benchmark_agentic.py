#!/usr/bin/env python3
"""Behavioral benchmark for Lewi's interactive agent loop.

Unlike the original benchmark, this scores transitions between actions and
observations, not merely whether an action appeared somewhere in a trace.
It is deliberately heuristic: the result is a regression signal, not an AGI
score.
"""
import json, os, re, time
from pathlib import Path
from urllib.request import Request, urlopen

API_URL = os.environ.get("LEWI_API_URL", "http://127.0.0.1:8000/chat")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "learning_curve" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("T01", "adaptive_compute", "What is 37 + 58? Answer directly. Do not use a tool.", {"required": ["final"], "forbidden": ["tool"]}),
    ("T02", "tool_selection", "Use the code execution tool to calculate 847 * 913. Verify the result from the tool observation, then give me the final answer.", {"required": ["code_exec", "verify", "final"]}),
    ("T03", "tool_selection", "Inspect the workspace and identify project files relevant to training. Do not guess filenames. Discover what exists, read relevant files, and summarize verified facts.", {"required": ["file_tool", "final"]}),
    ("T04", "recovery", "Try to read definitely_missing_benchmark_file.txt. When that fails, do not blindly repeat it. Use the error to choose a different strategy, discover the workspace, and report what you found.", {"required": ["file_tool", "recovery", "final"]}),
    ("T05", "memory", "Search memory for reusable guidance relevant to this project and Transformers training. Use relevant retrieved memory if present and distinguish it from facts you verified.", {"required": ["memory_search", "final"]}),
    ("T06", "memory", "Record a reusable memory that this benchmark tests whether observations change subsequent actions, and repeated tool failures should trigger recovery or replanning. Include evidence and applicability conditions. Then confirm what you stored.", {"required": ["memory_write", "final"]}),
    ("T07", "verification", "Create benchmark_verification.txt containing exactly LEWI_VERIFY_7319. Then verify by reading it back. Only claim success if readback confirms exact content.", {"required": ["file_write", "verify", "final"]}),
    ("T08", "replanning", "Find a project architecture document. If your first assumed filename is absent, update your plan from the observation instead of repeating it. Read an existing relevant document and summarize verified facts.", {"required": ["file_tool", "recovery", "replan", "final"]}),
    ("T09", "reasoning", "Tell me what you can verify from the workspace about the model used by this project. If something cannot be verified, explicitly say so rather than inventing it.", {"required": ["file_tool", "evidence", "final"]}),
    ("T10", "agent_loop", "Inspect the project, identify one concrete training improvement supported by what you find, save it as structured reusable memory with evidence and applicability, verify the write if possible, then summarize.", {"required": ["file_tool", "memory_write", "verify", "final"]}),
]
ACTIONS = {"memory_search", "memory_write", "code_exec", "file_read", "file_write", "workspace_list", "final"}
TOOL_ACTIONS = ACTIONS - {"final"}


def call(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
    req = Request(API_URL, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())


def steps(resp):
    x = resp.get("trace") or resp.get("steps") or []
    if isinstance(x, dict): x = x.get("events") or x.get("steps") or []
    return x if isinstance(x, list) else []


def normalize(st):
    """Return ordered events with action, observation, and nearby text."""
    out = []
    pending = None
    for s in st:
        if not isinstance(s, dict): continue
        action = None
        for k in ("action", "tool", "name"):
            if isinstance(s.get(k), str) and s[k] in ACTIONS: action = s[k]; break
        for k in ("model_action", "assistant_action"):
            if isinstance(s.get(k), dict) and s[k].get("action") in ACTIONS:
                action = s[k]["action"]; break
        text = " ".join(str(s.get(k, "")) for k in ("content", "reason", "plan", "output") if s.get(k) is not None)
        observation = " ".join(str(s.get(k, "")) for k in ("observation", "result", "tool_result", "error") if s.get(k) is not None)
        if action:
            out.append({"action": action, "text": text.lower(), "observation": observation.lower()})
            pending = out[-1]
        elif observation and pending:
            pending["observation"] += " " + observation.lower()
        elif text and out:
            out[-1]["text"] += " " + text.lower()
    return out


def output(resp):
    for k in ("reply", "response", "output", "message"):
        if isinstance(resp.get(k), str): return resp[k]
    return json.dumps(resp, ensure_ascii=False)


def has_failure(e):
    s = e["observation"]
    return any(x in s for x in ("error", "failed", "not found", "does not exist", "invalid_path", "path escapes", "no such file"))


def materially_changes(a, b):
    if not a or not b: return False
    return a["action"] != b["action"]


def score(task, resp):
    ev = normalize(steps(resp)); ac = [e["action"] for e in ev]; tx = output(resp).lower()
    checks = {}
    req = task[3]["required"]
    checks["required_actions"] = all(
        (r == "file_tool" and any(a in ac for a in ("file_read", "file_write", "workspace_list"))) or
        (r == "verify" and verification(ev)) or
        (r == "recovery" and recovery(ev)) or
        (r == "replan" and replan(ev)) or
        (r == "evidence" and any(x in tx for x in ("verified", "cannot verify", "not verified", "from the workspace", "according to"))) or
        (r in ac)
        for r in req
    )
    checks["no_action_loop"] = loop_rate(ac) < 0.60
    checks["observation_drives_next_action"] = transition_rate(ev) >= 0.50 if any(e["observation"] for e in ev) else True
    checks["terminates"] = "final" in ac and len(ac) < 10
    if task[3].get("forbidden"):
        checks["forbidden_respected"] = not any(a in TOOL_ACTIONS for a in ac)
    return {"id": task[0], "category": task[1], "prompt": task[2], "checks": checks,
            "score": sum(checks.values()) / len(checks), "steps": len(ev), "actions": ac,
            "events": ev, "stop_reason": resp.get("stop_reason"), "output": output(resp)}


def loop_rate(ac):
    if len(ac) < 3: return 0.0
    repeats = sum(a == b for a, b in zip(ac, ac[1:]))
    return repeats / (len(ac) - 1)


def transition_rate(ev):
    pairs = [(a, b) for a, b in zip(ev, ev[1:]) if a["observation"]]
    if not pairs: return 0.0
    return sum(materially_changes(a, b) for a, b in pairs) / len(pairs)


def recovery(ev):
    return any(has_failure(a) and i + 1 < len(ev) and materially_changes(a, ev[i+1]) for i, a in enumerate(ev))


def replan(ev):
    cues = ("plan", "replan", "instead", "since", "because", "observation")
    for i, e in enumerate(ev[:-1]):
        if has_failure(e) and materially_changes(e, ev[i+1]):
            return any(c in ev[i+1]["text"] for c in cues) or ev[i+1]["action"] != e["action"]
    return False


def verification(ev):
    for i, e in enumerate(ev):
        if e["action"] in ("file_write", "code_exec"):
            later = ev[i+1:]
            if any(x["action"] in ("file_read", "code_exec") for x in later) and any("lewi_verify_7319" in x["observation"] or "verified" in x["text"] for x in later):
                return True
    return any("lewi_verify_7319" in e["observation"] and i > 0 for i, e in enumerate(ev))


def main():
    run = time.strftime("benchmark-v2-%Y%m%d-%H%M%S"); start = time.time(); results = []
    print(f"Running {len(TASKS)} state-transition tasks against {API_URL}")
    for i, task in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] {task[0]} {task[1]}...", flush=True)
        try:
            results.append(score(task, call(task[2])))
            r = results[-1]
            print(f"  score={r['score']*100:.1f}% actions={','.join(r['actions']) or '-'}", flush=True)
        except Exception as e:
            results.append({"id": task[0], "category": task[1], "checks": {"api": False}, "score": 0, "steps": 0, "actions": [], "events": [], "output": str(e)})
    elapsed = time.time() - start
    overall = sum(r["score"] for r in results) / len(results)
    cats = {}
    for r in results: cats.setdefault(r["category"], []).append(r["score"])
    cats = {k: sum(v) / len(v) for k, v in cats.items()}
    payload = {"run_id": run, "api_url": API_URL, "elapsed_seconds": elapsed, "overall_score": overall, "category_scores": cats, "results": results}
    (OUT / f"{run}_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["LEWI INTERACTIVE BENCHMARK V2", "=" * 70, f"Run: {run}", f"Overall transition score: {overall*100:.1f}%", "", "This benchmark measures behavioral state transitions; it is not an AGI score.", ""]
    for r in results:
        lines.append(f"{r['id']} {r['category']:18s} {r['score']*100:6.1f}% actions={','.join(r['actions']) or '-'}")
        lines.extend(f"  {'PASS' if ok else 'FAIL'} {k}" for k, ok in r["checks"].items())
    (OUT / f"{run}_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__": main()
