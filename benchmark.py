#!/usr/bin/env python3

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


API_URL = os.environ.get(
    "LEWI_API_URL",
    "http://127.0.0.1:8000/chat",
)

BENCHMARK_DIR = Path("learning_curve/benchmarks")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


TASKS = [
    {
        "id": "T01",
        "category": "adaptive_compute",
        "prompt": (
            "What is 37 + 58? Answer directly. "
            "Do not use tools or memory."
        ),
        "criteria": ["no_tool", "final"],
    },
    {
        "id": "T02",
        "category": "tool_selection",
        "prompt": (
            "Use the code tool to calculate 847 * 913. "
            "Inspect the execution result, verify it, and then give "
            "the final answer."
        ),
        "criteria": ["code_exec", "verify", "final"],
    },
    {
        "id": "T03",
        "category": "tool_selection",
        "prompt": (
            "Inspect the workspace and determine what files are present. "
            "Use the appropriate file tool rather than guessing filenames. "
            "Then summarize what you found."
        ),
        "criteria": ["file_tool", "final"],
    },
    {
        "id": "T04",
        "category": "recovery",
        "prompt": (
            "Read workspace/nonexistent_file.txt. If that fails, "
            "inspect the workspace to determine what files actually exist, "
            "adapt your plan, and then report what happened."
        ),
        "criteria": ["file_tool", "recovery", "final"],
    },
    {
        "id": "T05",
        "category": "memory",
        "prompt": (
            "Search memory for any reusable lesson about recovering from "
            "repeated tool failures. Use the memory result to explain "
            "what strategy should be used."
        ),
        "criteria": ["memory_search", "final"],
    },
    {
        "id": "T06",
        "category": "memory",
        "prompt": (
            "Record a reusable memory that this benchmark tests whether "
            "observations change subsequent actions, and repeated tool "
            "failures should trigger recovery or replanning. Include "
            "evidence and applicability conditions. Then confirm what "
            "you stored."
        ),
        "criteria": ["memory_write", "final"],
    },
    {
        "id": "T07",
        "category": "verification",
        "prompt": (
            "Create a small text file in the workspace containing exactly "
            "VERIFICATION_TEST. Then read it back and verify that the "
            "contents match before reporting success."
        ),
        "criteria": ["file_write", "verify", "final"],
    },
    {
        "id": "T08",
        "category": "replanning",
        "prompt": (
            "Try to inspect workspace/project_architecture.txt. "
            "If the path is invalid or unavailable, do not repeatedly "
            "retry the same path. Inspect the workspace, identify a "
            "valid file instead, and update your plan."
        ),
        "criteria": ["file_tool", "recovery", "replan", "final"],
    },
    {
        "id": "T09",
        "category": "reasoning",
        "prompt": (
            "Inspect a workspace file and make a claim about the project "
            "only when the file contents provide evidence for that claim. "
            "Clearly distinguish verified information from uncertainty."
        ),
        "criteria": ["file_tool", "evidence", "final"],
    },
    {
        "id": "T10",
        "category": "agent_loop",
        "prompt": (
            "Inspect the workspace to understand what this project is "
            "building. Then store a concise reusable architectural memory "
            "with evidence and applicability conditions. Finally verify "
            "that the memory was stored and report the result."
        ),
        "criteria": ["file_tool", "memory_write", "verify", "final"],
    },
]


def call_api(prompt):
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8")

    return json.loads(body)


def extract_trace(response):
    """
    Current Lewi API schema:

    {
        "reply": "...",
        "trace": [...],
        "steps": 10,
        "memories_used": [...],
        "stop_reason": "..."
    }

    IMPORTANT:
    Prefer trace over steps because `steps` is an integer.
    """

    trace = response.get("trace")

    if isinstance(trace, list):
        return trace

    # Compatibility with possible older API formats.
    events = response.get("events")

    if isinstance(events, list):
        return events

    steps_value = response.get("steps")

    if isinstance(steps_value, list):
        return steps_value

    if isinstance(steps_value, dict):
        nested = (
            steps_value.get("steps")
            or steps_value.get("events")
            or steps_value.get("trace")
        )

        if isinstance(nested, list):
            return nested

    return []


def extract_actions(trace):
    actions = []

    for item in trace:
        if not isinstance(item, dict):
            continue

        action = item.get("action")

        if isinstance(action, dict):
            name = action.get("action")

            if name:
                actions.append(name)

        elif isinstance(action, str):
            actions.append(action)

    return actions


def extract_observations(trace):
    observations = []

    for item in trace:
        if not isinstance(item, dict):
            continue

        observation = item.get("observation")

        if observation is not None:
            observations.append(observation)

    return observations


def flatten_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.lower()

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        ).lower()
    except Exception:
        return str(value).lower()


def output_text(response):
    for key in (
        "reply",
        "response",
        "output",
        "message",
    ):
        value = response.get(key)

        if isinstance(value, str):
            return value

    return ""


def has_action(actions, name):
    return name in actions


def has_file_tool(actions):
    return any(
        action in {
            "file_read",
            "file_write",
        }
        for action in actions
    )


def has_memory_tool(actions, name):
    return name in actions


def has_verification(trace, response):
    text_parts = [
        output_text(response),
        flatten_text(response.get("reply")),
    ]

    for item in trace:
        text_parts.append(flatten_text(item))

    text = " ".join(text_parts)

    verification_terms = [
        "verif",
        "verified",
        "verify",
        "validation",
        "validated",
        "check",
        "checked",
        "matches",
        "confirmed",
    ]

    return any(term in text for term in verification_terms)


def has_recovery(trace):
    if len(trace) < 2:
        return False

    text = " ".join(
        flatten_text(item)
        for item in trace
    )

    recovery_terms = [
        "recover",
        "replan",
        "update the plan",
        "adapt",
        "instead",
        "different file",
        "failed",
        "failure",
        "invalid",
        "unavailable",
    ]

    return any(term in text for term in recovery_terms)


def has_replanning(trace):
    text = " ".join(
        flatten_text(item)
        for item in trace
    )

    terms = [
        "replan",
        "update the plan",
        "changed plan",
        "adapt",
        "instead",
        "new plan",
        "different path",
        "different file",
    ]

    return any(term in text for term in terms)


def has_evidence(trace, response):
    text = " ".join(
        [
            output_text(response),
            *[flatten_text(item) for item in trace],
        ]
    )

    terms = [
        "evidence",
        "observed",
        "file contents",
        "according to",
        "verified",
        "found",
        "contains",
        "read",
    ]

    return any(term in text for term in terms)


def evaluate_task(task, response):
    trace = extract_trace(response)
    actions = extract_actions(trace)
    observations = extract_observations(trace)

    text = output_text(response).lower()

    stop_reason = response.get(
        "stop_reason",
        "",
    )

    checks = {}

    # ------------------------------------------------------------
    # Adaptive compute
    # ------------------------------------------------------------

    if "no_tool" in task["criteria"]:
        checks["no_tool"] = not any(
            action in {
                "code_exec",
                "file_read",
                "file_write",
                "memory_search",
                "memory_write",
            }
            for action in actions
        )

    # ------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------

    if "code_exec" in task["criteria"]:
        checks["code_exec"] = has_action(
            actions,
            "code_exec",
        )

    if "file_tool" in task["criteria"]:
        checks["file_tool"] = has_file_tool(actions)

    if "memory_search" in task["criteria"]:
        checks["memory_search"] = has_memory_tool(
            actions,
            "memory_search",
        )

    if "memory_write" in task["criteria"]:
        checks["memory_write"] = has_memory_tool(
            actions,
            "memory_write",
        )

    if "file_write" in task["criteria"]:
        checks["file_write"] = has_action(
            actions,
            "file_write",
        )

    # ------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------

    if "verify" in task["criteria"]:
        checks["verify"] = has_verification(
            trace,
            response,
        )

    # ------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------

    if "recovery" in task["criteria"]:
        checks["recovery"] = has_recovery(trace)

    # ------------------------------------------------------------
    # Replanning
    # ------------------------------------------------------------

    if "replan" in task["criteria"]:
        checks["replan"] = has_replanning(trace)

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    if "evidence" in task["criteria"]:
        checks["evidence"] = has_evidence(
            trace,
            response,
        )

    # ------------------------------------------------------------
    # Final action
    # ------------------------------------------------------------

    if "final" in task["criteria"]:
        final_action = has_action(
            actions,
            "final",
        )

        # The final action is the actual criterion.
        # A natural-language answer is NOT enough.
        checks["final"] = final_action

    passed = sum(
        1
        for value in checks.values()
        if value
    )

    total = len(checks)

    return {
        "score": (
            passed / total
            if total
            else 0.0
        ),
        "checks": checks,
        "trace": trace,
        "actions": actions,
        "observations": observations,
        "steps": len(trace),
        "api_steps": response.get("steps"),
        "stop_reason": stop_reason,
        "reply": output_text(response),
        "memories_used": response.get(
            "memories_used",
            [],
        ),
    }


def run_benchmark():
    run_id = (
        "benchmark-"
        + datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
    )

    print(
        f"Running {len(TASKS)} interactive tasks "
        f"against {API_URL}"
    )

    results = []

    start_time = time.time()

    for index, task in enumerate(
        TASKS,
        start=1,
    ):
        print(
            f"[{index}/{len(TASKS)}] "
            f"{task['id']} "
            f"{task['category']}...",
            flush=True,
        )

        task_start = time.time()

        try:
            response = call_api(
                task["prompt"]
            )

            evaluated = evaluate_task(
                task,
                response,
            )

            elapsed = (
                time.time()
                - task_start
            )

            result = {
                "id": task["id"],
                "category": task["category"],
                "prompt": task["prompt"],
                "criteria": task["criteria"],
                "elapsed_seconds": elapsed,
                **evaluated,
            }

            results.append(result)

            print(
                "  "
                f"score={result['score'] * 100:.1f}% "
                f"steps={result['steps']} "
                f"actions="
                f"{','.join(result['actions']) or '-'}",
                flush=True,
            )

        except HTTPError as error:
            elapsed = (
                time.time()
                - task_start
            )

            body = ""

            try:
                body = (
                    error.read()
                    .decode("utf-8")
                )
            except Exception:
                pass

            result = {
                "id": task["id"],
                "category": task["category"],
                "prompt": task["prompt"],
                "criteria": task["criteria"],
                "score": 0.0,
                "checks": {},
                "trace": [],
                "actions": [],
                "observations": [],
                "steps": 0,
                "api_steps": None,
                "stop_reason": "http_error",
                "reply": body,
                "memories_used": [],
                "elapsed_seconds": elapsed,
                "error": str(error),
            }

            results.append(result)

            print(
                f"  ERROR HTTP {error.code}",
                flush=True,
            )

        except URLError as error:
            elapsed = (
                time.time()
                - task_start
            )

            result = {
                "id": task["id"],
                "category": task["category"],
                "prompt": task["prompt"],
                "criteria": task["criteria"],
                "score": 0.0,
                "checks": {},
                "trace": [],
                "actions": [],
                "observations": [],
                "steps": 0,
                "api_steps": None,
                "stop_reason": "connection_error",
                "reply": "",
                "memories_used": [],
                "elapsed_seconds": elapsed,
                "error": str(error),
            }

            results.append(result)

            print(
                f"  ERROR connection: {error}",
                flush=True,
            )

        except Exception as error:
            elapsed = (
                time.time()
                - task_start
            )

            result = {
                "id": task["id"],
                "category": task["category"],
                "prompt": task["prompt"],
                "criteria": task["criteria"],
                "score": 0.0,
                "checks": {},
                "trace": [],
                "actions": [],
                "observations": [],
                "steps": 0,
                "api_steps": None,
                "stop_reason": "benchmark_error",
                "reply": "",
                "memories_used": [],
                "elapsed_seconds": elapsed,
                "error": repr(error),
            }

            results.append(result)

            print(
                f"  ERROR: {error}",
                flush=True,
            )

    elapsed_total = (
        time.time()
        - start_time
    )

    # ------------------------------------------------------------
    # Overall score
    # ------------------------------------------------------------

    all_checks = []

    for result in results:
        all_checks.extend(
            result["checks"].values()
        )

    overall_score = (
        sum(all_checks) / len(all_checks)
        if all_checks
        else 0.0
    )

    # ------------------------------------------------------------
    # Category scores
    # ------------------------------------------------------------

    category_values = {}

    for result in results:
        category = result["category"]

        if category not in category_values:
            category_values[category] = []

        category_values[category].append(
            result["score"]
        )

    category_scores = {}

    for category, values in category_values.items():
        category_scores[category] = (
            sum(values) / len(values)
            if values
            else 0.0
        )

    # ------------------------------------------------------------
    # JSON result
    # ------------------------------------------------------------

    output = {
        "run": run_id,
        "api": API_URL,
        "tasks": len(TASKS),
        "elapsed_seconds": elapsed_total,
        "overall_criterion_score": overall_score,
        "category_scores": category_scores,
        "results": results,
    }

    json_path = (
        BENCHMARK_DIR
        / f"{run_id}_results.json"
    )

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Human-readable result
    # ------------------------------------------------------------

    lines = []

    lines.append(
        "LEWI INTERACTIVE BENCHMARK"
    )
    lines.append(
        "=" * 70
    )
    lines.append(
        f"Run: {run_id}"
    )
    lines.append(
        f"API: {API_URL}"
    )
    lines.append(
        f"Tasks: {len(TASKS)}"
    )
    lines.append(
        f"Elapsed: {elapsed_total:.1f}s"
    )
    lines.append(
        ""
    )
    lines.append(
        "Behavioral benchmark; not an AGI score."
    )
    lines.append(
        f"Overall criterion score: "
        f"{overall_score * 100:.1f}%"
    )
    lines.append(
        ""
    )

    lines.append(
        "CATEGORY SCORES"
    )
    lines.append(
        "-" * 70
    )

    for category in sorted(
        category_scores
    ):
        lines.append(
            f"{category:25s} "
            f"{category_scores[category] * 100:.1f}%"
        )

    lines.append("")
    lines.append("TASK RESULTS")
    lines.append("-" * 70)

    for result in results:
        lines.append(
            f"{result['id']} "
            f"{result['category']:18s} "
            f"{result['score'] * 100:6.1f}% "
            f"steps={result['steps']} "
            f"actions="
            f"{','.join(result['actions']) or '-'}"
        )

        for criterion, passed in (
            result["checks"].items()
        ):
            lines.append(
                f"  "
                f"{'PASS' if passed else 'FAIL'} "
                f"{criterion}"
            )

        reply = result.get(
            "reply",
            "",
        )

        if reply:
            lines.append(
                f"  output: {reply}"
            )

        if result.get("error"):
            lines.append(
                f"  error: "
                f"{result['error']}"
            )

    lines.append("")
    lines.append("TRACE SUMMARY")
    lines.append("-" * 70)

    for result in results:
        lines.append(
            f"\n{result['id']}:"
        )

        trace = result.get(
            "trace",
            [],
        )

        for item in trace:
            step = item.get(
                "step",
                "?",
            )

            action = item.get(
                "action",
                {},
            )

            if isinstance(action, dict):
                action_name = action.get(
                    "action",
                    "?",
                )
            else:
                action_name = str(
                    action
                )

            observation = item.get(
                "observation",
                {},
            )

            lines.append(
                f"  step {step}: "
                f"{action_name} "
                f"-> "
                f"{json.dumps(observation, ensure_ascii=False)}"
            )

    result_path = (
        BENCHMARK_DIR
        / f"{run_id}_result.txt"
    )

    result_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Optional graph
    # ------------------------------------------------------------

    graph_path = (
        BENCHMARK_DIR
        / f"{run_id}_graph.png"
    )

    try:
        import matplotlib.pyplot as plt

        categories = list(
            category_scores.keys()
        )

        values = [
            category_scores[x] * 100
            for x in categories
        ]

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            categories,
            values,
        )

        plt.ylabel(
            "Criterion score (%)"
        )

        plt.title(
            "Lewi Interactive Behavioral Benchmark"
        )

        plt.ylim(
            0,
            100,
        )

        plt.xticks(
            rotation=30,
            ha="right",
        )

        plt.tight_layout()

        plt.savefig(
            graph_path,
            dpi=160,
        )

        plt.close()

        graph_created = True

    except ImportError:
        graph_created = False

        print(
            "Graph generation skipped: "
            "matplotlib is not installed."
        )

    except Exception as error:
        graph_created = False

        print(
            f"Graph generation failed: "
            f"{error}"
        )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    print()
    print(
        "BENCHMARK COMPLETE"
    )
    print(
        "=" * 70
    )
    print(
        f"Overall criterion score: "
        f"{overall_score * 100:.1f}%"
    )

    print()
    print(
        f"JSON: {json_path.resolve()}"
    )

    print(
        f"TEXT: {result_path.resolve()}"
    )

    if graph_created:
        print(
            f"GRAPH: {graph_path.resolve()}"
        )

    return output


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print(
            "\nBenchmark interrupted."
        )
        sys.exit(130)