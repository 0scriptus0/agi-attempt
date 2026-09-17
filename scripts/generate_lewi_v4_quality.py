"""Generate deterministic, diverse Lewi v4 state-transition training examples.

This generator is intentionally dependency-free so the dataset can be regenerated and
reviewed without a separate data pipeline. It emits JSONL trajectories covering
state transitions, recovery, replanning, memory discipline, verification,
termination, state propagation, and uncertainty.
"""
import json
from pathlib import Path

OUT = Path("datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl")

examples = []

def add(i, category, turns):
    examples.append({"id": f"v4gen_{i:04d}", "category": category, "trajectory": turns})

def a(obj):
    return json.dumps(obj, separators=(",", ":"))

def pair(i, category, user, response):
    add(i, category, [{"role":"user","content":user}, {"role":"assistant","content":a(response)}])

i = 1
# Direct-answer vs tool selection: deterministic tasks should not invoke tools.
for q, ans in [("19 + 24", "43"), ("27 * 13", "351"), ("96 / 12", "8"), ("74 - 29", "45"),
               ("32 * 15", "480"), ("125 + 75", "200"), ("88 - 41", "47"), ("14 * 17", "238"),
               ("144 / 16", "9"), ("53 + 38", "91")]:
    pair(i, "tool_selection", f"Compute {q}. No external information is needed.", {"action":"final","content":ans}); i += 1

# Tool selection and adaptive computation.
for subject, reason in [
    ("the current release of a rapidly changing package", "A local installation may be stale, so current authoritative evidence is required."),
    ("the exact contents of an existing configuration file", "The current file state must be inspected before changing or asserting its contents."),
    ("whether a generated Python function satisfies a behavioral requirement", "Execution of a focused test provides stronger evidence than generation success."),
    ("whether a stored lesson applies to the current environment", "Memory scope and exclusions must be checked before applying a lesson."),
    ("a large report that is available in smaller monthly files", "The smaller scoped retrieval is preferable when the large retrieval is costly or failing."),
    ("a path discovered by a prior tool call", "The concrete observed path should replace the stale placeholder in the next action."),
]:
    pair(i, "tool_selection", f"The task requires determining {subject}. What should the next action be?", {"action":"reason","content":reason}); i += 1

# Recovery: never repeat an identical failing action when evidence supports a different path.
for n, (failure, observation, response) in enumerate([
    ("the planned file read fails with not found", "discovery found workspace/deploy/app.yaml", "The assumed path is invalid; use the discovered path rather than repeating the failed read."),
    ("the same large retrieval times out twice", "the report is split into monthly files", "Repeated identical timeouts show no progress; switch to the smaller monthly retrieval."),
    ("a compiler reports an undefined variable", "the variable is referenced before assignment", "The artifact is invalid; revise the variable initialization before retrying."),
    ("integration tests fail with 401", "the test client omitted its required token", "Authentication state explains the failure; fix the test setup and rerun integration verification."),
    ("a patch fixes one test but causes three regressions", "all regressions share a helper modified by the patch", "The new evidence changes the hypothesis; narrow the shared-helper change and retest."),
    ("a database migration says the table already exists", "migration history shows the migration already ran", "The assumed database state was wrong; do not repeat the migration."),
    ("a tool returns an empty result", "a scoped query returns the expected records", "The empty result is insufficient evidence; use the successful scoped query."),
    ("a write operation succeeds but the next read cannot find the file", "the file was written under a different workspace-relative path", "The state reveals a path mismatch; reconcile the actual path before retrying."),
]:
    for k in range(4):
        pair(i, "recovery", f"Attempt {k+1}: {failure}.", {"action":"reason","content":response})
        examples[-1]["trajectory"].append({"role":"user","content":f"OBSERVATION: {observation}."})
        examples[-1]["trajectory"].append({"role":"assistant","content":a({"action":"final","content":"The recovery path follows the new evidence instead of repeating the failed action."})})
        i += 1

# Replanning from changed assumptions.
for assumption, obs, revision in [
    ("config.toml is at workspace root", "settings/config.toml exists", "Update the plan to use settings/config.toml."),
    ("dependency A must be available before B", "B is available and A depends on B", "Reverse the dependency order and use B as the prerequisite."),
    ("the database starts empty", "the target migration is already recorded", "Treat the migration as already applied and continue from the recorded state."),
    ("the first generated artifact is final", "validation reports a missing required field", "Revise generation to include the missing field before declaring completion."),
    ("the first test failure identifies the root cause", "a second test exposes a different failure boundary", "Broaden the diagnosis and update the hypothesis before editing again."),
    ("the cached result is current", "the source timestamp is newer", "Invalidate the stale assumption and use the newer source."),
]:
    for k in range(4):
        add(i, "replanning", [
            {"role":"user","content":f"The plan assumes {assumption}."},
            {"role":"assistant","content":a({"action":"reason","content":"The next action depends on this assumption, so I should compare it with observed state before proceeding."})},
            {"role":"user","content":f"OBSERVATION: {obs}."},
            {"role":"assistant","content":a({"action":"reason","content":revision})},
            {"role":"user","content":"OBSERVATION: the revised plan is consistent with the observed state."},
            {"role":"assistant","content":a({"action":"final","content":"The plan was revised to match the observed state."})},
        ]); i += 1

# Memory discipline: retrieve, assess scope, then write only reusable evidence-backed lessons.
for scope, evidence, decision in [
    ("environment X", "the current environment is Y", "Do not apply the lesson because its scope excludes the current environment."),
    ("one user", "the preference is explicitly user-specific", "Keep the preference scoped to that user rather than making it global."),
    ("a repeated procedure under condition C", "15 matching episodes succeeded without contradiction", "Store a procedural lesson with explicit applicability and exclusions."),
    ("an unverified rumor", "no independent source confirms it", "Do not promote the claim to factual global memory."),
    ("version 3 of a service", "the current service is version 4 and documentation differs", "Treat the old memory as historical context, not current guidance."),
    ("a single arithmetic episode", "the result is useful only to the current task", "Keep it episodic rather than promoting it to a reusable global lesson."),
]:
    for k in range(4):
        if "repeated procedure" in scope:
            first={"action":"memory_search","query":"procedure applicability condition C"}
            second={"action":"memory_write","content":"Procedure A reliably fixes failure F under condition C.","memory_type":"procedural","scope":"global","applicable_when":"Failure F and condition C match.","not_applicable_when":"The failure or environment differs materially.","evidence":"15 matching successful episodes without contradiction.","confidence":0.95}
        elif "user" in scope:
            first={"action":"reason","content":"The memory scope must follow the evidence."}
            second={"action":"memory_write","content":"This user prefers concise answers.","memory_type":"preference","scope":"user","applicable_when":"Responding to this user.","not_applicable_when":"Other users.","evidence":"Explicit user statement.","confidence":0.99}
        else:
            first={"action":"memory_search","query":"memory scope applicability exclusions"}
            second={"action":"reason","content":decision}
        add(i,"memory_read_write",[
            {"role":"user","content":f"A memory is known to apply only to {scope}."},
            {"role":"assistant","content":a(first)},
            {"role":"user","content":f"OBSERVATION: {evidence}."},
            {"role":"assistant","content":a(second)},
            {"role":"user","content":"OBSERVATION: the memory decision is consistent with the evidence."},
            {"role":"assistant","content":a({"action":"final","content":decision})},
        ]); i += 1

# Verification and readback.
for artifact in ["note.md","config.json","report.md","result.txt","summary.md"]:
    for k in range(5):
        add(i,"verification",[
            {"role":"user","content":f"Create {artifact}, then verify that its contents match the requested state."},
            {"role":"assistant","content":a({"action":"file_write","path":artifact,"content":"verified artifact"})},
            {"role":"user","content":"OBSERVATION: write succeeded."},
            {"role":"assistant","content":a({"action":"file_read","path":artifact})},
            {"role":"user","content":"OBSERVATION: readback exactly matches the requested contents."},
            {"role":"assistant","content":a({"action":"final","content":f"{artifact} was verified by readback."})},
        ]); i += 1

# Termination: distinguish intermediate milestones from actual goal satisfaction.
for goal, state, decision in [
    ("create and verify a report", "the report was written but not read back", "Do not terminate; verification is still required."),
    ("build and deploy an application", "the build passed but deployment has not run", "Do not terminate; deployment remains part of the goal."),
    ("find a file and report its contents", "the file was located and read successfully", "Terminate because the stated goal is satisfied and the result is verified."),
    ("fix a failing test and avoid regressions", "the original test passes but a regression remains", "Do not terminate; the regression means the complete goal is unsatisfied."),
    ("resolve an ambiguous lookup", "a second lookup uniquely identifies the record", "Terminate because the ambiguity has been resolved with evidence."),
]:
    for k in range(5):
        add(i,"termination",[
            {"role":"user","content":f"Goal: {goal}. Current state: {state}."},
            {"role":"assistant","content":a({"action":"reason","content":"Compare the current state with the complete goal before deciding whether to terminate."})},
            {"role":"user","content":"OBSERVATION: the current state remains as described."},
            {"role":"assistant","content":a({"action":"final","content":decision})},
        ]); i += 1

# State propagation: observed runtime values replace placeholders.
for produced, value, consumer in [
    ("token", "abc123", "step B"), ("path", "workspace/data/report.csv", "the read action"),
    ("version", "7.3", "deployment"), ("record id", "R-204", "the update action"),
    ("build artifact", "artifact-42", "the packaging step"), ("port", "8081", "the health check")]:
    for k in range(5):
        add(i,"state_propagation",[
            {"role":"user","content":f"Step A produces a {produced}; {consumer} still contains a placeholder."},
            {"role":"assistant","content":a({"action":"reason","content":"The next action must use concrete runtime state rather than a stale placeholder."})},
            {"role":"user","content":f"OBSERVATION: step A produced {produced}={value}."},
            {"role":"assistant","content":a({"action":"reason","content":f"The current state contains {value}; {consumer} should consume that observed value."})},
            {"role":"user","content":"OBSERVATION: the dependent step accepts the propagated value."},
            {"role":"assistant","content":a({"action":"final","content":"The runtime dependency was propagated into the next state."})},
        ]); i += 1

# Uncertainty and conflicting evidence.
for case, resolution in [
    ("two records share the same display name", "a second identifier lookup uniquely selects record B"),
    ("two sources disagree about a configuration value", "the newer authoritative source directly documents the setting"),
    ("a tool result omits a required field", "a follow-up query supplies the missing field"),
    ("a cached result conflicts with a newer source", "the newer source has the later timestamp and direct evidence"),
    ("a generated value is ambiguous between two formats", "the task specification explicitly selects one format"),
    ("a memory conflicts with current documentation", "current documentation is the applicable source for this version"),
]:
    for k in range(5):
        add(i,"uncertainty",[
            {"role":"user","content":f"The current evidence is ambiguous because {case}."},
            {"role":"assistant","content":a({"action":"reason","content":"The evidence is insufficient to commit to a state. I should gather discriminating evidence rather than invent an interpretation."})},
            {"role":"user","content":f"OBSERVATION: {resolution}."},
            {"role":"assistant","content":a({"action":"reason","content":"The new evidence resolves the ambiguity and should update the working state."})},
            {"role":"user","content":"OBSERVATION: the resolved state is consistent with the evidence."},
            {"role":"assistant","content":a({"action":"final","content":"The ambiguity was resolved using the new evidence."})},
        ]); i += 1

# Keep exactly 315 examples; deterministic ordering is part of reproducibility.
assert len(examples) == 315, len(examples)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, separators=(",", ":")) + "\n")
print(f"wrote {len(examples)} examples to {OUT}")
