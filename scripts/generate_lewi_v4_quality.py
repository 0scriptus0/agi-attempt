"""Generate deterministic Lewi v4 state-transition SFT data.

The generated trajectories teach the runtime protocol rather than merely naming rules:
observe -> update state -> choose the next action -> execute/verify -> terminate only when
the goal is actually satisfied. No non-terminal state is encoded as ``action=final``.
"""
import json
from pathlib import Path

OUT = Path("datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl")
examples = []

def j(obj):
    return json.dumps(obj, separators=(",", ":"))

def add(i, category, turns):
    examples.append({"id": f"v4gen_{i:04d}", "category": category, "trajectory": turns})

i = 1

# Direct-answer/tool selection: deterministic tasks answer directly.
for q, ans in [
    ("19 + 24", "43"), ("27 * 13", "351"), ("96 / 12", "8"),
    ("74 - 29", "45"), ("32 * 15", "480"), ("125 + 75", "200"),
    ("88 - 41", "47"), ("14 * 17", "238"), ("144 / 16", "9"),
    ("53 + 38", "91"), ("61 + 29", "90"), ("18 * 16", "288"),
    ("144 - 57", "87"), ("81 / 9", "9"), ("23 * 14", "322"),
    ("37 + 58", "95"), ("847 * 913", "773311"), ("200 / 25", "8"),
    ("17 + 18 + 19", "54"), ("75 - 26", "49"),
]:
    add(i, "tool_selection", [
        {"role":"user","content":f"Compute {q}. No external information is needed."},
        {"role":"assistant","content":j({"action":"final","content":ans})},
    ])
    i += 1

# Tool selection: first action must match the evidence required.
tool_cases = [
    ("the current release of a rapidly changing package", "memory or local state may be stale; use authoritative current release data", {"action":"reason","content":"Use a current authoritative source because the requested value can change."}),
    ("the exact contents of an existing configuration file", "the current file state must be inspected", {"action":"file_read","path":"workspace/config.yaml"}),
    ("whether generated code satisfies a behavioral requirement", "generation success does not establish behavior", {"action":"code_exec","code":"run the focused behavioral test for the generated function"}),
    ("whether a stored lesson applies to this environment", "scope and exclusions determine applicability", {"action":"memory_search","query":"lesson applicability current environment exclusions"}),
    ("a large report available as smaller monthly files", "a smaller scoped retrieval can avoid an unnecessary broad operation", {"action":"file_read","path":"workspace/reports/month-01.md"}),
    ("a path discovered by a prior tool call", "the concrete observed path supersedes the placeholder", {"action":"file_read","path":"workspace/discovered/app.yaml"}),
    ("whether a deployment is healthy after rollout", "deployment completion is not health evidence", {"action":"code_exec","code":"run the deployment health check"}),
    ("whether a database migration is already applied", "migration history is direct state evidence", {"action":"code_exec","code":"inspect migration history for the target migration"}),
    ("whether two records refer to the same entity", "display names are insufficient identifiers", {"action":"reason","content":"Gather a discriminating identifier before choosing between the records."}),
    ("the exact value documented by the newest service release", "current documentation outranks stale local notes", {"action":"reason","content":"Use the current authoritative documentation rather than relying on stale memory."}),
]
for subject, evidence, action in tool_cases:
    add(i, "tool_selection", [
        {"role":"user","content":f"The task requires determining {subject}. What should happen next?"},
        {"role":"assistant","content":j(action)},
        {"role":"user","content":f"OBSERVATION: {evidence} is required to answer reliably."},
        {"role":"assistant","content":j({"action":"reason","content":"The next step follows the evidence requirement rather than using an unnecessary tool."})},
    ])
    i += 1

# Recovery: after failure, change the state and action instead of repeating the failure.
recovery_cases = [
    ("file_read","workspace/missing.yaml","workspace/deploy/app.yaml","The original path is invalid; use the discovered path.",{"action":"file_read","path":"workspace/deploy/app.yaml"}),
    ("code_exec","large report retrieval timed out","monthly files are available","Repeated identical timeouts show no progress; narrow the retrieval.",{"action":"file_read","path":"workspace/reports/month-01.md"}),
    ("code_exec","undefined variable x","x is referenced before assignment","The failure identifies an initialization defect; fix that state before retesting.",{"action":"code_exec","code":"initialize x before its first use, then rerun the focused test"}),
    ("code_exec","integration test returned 401","the test client omitted its token","Authentication setup explains the failure; repair the test setup before rerunning.",{"action":"code_exec","code":"provide the required test token and rerun the integration test"}),
    ("code_exec","patch passed one test but caused three regressions","all regressions share the modified helper","The new evidence changes the hypothesis; narrow the helper change and retest.",{"action":"code_exec","code":"revert the shared-helper change, isolate the intended behavior, and rerun affected tests"}),
    ("code_exec","migration says table already exists","migration history shows it already ran","The database state contradicts the assumption; do not repeat the migration.",{"action":"reason","content":"Treat the migration as already applied and continue from the recorded state."}),
    ("file_read","tool returned an empty broad query","a scoped query returns the expected records","The empty result is insufficient evidence; use the successful scoped query.",{"action":"memory_search","query":"scoped records for current task"}),
    ("file_write","write succeeded but readback cannot find the file","the file was written under a different workspace path","The observed path mismatch must be reconciled before retrying.",{"action":"file_read","path":"workspace/actual/artifact.txt"}),
    ("code_exec","health check failed on port 8080","runtime reports the service on port 8081","The runtime state changed the endpoint assumption.",{"action":"code_exec","code":"run health check against port 8081"}),
    ("file_read","requested file is empty","an archive contains the populated version","The empty file does not satisfy the evidence requirement; inspect the populated artifact.",{"action":"file_read","path":"workspace/archive/populated.txt"}),
]
for failure, observation, new_state, revision, next_action in recovery_cases:
    add(i, "recovery", [
        {"role":"user","content":f"The planned operation failed: {failure} at {observation}."},
        {"role":"assistant","content":j({"action":"reason","content":"Record the failure and identify what changed before selecting the next action."})},
        {"role":"user","content":f"OBSERVATION: {new_state}."},
        {"role":"assistant","content":j({"action":"reason","content":revision})},
        {"role":"assistant","content":j(next_action)},
        {"role":"user","content":"OBSERVATION: the revised action is available and addresses the new state."},
        {"role":"assistant","content":j({"action":"reason","content":"Proceed with the revised action; do not repeat the failed operation."})},
    ])
    i += 1

# Replanning: make the changed assumption explicit, then choose an action consistent with it.
replan_cases = [
    ("config.toml is at workspace root","settings/config.toml exists","Update the plan to use settings/config.toml.",{"action":"file_read","path":"workspace/settings/config.toml"}),
    ("dependency A must be available before B","B is available and A depends on B","Reverse the dependency order.",{"action":"reason","content":"Plan B first, then use B to satisfy A's dependency."}),
    ("the database starts empty","the target migration is already recorded","Treat the migration as already applied.",{"action":"reason","content":"Skip the already-applied migration and continue with the next required state."}),
    ("the first artifact is final","validation reports a missing required field","Regenerate with the missing field.",{"action":"code_exec","code":"regenerate the artifact with the required field and rerun validation"}),
    ("the first test failure identifies the root cause","a second test exposes a different boundary","Broaden the diagnosis before editing again.",{"action":"code_exec","code":"run the focused tests around both failure boundaries"}),
    ("the cached result is current","the source timestamp is newer","Invalidate the stale assumption.",{"action":"file_read","path":"workspace/current/source.json"}),
    ("deployment uses port 8000","runtime reports port 8081","Update downstream checks to the observed port.",{"action":"code_exec","code":"run the health check against port 8081"}),
    ("the output directory is empty","the build artifact is under workspace/dist","Use the observed artifact location.",{"action":"file_read","path":"workspace/dist/build-artifact"}),
    ("the old API endpoint is authoritative","current docs specify /v2/status","Update the plan to the current endpoint.",{"action":"code_exec","code":"query /v2/status and inspect the response"}),
    ("all records are unique","two IDs appear twice","Add deduplication before finalizing the record set.",{"action":"code_exec","code":"deduplicate by stable record ID and rerun validation"}),
]
for assumption, obs, revision, next_action in replan_cases:
    add(i, "replanning", [
        {"role":"user","content":f"The plan assumes {assumption}."},
        {"role":"assistant","content":j({"action":"reason","content":"Compare the assumption with observed state before committing to the plan."})},
        {"role":"user","content":f"OBSERVATION: {obs}."},
        {"role":"assistant","content":j({"action":"reason","content":revision})},
        {"role":"assistant","content":j(next_action)},
        {"role":"user","content":"OBSERVATION: the revised action is consistent with the updated plan."},
        {"role":"assistant","content":j({"action":"reason","content":"Continue from the updated state; the original assumption is no longer controlling."})},
    ])
    i += 1

# Memory discipline: search before applying; write only reusable evidence-backed knowledge.
memory_cases = [
    ("environment X","current environment is Y",{"action":"reason","content":"Do not apply the memory because its scope excludes the current environment."}),
    ("one user","the preference is explicitly user-specific",{"action":"memory_write","content":"This user prefers concise answers.","memory_type":"preference","scope":"user","applicable_when":"Responding to this user.","not_applicable_when":"Other users.","evidence":"Explicit user statement.","confidence":0.99}),
    ("repeated procedure under condition C","15 matching episodes succeeded without contradiction",{"action":"memory_write","content":"Procedure A reliably fixes failure F under condition C.","memory_type":"procedural","scope":"global","applicable_when":"Failure F and condition C match.","not_applicable_when":"The failure or environment differs materially.","evidence":"15 matching successful episodes without contradiction.","confidence":0.95}),
    ("an unverified rumor","no independent source confirms it",{"action":"reason","content":"Do not promote the claim to factual global memory without evidence."}),
    ("version 3 of a service","current service is version 4",{"action":"reason","content":"Treat the old memory as historical context rather than current guidance."}),
    ("a single arithmetic episode","the result is useful only to the current task",{"action":"reason","content":"Keep the fact episodic and task-scoped rather than promoting it to global memory."}),
    ("a prior recovery strategy","three matching episodes succeeded and one contradicted it",{"action":"reason","content":"The contradiction lowers confidence; do not promote the strategy without resolving the conflict."}),
    ("a memory with explicit exclusions","the current case matches an exclusion",{"action":"reason","content":"Respect the exclusion and do not apply the memory to this case."}),
    ("two similar lessons","one is scoped to the current version",{"action":"memory_search","query":"current version lesson applicability"}),
    ("a proposed global lesson","only one weakly supported episode exists",{"action":"reason","content":"Keep it episodic until repeated evidence supports generalization."}),
]
for scope, evidence, decision in memory_cases:
    add(i, "memory_read_write", [
        {"role":"user","content":f"A memory is known to apply only to {scope}."},
        {"role":"assistant","content":j({"action":"memory_search","query":"memory scope applicability exclusions evidence"})},
        {"role":"user","content":f"OBSERVATION: {evidence}."},
        {"role":"assistant","content":j(decision)},
        {"role":"user","content":"OBSERVATION: the memory decision is consistent with the evidence."},
        {"role":"assistant","content":j({"action":"reason","content":"Preserve the observed scope and evidence; do not invent broader applicability."})},
    ])
    i += 1

# Verification/readback: writing is intermediate; readback is the evidence that closes the loop.
for artifact in ["note.md","config.json","report.md","result.txt","summary.md","manifest.json","output.txt","plan.md"]:
    add(i, "verification", [
        {"role":"user","content":f"Create {artifact}, then verify that its contents match the requested state."},
        {"role":"assistant","content":j({"action":"file_write","path":artifact,"content":"verified artifact"})},
        {"role":"user","content":"OBSERVATION: write succeeded."},
        {"role":"assistant","content":j({"action":"file_read","path":artifact})},
        {"role":"user","content":"OBSERVATION: readback exactly matches the requested contents."},
        {"role":"assistant","content":j({"action":"final","content":f"{artifact} was verified by readback."})},
    ])
    i += 1

# Termination: non-terminal states never use action=final.
termination_cases = [
    ("create and verify a report","report written but not read back",{"action":"file_read","path":"workspace/report.md"}),
    ("build and deploy an application","build passed but deployment has not run",{"action":"code_exec","code":"run the deployment step"}),
    ("find a file and report its contents","file located and read successfully",{"action":"final","content":"The file was located, read, and its contents can now be reported."}),
    ("fix a failing test and avoid regressions","original test passes but a regression remains",{"action":"code_exec","code":"run the regression test and inspect the shared failure boundary"}),
    ("resolve an ambiguous lookup","a second lookup uniquely identifies record B",{"action":"final","content":"The ambiguity is resolved with a unique identifier and supporting evidence."}),
    ("publish a package","package built but registry upload has not completed",{"action":"code_exec","code":"upload the verified package to the target registry"}),
    ("produce a correct dataset","generation completed but schema validation has not run",{"action":"code_exec","code":"run schema validation over the generated dataset"}),
    ("answer a current factual question","current authoritative source was retrieved and checked",{"action":"final","content":"The requested current fact is supported by the retrieved authoritative evidence."}),
    ("repair a service","service restarted but health check has not passed",{"action":"code_exec","code":"run the service health check"}),
    ("complete a migration","migration recorded but post-migration verification is pending",{"action":"code_exec","code":"run the post-migration verification query"}),
]
for goal, state, next_action in termination_cases:
    add(i, "termination", [
        {"role":"user","content":f"Goal: {goal}. Current state: {state}."},
        {"role":"assistant","content":j({"action":"reason","content":"Compare the current state with the complete goal before deciding whether to terminate."})},
        {"role":"assistant","content":j(next_action)},
    ])
    i += 1

# State propagation: observed runtime values replace placeholders in dependent actions.
propagation_cases = [
    ("token","abc123","step B","use the observed token"), ("path","workspace/data/report.csv","the read action","use the observed path"),
    ("version","7.3","deployment","use the observed version"), ("record id","R-204","the update action","use the observed record ID"),
    ("build artifact","artifact-42","the packaging step","use the observed artifact"), ("port","8081","the health check","use the observed port"),
    ("branch","release/2026-09","the deployment action","use the observed branch"), ("checksum","sha256:abcd","verification","verify against the observed checksum"),
    ("job id","J-91","the status query","query the observed job ID"), ("output directory","workspace/out-v2","the packaging step","package from the observed directory"),
]
for produced, value, consumer, instruction in propagation_cases:
    add(i, "state_propagation", [
        {"role":"user","content":f"Step A produces a {produced}; {consumer} still contains a placeholder."},
        {"role":"assistant","content":j({"action":"reason","content":"The next action must use concrete runtime state rather than a stale placeholder."})},
        {"role":"user","content":f"OBSERVATION: step A produced {produced}={value}."},
        {"role":"assistant","content":j({"action":"reason","content":f"The current state contains {value}; {consumer} must {instruction}."})},
        {"role":"assistant","content":j({"action":"code_exec","code":f"run {consumer} using {value}"})},
        {"role":"user","content":"OBSERVATION: the dependent step accepted the propagated value."},
    ])
    i += 1

# Uncertainty/conflicting evidence: gather discriminating evidence, then update state.
uncertainty_cases = [
    ("two records share the same display name","a second identifier lookup uniquely selects record B"),
    ("two sources disagree about a configuration value","the newer authoritative source directly documents the setting"),
    ("a tool result omits a required field","a follow-up query supplies the missing field"),
    ("a cached result conflicts with a newer source","the newer source has the later timestamp and direct evidence"),
    ("a generated value is ambiguous between two formats","the task specification explicitly selects one format"),
    ("a memory conflicts with current documentation","current documentation is the applicable source for this version"),
    ("two test failures suggest different causes","a focused reproduction isolates the actual failure boundary"),
    ("a package name maps to two repositories","the canonical repository metadata identifies the intended package"),
    ("a deployment reports two possible endpoints","the runtime service metadata identifies the active endpoint"),
    ("a file exists in two locations","the task's current manifest identifies the authoritative path"),
]
for case, resolution in uncertainty_cases:
    add(i, "uncertainty", [
        {"role":"user","content":f"The current evidence is ambiguous because {case}."},
        {"role":"assistant","content":j({"action":"reason","content":"The evidence is insufficient to commit to a state; gather discriminating evidence rather than inventing an interpretation."})},
        {"role":"user","content":f"OBSERVATION: {resolution}."},
        {"role":"assistant","content":j({"action":"reason","content":"The new evidence resolves the ambiguity and should update the working state."})},
        {"role":"assistant","content":j({"action":"final","content":"The ambiguity was resolved using the new evidence."})},
    ])
    i += 1

# Counterfactual state-transition variants. These deliberately vary observed state so the
# correct next action changes instead of rewarding a single memorized response.
variant_templates = [
    ("recovery", "A read of {path} failed.", "discovery found {alt}.", {"action":"file_read","path":"{alt}"}),
    ("replanning", "The plan assumes {assumption}.", "runtime reports {observed}.", {"action":"reason","content":"Revise the plan to use the observed state: {observed}."}),
    ("verification", "The artifact was written.", "readback reports {result}.", {"action":"reason","content":"Compare the readback with the requested state before terminating."}),
    ("termination", "The goal is {goal}.", "current state is {state}.", {"action":"reason","content":"The goal is not complete until the remaining state is satisfied."}),
    ("state_propagation", "Step A produced {key}={value}.", "Step B still has {placeholder}.", {"action":"reason","content":"Replace the placeholder with the observed {key}={value}."}),
    ("memory_read_write", "A lesson applies to {scope}.", "the current case is {current}.", {"action":"reason","content":"Check scope before applying or generalizing the lesson."}),
    ("uncertainty", "Evidence is split between {a} and {b}.", "new evidence supports {resolved}.", {"action":"reason","content":"Update the working state to {resolved} based on the new evidence."}),
]
variants = [
    {"path":"workspace/a.yaml","alt":"workspace/deploy/a.yaml","assumption":"the root config is current","observed":"settings/config.yaml is authoritative"},
    {"path":"workspace/b.json","alt":"workspace/archive/b.json","assumption":"the cache is current","observed":"source timestamp is newer"},
    {"path":"workspace/c.txt","alt":"workspace/data/c.txt","assumption":"the first artifact is final","observed":"validation found a missing field"},
    {"path":"workspace/d.md","alt":"workspace/reports/d.md","assumption":"the broad report retrieval will succeed","observed":"the report is split into monthly files"},
    {"path":"workspace/e.csv","alt":"workspace/out/e.csv","assumption":"deployment uses port 8000","observed":"runtime reports port 8081"},
    {"path":"workspace/f.py","alt":"workspace/src/f.py","assumption":"dependency A comes first","observed":"A depends on B and B is available"},
    {"path":"workspace/g.toml","alt":"workspace/config/g.toml","assumption":"the database starts empty","observed":"migration history shows the target already ran"},
    {"path":"workspace/h.log","alt":"workspace/logs/h.log","assumption":"the first test failure is the root cause","observed":"a second test exposes another boundary"},
    {"path":"workspace/i.json","alt":"workspace/current/i.json","assumption":"old documentation is authoritative","observed":"current documentation specifies a new endpoint"},
]
goals = [
    "create and verify the artifact","build and deploy the service","find and report the file contents",
    "fix the failing test without regressions","publish the verified package","complete the migration",
    "answer the current factual question","repair the service and confirm health","produce a schema-valid dataset",
]
states = [
    "write complete but verification pending","build complete but deployment pending",
    "file located and read successfully","original test passes but a regression remains",
    "package built but upload pending","migration recorded but post-check pending",
    "authoritative source retrieved and checked","service restarted but health check pending",
    "generation complete but validation pending",
]
for idx in range(207):
    category, user_t, obs_t, action_t = variant_templates[idx % len(variant_templates)]
    v = variants[idx % len(variants)]
    values = {
        **v,
        "key": ["token","path","version","record_id","artifact","port","branch","checksum","job_id","output_dir"][idx % 10],
        "value": ["tok-17","workspace/out/report.csv","7.4","R-311","artifact-77","8082","release/next","sha256:ef01","J-108","workspace/out-v3"][idx % 10],
        "placeholder": ["TOKEN","/tmp/old.csv","VERSION","RECORD","ARTIFACT","PORT","BRANCH","CHECKSUM","JOB","OUTPUT"][idx % 10],
        "scope": ["environment X","one user","version 3","current deployment","one task","global","service A"][idx % 7],
        "current": ["environment Y","another user","version 4","service B","a different task","a scoped workflow","service B"][idx % 7],
        "a": chr(65 + (idx % 26)), "b": chr(90 - (idx % 26)),
        "resolved": ["record B","the newer source","the supplied field","the later timestamp","the specified format","current documentation","the focused reproduction"][idx % 7],
        "goal": goals[idx % len(goals)], "state": states[(idx * 3) % len(states)],
        "result": ["the requested contents","a mismatch","the expected checksum","an empty response","the correct schema"][idx % 5],
    }
    def render(text):
        for key, val in values.items():
            text = text.replace("{" + key + "}", str(val))
        return text
    turns = [
        {"role":"user","content":render(user_t)},
        {"role":"assistant","content":j({"action":"reason","content":"Update the working state from the observed evidence before choosing the next action."})},
        {"role":"user","content":f"OBSERVATION: {render(obs_t)}"},
    ]
    if category == "termination":
        turns.append({"role":"assistant","content":j({"action":"reason","content":"Do not terminate yet; identify the remaining required state and act on it."})})
        turns.append({"role":"assistant","content":j({"action":"code_exec","code":"continue the workflow until the remaining goal state is verified"})})
    elif category == "verification":
        turns.append({"role":"assistant","content":j({"action":"file_read","path":v["alt"]})})
        turns.append({"role":"user","content":"OBSERVATION: readback is available for comparison."})
        turns.append({"role":"assistant","content":j({"action":"reason","content":"Use the readback evidence to determine whether the requested state is satisfied."})})
    else:
        rendered_text = render(j(action_t))
        turns.append({"role":"assistant","content":rendered_text})
    add(i, category, turns)
    i += 1

assert len(examples) == 315, len(examples)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, separators=(",", ":")) + "\n")
print(f"wrote {len(examples)} examples to {OUT}")
