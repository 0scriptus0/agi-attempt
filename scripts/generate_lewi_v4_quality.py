import json
from pathlib import Path
import os

OUT = Path(os.environ.get("LEWI_V4_OUTPUT", "datasets/reasoning/lewi_agentic_state_transition_sft_v4_generated.jsonl"))
rows = []

def j(x): return json.dumps(x, separators=(",", ":"))
def add(category, turns): rows.append({"id": f"v4gen_{len(rows)+1:04d}", "category": category, "trajectory": turns})

arithmetic = [("19+24","43"),("27*13","351"),("96/12","8"),("74-29","45"),("32*15","480"),("125+75","200"),("88-41","47"),("14*17","238"),("144/16","9"),("53+38","91"),("61+29","90"),("18*16","288"),("144-57","87"),("81/9","9"),("23*14","322"),("37+58","95"),("847*913","773311"),("200/25","8"),("17+18+19","54"),("75-26","49")]
for n in range(3):
    for q,a in arithmetic:
        add("tool_selection", [{"role":"user","content":f"Compute {q}. No external information is needed."},{"role":"assistant","content":j({"action":"final","content":a})}])

tool_cases = [
("current release of a rapidly changing package", {"action":"reason","content":"Use current authoritative release data; local memory may be stale."}),
("exact contents of an existing configuration file", {"action":"file_read","path":"workspace/config.yaml"}),
("behavior of generated code", {"action":"code_exec","code":"run the focused behavioral test"}),
("whether a memory applies here", {"action":"memory_search","query":"applicability scope exclusions evidence"}),
("a discovered artifact path", {"action":"file_read","path":"workspace/discovered/app.yaml"}),
("whether a migration already ran", {"action":"code_exec","code":"inspect migration history"}),
("whether two records are the same entity", {"action":"reason","content":"Gather a stable discriminating identifier before choosing."}),
("a value documented by the newest release", {"action":"reason","content":"Use current authoritative documentation rather than stale local notes."}),
("deployment health after rollout", {"action":"code_exec","code":"run the deployment health check"}),
("a large report available as monthly files", {"action":"file_read","path":"workspace/reports/month-01.md"})]
for n in range(3):
    for subject,action in tool_cases:
        add("tool_selection", [{"role":"user","content":f"Determine {subject}. What should happen next?"},{"role":"assistant","content":j(action)},{"role":"user","content":"OBSERVATION: the evidence requirement is confirmed."},{"role":"assistant","content":j({"action":"reason","content":"Choose from observed evidence, not habit."})}])

recovery = [
("missing.yaml","deploy/app.yaml exists","The assumed path is invalid; use the discovered path.",{"action":"file_read","path":"workspace/deploy/app.yaml"}),
("large retrieval timed out twice","monthly files are available","Repeated failure shows no progress; narrow retrieval.",{"action":"file_read","path":"workspace/reports/month-01.md"}),
("undefined variable","referenced before assignment","Fix initialization before retesting.",{"action":"code_exec","code":"initialize the variable and rerun the focused test"}),
("401 integration failure","test client omitted its token","Repair authentication setup before rerunning integration.",{"action":"code_exec","code":"provide the test token and rerun integration"}),
("one test fixed, three regressions","all regressions share a helper","Revise the hypothesis and isolate the helper change.",{"action":"code_exec","code":"narrow the helper change and rerun affected tests"}),
("migration says table exists","history shows it already ran","Treat the migration as applied; do not repeat it.",{"action":"reason","content":"Continue from the recorded migration state."}),
("empty broad query","scoped query returns records","Use the successful scoped query as new evidence.",{"action":"memory_search","query":"scoped records for current task"}),
("write succeeded, readback misses file","actual workspace path differs","Reconcile the observed path before retrying.",{"action":"file_read","path":"workspace/actual/artifact.txt"}),
("health check fails on 8080","runtime reports 8081","Use the observed runtime endpoint.",{"action":"code_exec","code":"run health check against port 8081"}),
("requested file is empty","archive contains populated version","Inspect the populated artifact instead.",{"action":"file_read","path":"workspace/archive/populated.txt"})]
for n in range(4):
    for failed,observed,revision,action in recovery:
        add("recovery", [{"role":"user","content":f"The planned operation failed: {failed}."},{"role":"assistant","content":j({"action":"reason","content":"Record the failure and identify changed state before acting again."})},{"role":"user","content":f"OBSERVATION: {observed}."},{"role":"assistant","content":j({"action":"reason","content":revision})},{"role":"assistant","content":j(action)},{"role":"user","content":"OBSERVATION: the revised action is available."}])

plans = [
("config is at root","settings/config.toml exists",{"action":"file_read","path":"workspace/settings/config.toml"}),
("A must precede B","A depends on B",{"action":"reason","content":"Plan B first, then satisfy A's dependency."}),
("database starts empty","target migration is recorded",{"action":"reason","content":"Skip the already-applied migration and continue."}),
("first artifact is final","validation reports a missing field",{"action":"code_exec","code":"regenerate with the required field and validate again"}),
("first failure is root cause","second test exposes another boundary",{"action":"code_exec","code":"test both failure boundaries before editing"}),
("cache is current","source timestamp is newer",{"action":"file_read","path":"workspace/current/source.json"}),
("deployment uses 8000","runtime reports 8081",{"action":"code_exec","code":"run health check against port 8081"}),
("output directory is empty","artifact is under workspace/dist",{"action":"file_read","path":"workspace/dist/build-artifact"}),
("old endpoint is authoritative","current docs specify /v2/status",{"action":"code_exec","code":"query /v2/status"}),
("records are unique","two IDs are duplicated",{"action":"code_exec","code":"deduplicate by stable ID and rerun validation"})]
for n in range(4):
    for assumption,obs,action in plans:
        add("replanning", [{"role":"user","content":f"The plan assumes {assumption}."},{"role":"assistant","content":j({"action":"reason","content":"Compare the assumption with observed state before committing."})},{"role":"user","content":f"OBSERVATION: {obs}."},{"role":"assistant","content":j({"action":"reason","content":"Update the plan to match the observed state."})},{"role":"assistant","content":j(action)},{"role":"user","content":"OBSERVATION: the revised action is consistent with the updated plan."}])

memory = [
("environment X","current environment Y",{"action":"reason","content":"Do not apply the memory; its scope excludes the current environment."}),
("one user","preference is explicitly user-specific",{"action":"memory_write","content":"User prefers concise answers.","memory_type":"preference","scope":"user","applicable_when":"This user","not_applicable_when":"Other users","evidence":"Explicit user statement","confidence":0.99}),
("repeated procedure C","15 matching successes, no contradiction",{"action":"memory_write","content":"Procedure A fixes failure F under C.","memory_type":"procedural","scope":"global","applicable_when":"F and C match","not_applicable_when":"Materially different state","evidence":"15 matching successful episodes","confidence":0.95}),
("unverified rumor","no independent confirmation",{"action":"reason","content":"Do not promote an unsupported claim to global factual memory."}),
("version 3","current service is version 4",{"action":"reason","content":"Treat the old memory as historical context, not current guidance."}),
("single arithmetic episode","useful only to current task",{"action":"reason","content":"Keep the fact episodic and task-scoped."}),
("prior recovery strategy","one episode contradicts three successes",{"action":"reason","content":"Resolve the contradiction before generalizing."}),
("memory with exclusions","current case matches an exclusion",{"action":"reason","content":"Respect the exclusion and do not apply the memory."}),
("two similar lessons","one is scoped to current version",{"action":"memory_search","query":"current version lesson applicability"}),
("proposed global lesson","one weak episode supports it",{"action":"reason","content":"Keep it episodic until repeated evidence supports generalization."})]
for n in range(4):
    for scope,evidence,action in memory:
        add("memory_read_write", [{"role":"user","content":f"A memory is known to apply only to {scope}."},{"role":"assistant","content":j({"action":"memory_search","query":"memory scope applicability exclusions evidence"})},{"role":"user","content":f"OBSERVATION: {evidence}."},{"role":"assistant","content":j(action)},{"role":"user","content":"OBSERVATION: the memory decision is consistent with evidence."},{"role":"assistant","content":j({"action":"reason","content":"Preserve observed scope; do not broaden without support."})}])

artifacts = ["note.md","config.json","report.md","result.txt","manifest.json","output.txt","plan.md","summary.md"]
for n in range(25):
    artifact = artifacts[n % len(artifacts)]
    add("verification", [{"role":"user","content":f"Create {artifact} and verify its contents."},{"role":"assistant","content":j({"action":"file_write","path":artifact,"content":"verified artifact"})},{"role":"user","content":"OBSERVATION: write succeeded."},{"role":"assistant","content":j({"action":"file_read","path":artifact})},{"role":"user","content":"OBSERVATION: readback matches the requested contents."},{"role":"assistant","content":j({"action":"final","content":"The artifact is verified by readback."})}])

termination = [
("create and verify a report","written but not read back",{"action":"file_read","path":"workspace/report.md"}),
("build and deploy","build passed, deployment pending",{"action":"code_exec","code":"run deployment"}),
("find and report file","located and read successfully",{"action":"final","content":"The requested file was located, read, and is ready to report."}),
("fix test without regressions","original passes, regression remains",{"action":"code_exec","code":"run the regression test"}),
("resolve ambiguous lookup","unique identifier confirms record B",{"action":"final","content":"The ambiguity is resolved with supporting evidence."}),
("publish package","build complete, upload pending",{"action":"code_exec","code":"upload the verified package"}),
("produce valid dataset","generated, schema not validated",{"action":"code_exec","code":"run schema validation"}),
("answer current fact","authoritative current source checked",{"action":"final","content":"The answer is supported by current authoritative evidence."}),
("repair service","restarted, health pending",{"action":"code_exec","code":"run service health check"}),
("complete migration","recorded, post-check pending",{"action":"code_exec","code":"run post-migration verification"})]
for n in range(1):
    for goal,state,action in termination:
        add("termination", [{"role":"user","content":f"Goal: {goal}. Current state: {state}."},{"role":"assistant","content":j({"action":"reason","content":"Compare current state with the complete goal before terminating."})},{"role":"assistant","content":j(action)}])

props = [("token","abc123","step B"),("path","workspace/data/report.csv","read"),("version","7.3","deployment"),("record_id","R-204","update"),("artifact","artifact-42","packaging"),("port","8081","health check"),("branch","release/next","deployment"),("checksum","sha256:abcd","verification"),("job_id","J-91","status query"),("output_dir","workspace/out-v2","packaging")]
for n in range(4):
    for key,value,consumer in props:
        add("state_propagation", [{"role":"user","content":f"Step A produces {key}={value}; {consumer} still has a placeholder."},{"role":"assistant","content":j({"action":"reason","content":"Use concrete runtime state instead of the stale placeholder."})},{"role":"user","content":f"OBSERVATION: {key}={value}."},{"role":"assistant","content":j({"action":"reason","content":f"Propagate {value} into {consumer}."})},{"role":"assistant","content":j({"action":"code_exec","code":f"run {consumer} using {value}"})}])

unc = [("two records share a name","identifier lookup selects record B"),("sources disagree","newer authoritative source documents the value"),("tool omits a field","follow-up query supplies it"),("cache conflicts with source","newer timestamp supports source"),("two formats are possible","task specification selects one"),("memory conflicts with docs","current docs apply"),("two test causes are plausible","focused reproduction isolates the boundary"),("two repositories match","canonical metadata identifies one"),("two endpoints appear active","runtime metadata identifies active endpoint"),("file exists twice","manifest identifies authoritative path")]
for n in range(3):
    for ambiguity,resolution in unc:
        add("uncertainty", [{"role":"user","content":f"Evidence is ambiguous because {ambiguity}."},{"role":"assistant","content":j({"action":"reason","content":"Gather discriminating evidence instead of inventing a state."})},{"role":"user","content":f"OBSERVATION: {resolution}."},{"role":"assistant","content":j({"action":"reason","content":"Update the working state from the new evidence."})},{"role":"assistant","content":j({"action":"final","content":"The ambiguity is resolved using the new evidence."})}])

assert len(rows) == 315, len(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} examples to {OUT}")
