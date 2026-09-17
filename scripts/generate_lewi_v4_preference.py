"""Generate deterministic Lewi v4 preference pairs focused on state-transition failures."""
import json
from pathlib import Path

OUT = Path("datasets/reasoning/lewi_agentic_state_transition_preference_v4_generated.jsonl")
CASES = [
    ("A planned file read failed because the path does not exist; discovery found workspace/deploy/app.yaml.",
     {"action":"reason","content":"The failed read invalidates the path assumption; use the discovered path instead of repeating the same read."},
     {"action":"file_read","path":"app.yaml"}),
    ("A retrieval timed out twice; the report is available as smaller monthly files.",
     {"action":"reason","content":"Repeated identical timeouts show no progress; switch to the smaller monthly retrieval."},
     {"action":"reason","content":"Retry the same large request."}),
    ("An artifact was written successfully, but its contents have not been checked.",
     {"action":"file_read","path":"artifact.txt"},
     {"action":"final","content":"The artifact is correct because the write succeeded."}),
    ("A memory is explicitly scoped to environment X while the current environment is Y.",
     {"action":"reason","content":"The memory is out of scope for the current environment and should not be applied."},
     {"action":"memory_write","content":"The procedure is globally valid.","memory_type":"lesson"}),
    ("A build milestone is complete, but deployment is still required by the user's goal.",
     {"action":"reason","content":"The milestone is intermediate; the overall goal is not satisfied, so continue to the deployment state."},
     {"action":"final","content":"The task is complete because the build passed."}),
    ("Two records share a display name and the first lookup is ambiguous.",
     {"action":"reason","content":"The evidence does not identify the record, so gather a discriminating identifier before choosing."},
     {"action":"final","content":"The first record is probably the requested one."}),
    ("Step A produced token T=abc123 and step B requires that token.",
     {"action":"reason","content":"The runtime state contains abc123; step B must use that concrete value rather than a placeholder."},
     {"action":"reason","content":"Keep the placeholder for step B."}),
    ("A patch passes lint, but integration tests fail with 401 because the test client omitted its token.",
     {"action":"reason","content":"The integration failure must be corrected and rerun before claiming behavioral verification."},
     {"action":"final","content":"Lint passed, so the patch is verified."}),
    ("A patch fixes one test but causes three regressions sharing a modified helper.",
     {"action":"reason","content":"The regression evidence changes the hypothesis; narrow the shared-helper change and retest."},
     {"action":"reason","content":"The original test passes, so keep the patch unchanged."}),
    ("A current documentation page conflicts with an old memory for the same service version.",
     {"action":"reason","content":"The current authoritative documentation should update the working state; the old memory is historical context."},
     {"action":"memory_write","content":"The old memory overrides current documentation.","memory_type":"lesson"}),
]
rows=[]
for i in range(1, 101):
    prompt, chosen, rejected = CASES[(i-1)%len(CASES)]
    rows.append({"id":f"v4pref_{i:04d}","category":["recovery","verification","memory_read_write","termination","uncertainty","state_propagation","replanning","tool_selection"][i%8],"prompt":prompt,"chosen":json.dumps(chosen,separators=(",",":")),"rejected":json.dumps(rejected,separators=(",",":"))})
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w",encoding="utf-8") as f:
    for row in rows: f.write(json.dumps(row,separators=(",",":"))+"\n")
print(f"wrote {len(rows)} preference pairs to {OUT}")
