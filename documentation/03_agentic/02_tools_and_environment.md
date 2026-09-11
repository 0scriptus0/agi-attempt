# Tools and Environment Interface

## Principle

Tools are not merely capabilities attached to a model. They are part of the agent's decision process.

The agent should learn when to use a tool, which tool to use, and whether the result justified the action.

## Tool categories

Possible interfaces include:

```text
search()
open_source()
read_file()
run_code()
run_tests()
run_shell()
inspect_state()
query_database()
call_api()
```

## High-level actions

Where possible, tools should expose meaningful operations instead of unnecessary low-level complexity.

For example, `run_tests()` can be more useful to a reasoning policy than forcing it to reconstruct every shell command needed to execute a test suite.

## Environment feedback

Every tool call should produce a structured observation:

```text
OBSERVED
INFERRED
UNVERIFIED
ERROR
CONTRADICTION
```

This prevents raw tool output from becoming an undifferentiated context dump.
