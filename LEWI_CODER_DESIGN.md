# Lewi-Coder design

Goal: turn the OpenClaude runtime into a persistent coding agent while preserving its existing model providers, streaming UI, terminal integration, and tool implementations.

Core adapter contract:
1. plan(task, checkpoint)
2. act(action)
3. observe(tool_result)
4. verify(task_state)
5. finalize(task_state)

Loop:
- load checkpoint
- inspect unfinished objectives
- request one next action
- execute it
- capture stdout, stderr, exit status, and diffs
- feed the observation back to the model
- update state
- verify relevant acceptance criteria
- recover/re-plan after failure
- checkpoint before continuing
- mark completed only when evidence passes

There is intentionally no arbitrary max_steps field in the task state machine. Resource policies may still stop execution.

24/7 operation must survive model failures, malformed output, tool failures, provider outages, retries, and restarts. Scheduled automation should enqueue ordinary tasks into the same queue instead of creating a separate execution engine.

Do not implement permission bypass. The useful target is explicit user-authorized execution with no redundant agent-level restrictions while retaining operating-system security boundaries.

First milestone: streaming output, persistent queue, checkpoints, resume/cancel, tool execution, observation feedback, verification, recoverable retries, indefinite worker operation while runnable work exists, UI task/event status, and scheduled task creation.
