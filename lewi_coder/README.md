# Lewi-Coder

Persistent autonomous coding orchestration built around the Lewi research runtime.

The execution model is task queue -> model action -> tool -> observation -> state update -> verification -> continue/recover/finalize.

The important change from the current Lewi API is that task lifetime is not tied to a fixed model-step limit. Work continues until acceptance criteria are verified, the task is cancelled, or a resource/safety policy stops it.

Permission profiles are explicit: restricted, workspace, and unrestricted. Unrestricted removes arbitrary application-level restrictions but does not bypass OS ownership, authentication, sudo, sandboxing, network controls, or account permissions.

State lives under .lewi/ so unfinished tasks can resume after process restarts. The OpenClaude-derived adapter should provide model planning, tool execution, observation feedback, verification, streaming, and finalization without duplicating existing provider/UI infrastructure.
