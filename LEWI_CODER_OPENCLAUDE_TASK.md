# OpenClaude -> Lewi-Coder migration task

You are modifying the existing OpenClaude codebase in this checkout. Do not build a second coding agent beside it. Reuse the existing provider layer, streaming/token display, terminal integration, tool registry, configuration, and UI wherever possible.

Target: make the existing OpenClaude runtime become Lewi-Coder.

Requirements:

1. Add a persistent task queue with task IDs, priorities, dependencies, checkpoints, events, retries, cancellation, and resume-after-restart.
2. Replace the fixed conversational turn loop with a task state machine. A task can perform arbitrarily many model/tool iterations. Do not use an arbitrary max-step limit as the completion mechanism.
3. Every iteration must follow: inspect state -> model chooses one action -> execute -> capture observation -> update state -> verify -> continue/recover/finalize.
4. Never accept the model saying 'done' as sufficient. Completion requires explicit verification evidence tied to the task acceptance criteria.
5. On failure, provide the exact tool result/error/diff to the next model iteration and require a changed strategy when appropriate. Avoid blind repetition.
6. Persist a checkpoint before moving to the next iteration so a process crash can resume the task.
7. Recover stale running tasks after restart using a lease/heartbeat mechanism.
8. Add a scheduler that creates normal queue tasks. Scheduled jobs must use the same executor, verification, logging, and permission system.
9. Preserve real-time model token streaming in the existing OpenClaude UI.
10. Expose task status, current action, verification status, retry state, and event history in the existing UI without making it visually busy.
11. Keep the existing model/provider abstraction. Lewi-Coder must work with the currently supported OpenClaude backends instead of hard-coding one provider.
12. Integrate the Lewi research runtime from lewi_coder/ where appropriate rather than duplicating TaskStore/verification concepts.
13. Permission handling must be explicit and configurable. Do not bypass OS permissions, authentication, sudo, sandboxing, filesystem ownership, or network/account security. 'Unrestricted' means the agent does not add unnecessary application-level restrictions when the user has already authorized the operation.
14. Add graceful shutdown and automatic resume.
15. Add tests for queue persistence, dependency scheduling, checkpoint recovery, retry behavior, verification failure, cancellation, and restart recovery.
16. Add a dry-run/test mode so the state machine can be tested without executing destructive commands.
17. Log every model action, tool input, observation, verification result, state transition, and terminal result.
18. Avoid recursive self-invocation. The 24/7 loop must be a durable worker/event loop with bounded resource usage per individual process operation.

Acceptance test:
- Queue a multi-file coding task.
- Have the model edit files, run tests, observe a failure, change the implementation, rerun tests, and continue without a fixed step ceiling.
- Kill the process during execution.
- Restart it.
- Confirm the same task resumes from its checkpoint rather than starting over.
- Confirm it only becomes completed after verification passes.
- Confirm a scheduled task enters the same queue and follows the same lifecycle.

When implementation is complete, run the project's tests/lint/type checks that are available and report actual results. Do not claim tests passed unless they were executed.
