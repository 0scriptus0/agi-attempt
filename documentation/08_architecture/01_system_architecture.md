# System Architecture

## High-level design

```text
                         USER TASK
                            |
                            v
                     TASK ANALYZER
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
         MEMORY SEARCH   WEB/DOCS      TOOLS/TESTS
              |             |             |
              +-------------+-------------+
                            |
                            v
                    EVIDENCE FUSION
                            |
                            v
                   REASONING ENGINE
                            |
                            v
                         VERIFIER
                            |
                            v
                          ANSWER
                            |
                            v
                         OUTCOME
                            |
                            v
                    LEARNING ENGINE
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
           ADD           UPDATE        SUPERSEDE
             |              |              |
             +--------------+--------------+
                            |
                            v
                     LONG-TERM MEMORY
```

## Major modules

1. Foundation model
2. Reasoning controller
3. Agent planner
4. Memory controller
5. Retrieval engine
6. External search interface
7. Tool manager
8. Evidence fusion layer
9. Verifier stack
10. Long-term memory store
11. Personalization layer
12. Evaluation and telemetry system

## Boundary principle

The neural model is the reasoning engine, not the entire system.

The surrounding architecture provides memory, tools, verification, and learning control.
