# AGI Architecture Research Documentation

This repository documents the architectural ideas, engineering decisions, experiments, and discoveries behind the project.

The goal is not to claim that any single component solves AGI. The goal is to document an evolving system for building a capable, continually improving agent whose long-term knowledge can grow without requiring the core neural model to absorb every experience into its parameters.

## Documentation map

- `01_vision/` — project goals and core principles
- `02_reasoning/` — general-purpose reasoning and adaptive inference
- `03_agentic/` — planning, action, observation, recovery, and tool use
- `04_memory/` — lifelong memory, compression, consolidation, and temporal knowledge
- `05_retrieval/` — memory search, hierarchical retrieval, and retrieval control
- `06_learning/` — feedback, verification, post-training, and continual improvement
- `07_personalization/` — separation of global knowledge, user knowledge, and task state
- `08_architecture/` — complete system architecture and module boundaries
- `09_scaling/` — compute efficiency, parameter growth, and long-horizon scaling
- `10_research/` — research log, hypotheses, experiments, and discoveries

## Core principle

> Weights should learn general cognitive capabilities. Memory should learn experience.

The architecture is designed around this separation so that lifetime experience can grow far faster than the active neural model.

## Status

This is a living research document. Architectural decisions may change as experiments provide better evidence.