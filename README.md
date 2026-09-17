# AGI Architecture Research Documentation

This repository documents the architectural ideas, engineering decisions, experiments, and discoveries behind the project.

The goal is not to claim that any single component solves AGI. The goal is to document an evolving system for building a capable, continually improving agent whose long-term knowledge can grow without requiring the core neural model to absorb every experience into its parameters.

## Core principle

> Weights should learn general cognitive capabilities. Memory should learn experience.

The current experiment separates the post-trained model from the runtime cognitive machinery:

`post-trained model -> reasoning/action -> tool execution -> observation -> recovery/reasoning -> verification -> final -> episodic trace/memory`

The model supplies learned behavior. `api.py` supplies the executable environment and feedback loop. `learning_curve/` stores the experiment's memory and interaction traces until a server database is added later.

## Local hypothesis test

No hosted-model API key is required.

1. Train the post-trained model:

```bash
python train.py --stage sft --base-model Qwen/Qwen2.5-7B-Instruct --output-dir checkpoints/lewi1-sft
```

2. Optional preference optimization after SFT:

```bash
python train.py --stage dpo --base-model Qwen/Qwen2.5-7B-Instruct --sft-adapter checkpoints/lewi1-sft --output-dir checkpoints/lewi1-dpo
```

3. Run the local model as the agent:

```bash
export LEWI_MODEL=checkpoints/lewi1-sft
export LEWI_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
python api.py
```

The API loads a PEFT adapter when `LEWI_MODEL` contains `adapter_config.json`; otherwise it treats `LEWI_MODEL` as a full local model directory.

## What the experiment measures

Every run is written to `learning_curve/interactions/YYYY-MM-DD.jsonl` with:

- model actions
- tool inputs and observations
- step count
- memory retrieved
- final result
- stop reason

Durable memories are written to `learning_curve/memory/memories.jsonl`.

The important hypothesis is not whether the agent can call a tool once. It is whether **observations change subsequent decisions**, whether failures cause strategy changes, and whether verified experience can be retrieved later without retraining the neural weights.

## Dataset design

The SFT configuration includes the original reasoning/memory material plus `datasets/reasoning/lewi_cognitive_loop_sft.jsonl`, which explicitly teaches the runtime action protocol: reasoning updates, memory retrieval, memory writes with applicability boundaries, tool execution, recovery, verification, and finalization.

The optional DPO configuration also includes `datasets/reasoning/lewi_cognitive_loop_preference.jsonl`, which contrasts correct recovery/memory/verification behavior against plausible but incorrect agent behavior.

The held-out evaluation set remains separate from SFT data.

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

## Status

This is a living research document. Architectural decisions may change as experiments provide better evidence.
