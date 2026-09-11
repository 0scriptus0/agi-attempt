# Discovery Log

This file records major conceptual discoveries from the project's development discussions.

## Discovery 001 — Memory does not need to fit in context

A very large lifetime memory can remain external and searchable. The active model only needs a small relevant subset.

## Discovery 002 — Memory should behave like a search engine

The system should build indexes over a growing experience corpus and retrieve relevant information progressively rather than loading the entire corpus.

## Discovery 003 — Retrieval and reasoning should be coupled

For hard problems, the agent should be able to search, inspect, identify information gaps, and search again.

## Discovery 004 — External information should not blindly overwrite memory

Current documentation and external sources should create evidence and conflict signals. Updates need provenance and validation.

## Discovery 005 — Truth and applicability are different

A memory may be correct in general but irrelevant to the present task. Confidence therefore needs multiple dimensions.

## Discovery 006 — User memory must be isolated from global knowledge

Preferences and personal context should remain scoped to the user unless independently validated as general knowledge.

## Discovery 007 — Weights and experience should serve different roles

The neural model should primarily learn general cognitive skills. Long-term memory should primarily contain experiences and mutable knowledge.

## Discovery 008 — Deduplication and consolidation are first-class systems

A lifelong memory cannot remain a flat append-only log. Repeated experiences should be merged or compressed while preserving evidence and exceptions.

## Discovery 009 — Learning requires a controller

The model should not have unrestricted authority to modify global long-term knowledge. A memory/learning controller should manage ADD, UPDATE, MERGE, SUPERSEDE, ARCHIVE, and REJECT operations.

## Discovery 010 — The hardest problem moves rather than disappears

External memory solves parameter-growth and context-size problems, but introduces retrieval, organization, temporal reasoning, contradiction, and continual-learning challenges. Those must become explicit evaluation targets.
