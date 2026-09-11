# General Reasoning Architecture

## Problem

Strong reasoning should not be implemented as a fixed-length chain of thought that is equally expensive for every task.

Different tasks have different difficulty, uncertainty, and verification requirements.

## Proposed reasoning loop

```text
TASK
  -> understand
  -> estimate difficulty and uncertainty
  -> choose compute budget
  -> choose reasoning strategy
  -> generate candidate approaches
  -> verify or test
  -> select / backtrack
  -> synthesize
```

## Adaptive inference

The system should estimate whether additional computation is likely to improve the outcome.

Possible actions include:

- answer directly,
- think longer,
- branch into multiple hypotheses,
- search for evidence,
- run code,
- execute a tool,
- request additional information,
- verify an intermediate result.

## Search over reasoning paths

For difficult tasks, the reasoner should be able to maintain multiple candidate solutions rather than committing immediately to the first plausible path.

A candidate can be:

- continued,
- verified,
- pruned,
- revised,
- or abandoned.

## Verification as a first-class operation

Verification should occur whenever correctness can be tested.

Examples:

- mathematics -> symbolic or numeric verification,
- code -> tests and execution,
- research -> source/evidence verification,
- structured output -> schema validation,
- planning -> environment outcome.

## Core rule

The system should optimize for verified progress, not for producing a convincing-looking reasoning trace.
