# SOTA Thinking + SOTA Agentic Thinking: Research Blueprint

## Executive thesis

A strong 2026-era reasoning/agentic system should not be designed as a single long chain-of-thought generator. The emerging state of the art is better described as a **closed-loop cognitive architecture**:

**understand -> decompose -> allocate compute -> generate candidate reasoning -> verify -> act/use tools -> observe -> update belief/state -> recover/re-plan -> verify outcome -> synthesize**

General reasoning and agentic reasoning share a core engine, but the agent adds an environment loop, persistent state, tool planning, evidence management, execution feedback, and trajectory-level learning.

## 1. What SOTA general reasoning is actually using

### 1.1 Test-time compute is a first-class scaling axis

Modern reasoning systems improve by spending more computation at inference rather than relying only on larger parameter counts. Compute allocation should be **adaptive to problem difficulty**, rather than using one fixed thinking budget for every prompt.

Snell et al. show that compute-optimal test-time strategies can substantially outperform naive best-of-N allocation, including more than 4x efficiency improvements in their math experiments and cases where a smaller model with more inference compute beat a 14x larger model under matched FLOPs.

### 1.2 Reinforcement learning can induce reasoning behaviors

DeepSeek-R1 is a major empirical demonstration: reinforcement learning with verifiable rewards can induce behaviors such as self-reflection, verification and dynamic strategy adaptation without requiring human-written reasoning trajectories for the core reasoning signal. Its training uses outcome verification heavily, with rule-based correctness rewards for domains such as mathematics and coding.

The architectural lesson is more important than the exact training recipe: **reward the thing you can verify, then let the model discover useful internal strategies.**

### 1.3 Search over reasoning paths beats a single path

Tree-of-Thoughts demonstrated the benefit of exploring multiple coherent reasoning paths, evaluating intermediate states, and backtracking instead of committing to one token-by-token path. Later test-time-scaling work generalizes the same idea with sampling, verifiers, self-evaluation and other search procedures.

A practical general-reasoning controller should therefore support:

- candidate generation
- branch scoring
- partial-state verification
- pruning
- backtracking
- final candidate selection

### 1.4 Verification is a separate capability

A recurring SOTA pattern is to separate **generation** from **verification**. A verifier may evaluate an entire answer, an intermediate reasoning state, tool results, or a predicted action. The verifier can be a rule-based checker, compiler/test suite, retrieval/evidence check, process reward model, model-based critic, or another reasoning model.

The system should not ask only: "What answer can I generate?" It should ask: **"How can I falsify or independently verify the answer I generated?"**

### 1.5 Self-correction is useful when grounded in feedback

Self-reflection works best when it has an external signal. Reflexion demonstrated a lightweight way to store verbal lessons from previous attempts and use them to change behavior on subsequent attempts. More recent self-verification/self-correction work combines generation, verification, and correction into a test-time loop.

The key design principle is:

**reflection should be triggered by evidence, not performed ceremonially.**

A model that produces "let me reconsider" without new evidence is not necessarily reasoning better.

## 2. What SOTA agentic reasoning adds

### 2.1 Reasoning and acting must be interleaved

ReAct established the basic pattern of alternating reasoning and action so observations from the environment can update the next decision. This is stronger than either pure chain-of-thought or pure action prediction because the agent can use the world as a source of information and correction.

The modern version should go further:

**reason -> action -> observation -> state update -> verification -> reason**

rather than:

**reason -> execute a prewritten workflow**

### 2.2 Long-horizon agents need explicit state

A robust agent should maintain at least five kinds of state:

1. **Task state** — what the user ultimately wants.
2. **Plan state** — current subgoals, dependencies, and completion status.
3. **World state** — observations from tools/environment.
4. **Evidence state** — claims, sources, confidence, contradictions, provenance.
5. **Memory state** — reusable lessons, prior failures, stable preferences, and learned strategies.

Do not treat the raw conversation transcript as the only memory structure.

### 2.3 Tools should expose useful affordances, not merely raw APIs

SWE-agent provides a useful architectural lesson: agent performance depends strongly on the interface between the model and the environment. A tool interface can be designed to make valid actions easier, expose relevant context, and provide useful feedback.

For our blueprint, tools should be typed and state-aware. A tool result should tell the reasoner not just what happened, but enough structured information to support the next decision.

### 2.4 Deep research requires strategic browsing, not just retrieval

BrowseComp is deliberately designed so that basic search is insufficient. The benchmark shows that a strong browsing agent must persist, reformulate searches, reason about factuality, and assemble fragmented evidence. In OpenAI's reported evaluation, Deep Research substantially outperformed ordinary browsing models on this benchmark.

DeepResearcher goes further by training agents with reinforcement learning in authentic, noisy web environments. Its reported emergent behaviors include planning, multi-source cross-validation, self-reflection for research redirection, and calibrated inability to find a definitive answer.

This supports a major design rule:

**the research agent should learn when to search, what to search, when to pivot, and when evidence is sufficient.**

### 2.5 Structured thinking at the tool layer can help

Research on structured reasoning tools for deep research indicates that exposing explicit planning, query-monitoring, and evidence-processing operations can improve agent robustness and efficiency. This suggests a useful hybrid: keep the underlying model capable of free-form reasoning while giving it typed cognitive operations for high-value state transitions.

Examples:

- `plan_subtasks`
- `compare_hypotheses`
- `generate_queries`
- `check_evidence`
- `resolve_conflict`
- `assess_completion`
- `decide_replan`

These are not separate "brains"; they are structured control points around the same reasoning policy.

## 3. Proposed unified architecture

### Layer A — Foundation reasoner

The base model should be strong at language, abstraction, coding, mathematics, multimodal understanding, and general knowledge. It should support configurable reasoning effort.

Output: candidate hypotheses/plans/answers rather than immediate commitment.

### Layer B — Thinking controller

The controller decides **how much and what kind of thinking is warranted**.

Core functions:

- difficulty estimation
- compute budget selection
- reasoning-mode selection
- branch count selection
- verification policy selection
- stop/continue decision

This should implement compute-optimal behavior rather than a constant "think for N tokens" rule.

### Layer C — Search / deliberation engine

Use a mixture of:

- single-path reasoning
- best-of-N sampling
- tree/beam search
- hypothesis comparison
- backtracking
- verifier-guided pruning

Escalate search only when uncertainty or disagreement warrants it.

### Layer D — Verifier stack

Use multiple verifier classes rather than one universal critic.

**Deterministic verifier**
- compiler
- unit tests
- exact answer checker
- schema validator
- mathematical checker

**Evidence verifier**
- source quality
- citation support
- temporal consistency
- claim/source entailment
- conflict detection

**Process verifier**
- intermediate-state correctness
- tool choice quality
- subgoal validity
- plan coherence

**Outcome verifier**
- task completion
- user-defined success criteria
- environment state

### Layer E — Agent loop

The loop is:

1. Interpret task and constraints.
2. Build a task graph.
3. Estimate uncertainty and difficulty.
4. Select an execution/reasoning policy.
5. Generate plan candidates.
6. Verify plan feasibility.
7. Execute the highest-value action.
8. Observe environment/tool output.
9. Update world/evidence/task state.
10. Check whether the current hypothesis still survives.
11. Re-plan if necessary.
12. Verify completion.
13. Produce final answer with evidence and uncertainty.

### Layer F — Memory

Separate memory into:

- episodic trajectory summaries
- procedural lessons
- tool affordance knowledge
- user/task preferences
- reusable verified facts
- failure cases

Memory writes should themselves be gated and verified. Otherwise the agent can permanently encode a hallucinated lesson.

## 4. A concrete thinking policy

### Phase 0: Task parsing

Extract:

- objective
- constraints
- desired output
- success test
- time/cost limits
- allowed tools
- risk level

### Phase 1: Situation model

Construct a compact state representation:

`Goal + Known facts + Unknowns + Constraints + Dependencies + Evidence + Risks`

### Phase 2: Strategy selection

Choose among:

- direct answer
- decomposition
- retrieval
- simulation
- code execution
- multi-hypothesis reasoning
- external research
- iterative execution

Do not use expensive agent loops for trivial tasks.

### Phase 3: Candidate generation

Generate multiple candidates only when:

- uncertainty is high,
- multiple approaches are plausible,
- the consequence of failure is significant, or
- verification is cheap relative to generation.

### Phase 4: Verification

For each candidate ask:

- What observations support it?
- What would falsify it?
- Can it be independently checked?
- Are sources authoritative enough?
- Are there contradictions?
- Is the reasoning internally consistent?

### Phase 5: Action selection

Select the action with the best expected information/value gain subject to cost and risk.

A useful conceptual objective is:

`utility(action) = expected task progress + expected information gain - execution cost - risk`

### Phase 6: Observation update

Never blindly append tool output. Parse it into structured state and distinguish:

- observed fact
- inferred fact
- unverified claim
- error
- contradiction

### Phase 7: Recovery

Failures should produce strategy changes, not just retries.

Examples:

- search result irrelevant -> reformulate query
- source contradictory -> find independent source
- tool error -> change tool/path
- plan blocked -> revise dependency graph
- repeated reasoning failure -> spawn alternative approach

### Phase 8: Stop condition

Stop when:

- task success criteria are satisfied,
- remaining uncertainty is immaterial,
- expected information gain is lower than its cost, or
- the system reaches an explicit uncertainty/abstention threshold.

## 5. Training blueprint

### Stage 1 — General foundation

Train the base model with broad multimodal/code/science/data mixtures.

### Stage 2 — Reasoning SFT

Use high-quality demonstrations for:

- decomposition
- verification
- tool interpretation
- mathematical/code reasoning
- error correction
- calibrated uncertainty

SFT should teach protocols, not overconstrain internal reasoning.

### Stage 3 — Verifiable RL

Use outcome rewards wherever possible.

Examples:

- math -> exact verifier
- coding -> tests
- web research -> claim/evidence grading
- planning -> environment outcome
- structured outputs -> validator

Prefer hard verifiers when available; use learned reward models where the task cannot be deterministically checked.

### Stage 4 — Agentic RL

Train directly in interactive environments.

Each step should expose:

`state -> action -> observation -> reward -> next state`

rather than pretending the entire interaction is one giant sequence.

Use step-level or segment-level credit assignment for tool choices and recoveries.

### Stage 5 — Research/environment diversity

Train in multiple environments:

- web
- browser
- shell
- code repository
- APIs
- files
- structured databases
- multimodal environments

Avoid overfitting to one benchmark interface.

### Stage 6 — Adversarial verification training

Train the model to attack its own solutions.

Useful tasks:

- find a counterexample
- identify unsupported claims
- find the missing source
- detect stale information
- break the proposed code
- locate the incorrect assumption

### Stage 7 — Distillation

Distill long expensive trajectories into smaller models for low-latency execution, while retaining an escalation path to a stronger reasoner.

## 6. Reward architecture

A practical reward hierarchy should be:

`R_total = R_outcome + w_p R_process + w_e R_evidence + w_c R_cost + w_s R_safety`

But weights should be task-dependent.

### Outcome reward

Did the task actually succeed?

### Process reward

Were intermediate decisions useful/correct?

### Evidence reward

Were important claims supported by appropriate evidence?

### Cost reward

Did the agent solve the task without wasteful tool calls or excessive inference compute?

### Safety/constraint reward

Did it respect task boundaries and tool permissions?

Avoid optimizing heavily for stylistic "good reasoning." Reward verifiable progress.

## 7. Compute allocation policy

Use a staged escalation ladder:

**Tier 0:** direct generation

**Tier 1:** short internal deliberation

**Tier 2:** candidate sampling + verification

**Tier 3:** branching search / backtracking

**Tier 4:** tool-assisted investigation

**Tier 5:** multi-agent or specialist decomposition

The agent should climb the ladder only when confidence, disagreement, or task difficulty justifies it.

This is the core mechanism for making the system both strong and economical.

## 8. Multi-agent design

Do not default to many agents.

A strong single reasoner with explicit state and verification should be the default. Add specialized agents only when specialization creates measurable gains.

Useful specialist roles:

- planner
- researcher/browser
- coder/executor
- verifier/critic
- synthesizer

A central coordinator should maintain shared task state and enforce evidence/provenance requirements.

Parallel agents should have deliberately different search strategies or hypotheses; cloning the same policy adds less value.

## 9. Memory design

Memory should not simply be a transcript.

Recommended record:

`memory = {context, lesson, evidence, confidence, source, timestamp, reuse_conditions}`

Write memory only when:

- the lesson is verified,
- it is reusable,
- provenance can be retained,
- it will likely reduce future cost.

Use retrieval plus contradiction checking before trusting old memory.

## 10. Evaluation blueprint

A real SOTA system needs a portfolio, not one benchmark.

### General reasoning

- Humanity's Last Exam — expert knowledge/reasoning
- ARC-AGI-2 — abstraction and novel-task generalization
- difficult mathematics/science benchmarks
- coding/algorithmic reasoning

HLE is valuable because it remains difficult for frontier models despite saturation of many older benchmarks. ARC-AGI-2 is valuable because it specifically stresses novel abstraction and generalization.

### Agentic reasoning

- BrowseComp — persistent hard-to-find web research
- GAIA — broad general-assistant tool use
- Terminal-Bench 2.0 — long-horizon terminal execution
- SWE-bench / coding-agent benchmarks
- real interactive environments

### Internal capability probes

Measure:

- planning depth
- recovery rate after tool failure
- contradiction detection
- evidence completeness
- calibration
- unnecessary tool calls
- compute used per solved task
- successful task completion per unit cost
- ability to recognize unsolvable/underspecified tasks

## 11. Failure modes the blueprint should explicitly target

### Premature commitment

The model chooses a plausible answer before exploring alternatives.

Countermeasure: candidate branching + verification.

### Search drift

The agent keeps browsing but stops making progress.

Countermeasure: progress monitor + novelty tracking + re-plan trigger.

### Evidence laundering

A weak source is repeated many times and treated as corroborated.

Countermeasure: source independence graph.

### Reward hacking

The agent finds a way to score without actually completing the task.

Countermeasure: independent outcome checks and adversarial evaluation.

### Infinite deliberation

The agent spends computation without increasing expected success.

Countermeasure: marginal-information-gain stopping rule.

### Reflection theater

The model produces elaborate self-critique without changing its behavior.

Countermeasure: reflections must result in a changed hypothesis, action, or confidence state.

### Memory poisoning

A previous hallucination becomes a future "fact."

Countermeasure: provenance, confidence, freshness, contradiction checks.

### Tool overuse

The model invokes tools because they exist rather than because they improve the solution.

Countermeasure: tool-value prediction and cost-aware action selection.

## 12. The target cognitive loop

The final design should look like this:

```text
                 ┌───────────────────────┐
                 │      USER TASK        │
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │   TASK / STATE MODEL  │
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │ DIFFICULTY + UNCERTAINTY│
                 │   + COMPUTE BUDGET     │
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │ STRATEGY / PLAN SEARCH │
                 └──────────┬────────────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
      ┌───────────────┐           ┌───────────────┐
      │ REASONING     │           │ TOOL / ACTION │
      │ CANDIDATES    │           │ CANDIDATES    │
      └───────┬───────┘           └───────┬───────┘
              └─────────────┬─────────────┘
                            ↓
                 ┌───────────────────────┐
                 │   VERIFIER STACK     │
                 │ outcome/process/evidence│
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │ SELECT / PRUNE /      │
                 │ BACKTRACK / REPLAN    │
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │ EXECUTE + OBSERVE     │
                 └──────────┬────────────┘
                            ↓
                 ┌───────────────────────┐
                 │ STATE + MEMORY UPDATE │
                 └──────────┬────────────┘
                            │
                    success? ├──── no ─────────┐
                            ↓                  │
                 ┌───────────────────────┐     │
                 │ FINAL VERIFICATION    │     │
                 └──────────┬────────────┘     │
                            ↓                  │
                 ┌───────────────────────┐     │
                 │ ANSWER / ACTION       │     │
                 │ + EVIDENCE + CONFIDENCE│     │
                 └───────────────────────┘     │
                            ↑                  │
                            └──────────────────┘
```

## 13. Recommended implementation order

### Version 1 — Strong thinker

Build:

- base reasoning model
- adaptive thinking budget
- candidate sampling
- verifier
- self-correction loop

### Version 2 — Tool-using reasoner

Add:

- typed tools
- structured state
- action selection
- observation parsing
- recovery/replanning

### Version 3 — Research agent

Add:

- browser/search tools
- evidence graph
- source quality scoring
- claim verification
- persistent research state

### Version 4 — Autonomous agent

Add:

- shell/code execution
- long-horizon planning
- memory
- multi-agent specialists where justified
- trajectory-level RL

### Version 5 — SOTA training stack

Add:

- environment-scale RL
- process + outcome rewards
- compute-optimal inference controller
- adversarial self-verification
- automatic curriculum generation
- distillation from expensive trajectories

## 14. Design principles to preserve

1. **Reasoning is a budgeted search problem, not merely a longer prompt.**
2. **Verification should be independent wherever possible.**
3. **Actions must produce observations that change state.**
4. **The agent should adapt compute to difficulty.**
5. **Reflection must produce behavioral change or confidence updates.**
6. **Memory must have provenance.**
7. **The environment is part of the reasoning process.**
8. **Outcome-grounded RL is preferable to optimizing reasoning style.**
9. **Specialists should be added only when they provide measurable specialization gains.**
10. **The system must know when it does not know.**

## 15. Research basis

- OpenAI, BrowseComp (2025): hard browsing requires persistence, strategic search, and reasoning; Deep Research substantially outperformed ordinary browsing baselines on the benchmark.
- Snell et al., ICLR 2025: compute-optimal test-time scaling can outperform naive inference scaling and, in matched FLOPs settings, beat much larger models.
- Guo et al., Nature 2025, DeepSeek-R1: reinforcement learning can induce self-reflection, verification, and dynamic strategy adaptation on verifiable reasoning tasks.
- Yao et al., Tree of Thoughts (2023): deliberate search over alternative reasoning paths and backtracking can outperform a single CoT path.
- Yao et al., ReAct (2023): interleaving reasoning and acting improves tool-using agents and creates feedback loops from the environment.
- Shinn et al., Reflexion / NeurIPS 2023: language-based reflection plus episodic memory can improve subsequent attempts using task feedback.
- Yang et al., SWE-agent (2024): the agent-computer interface materially affects agent performance, motivating specialized, feedback-rich tool interfaces.
- Mialon et al., GAIA / ICLR 2024: general assistants require multimodal reasoning, browsing, and tool-use proficiency.
- Zheng et al., DeepResearcher / EMNLP 2025: end-to-end RL in real web environments produced gains and emergent planning, cross-validation, self-reflection, and honest failure behaviors.
- Search-R1 (2025): RL can train models to perform multi-turn search interactions instead of treating retrieval as a static augmentation step.
- Huang et al., Deep Research Agents survey (2025): modern research agents combine dynamic reasoning, long-horizon planning, multi-hop retrieval, tool use, and structured report synthesis.
- Anthropic Think Tool research (2025): explicit structured thinking space can improve complex tool-use performance; later extended thinking reduced the need for a separate think tool in many cases.
- ARC Prize 2025/2026 reports: novel-task generalization remains a major frontier; refinement loops and test-time adaptation are important to progress on ARC-AGI-2.
- Humanity's Last Exam (Nature 2026): frontier academic reasoning remains substantially unsolved on a broad expert-level benchmark.
- Terminal-Bench 2.0 (2025): long-horizon terminal agents require realistic execution environments and robust evaluation infrastructure.

## Bottom line

The target should not be a model that merely "thinks harder."

The target should be a model that knows **when to think, how to branch, when to verify, when to act, what to remember, when to recover, when to stop, and when to admit uncertainty**.

That is the most defensible synthesis of the current SOTA trajectory across reasoning models and agentic systems.

## Sources / primary research links

1. OpenAI. “BrowseComp: a benchmark for browsing agents.” 2025. https://openai.com/index/browsecomp/
2. Snell, Charlie Victor, et al. “Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Parameters for Reasoning.” ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html
3. Guo, Daya, et al. “DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.” Nature, 2025. https://www.nature.com/articles/s41586-025-09422-z
4. Yao, Shunyu, et al. “Tree of Thoughts: Deliberate Problem Solving with Large Language Models.” 2023. https://arxiv.org/abs/2305.10601
5. Google Research. “ReAct: Synergizing Reasoning and Acting in Language Models.” https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/
6. Shinn, Noah, et al. “Reflexion: Language Agents with Verbal Reinforcement Learning.” NeurIPS 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
7. Yang, John, et al. “SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.” 2024. https://arxiv.org/abs/2405.15793
8. Mialon, Grégoire, et al. “GAIA: a benchmark for General AI Assistants.” ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/25ae35b5b1738d80f1f03a8713e405ec-Abstract-Conference.html
9. Zheng, Yuxiang, et al. “DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments.” EMNLP 2025. https://aclanthology.org/2025.emnlp-main.22/
10. Jin, Bowen, et al. “Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.” 2025. https://arxiv.org/abs/2503.09516
11. Huang, Yuxuan, et al. “Deep Research Agents: A Systematic Examination And Roadmap.” 2025. https://arxiv.org/abs/2506.18096
12. Anthropic. “The ‘think’ tool: Enabling Claude to stop and think in complex tool use situations.” 2025. https://www.anthropic.com/engineering/claude-think-tool
13. ARC Prize. “ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems.” 2025. https://arcprize.org/blog/arc-agi-2-technical-report
14. Chollet, François, et al. “ARC Prize 2025: Technical Report.” 2026. https://arxiv.org/abs/2601.10904
15. Center for AI Safety, Scale AI, HLE Contributors Consortium. “A benchmark of expert-level academic questions to assess AI capabilities.” Nature, 2026. https://doi.org/10.1038/s41586-025-09962-4
16. Terminal-Bench. “Introducing Terminal-Bench 2.0 and Harbor.” 2025. https://www.tbench.ai/news/announcement-2-0
17. Venktesh, V., et al. “Trust but Verify! A Survey on Verification Design for Test-time Scaling.” 2025. https://arxiv.org/abs/2508.16665
18. Chen, Jiefeng, et al. “SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling.” 2025. https://arxiv.org/abs/2501.19306
19. Cheng, Mingyue, et al. “Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning.” 2025. https://arxiv.org/abs/2511.14460
