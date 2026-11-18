---
id: "09.06"
course: "ai-engineering-course"
module: "09"
slug: "multi-agent-systems"
title: "Multi-agent systems"
keyIdeas: ["A multi-agent system runs more than one model loop, each with its own context; the main patterns are orchestrator-workers, handoffs and parallel fan-out.", "It pays when work splits into independent, read-heavy parts: separate contexts stay small and parallel workers cut wall-clock time.", "The price is tokens: Anthropic measured multi-agent research at about 15x the tokens of a chat; the multiplier is roughly total tokens across all loops over a single agent's.", "Coordination fails in predictable ways: vague task specs, duplicated work, conflicting writes, lost context between agents and nobody verifying the result.", "Parallelise reads, serialise writes, give each worker a full task spec, and keep a single agent when steps are tightly coupled."]
question: "A team proposes splitting its single coding agent into a planner, five parallel coder agents and a reviewer. When would that help, when would it hurt, and how would you decide?"
answer: "It helps when the work splits into parts that are independent and read-heavy, for example investigating five unrelated modules or searching several sources, because each worker keeps a small clean context and they run in parallel. It hurts for a typical code change, because edits to one codebase are tightly coupled: parallel coders make conflicting assumptions and writes, the planner's summary drops details they need, and the token bill rises several-fold. Decide with an eval: run both designs on the same task set and compare success rate, cost per solved task and wall-clock time, reading traces for duplicated work and conflicts. A middle ground often wins: one agent that writes, with read-only subagents for exploration that return short summaries, and a separate verifier that runs tests rather than another model's opinion."
rubric: ["Identifies independence and read-heaviness as the condition for gains", "Names context isolation and parallel wall-clock as the benefits", "Quantifies or explains the token cost multiplier", "Names coordination failures: conflicting writes, lost context in handoffs, duplicated work, unverified outputs", "Proposes deciding with an eval on success, cost per solved task and latency, not by architecture preference"]
sources: [{"title": "Anthropic: How we built our multi-agent research system", "url": "https://www.anthropic.com/engineering/multi-agent-research-system", "note": "Orchestrator-workers for research; 90.2% gain on an internal eval; about 15x chat tokens; token use explains 80% of BrowseComp variance."}, {"title": "Cognition: Don't Build Multi-Agents", "url": "https://cognition.ai/blog/dont-build-multi-agents", "note": "Share full context and traces; actions carry implicit decisions that parallel agents contradict."}, {"title": "Cemri et al.: Why Do Multi-Agent LLM Systems Fail? (2025)", "url": "https://arxiv.org/abs/2503.13657", "note": "MAST taxonomy: 14 failure modes in 3 categories from 1,600+ traces across 7 frameworks."}, {"title": "A2A protocol specification", "url": "https://a2a-protocol.org/latest/specification/", "note": "A protocol for agents from different systems to delegate tasks to each other."}]
practice: ["run-react-loop", "evaluate-agent-trajectory", "react-agent-capstone", "dispatch-tool-call"]
tools: ["OpenAI Agents SDK", "LangGraph", "A2A"]
tags: ["Agents"]
---

## What counts as multi-agent

One agent is one model loop with one context ([09.02](lesson:aieng:09.02)). A multi-agent system runs several loops, each with its own context window and often its own prompt, tools and model. The differences between designs are who decides what runs next and what information crosses between loops.

| Pattern | Control | What crosses between agents | Typical use |
| --- | --- | --- | --- |
| Orchestrator-workers | A lead plans, spawns workers, merges results | Task spec down, summary up | Research, broad search, per-file analysis |
| Handoff | Control transfers to a specialist, which talks to the user | The conversation so far | Support triage to billing or refunds |
| Parallel fan-out (map-reduce) | Code splits input; same prompt per chunk | Chunk in, result out | Review 300 contracts, label a dataset |
| Pipeline / roles | Fixed sequence: planner, writer, reviewer | Artifact passed along | Draft then critique ([09.02](lesson:aieng:09.02) evaluator-optimizer) |

Fan-out and pipelines are really workflows with several prompts; the code decides. Orchestrator-workers and handoffs let a model decide, which is where the new failure modes live. When agents from different vendors or organisations must delegate to each other, protocols such as A2A define task exchange between agents, while MCP ([09.03](lesson:aieng:09.03)) connects an agent to tools.

~~~dg-flow Orchestrator-workers: specs go down, short summaries come up.
User query -(plan)-> *Lead agent -(task spec 1)-> Worker A
Lead agent -(task spec 2)-> Worker B
Lead agent -(task spec 3)-> Worker C
Worker A -(summary + sources)-> Synthesis by lead
Worker B -(summary + sources)-> Synthesis by lead
Worker C -(summary + sources)-> Synthesis by lead
~~~

## Why it can help

Three mechanisms, all visible in Anthropic's write-up of its research system:

- **More total work.** On BrowseComp, Anthropic found token usage alone explained 80% of performance variance (with tool-call count and model choice explaining most of the rest). A team of agents is a way to spend more tokens on a problem than one context can hold. Its lead-plus-subagents system beat a single agent on the same lead model by 90.2% on an internal research eval.
- **Context isolation.** Each worker explores with a clean, small context and returns a condensed result, so the lead's context does not fill with raw search output ([09.01](lesson:aieng:09.01) on context rot).
- **Parallel wall-clock.** Running 3–5 subagents at once, each making several tool calls in parallel, cut research time by up to 90% on complex queries in the same report.

## The cost multiplier

Anthropic's data: agents use about 4x the tokens of a chat interaction, and multi-agent systems about 15x. For your own design the multiplier is simply

$$
m = \frac{T_{\text{lead}} + \sum_i T_{\text{worker}, i}}{T_{\text{single}}}
$$

in tokens, or in dollars once each loop's model price is applied.

<div class="ex">
<div class="lab">Worked example</div>

**A research report, single agent versus lead plus five workers.** Illustrative prices: large model \$5 / \$25 per million input / output tokens (cache reads \$0.50), small model \$1 / \$5 (cache reads \$0.10). Assume 85% of each loop's input is read from the prompt cache.

| Design | Loops | Input tokens | Output tokens | Cost |
| --- | --- | --- | --- | --- |
| Single agent, large model | 20 steps, average context 25k | 500k | 12k | \$0.89 |
| Lead (large) | 8 steps, average context 20k | 160k | 8k | \$0.39 |
| 5 workers, large model | 15 steps each, average context 25k | 1,875k | 45k | \$3.33 |
| 5 workers, small model | same | 1,875k | 45k | \$0.67 |

Total tokens: 2.09M versus 0.51M, a **4.1x** token multiplier. In dollars, lead plus large workers costs \$3.72, **4.2x** the single agent; with small workers it costs \$1.05, **1.2x**.

Wall-clock at about 6 s per step: the single agent takes 20 × 6 = 120 s. The multi-agent run takes about 8 lead steps × 6 s plus 15 worker steps × 5 s in parallel, about 123 s, while doing four times the work. A single agent doing the same 83 steps in sequence would take over 8 minutes, and its context would pass 100k tokens.

</div>

Two cost details people miss. Each worker starts a new context, so its first call is a cache write, not a read; short-lived workers get less caching benefit than one long loop. And worker output flows back into the lead's context, so ask for summaries with sources, not transcripts.

## Coordination: what goes wrong

The MAST study (Cemri et al., 2025) annotated over 1,600 traces from seven multi-agent frameworks and found 14 failure modes in three groups: **system design** (bad role or task specification, step repetition, losing conversation history, not knowing when to stop), **inter-agent misalignment** (withholding information, ignoring other agents' input, task derailment, reasoning that does not match actions) and **task verification** (premature termination, no or incomplete checking). In practice that looks like:

- **Vague task specs.** "Research the semiconductor shortage" sent to three workers produces three overlapping searches. Anthropic's fix was a spec per worker: objective, output format, tools and sources to use, and clear boundaries.
- **Conflicting decisions.** Cognition's argument: every action carries implicit decisions (style, data format, assumptions), and parallel agents that cannot see each other's actions make incompatible ones. Two coders writing parts of one feature is the classic case.
- **Telephone game.** Each handoff summarises; details the next agent needs get dropped. Let workers write results to shared storage (files, a database) and pass references, rather than relaying everything through the lead.
- **Nobody verifies.** The lead trusts a worker's "done". Verification should be external (tests, citations that resolve, a checker), not another agent's opinion.
- **Runaway delegation.** Agents spawning agents without a budget. Cap depth, worker count and total tokens in code.

Rules that follow: **parallelise reads, serialise writes** (one agent owns each artifact), share full context or traces with any agent that must make decisions consistent with others, and give the system one owner of the final answer.

## Shared state

Workers need some state in common: the task list, what has been found, what is claimed. Options in rising order of coupling: no sharing (pure fan-out, results merged by code); a shared, append-only scratchpad or file store that workers read and write by key; a shared task board with claims to stop duplicate work. Durable execution from [09.02](lesson:aieng:09.02) matters more here: a lead that crashes after spawning five workers should resume and collect their results, not start over.

<div class="take">
<div class="lab">Key takeaway</div>

Use several agents when work splits into independent, read-heavy parts that would overflow one context or take too long in sequence. Expect a several-fold token bill, cut it with cheaper workers and short summaries, and keep writes and final verification with one owner.

</div>

## When a single agent wins

- **Tightly coupled work**: most code changes, a single document, anything where step 7 depends on the details of step 3.
- **Low value per task**: a 4x bill must buy a 4x-valuable improvement.
- **Shared context needed everywhere**: if every worker needs the whole history, isolation buys nothing.
- **Hard-to-check outputs**: more agents produce more unverified claims.

A common middle ground is one agent that makes all edits, with read-only helper agents for exploration that return summaries.

## Evaluating it

Evaluate the whole system on outcomes and on cost per solved task ([10.05](lesson:aieng:10.05)), then read per-agent traces for duplicated work, ignored results and early stops. Compare against a single-agent baseline on the same tasks; the architecture has to earn its multiplier.

Interviewers probe exactly that: "why not one agent?", "what does this cost per task?", "how do workers avoid duplicating work?", "who resolves conflicts between two agents' outputs?", and "how do you test it?".
