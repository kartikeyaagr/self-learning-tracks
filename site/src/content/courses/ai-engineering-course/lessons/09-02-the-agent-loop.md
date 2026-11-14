---
id: "09.02"
course: "ai-engineering-course"
module: "09"
slug: "the-agent-loop"
title: "The agent loop"
keyIdeas: ["An agent is an LLM using tools in a loop: decide, act, observe the result, repeat until done.", "Anthropic distinguishes workflows (LLM calls in a path fixed by code) from agents (the model directs its own steps and tool use).", "Workflow patterns: prompt chaining, routing, parallelisation, orchestrator-workers, evaluator-optimizer.", "Use workflows when the steps are known in advance; agents when the number and order of steps can't be predicted.", "Agents need stopping conditions (task done, max steps, budget), error recovery, and checkpoints for human review on risky actions.", "Start with the simplest thing that works; add autonomy only when it measurably helps, because agents cost more and their errors compound."]
question: "When should you build a fixed workflow instead of an autonomous agent? Give two criteria and an example of each."
answer: "Use a workflow when the steps are known and fixed in advance (e.g. a weekly report: fetch metrics, summarise, format), and when errors are costly or hard to detect, so predictability matters. Use an agent when the number and order of steps depend on what's discovered (e.g. debugging an unfamiliar codebase) and when results can be checked by real feedback such as tests, so the loop can self-correct."
sources: [{"title": "Anthropic: Building effective agents", "url": "https://www.anthropic.com/engineering/building-effective-agents", "note": "Workflows vs agents, and five composable patterns."}, {"title": "Lilian Weng: LLM Powered Autonomous Agents", "url": "https://lilianweng.github.io/posts/2023-06-23-agent/", "note": "Planning, memory and tool use."}, {"title": "Yao et al.: ReAct (2022)", "url": "https://arxiv.org/abs/2210.03629", "note": "Reason, act, observe as a loop."}]
---

## What an agent is

Lilian Weng's overview describes an agent as an LLM brain plus planning, memory and tool use. Anthropic's "Building effective agents" gives the simplest working definition: an LLM using tools in a loop, deciding its own next step based on feedback from the environment.

~~~dg-flow The agent loop.
Goal -(model decides)-> *Tool call -(execute)-> Observation -(append to context)-> Model decides again -(until done or limit)-> Final answer
~~~

Each turn the model sees the goal, the history and the latest tool results, then either calls another tool or finishes. ReAct (lesson 08.03) is the same loop with explicit reasoning between actions.

## Workflows vs agents

Anthropic splits agentic systems into two kinds:

- **Workflows**: LLM calls and tools connected by a path you write in code.
- **Agents**: the model decides the path itself, dynamically.

Common workflow patterns:

| Pattern | What it does | Example |
| --- | --- | --- |
| Prompt chaining | Step 1's output feeds step 2, with checks between | Outline, check outline, write draft |
| Routing | Classify input, send to a specialised prompt or model | Refund vs technical vs sales queries |
| Parallelisation | Run independent subtasks at once, or vote | Several guardrail checks in parallel |
| Orchestrator-workers | One LLM splits work and delegates dynamically | Multi-file code change |
| Evaluator-optimizer | One LLM drafts, another critiques, repeat | Translation with review |

## When to use which

- **Steps known in advance** -> workflow. Generating a weekly report from fixed sources is always fetch, summarise, format.
- **Steps unknown, depend on what's found** -> agent. Fixing a bug in an unfamiliar codebase: you don't know which files to read until you've read some.
- **Errors are cheap to detect** (tests, compilers) -> agents work well, because the loop can self-correct from real feedback.
- **Errors are costly or silent** -> keep humans in the loop or stay with workflows.

## Making the loop safe

- **Stopping conditions**: task complete, max iterations, token or cost budget.
- **Ground truth each step**: tool results, test outputs, not the model's own claims.
- **Checkpoints**: pause for human approval before irreversible actions.
- **Sandboxing**: run code and file operations in an isolated environment.
- **Good tools**: Anthropic stresses designing the agent-computer interface as carefully as a human UI: clear tool docs, examples, formats that are hard to misuse.

## Why simple first

Agents trade cost and latency for flexibility, and small error rates compound over many steps: 95% per-step reliability over 20 steps is about 36% end to end. Start with a single call plus retrieval, move to a workflow, and only then to an agent when the task really needs it.

## Why long tasks fail

If each step of an agent succeeds with probability p, a task of n independent steps succeeds with p^n. At 95% per step, 20 steps succeed 36% of the time. The figure adds the two things that change this: detecting errors and retrying.

~~~dg-raw
%%widget reliability Task success versus number of steps, with and without error detection and retries.
~~~

The lesson is about detection, not retries. A retry only helps if the loop notices the step failed. Undetected errors (a wrong file edited, a plausible but wrong number) can't be retried, so they compound. Set detection to 0% and retries do nothing.

Ways to detect errors in an agent loop:

- **Check tool results**: non-zero exit codes, empty search results, HTTP errors. Feed them back to the model as plain text.
- **Run tests or validators** after every code change, not just at the end.
- **Assert invariants** in code: the refund is not larger than the order, the file still parses.
- **Ask for evidence**: before finishing, the agent must cite the test output or record that shows the task is done.

## The loop in code

```
messages = [system, user_task]
for step in range(MAX_STEPS):                  # hard cap on steps
    reply = model(messages, tools=TOOLS)
    if reply.stop_reason != "tool_use":
        return reply.text                       # finished
    for call in reply.tool_calls:
        if needs_approval(call): wait_for_human(call)
        result = run_tool(call)                 # validated, sandboxed, time-limited
        messages += [reply, tool_result(call.id, result)]
    if spent_tokens() > BUDGET: return escalate("budget exceeded")
return escalate("step limit reached")
```

Every production loop has the same guards: a step cap, a token or cost budget, approval for risky actions, and a way to hand off to a human.

## Workflow first

Anthropic's guide "Building effective agents" (December 2024) separates workflows, where code fixes the sequence of LLM calls (prompt chaining, routing, parallel calls, orchestrator-workers, evaluator-optimizer), from agents, where the model chooses its own steps. Start with the simplest version that meets the eval. A three-step chain that always runs in the same order is easier to test, cheaper and more predictable than an agent. Use an agent when the number and order of steps truly can't be known in advance.

<div class="take">
<div class="lab">Key takeaway</div>

an agent is a model calling tools in a loop until it decides it's done. Prefer fixed workflows when the steps are known; use agents when they aren't, with limits, real feedback and checkpoints.

</div>
