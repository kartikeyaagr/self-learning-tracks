---
id: "09.02"
course: "ai-engineering-course"
module: "09"
slug: "the-agent-loop"
title: "The agent loop"
keyIdeas: ["An agent is an LLM using tools in a loop: decide, act, observe the result, repeat until done or a limit is hit.", "Workflows fix the path in code (chaining, routing, parallelisation, orchestrator-workers, evaluator-optimizer); agents let the model choose. Use workflows when the steps are known.", "Durable execution: persist the loop's state after every step so a crash, deploy or long wait resumes instead of restarting, and make side-effecting tools idempotent so a replayed step does not repeat them.", "Human-in-the-loop approval is a persisted pause, not a blocking call: save state, show the exact action, resume on approve, edit or reject.", "A plain loop is enough for short runs; frameworks (vendor agent SDKs, LangGraph) and durable workflow engines (Temporal) pay when runs are long, pause for people, or move money."]
question: "You are building an agent that resolves refund tickets in about 30 tool calls, needs a manager's approval for refunds over 200 USD, and runs on autoscaled workers that restart on every deploy. Design the loop: what state you persist, how approvals work, how you avoid double refunds, and whether you would use a framework."
answer: "Persist the run after every step: messages, tool calls and results, step count, spend, and any pending approval, keyed by run id in a database. On restart, load the latest checkpoint and continue, so a deploy costs at most one step. The refund tool checks the amount; above 200 USD it does not execute but records a pending approval with the exact arguments and ends the worker's turn. A manager approves, edits or rejects in a UI; that event resumes the run with the decision as the tool result. The refund call carries an idempotency key derived from run id and call id, so a step replayed after a crash cannot pay twice. Keep step, budget and retry caps. A plain loop plus a database is fine to start; if runs span days or need timers and many approval types, use a checkpointing framework such as LangGraph or a durable workflow engine such as Temporal rather than rebuilding them."
rubric: ["Persists run state after each step and resumes from the latest checkpoint after a restart.", "Implements approval as a stored pause with the exact arguments, resumed by a human decision (approve, edit or reject), not a blocking call.", "Prevents duplicate side effects with idempotency keys or a check before re-executing.", "Keeps loop guards: step cap, budget, retry limits, escalation.", "Gives a reasoned framework choice tied to run length, pauses and risk."]
sources: [{"title": "Anthropic: Building effective agents", "url": "https://www.anthropic.com/engineering/building-effective-agents", "note": "Workflows vs agents, and five composable patterns."}, {"title": "LangGraph: Durable execution", "url": "https://docs.langchain.com/oss/python/langgraph/durable-execution", "note": "Checkpointers, threads, resuming after failures and interrupts."}, {"title": "Temporal: Understanding Temporal", "url": "https://docs.temporal.io/evaluate/understanding-temporal", "note": "Durable execution: event history, replay, retried activities."}, {"title": "OpenAI Agents SDK", "url": "https://openai.github.io/openai-agents-python/", "note": "A vendor agent SDK: agents, handoffs, guardrails, sessions, tracing."}]
practice: ["run-react-loop", "advance-agent-state", "authorize-tool-call", "react-agent-capstone", "dispatch-tool-call"]
tools: ["LangGraph", "Temporal", "OpenAI Agents SDK", "Claude Agent SDK"]
---

## What an agent is

Anthropic's "Building effective agents" gives the working definition: an LLM using tools in a loop, choosing its next step from feedback from the environment.

~~~dg-flow The agent loop.
Goal -(model decides)-> *Tool call -(execute)-> Observation -(append to context)-> Model decides again -(until done or limit)-> Final answer
~~~

Each turn the model sees the goal, the history and the latest tool results, then calls another tool or finishes. ReAct is the same loop with explicit reasoning between actions; the mechanics of a single call are in [Tool calling that holds up](lesson:aieng:08.03).

## Workflows vs agents

- **Workflows**: LLM calls and tools connected by a path you write in code.
- **Agents**: the model decides the path itself.

| Pattern | What it does | Example |
| --- | --- | --- |
| Prompt chaining | Step 1's output feeds step 2, with checks between | Outline, check outline, write draft |
| Routing | Classify input, send to a specialised prompt or model | Refund vs technical vs sales queries |
| Parallelisation | Run independent subtasks at once, or vote | Several guardrail checks in parallel |
| Orchestrator-workers | One LLM splits work and delegates dynamically | Multi-file code change |
| Evaluator-optimizer | One LLM drafts, another critiques, repeat | Translation with review |

Choose by the task. **Steps known in advance** means a workflow: a weekly report is always fetch, summarise, format. **Steps that depend on what is found** means an agent: in an unfamiliar codebase you don't know which files to read until you have read some. Agents work best where **errors are cheap to detect** (tests, compilers), so the loop can correct itself from real feedback; where errors are **costly or silent**, keep a workflow or a human in the loop. Start with the simplest version that meets the eval, and add autonomy only when it measurably helps. Splitting work across several agents is its own decision, covered in [Multi-agent systems](lesson:aieng:09.06).

## The loop in code

```python
state = load(run_id) or new_run(system, user_task)       # resume if it exists
while state.step < MAX_STEPS:                            # hard cap on steps
    reply = model(state.messages, tools=TOOLS)
    state.messages.append(reply)
    if reply.stop_reason != "tool_use":
        return finish(state, reply.text)                 # model is done
    results = []
    for call in reply.tool_calls:
        if needs_approval(call):
            state.pending = call
            save(state)
            return request_approval(run_id, call)        # pause; resumed later
        results.append(tool_result(call.id, run_tool(call)))  # validated, sandboxed, time-limited
    state.messages.append(results)                       # one result per call, one message
    state.step += 1
    save(state)                                          # checkpoint every step
    if state.spent > BUDGET:
        return escalate(state, "budget exceeded")
return escalate(state, "step limit reached")
```

Every production loop has the same guards: a step cap, a token or cost budget, approval for risky actions, a checkpoint after each step, and a way to hand off to a human.

## Why long tasks fail

If each step succeeds with probability $p$, a task of $n$ independent steps succeeds with $p^n$. At 95% per step, 20 steps succeed 36% of the time. The figure adds the two things that change this: detecting errors and retrying.

~~~dg-raw
%%widget reliability Task success versus number of steps, with and without error detection and retries.
~~~

The lesson is about detection, not retries. A retry only helps if the loop notices the step failed; a wrong file edited or a plausible but wrong number cannot be retried, so it compounds. Detect errors by checking tool results (exit codes, empty results, HTTP errors, fed back as plain text), running tests or validators after every change, asserting invariants in code (the refund is not larger than the order), and requiring evidence before the agent declares success.

## Durable execution and state

The loop above lives in a process. Processes die: deploys, autoscaling, out-of-memory kills, a provider outage that outlasts your retries. If the run's state lives only in memory, a crash at step 25 of 30 restarts from step 1, repeating every model call and every side effect. **Durable execution** means the run survives that: state is persisted at each step and a new worker picks up where the old one stopped.

The state to persist is small and explicit: the message list (including tool calls and results), step count, spend, any pending approval, and the ids of side effects already performed. Store it keyed by run id after each step. Three rules make resuming safe:

- **Idempotent side effects.** A step can run twice: the refund API call succeeded, then the worker died before saving. Pass an idempotency key built from run id and call id so the second attempt is a no-op, or check "was this already done?" before acting.
- **Record results, don't recompute them.** On resume, replay saved tool results instead of calling the tools again.
- **Bound the wait.** A run waiting on a person or a slow job should hold no process at all, only a row in a database.

<div class="ex">
<div class="lab">Worked example</div>

**what a restart costs without checkpoints**

A 100-step run where each step has a 1% chance of a transient failure that kills the worker. Without checkpoints, any failure restarts from step 1. The expected number of steps executed to finish $n$ steps in a row, with per-step success $p$, is

$$
E = \frac{1 - p^n}{(1 - p)\,p^n}
$$

With $p = 0.99$ and $n = 100$: $p^{100} = 0.366$, so $E = \frac{0.634}{0.01 \times 0.366} = 173$ steps. With a checkpoint after every step, only the failed step repeats: $\frac{100}{0.99} = 101$ steps. At an illustrative \$0.05 of model calls per step that is \$8.66 against \$5.05 per run, 71% more spend for no benefit, and every repeated step is a chance to repeat a side effect. For a 30-step run the gap is smaller (35 versus 30 steps); durability matters more the longer runs get.

</div>

## Human-in-the-loop approvals

Some actions need a person first: payments, deletions, external messages, anything irreversible. An approval is a **persisted pause**, not a blocking call:

1. The loop reaches a gated tool call and saves state with the call marked pending.
2. The reviewer sees the exact tool name and arguments (not the model's summary of them) and why it was proposed.
3. They approve, edit the arguments, or reject with a reason.
4. That decision resumes the run. Approve runs the call; edit runs the edited call; reject is returned to the model as a tool result ("rejected by manager: refund only the damaged item") so it can re-plan.

Put the gate in code around the tool, decided by rules (tool type, amount, recipient), not by asking the model whether it needs approval. Set a timeout with a default outcome (usually reject and notify), and log who approved what. LangGraph implements this with `interrupt()` inside a node and a checkpointer keyed by thread id; Temporal uses signals to a waiting workflow. Note that LangGraph re-runs the interrupted node from its start when it resumes, so any side effect placed before the interrupt in that node runs twice.

## Frameworks vs a plain loop

| Option | What you get | What it costs | Pays when |
| --- | --- | --- | --- |
| Plain loop over the provider SDK | Full control, easy debugging, no dependency | You build persistence, tracing, approvals | Short runs, one process, most first versions |
| Vendor agent SDK (OpenAI Agents SDK, Claude Agent SDK, Google ADK) | The loop, tool plumbing, handoffs or sub-agents, tracing, guardrails; the Claude Agent SDK ships Claude Code's file, shell and compaction tooling | Tied to one provider's conventions; prompts partly hidden | You want the vendor's built-in tools and tracing and are staying with that provider |
| Graph framework (LangGraph) | Explicit state graph, checkpoint per step, interrupt and resume, replay from any checkpoint | A graph abstraction to learn; overkill for a simple loop | Stateful flows with human pauses, several providers |
| Durable workflow engine (Temporal and similar) | Event history replay, automatic activity retries, timers, signals, runs lasting days | Operational weight; workflow code must be deterministic, so model and tool calls go in activities | Long-running, money-moving or SLA-bound processes |

The honest default is a plain loop with state in your database. Move up when you find yourself rebuilding checkpointing, resumable approvals or retry policies; those are what the frameworks have already debugged.

## Where this breaks

- **No ground truth per step.** If the loop trusts the model's own claims ("tests pass") instead of tool output, errors compound silently.
- **Approvals that rubber-stamp.** Reviewers shown 50 requests a day stop reading. Gate fewer actions, show diffs, and sample approved actions for audit.
- **State that is not really persisted.** Caches, open connections and in-memory objects referenced by the run do not survive a restart. Save everything the next step needs.

<div class="take">
<div class="lab">Key takeaway</div>

An agent is a model calling tools in a loop until it is done or a limit is hit. Prefer workflows when the steps are known. When you do run an agent, checkpoint every step, make side effects idempotent, treat approvals as persisted pauses, and reach for a framework or durable engine when runs get long or pause for people.

</div>
