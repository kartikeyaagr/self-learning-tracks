---
id: "26.04"
course: "inference-eng-course"
module: "26"
slug: "agentic-multi-turn-inference"
title: "Agentic & Multi-Turn Inference"
keyIdeas: ["Agents: many dependent calls per task, long shared prefixes, short outputs.", "Prefix caching and session-affinity routing give large wins.", "Optimize per-call TTFT and E2E; task latency adds up.", "Tool execution gaps: keep KV warm (or offloaded) between calls.", "Priorities: an agent's later calls should not starve behind new tasks."]
question: "How do agent workloads differ from chat for an inference service?"
answer: "Agents make many sequential calls per task, each with a long, growing prompt (instructions, tool definitions, history, tool outputs) and often short outputs, and the next call waits for the previous one plus tool execution. So prefix reuse is very high, TTFT and E2E per call matter more than streaming speed, and total task latency is the sum of many calls. Bursty fan-out (parallel tool calls) is also common."
sources: [{"title": "SGLang paper", "url": "https://arxiv.org/abs/2312.07104", "note": "Programs with many LLM calls sharing prefixes."}, {"title": "Mooncake", "url": "https://arxiv.org/abs/2407.00079", "note": "KV reuse for long multi-turn contexts."}]
tags: ["agents"]
---

## Workload shape

| Property | Chat | Agent |
| --- | --- | --- |
| Model calls per task | 1 | 5 to 100 or more |
| Prompt length | short to medium | long and growing |
| Output | medium to long | often short (tool calls) |
| Prefix reuse | moderate | very high |
| Key metric | TTFT, TPOT | per-call latency, task completion time |

An agent calls the model, runs a tool, appends the tool's result to the context, and calls the model again. Each call's prompt is the previous call's prompt plus a little more.

~~~dg-raw
%%widget agent Set 40 calls with a 10k starting context and compare the two lines. Without reuse, prefill grows with the square of the number of calls.
n = 20
c0 = 6000
g = 800
%%
~~~

## What helps

- **Prefix caching with sticky routing:** most of each call's prompt was in the previous call. Route all calls of one task to the same replica.
- **KV retention between calls:** tools often take seconds. If the engine evicts the session's KV during that time, the next call pays the full prefill again. Offload it to CPU memory instead of dropping it.
- **TTFT focus:** outputs are short, so TTFT dominates each call. Prefill speed and cache hits matter more than decode speed.
- **Structured output** for reliable tool calls.
- **Scheduling:** favor in-progress tasks, so tasks finish rather than all advancing slowly together.

<div class="ex">
<div class="lab">Worked example</div>

**one coding agent task**

30 calls, starting from a 10,000-token context, adding about 1,500 tokens per call (file contents, test output, the model's edits).

Without reuse, the engine prefills 10,000 + 11,500 + ... about 950,000 tokens in total. On one GPU at 20,000 tokens per second, that is about 48 seconds of prefill.

With a cache hit on every call, it prefills about 10,000 + 29 × 1,500 ≈ 53,500 tokens, under 3 seconds.

</div>

## Deep dive: KV held while the tool runs

An agent alternates model calls and tool calls. During a tool call, the conversation's KV sits idle in GPU memory. Over a task, the share of KV-time that is idle is

⟦M1111⟧

~~~dg-raw
%%timeline One agent task: model calls and tool calls. KV stays resident throughout (s, illustrative).
unit = s
tick = 2
Model | 0-1.2 plan c1 ; 3.2-4.0 read result c1 ; 7.0-7.9 edit c1 ; 9.9-10.6 answer c1
Tools | 1.2-3.2 search flame ; 4.0-7.0 run tests flame ; 7.9-9.9 run tests flame
KV in GPU memory | 0-10.6 held c3
%%
~~~

Here the model runs for 3.6 s out of 10.6: KV is idle 66% of the time. Options: keep it (fast resume, low concurrency), offload it to CPU during long tool calls (20.04), or drop it and rely on the prefix cache to recompute. The right choice depends on tool duration against reload time.

<div class="take">
<div class="lab">Key takeaway</div>

agent traffic is long, repetitive prompts with short outputs in dependent chains. Prefix reuse, KV retention and TTFT matter most.

</div>
