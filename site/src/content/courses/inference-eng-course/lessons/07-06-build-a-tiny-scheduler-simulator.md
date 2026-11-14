---
id: "07.06"
course: "inference-eng-course"
module: "07"
slug: "build-a-tiny-scheduler-simulator"
title: "Build a Tiny Scheduler Simulator"
keyIdeas: ["Simulate the engine loop with a simple step-time model instead of a GPU.", "Step time ~ a + b x decode tokens + c x prefill tokens (+ attention term).", "Fit the coefficients from a few real measurements.", "Replay request traces under different policies and compare TTFT and TPOT.", "Simulators are fast and let you try ideas before touching an engine."]
question: "What inputs and outputs should a minimal continuous-batching simulator have to answer \"what happens to TTFT if I halve max_num_batched_tokens?\""
answer: "Inputs: a request trace (arrival time, prompt length, output length), a cost model for step time as a function of decode tokens and prefill tokens, KV capacity in tokens, and scheduler parameters (token budget, max seqs, chunked prefill on or off). Outputs: per-request TTFT, TPOT, E2E, and per-step batch composition, so you can compare percentiles across settings."
sources: [{"title": "Vidur: A Large-Scale Simulation Framework for LLM Inference", "url": "https://arxiv.org/abs/2405.05465", "note": "A full-featured inference simulator."}, {"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Scheduling policies worth simulating."}]
tags: ["scheduling", "lab"]
---

## Why simulate

A real rate sweep takes hours of GPU time, and testing a new scheduling idea in a real engine means changing complex code. A simulator replays thousands of requests in seconds and lets you test ideas safely. Production teams use simulators such as Vidur to explore configurations before touching hardware. A simple one fits in a page of Python, and building it explains most scheduler trade-offs better than reading about them.

## A cost model for one step

The simulator needs to know how long a step takes. A linear model fitted to a few real measurements works surprisingly well:

step time = a + b × decode tokens + c × prefill tokens + d × total context read

- **a**: fixed overhead per step, including the weight read at small batch. For an 8B model on an H100, about 5 to 7 ms.
- **b**: extra cost per decode sequence, roughly tens of microseconds.
- **c**: cost per prefill token, about 16 GFLOP ÷ achieved FLOP/s, or about 25 to 30 µs per token.
- **d**: cost per cached token read by attention, from KV bytes per token ÷ bandwidth.

Fit the four numbers by timing a handful of real steps with different mixes. Real step time is not exactly linear, especially near the compute roof, but this is good enough for comparing policies.

## The loop

```
t = 0.0
waiting, running, done = [], [], []
while trace or waiting or running:
    waiting += arrivals_until(t)
    budget = max_batched_tokens
    decodes = running[:max_seqs]
    budget -= len(decodes)
    prefills = []
    for r in list(waiting):
        chunk = min(r.remaining_prompt, budget)
        if chunk == 0 or not kv_fits(r, chunk):
            break
        prefills.append((r, chunk))
        budget -= chunk
    dt = step_time(decodes, prefills)
    t += dt
    advance(decodes, prefills, t)
```

`advance` does the bookkeeping: it moves finished prompts from waiting to running and records their first-token time, appends a token to each decode, records token times, and moves finished requests to `done`. `kv_fits` tracks blocks used, so you can model memory limits and preemption too.

## What to record

For every request: arrival time, first-token time, every token time and finish time. From these, compute the same metrics as your benchmark harness: TTFT, TPOT, ITL, E2E, throughput and goodput. Reusing the harness's summary code keeps the two comparable.

## Experiments to run

1. **Queue policy.** FCFS against shortest-prompt-first on p50 and p95 TTFT. Shortest-first should lower the median and raise the tail for long prompts.
2. **Token budget.** 512, 2,048 and 8,192 on TTFT and p99 ITL. You should see the trade-off from lesson 07.04.
3. **Chunked prefill.** Turn chunking off by allowing a whole prompt in one step. Mix in 2% of 30,000-token prompts and watch p99 ITL.
4. **Memory.** Shrink the KV pool and add preemption. Watch how tail E2E reacts when the pool is 10% too small.

<div class="ex">
<div class="lab">Worked example</div>

**what the simulator shows about budgets**

Workload: 8 requests per second, 90% at 500 tokens in, 10% at 8,000 tokens in, all with 200 tokens out.

Budget 8,192: p95 TTFT 210 ms, p99 ITL 330 ms. The long prompts prefill in one step and stall everyone.

Budget 1,024: p95 TTFT 420 ms, p99 ITL 48 ms. Long prompts take 8 steps, but streams stay smooth.

Budget 2,048: p95 TTFT 290 ms, p99 ITL 75 ms. Often the best compromise.

</div>

## Validate before you trust it

Run one configuration in both the simulator and a real engine with the same trace. If p50 and p95 TTFT and TPOT agree within 10 to 20%, trust the simulator for relative comparisons: which policy is better, and roughly by how much. Do not trust it for absolute capacity numbers without checking them on hardware.

Common reasons for disagreement: scheduler overhead not in the model, CUDA graph padding to fixed batch sizes, and non-linear step time near the compute roof.

## Deep dive: fitting the simulator's cost model

Log real steps with their composition, then fit a linear model by least squares:

⟦M517⟧

- ⟦m518⟧ is the fixed cost: weight reads and launch overhead.
- ⟦m519⟧ is the per-prefill-token cost above the ridge.
- ⟦m520⟧ is the KV read cost per cached token.
- ⟦m521⟧ is the per-sequence overhead: sampling and metadata.

<div class="ex">
<div class="lab">Worked example</div>

**a fit and its check**

From 20,000 logged steps: ⟦m522⟧ ms, ⟦m523⟧ µs per token, ⟦m524⟧ µs per cached token, ⟦m525⟧ µs per sequence.

Sanity check: ⟦m526⟧ should be about ⟦m527⟧ µs. It is. ⟦m528⟧ should be about ⟦m529⟧ µs. It is.

On a held-out hour of steps, the mean absolute error is 4%. Errors larger than about 10% usually mean a missing term, such as attention compute for long prefills.

</div>

A max() model, as in 07.04, is more faithful near the ridge. A linear fit is simpler and good enough when most steps are either clearly below or clearly above it.

<div class="take">
<div class="lab">Key takeaway</div>

a few dozen lines of simulation explain most scheduler trade-offs. Fit a step-time model to real measurements, reuse your harness's metrics, and validate against one real run before trusting the results.

</div>
