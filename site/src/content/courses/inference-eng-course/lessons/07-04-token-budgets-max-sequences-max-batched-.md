---
id: "07.04"
course: "inference-eng-course"
module: "07"
slug: "token-budgets-max-sequences-max-batched-tokens"
title: "Token Budgets, Max Sequences & Max Batched Tokens"
keyIdeas: ["Token budget (max batched tokens): tokens processed per step, prefill plus decode.", "Max sequences (max num seqs): how many sequences can be in the running batch.", "Budget too small: slow prefills, higher TTFT. Too large: long steps, ITL spikes, activation memory.", "Max seqs too high: per-user TPOT suffers; too low: throughput left on the table.", "Tune both against your SLO with your harness."]
question: "What does max_num_batched_tokens limit, and what happens to TTFT and ITL if you set it very low or very high?"
answer: "It caps how many tokens (decode tokens plus prompt tokens) one step may process. Very low: long prompts are split into many small chunks, so TTFT rises, but decode steps stay short and ITL stays smooth. Very high: big prefills run in one step, lowering TTFT for that request but causing long steps and ITL spikes for everyone else, plus a higher activation memory peak."
sources: [{"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Chunked prefill and batch size tuning."}, {"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Token budgets for stall-free scheduling."}]
tags: ["scheduling", "tuning"]
---

## Two limits on every step

The scheduler fills each step under two limits:

- **max_num_seqs**: the maximum number of sequences that can be running. It caps the decode batch size.
- **max_num_batched_tokens**: the maximum total tokens in one forward pass, the **token budget**. Each decoding sequence contributes one token. A prefilling sequence contributes its prompt length, or the size of its current chunk.

A third limit is implicit: **free KV blocks**. A request cannot be admitted unless its prompt fits in the pool.

## How the scheduler fills a step

The usual policy, used in vLLM with chunked prefill on, is decode first:

1. Give one token to each running decode sequence, up to max_num_seqs.
2. Spend the remaining budget on prompt tokens: first on requests that are part-way through a chunked prefill, then on new requests from the waiting queue.
3. Stop when the budget or the free KV blocks run out.

Putting decodes first keeps TPOT steady for users who are already streaming. Prefills get whatever is left.

<div class="ex">
<div class="lab">Worked example</div>

**a budget of 2,048 with 100 running sequences**

Decodes take 100 tokens. That leaves 1,948 for prefill.

A new 6,000-token prompt is split across 4 steps: 1,948 + 1,948 + 1,948 + 156 tokens.

Its TTFT is about 4 steps. Each of those steps costs roughly the same, so no decoding user sees a large stall.

With a budget of 8,192, the same prompt would finish in one step, but that step would take about four times as long, and the 100 decoding users would feel it.

</div>

## What each setting trades

| Setting | Low value | High value |
| --- | --- | --- |
| max_num_batched_tokens | smooth ITL, slower TTFT for long prompts, less activation memory | fast TTFT, ITL spikes, more activation memory |
| max_num_seqs | lower TPOT per user, lower peak throughput | higher throughput, higher TPOT, more KV pressure |

There is a floor on the budget. It must be at least max_num_seqs, or decodes alone could exceed it. In practice it should be several times larger, so some prefill can always make progress.

## How step time depends on the budget

Roughly, the time of a mixed step is:

step time ≈ fixed overhead + weight read time + KV read time + prefill tokens × compute per token

At small budgets, the weight read dominates and adding prefill tokens is almost free, because the matmuls are memory bound. Past a few hundred tokens, the matmuls become compute bound and each extra prefill token adds time. This is why budgets of 512 to 2,048 often give the best balance for interactive workloads: big enough to use the compute that decode leaves idle, small enough not to stretch the step much.

~~~dg-raw
%%widget roof Switch to Prefill mode. Tokens per step below the ridge are nearly free because the step is still memory bound.
mode = Prefill
n = 8
%%
~~~

## A tuning procedure

1. **Write down the SLOs**, for example p95 TTFT under 800 ms and p95 TPOT under 40 ms.
2. **Fix a realistic load**, such as the expected peak request rate, with your real workload mix.
3. **Sweep the token budget**, for example 512, 1,024, 2,048, 4,096 and 8,192. Record p95 TTFT, p95 TPOT and p99 ITL.
4. **Pick the smallest budget that meets the TTFT SLO.** Smaller budgets give smoother streams.
5. **Sweep max_num_seqs** at that budget, for example 64, 128, 256 and 512, and pick the largest that still meets the TPOT SLO.
6. **Rerun a rate sweep** with the chosen pair to find the new SLO-safe capacity.

## Common mistakes

- **Tuning on an idle server.** These settings only matter under load. Always tune at the target rate.
- **Tuning on the wrong workload.** A budget tuned on short prompts may be far too small for a RAG workload with 8k-token prompts.
- **Raising max_num_seqs beyond KV capacity.** If the pool holds only 120 sequences of your typical size, a cap of 512 only produces preemptions.

## Deep dive: the step-time curve and the ridge in tokens

A step processing ⟦m505⟧ tokens in total reads the weights once and does ⟦m506⟧ FLOPs. Ignoring KV reads:

⟦M504⟧

For BF16 weights on an H100 at 600 TFLOP/s and 2.8 TB/s achieved, ⟦m507⟧ tokens. Below it, extra tokens in a step are almost free. Above it, step time grows linearly, 25 µs per token for an 8B model.

~~~dg-raw
%%chart Predicted step time against tokens in the step (Llama 3 8B, H100).
x = 1, 64, 128, 256, 512, 1024, 2048, 4096
logx = 1
logy = 1
xlab = Tokens in the step
ylab = Step time (ms)
Step time: 6.75, 6.75, 6.75, 7.4, 13.8, 26.6, 52.2, 103.4
%%
~~~

## Deep dive: what the budget does to TTFT and ITL

For a 6,000-token prompt joining 100 decodes, the prompt needs ⟦m508⟧ steps. Adding up the step times from the curve above:

| Budget | Steps to finish prefill | TTFT of the long prompt | Worst ITL for the 100 others |
| --- | --- | --- | --- |
| 256 | 39 | 288 ms | 7.4 ms |
| 512 | 15 | 203 ms | 13.8 ms |
| 1,024 | 7 | 175 ms | 26.6 ms |
| 2,048 | 4 | 164 ms | 52.2 ms |
| 4,096 | 2 | 157 ms | 103 ms |
| 8,192 | 1 | 154 ms | 154 ms |

~~~dg-raw
%%chart The same table as a trade-off curve.
x = 256, 512, 1024, 2048, 4096, 8192
logx = 1
logy = 1
xlab = Token budget per step
ylab = ms
TTFT of the long prompt: 288, 203, 175, 164, 157, 154
Worst ITL for others: 7.4, 13.8, 26.6, 52.2, 103, 154
%%
~~~

Above the ridge, the prompt's total compute is fixed, so its TTFT barely moves: from 175 to 154 ms. The budget mostly decides how that compute is spread, and the others' worst ITL scales with it. Below the ridge, steps are memory-bound and TTFT rises fast, because each step pays the full weight read for only a few prompt tokens. Under load, queueing adds to the TTFT column, which is why the simulator in 07.06 shows larger TTFT differences.

<div class="take">
<div class="lab">Key takeaway</div>

the token budget trades TTFT against smooth streaming, and max sequences trades throughput against per-user speed. Fill decodes first, tune both at the target load against the SLO, and keep the cap within KV capacity.

</div>
