---
id: "21.02"
course: "inference-eng-course"
module: "21"
slug: "size-prefill-decode-worker-pools"
title: "Size Prefill & Decode Worker Pools"
keyIdeas: ["Prefill demand = request rate x prompt tokens (minus cache hits).", "Decode demand = request rate x output tokens, constrained by KV capacity and TPOT.", "Measure each pool's SLO-safe capacity separately.", "Ratio of pools follows the input/output token ratio of traffic.", "Re-balance as traffic changes; some systems let workers switch roles."]
question: "Traffic is 5 requests/s with 3,000-token prompts and 300-token outputs. A prefill GPU processes 20,000 tokens/s within TTFT SLO; a decode GPU sustains 2,500 output tokens/s within TPOT SLO. How many of each do you need?"
answer: "Prefill demand is 5 x 3,000 = 15,000 tokens/s, so 0.75 GPUs: round up to 1 (plus headroom, so 1 to 2). Decode demand is 5 x 300 = 1,500 tokens/s, so 0.6 GPUs: 1. With headroom for bursts, about 2 prefill and 1 to 2 decode GPUs. The ratio shifts as prompt and output lengths change."
sources: [{"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Placement and sizing algorithm."}, {"title": "Splitwise", "url": "https://arxiv.org/abs/2311.18677", "note": "Pool provisioning."}]
tags: ["disaggregation", "capacity"]
---

## Demand

Each pool has its own demand, computed from traffic:

```
prefill_tokens_per_s = rate × avg_prompt_tokens × (1 − prefix_hit_rate)
decode_tokens_per_s  = rate × avg_output_tokens
decode_concurrency   = rate × avg_decode_time      # Little's law, for KV sizing
```

## Capacity per GPU

Measure each separately, under its SLO:

- **Prefill GPU:** the highest prompt tokens per second that keeps TTFT p95 within the SLO, including queueing time.
- **Decode GPU:** the highest output tokens per second that keeps TPOT p95 within the SLO, and the most concurrent sequences its KV memory holds.

## Worked example

| Item | Value |
| --- | --- |
| Rate | 5 requests/s |
| Average prompt, average output | 3,000 tokens, 300 tokens |
| Prefill demand | 15,000 tokens/s |
| Prefill GPU capacity | 20,000 tokens/s |
| Decode demand | 1,500 tokens/s |
| Decode GPU capacity | 2,500 tokens/s |
| Decode concurrency at 30 ms TPOT | 5 × 300 × 0.03 s ≈ 45 sequences |
| Plan | 1 to 2 prefill GPUs, 1 decode GPU, plus headroom |

~~~dg-raw
%%widget pools Change the prompt and output lengths and watch the ratio of prefill to decode GPUs move. Then raise the prefix cache hit rate.
r = 5
p = 11.5
n = 8.25
h = 0
%%
~~~

## Shifting ratios

- **RAG traffic** (long prompts, short answers) needs more prefill GPUs. At 8k prompts and 200-token answers, the plan above becomes about 2 prefill GPUs per decode GPU at 5 requests/s.
- **Creative writing or reasoning** (short prompts, long outputs) needs more decode GPUs.
- **Prefix caching** cuts prefill demand directly. A 60% hit rate removes more than half the prefill pool.

Traffic mix changes during the day, so monitor both demands and adjust pool sizes on a schedule or automatically.

## Deep dive: the prefill-to-decode ratio from first principles

At arrival rate ⟦m992⟧, with per-GPU prefill capacity ⟦m993⟧ tokens/s and decode capacity ⟦m994⟧ output tokens/s:

⟦M991⟧

For Llama 3 8B on H100s: ⟦m995⟧ tokens/s (600 TFLOP/s at 15 GFLOP per token) and ⟦m996⟧ tokens/s at batch 128 and 1k context. With equal utilization targets, ⟦m997⟧.

~~~dg-raw
%%bars Prefill GPUs needed per decode GPU, by workload shape (equal utilization, 1k decode context).
unit = ×
Chat (500 in, 500 out) | 0.25 | c3
RAG (4,000 in, 200 out) | 5.0 | c2
Summarization (8,000 in, 300 out) | 6.6 | flame
Code completion (1,500 in, 50 out) | 7.5 | c1
Agent step (12,000 in, 150 out) | 19.9 | wine
%%
~~~

Long decode contexts lower ⟦m998⟧ (05.03), which pulls these ratios down, and prefix caching lowers effective ⟦m999⟧. Measure both before fixing the pool sizes. The spread is the point: a single ratio cannot serve chat and agents well.

<div class="take">
<div class="lab">Key takeaway</div>

size each pool from its own demand and its measured SLO-safe capacity. The prefill-to-decode ratio follows the traffic's ratio of uncached prompt tokens to output tokens.

</div>
