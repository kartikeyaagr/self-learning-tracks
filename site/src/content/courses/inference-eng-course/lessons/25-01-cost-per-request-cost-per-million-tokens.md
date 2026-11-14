---
id: "25.01"
course: "inference-eng-course"
module: "25"
slug: "cost-per-request-cost-per-million-tokens"
title: "Cost per Request & Cost per Million Tokens"
keyIdeas: ["Cost per token = GPU cost per hour / tokens served per hour.", "Use SLO-safe throughput and real average utilization, not peak benchmark numbers.", "Input and output tokens cost differently; price or attribute them separately.", "Include all costs: idle capacity, headroom, networking, storage, engineering.", "Cost per request depends on its token mix."]
question: "A GPU costs \\$2.50 per hour and sustains 3,000 output tokens/s at the SLO while running at 70% average utilization. What is the cost per million output tokens?"
answer: "Effective throughput is 3,000 x 0.7 = 2,100 tokens/s, or 7.56 million tokens per hour. \\$2.50 / 7.56 is about \\$0.33 per million output tokens. Input tokens add cost too; a complete model would attribute GPU time to both prefill and decode."
sources: [{"title": "Artificial Analysis", "url": "https://artificialanalysis.ai/", "note": "Public price and speed comparisons of LLM APIs."}, {"title": "SemiAnalysis: inference cost articles", "url": "https://semianalysis.com/", "note": "Industry analysis of inference economics (some paywalled)."}]
tags: ["cost"]
---

## The formula

```
cost_per_M_tokens = gpu_cost_per_hour / (tokens_per_second × 3600 × utilization) × 1e6
```

Each term matters as much as the others:

- **gpu_cost_per_hour:** what you actually pay, including any commitment discount.
- **tokens_per_second:** SLO-safe throughput at full load, measured with your traffic.
- **utilization:** the share of paid hours the GPU spends doing useful work, averaged over the whole day.

~~~dg-raw
%%widget cost Set throughput to about 2,000 tokens/s, then move utilization from 70% down to 25%. Cost per token rises as fast as it would if the GPU got slower.
p = 3
t = 3.3
u = 0.5
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one H100 serving an 8B model**

Price: \$3 per hour. SLO-safe throughput: 2,500 output tokens per second. Average utilization over the day: 40%.

Cost per million output tokens = 3 ÷ (2,500 × 3,600 × 0.4) × 1,000,000 ≈ \$0.83.

At 80% utilization, the same GPU costs about \$0.42 per million. Doubling utilization halved the cost, with no change to the engine.

</div>

## Input versus output tokens

Prefill tokens are cheaper per token than decode tokens, because they are processed in parallel and the GPU's compute is used well. API prices reflect this with lower input prices, often 3 to 5 times lower.

To split your own costs:

1. Measure the GPU-seconds spent on prefill and on decode, from engine metrics or profiles.
2. Divide each by the number of tokens of that kind.

## Cost per request

```
cost = prompt_tokens × c_in + output_tokens × c_out
```

<div class="ex">
<div class="lab">Worked example</div>

**RAG versus chat**

Say c_in is \$0.10 and c_out is \$0.50 per million tokens.

A RAG request with 6,000 input and 200 output tokens costs 6,000 × 0.10 + 200 × 0.50 = 700 millionths of a dollar, about \$0.0007.

A chat request with 300 input and 600 output tokens costs 30 + 300 = 330 millionths, about \$0.0003.

The RAG request is shorter to read but costs twice as much, because of its long prompt.

</div>

## What people forget

| Cost | Effect |
| --- | --- |
| Headroom (running at 70%) | about 43% more per useful token |
| Idle off-peak hours | lower average utilization |
| Warm standby replicas | paid for but unused |
| Storage, network, egress | small but real |
| Engineering and operations time | often the largest cost of all |

## Deep dive: why output tokens cost more than input tokens

Charge each token for the GPU time it uses. A prompt token in prefill uses about ⟦m1090⟧ of exclusive GPU time. An output token in decode uses a ⟦m1091⟧ share of a memory-bound step:

⟦M1089⟧

For Llama 3 8B on an H100 at \$2.50 per hour, 25 µs per prompt token, and 0.2 ms per output token at batch 64:

~~~dg-raw
%%bars GPU cost per million tokens by type (Llama 3 8B, H100, $2.50/h, batch 64).
unit = $
Input (prefill) | 0.017 | c3
Output (decode) | 0.139 | c1
%%
~~~

Output is about 8 times as expensive here. The ratio shrinks at larger batch (decode gets cheaper per token) and grows at long contexts (decode reads more KV). It is the physical reason that API prices for output tokens are typically several times those for input.

<div class="take">
<div class="lab">Key takeaway</div>

cost per token is GPU cost divided by tokens actually served at the SLO, including idle time. Split input and output costs, and count everything you pay for.

</div>
