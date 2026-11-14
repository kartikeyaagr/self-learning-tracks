---
id: "20.02"
course: "inference-eng-course"
module: "20"
slug: "long-context-prefill-attention-bottlenecks"
title: "Long-Context Prefill & Attention Bottlenecks"
keyIdeas: ["At long lengths, attention dominates prefill FLOPs.", "Prefill of very long prompts can take tens of seconds on one GPU.", "Chunked prefill keeps other users' decode flowing but does not cut total work.", "Options: faster attention kernels, FP8 attention, context parallelism, prefix caching for repeated documents.", "Sparse or sliding-window attention models reduce cost by design."]
question: "For a 128k-token prompt, roughly what fraction of prefill FLOPs is attention for an 8B model with 32 layers and hidden size 4096?"
answer: "Linear layers cost about 2 x 8B x 128k, about 2e15 FLOPs. Attention scores and weighted sum cost about 4 x layers x seq^2 x hidden / 2 for causal, roughly 2 x 32 x (128k)^2 x 4096, about 4.4e15 FLOPs. So attention is around two thirds of prefill FLOPs at this length."
sources: [{"title": "FlashAttention-3", "url": "https://arxiv.org/abs/2407.08608", "note": "Fast attention for long sequences."}, {"title": "Ring Attention", "url": "https://arxiv.org/abs/2310.01889", "note": "Splitting long sequences across GPUs."}]
tags: ["long-context", "prefill"]
---

## The arithmetic

For a prompt of T tokens:

- **Linear layers:** about 2 × parameters × T FLOPs.
- **Causal attention:** about 2 × layers × T² × hidden size FLOPs. This counts QKᵀ and PV, halved because the causal mask skips the upper triangle.

For an 8B model (32 layers, hidden size 4096):

| Prompt | Linear FLOPs | Attention FLOPs | Attention share |
| --- | --- | --- | --- |
| 8k | 1.3e14 | 1.7e13 | about 12% |
| 32k | 5.1e14 | 2.7e14 | about 35% |
| 128k | 2.0e15 | 4.4e15 | about 68% |

Doubling the prompt doubles the linear work and quadruples the attention work. Past about 60k tokens on this model, attention costs more than all the linear layers together.

<div class="ex">
<div class="lab">Worked example</div>

**TTFT for a 128k prompt**

Total FLOPs are about 6.4e15. An H100 delivers about 989 TFLOP/s in BF16. At a realistic 50% utilization, prefill takes about 6.4e15 ÷ 4.9e14 ≈ 13 seconds.

At 32k tokens the same arithmetic gives about 1.6 seconds. Four times the prompt costs eight times the time.

</div>

~~~dg-raw
%%widget roof Switch to Prefill and push the prompt length up. Long prefill is compute bound, so the only ways to cut TTFT are fewer FLOPs, faster kernels or more GPUs.
gpu = H100 SXM (80 GB)
mode = Prefill
n = 11
%%
~~~

## Consequences

- **TTFT grows quickly past 32k.** Interactive products that accept very long prompts need a TTFT target per length band, not one number.
- **Long prefills monopolize GPU time.** Chunked prefill spreads them across steps so decode keeps flowing, but the total work is the same.
- **Prefix caching is especially valuable.** A cached 100k-token document saves about 10 seconds of H100 time on every reuse.

## Options

| Option | Effect | Trade-off |
| --- | --- | --- |
| FlashAttention-3, FP8 attention | faster attention kernels, up to about 1.5 to 2 times | needs recent hardware and kernel support |
| Context parallelism | split one prompt across GPUs, cutting TTFT | communication of K and V between GPUs |
| Prefix caching and KV reuse | skip repeated prefill entirely | memory, needs reuse |
| Sliding-window or sparse attention models | less attention by design | model choice, may lose long-range recall |

## Context parallelism in one picture

~~~dg-flow Ring attention across four GPUs.
Split prompt in 4 chunks -> Each GPU computes Q for its chunk -> *Pass K and V around the ring -> Each GPU attends to every chunk -> Concatenate outputs
~~~

Each GPU holds a quarter of the tokens. K and V blocks rotate around the ring while each GPU computes attention against the block it currently holds. When compute per block is longer than the transfer, the communication hides behind it.

## Deep dive: one long request in units of short ones

Prefill FLOPs grow as ⟦m964⟧. Relative to a 1k-token request:

⟦M963⟧

~~~dg-raw
%%bars Prefill work of one request, in multiples of a 1,024-token request (Llama 3 8B).
unit = ×
8k tokens | 9 | c3
32k tokens | 50 | c2
128k tokens | 415 | c1
%%
~~~

A 128k prompt is 128 times longer but 415 times more work, because of the quadratic attention term. It also holds 128 times the KV for its whole life. For capacity planning, one such request is worth hundreds of ordinary ones, and its 11-second prefill (at 600 TFLOP/s) is a stall for everyone else unless chunked.

<div class="take">
<div class="lab">Key takeaway</div>

past a few tens of thousands of tokens, attention dominates prefill. Speed it up with better kernels and more GPUs, or avoid it with caching.

</div>
