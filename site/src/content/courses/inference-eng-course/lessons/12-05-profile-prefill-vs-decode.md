---
id: "12.05"
course: "inference-eng-course"
module: "12"
slug: "profile-prefill-vs-decode"
title: "Profile Prefill vs Decode"
keyIdeas: ["Profile a pure prefill step and a pure decode step separately.", "Prefill: GEMMs dominate, compute bound; attention grows with length squared.", "Decode: memory-bound GEMMs, KV-reading attention, more relative overhead.", "The same model has very different optimization targets per phase.", "Mixed steps (continuous batching) blend both profiles."]
question: "In a profile of prefill and decode for the same model, which kernels' share of time changes most, and why?"
answer: "In prefill, large GEMMs dominate and attention grows with prompt length squared, both compute bound. In decode, GEMMs remain large in share but become memory bound and much shorter per token, attention becomes a bandwidth-bound read of the KV cache, and small kernels (norms, RoPE, sampling) plus launch overhead take a much larger share of the step."
sources: [{"title": "Splitwise", "url": "https://arxiv.org/abs/2311.18677", "note": "Phase-level characterization."}, {"title": "PyTorch Profiler recipe", "url": "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", "note": "Tool for both captures."}]
tags: ["profiling"]
---

## Two different workloads on one GPU

Prefill and decode run the same model code but stress the GPU in opposite ways. Prefill is compute bound with large matmuls. Decode is memory bound with small ones and much more overhead. In a real server they are mixed in each step, which blurs the picture. Profiling them separately makes each part's cost clear.

## Two captures

1. **Prefill only.** One request with a 4,000-token prompt and `max_tokens = 1`, on an idle server. The whole step is prefill.
2. **Decode only.** 32 sequences at about 1,000 tokens of context, all decoding. Capture a few steady-state steps after their prefills are done.

Capture both with Nsight Systems and the same NVTX ranges, then group kernel time by type.

## A typical breakdown

| Kernel group | Share of prefill step | Share of decode step |
| --- | --- | --- |
| QKV and output projections | about 25% | about 25% |
| MLP matmuls | about 50% | about 45% |
| Attention | about 15%, grows with prompt length | about 15 to 30%, grows with context × batch |
| Norms, RoPE, residual adds | about 5% | about 5 to 10% |
| LM head and sampling | small | about 5% |
| GPU idle | small | 10% or more without CUDA graphs |

The numbers are illustrative. The differences between the two columns are the point:

- Decode has more overhead and idle time, because each kernel is small.
- Decode attention is memory bound and grows with the total cached context in the batch.
- Prefill's matmuls dominate even more, and they are compute bound.

~~~dg-raw
%%widget roof Switch between Decode and Prefill. The same model sits far left of the ridge in decode and on the compute roof in prefill.
mode = Decode
n = 5
%%
~~~

## What to optimize in each phase

| Phase | Main targets |
| --- | --- |
| Prefill | FP8 tensor cores, FlashAttention for long prompts, chunk sizing, prefix caching to skip work |
| Decode | fewer bytes (quantized weights and KV), CUDA graphs, fused small ops, efficient paged attention, larger batches |

An optimization that helps one phase may do nothing for the other. INT4 weight-only quantization speeds decode at small batch and does nothing for prefill. FP8 activations speed prefill and help decode less.

<div class="ex">
<div class="lab">Worked example</div>

**two profiles, two plans**

Prefill of 4,000 tokens on an 8B model takes 120 ms. Matmuls are 78% of it, running at 62% of peak BF16 FLOPs. Plan: FP8 W8A8, which should cut prefill to about 75 ms.

A decode step at batch 32 takes 9 ms. Matmuls are 5.2 ms at 85% of peak bandwidth, attention is 1.6 ms, small ops 0.8 ms and idle time 1.4 ms. Plan: enable CUDA graphs for the idle time, then FP8 weights for the matmuls.

Without separate captures, the mixed-step profile would have shown matmuls at "about 65% of something", and neither plan would have been obvious.

</div>

## Mixed steps in production

Once you understand both phases separately, profile real traffic. A mixed step's time is roughly the decode step's time plus the cost of its prefill chunk, minus some overlap because the chunk's matmuls use compute the decode left idle. Comparing mixed steps with your two reference profiles shows whether chunk sizing and scheduling behave as expected.

## Deep dive: arithmetic intensity of the two phases

For a weight matmul with ⟦m696⟧ tokens in the step:

⟦M695⟧

| Phase | Tokens per step ⟦m697⟧ | Intensity (FLOP/byte) | Where on an H100 (ridge about 295) |
| --- | --- | --- | --- |
| Decode, batch 1 | 1 | 1 | Deep in the memory-bound region |
| Decode, batch 64 | 64 | 62 | Still memory bound |
| Decode, batch 256 | 256 | 228 | Near the ridge |
| Prefill, 4k tokens | 4,096 | 1,365 | Compute bound |

Decode attention is different: its intensity is about the GQA group size ⟦m698⟧ (4 for Llama 3 8B) at any batch, because each sequence reads its own KV. Batching does not move it toward the ridge.

<div class="take">
<div class="lab">Key takeaway</div>

prefill and decode stress the GPU differently: prefill is compute bound, decode is memory and overhead bound. Profile them separately so you know which optimization helps which phase, then check mixed steps against both.

</div>
