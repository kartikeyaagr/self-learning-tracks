---
id: "19.05"
course: "inference-eng-course"
module: "19"
slug: "all-to-all-communication-grouped-gemm"
title: "All-to-All Communication & Grouped GEMM"
keyIdeas: ["All-to-all dispatches tokens to experts and combines results: twice per MoE layer.", "Prefill: large messages, bandwidth bound. Decode: small messages, latency bound.", "Specialized libraries offer high-throughput (prefill) and low-latency (decode) all-to-all modes.", "Grouped GEMM runs many variable-size expert matmuls in one kernel.", "Overlap communication with computation (e.g. split batches) to hide it."]
question: "Why does MoE need grouped GEMM kernels, and why is all-to-all latency critical for decode?"
answer: "Each expert gets a different, often small, number of tokens; launching one GEMM per expert wastes time and underuses the GPU. A grouped GEMM runs all experts' matmuls in one kernel with per-group sizes. In decode, few tokens per step means small messages, so all-to-all time is dominated by latency, and with two all-to-alls per MoE layer over dozens of layers it can rival compute time; low-latency all-to-all kernels and overlapping with compute are needed."
sources: [{"title": "DeepEP", "url": "https://github.com/deepseek-ai/DeepEP", "note": "Efficient expert-parallel communication library."}, {"title": "MegaBlocks", "url": "https://arxiv.org/abs/2211.15841", "note": "Block-sparse kernels for MoE."}]
tags: ["moe", "kernels"]
---

## Two kinds of all-to-all

| Phase | Message size | Bound by | Approach |
| --- | --- | --- | --- |
| Prefill | large: thousands of tokens per GPU | bandwidth | high-throughput kernels over NVLink and RDMA |
| Decode | small: a few tokens per GPU | latency | low-latency kernels, fewer hops, no extra copies |

Libraries such as DeepEP provide both modes and plug into serving engines.

<div class="ex">
<div class="lab">Worked example</div>

**dispatch volume in decode**

128 tokens per GPU, k = 8, hidden size 7168, FP8 activations (1 byte each).

Each token sends 8 copies of 7 KB, so each GPU sends about 128 × 8 × 7 KB = 7.3 MB per MoE layer, and the combine sends the same back in BF16 (about 14.7 MB).

Over 400 Gb/s RDMA (50 GB/s), that is roughly 0.15 + 0.3 ms per layer. With 58 MoE layers, communication alone would be about 26 ms per step unless it overlaps with compute.

</div>

## Overlap

Split the batch into two micro-batches. While one micro-batch is in all-to-all, the other runs attention or experts. Done well, this hides most of the communication time.

~~~dg-seq Two micro-batches overlapping.
Compute > Compute : attention, batch 1
Compute > Network : dispatch batch 1
Compute > Compute : attention, batch 2
Network > Compute : batch 1 arrives
~~~

## Grouped GEMM

Each expert has its own weights and its own variable number of tokens.

```
for each expert e:
    Y_e = X_e @ W_e      # many small GEMMs, each with a different row count
```

Launching one kernel per expert is slow: 256 launches per layer, most with tiny work. A grouped GEMM kernel takes all groups and their offsets and handles them in one launch, assigning tiles to experts in proportion to their token counts.

Benefits:

- One launch instead of hundreds.
- Better SM use, since busy experts get more tiles.
- Some implementations fuse the permutation and the activation function.

## Decode efficiency

At small batch, each expert sees few tokens, so its GEMM is memory bound and the weight read buys little compute. The fix is to aggregate more tokens across the deployment: data-parallel attention on many GPUs feeds one large expert-parallel pool, so each expert sees tokens from all of them.

## Deep dive: hiding all-to-all with two micro-batches

The dispatch and combine all-to-alls in each MoE layer can be as long as the expert compute itself. Splitting the batch into two halves lets one half communicate while the other computes:

~~~dg-raw
%%timeline One MoE layer, without and with two-micro-batch overlap (illustrative, µs).
unit = µs
tick = 100
Compute, serial | 0-100 attn c3 ; 220-380 experts c3
Network, serial | 100-220 dispatch c1 ; 380-500 combine c1
Compute, overlapped | 0-50 attn A c3 ; 50-100 attn B c3 ; 160-240 experts A c3 ; 240-320 experts B c3
Network, overlapped | 50-110 dispatch A c1 ; 110-170 dispatch B c1 ; 240-300 combine A c1 ; 320-380 combine B c1
%%
~~~

⟦M961⟧

instead of their sum. In the illustration, the layer drops from 500 to 380 µs. The overlap is never perfect, because the first dispatch and last combine have nothing to hide behind.

<div class="take">
<div class="lab">Key takeaway</div>

MoE serving depends on two kernels besides attention: all-to-all and grouped GEMM. Decode needs low-latency communication, prefill needs bandwidth, and both need overlap.

</div>
