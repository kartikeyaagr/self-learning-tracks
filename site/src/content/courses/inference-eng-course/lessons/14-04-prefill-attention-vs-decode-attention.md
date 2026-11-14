---
id: "14.04"
course: "inference-eng-course"
module: "14"
slug: "prefill-attention-vs-decode-attention"
title: "Prefill Attention vs Decode Attention"
keyIdeas: ["Prefill: many queries per sequence, high reuse, compute bound.", "Decode: one query per sequence, low reuse, memory bound.", "Decode at small batch and long context underuses the GPU without splitting KV.", "Split-KV / flash-decoding: parallelize over the KV length, then reduce.", "GQA lets one KV load serve several query heads; good kernels exploit it."]
question: "Why is a FlashAttention kernel tuned for prefill a poor fit for decode, and what technique helps long-context decode?"
answer: "Prefill kernels tile over many queries per sequence, so each loaded K/V tile is reused by a big Q tile and tensor cores stay busy. In decode there is one query per sequence (or a few per KV head with GQA), so there is little reuse and too few query tiles to fill the GPU at small batch. Split-KV (flash-decoding) partitions the long KV sequence across many thread blocks, computes partial results in parallel, and reduces them, keeping all SMs busy."
sources: [{"title": "Flash-Decoding for long-context inference", "url": "https://pytorch.org/blog/flash-decoding/", "note": "Split-KV decode attention."}, {"title": "FlashInfer", "url": "https://github.com/flashinfer-ai/flashinfer", "note": "Kernels specialized for prefill, decode and paged KV."}]
tags: ["attention"]
---

## The shapes differ

| Property | Prefill | Decode |
| --- | --- | --- |
| Queries per sequence | prompt length | 1 per step |
| Keys per sequence | prompt length | the full cached context |
| Bound by | compute | memory bandwidth |
| Main source of parallelism | many query tiles | batch × heads, small at low batch |

A kernel tuned for one is poor at the other. FlashAttention's prefill tiling groups 64 to 128 queries per program. In decode there is only one query per sequence and head, so most of that tile would be empty.

## The decode parallelism problem

A natural decode kernel assigns one program to each (sequence, KV head) pair, and loops over that sequence's whole context inside the program.

<div class="ex">
<div class="lab">Worked example</div>

**an idle GPU**

Batch 4, 8 KV heads: 32 programs. An H100 has 132 SMs.

100 SMs have nothing to do. The 32 busy ones each read a 32,000-token KV sequence alone, serially.

The kernel reaches perhaps 20% of peak bandwidth, and attention dominates the step. Adding more sequences would help, but at low traffic there are none to add.

</div>

## Split-KV, or flash-decoding

The fix is to split each sequence's KV along its length and process the pieces in parallel.

~~~dg-flow Split-KV decode attention for one sequence and head.
KV sequence -> *Split into chunks, one program each -> Partial output, max and sum per chunk -> Combine with log-sum-exp -> Output
~~~

1. Divide the context into chunks, for example 8 chunks of 4,000 tokens for a 32,000-token sequence.
2. Each chunk's program computes a partial attention output over its keys, plus its local maximum m and denominator l.
3. A small reduction combines the partial results exactly, using the same rescaling as online softmax: each partial output is weighted by l_i × exp(m_i − m_global), and the sum is divided by the total denominator.

Now the example above launches 32 × 8 = 256 programs, enough to fill every SM. The reduction adds a little extra work, which is negligible at long context.

## GQA-aware decode

With grouped-query attention, several query heads share one KV head. Llama 3 8B has 4 query heads per KV head, and Llama 3 70B has 8. A naive kernel that handles each query head separately loads the same K and V from HBM four or eight times.

A GQA-aware kernel loads each KV tile once and computes scores for all the query heads in its group together. This turns several matrix-vector products into one small matrix-matrix product, multiplying the reuse of every loaded byte by the group size. For a bandwidth-bound kernel, that is a direct speedup.

| Technique | Problem it solves | Typical gain |
| --- | --- | --- |
| Split-KV | too few programs at small batch and long context | 2 to 5 times at batch 1 to 8 and 16k+ context |
| GQA grouping | repeated loads of the same KV head | up to the group size, in KV traffic |
| FP8 KV | bytes per cached token | up to 2 times in attention time |
| Paged layouts with large blocks | many short, scattered reads | modest, kernel dependent |

~~~dg-raw
%%widget batch Raise the context to 32k with a small batch. The KV read dominates the step, which is exactly where split-KV and FP8 KV pay off.
model = Llama 3 70B
gpu = H100 SXM (80 GB)
ctx = 15
b = 2
%%
~~~

## Mixed batches

With chunked prefill, one step contains decode tokens and prefill chunks together. Engines handle this either with one kernel that supports variable query lengths per sequence, or by splitting the batch into a prefill part and a decode part and calling a specialized kernel for each. Both are common. The second gives each phase its best kernel at the cost of two launches.

## Deep dive: how many splits to use

Decode attention without splitting launches one program per (sequence, KV head): ⟦m789⟧. To fill ⟦m790⟧ SMs with about ⟦m791⟧ programs each:

⟦M788⟧

At batch 1 with 8 KV heads on 132 SMs and ⟦m792⟧: 33 splits. A 32k context gives each split about 1,000 tokens. At batch 64, ⟦m793⟧ already fills the GPU and one split is enough.

~~~dg-raw
%%chart Decode attention time per layer at batch 1 against context (illustrative: about 15% of peak bandwidth unsplit, 75% split).
x = 4096, 16384, 32768, 131072
logx = 1
logy = 1
xlab = Context (tokens)
ylab = µs per layer
No split: 34, 134, 268, 1074
Split-KV: 6.7, 27, 54, 215
%%
~~~

Times 32 layers, the 32k case is 8.6 ms against 1.7 ms of attention per step: the difference between a 14 ms and a 7.5 ms TPOT.

<div class="take">
<div class="lab">Key takeaway</div>

decode attention needs its own kernels. Split the KV length across programs so every SM has work, load each KV head once for all its query heads, and aim for full memory bandwidth.

</div>
