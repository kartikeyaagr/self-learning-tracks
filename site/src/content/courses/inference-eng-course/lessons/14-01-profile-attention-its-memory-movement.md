---
id: "14.01"
course: "inference-eng-course"
module: "14"
slug: "profile-attention-its-memory-movement"
title: "Profile Attention & Its Memory Movement"
keyIdeas: ["Naive attention materializes the S x S score matrix in HBM.", "Its memory traffic grows with sequence length squared and dwarfs Q, K and V.", "Profile: attention share of step time vs sequence length.", "In decode, attention reads the KV cache: traffic proportional to context x batch.", "The fix for prefill is IO-aware tiling (FlashAttention); for decode, efficient KV reads."]
question: "How much HBM traffic does naive attention generate for one head with sequence length 8k and head dim 128 in BF16, compared with the inputs?"
answer: "Q, K and V are each 8k x 128 x 2 bytes = 2 MB, 6 MB total. The score matrix is 8k x 8k x 2 bytes = 128 MB, written and read back at least twice (scores, softmax output), so hundreds of MB of extra traffic, about 50 to 100 times the inputs. That traffic dominates naive attention time."
sources: [{"title": "FlashAttention", "url": "https://arxiv.org/abs/2205.14135", "note": "IO analysis of attention."}, {"title": "PyTorch: scaled_dot_product_attention", "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", "note": "Fused attention entry point."}]
tags: ["attention", "profiling"]
---

## Naive attention's data flow

The textbook formula for attention is softmax(Q Kᵀ ÷ √d) V. A direct implementation runs it as separate steps, and each step writes its full result to HBM before the next step reads it back.

~~~dg-flow Naive attention round-trips through HBM at every step. Press Play to follow the data.
Q, K -> Scores S×S in HBM -> Softmax S×S in HBM -> *Multiply by V -> Output
~~~

1. Compute scores = Q Kᵀ, an S × S matrix, and write it to HBM.
2. Read the scores, apply the causal mask and the softmax, and write the probabilities to HBM.
3. Read the probabilities and V, multiply, and write the output.

## The traffic, for one head

With S = 8,192 tokens, head size 128 and BF16:

| Tensor | Size |
| --- | --- |
| Q, K and V | 2 MB each |
| Scores (S × S) | 128 MB |
| Probabilities (S × S) | 128 MB |
| Output | 2 MB |

The S × S matrices dominate. Writing and reading them costs about 512 MB of traffic for one head in one layer, against 8 MB for the actual inputs and output. Across 32 query heads and 32 layers, that is about 520 GB of traffic for one 8k-token prefill, which would take about 160 ms on an H100 just to move.

The memory footprint is a problem too. At S = 32,768, one head's score matrix is 2 GB. Holding several heads' scores at once quickly exhausts GPU memory.

## Profile it

Run prefill with an eager attention implementation at prompt lengths of 1k, 4k and 16k tokens. Measure attention's share of the step:

| Prompt length | Attention share (eager, illustrative) |
| --- | --- |
| 1,024 | about 10% |
| 4,096 | about 30% |
| 16,384 | about 60%, or out of memory |

Then switch to PyTorch's `scaled_dot_product_attention`, which dispatches to a fused kernel when it can, and repeat. The attention share falls sharply, and memory use stops growing with S².

<div class="ex">
<div class="lab">Worked example</div>

**one profile, two implementations**

An 8B model, 8,192-token prefill on an H100.

Eager attention: 610 ms total, of which attention is 430 ms. Nsight Compute shows the softmax kernel at 85% of peak bandwidth: it is efficient, but it is moving a huge amount of data.

Fused attention: 190 ms total, attention 25 ms. The math is identical. Only the memory traffic changed.

</div>

## Decode is a different problem

In decode, each sequence has one new query. The score matrix is 1 × S per head, which is tiny. The cost is reading the K and V of every cached token, every step:

KV bytes per step ≈ batch size × context length × KV bytes per token

For 64 sequences at 4,000 tokens on Llama 3 8B, that is 64 × 4,000 × 128 KB ≈ 33 GB per step, twice the weights. The optimization target in decode is bandwidth efficiency and fewer KV bytes, not avoiding a score matrix.

## Two problems, two kinds of kernels

- **Prefill attention** must avoid materializing S × S, and should keep the tensor cores busy because it has plenty of math. FlashAttention solves this.
- **Decode attention** must read the KV cache at close to full bandwidth, with enough parallelism to use every SM even at small batch. Split-KV and GQA-aware kernels solve this.

The rest of this module covers both.

## Deep dive: the size of the score matrix

Naive attention materializes ⟦m760⟧ and ⟦m761⟧, each ⟦m762⟧ per head. Their size alone is

⟦M758⟧

For 32 heads in BF16 at ⟦m763⟧: ⟦m764⟧ GB per layer. At ⟦m765⟧ it is 69 GB per layer, which does not fit at all.

Each pass over ⟦m766⟧ or ⟦m767⟧ (write scores, read for softmax, write probabilities, read for ⟦m768⟧) costs ⟦m769⟧ bytes of traffic. Four passes against the causal FLOPs ⟦m770⟧:

⟦M759⟧

~~~dg-raw
%%chart Attention time per layer against sequence length, Llama 3 8B shapes on an H100 (naive: memory time of 4 passes; fused: compute time at 600 TFLOP/s).
x = 2048, 4096, 8192, 16384, 32768
logx = 1
logy = 1
xlab = Sequence length
ylab = ms per layer
Naive (score-matrix traffic): 0.32, 1.28, 5.13, 20.5, 82
Fused (FLOP-bound): 0.057, 0.23, 0.92, 3.67, 14.7
%%
~~~

Both grow with ⟦m771⟧, so the ratio stays near 5.6 times. The difference is that the naive version is bound by memory traffic it did not need, and runs out of memory long before 32k.

<div class="take">
<div class="lab">Key takeaway</div>

naive prefill attention is dominated by writing and reading the S × S score matrix, while decode attention is dominated by reading the KV cache. They are different problems that need different kernels.

</div>
