---
id: "14.02"
course: "inference-eng-course"
module: "14"
slug: "tiled-attention-online-softmax"
title: "Tiled Attention & Online Softmax"
keyIdeas: ["Split K and V into blocks; process one block at a time for a tile of queries.", "Keep running max, running sum and running output per query row.", "When the max changes, rescale earlier sums and outputs.", "The final result is exact, not an approximation.", "This makes attention memory O(S) instead of O(S^2)."]
question: "Explain how online softmax lets you process attention in blocks of keys without storing all scores."
answer: "For each query row you keep a running max m, a running sum of exponentials l, and a running output accumulator. For each new block of keys you compute block scores, update the max, rescale the old sum and accumulator by exp(old_m - new_m), add the new block's exponentials and weighted values, and move on. At the end you divide the accumulator by l. Only one block of scores exists at a time."
sources: [{"title": "Online normalizer calculation for softmax", "url": "https://arxiv.org/abs/1805.02867", "note": "The online softmax trick."}, {"title": "Self-attention Does Not Need O(n^2) Memory", "url": "https://arxiv.org/abs/2112.05682", "note": "Chunked attention with constant memory."}]
tags: ["attention"]
---

## The obstacle: softmax needs the whole row

Avoiding the S × S matrix means processing keys in blocks, without ever holding all scores for a query at once. The difficulty is the softmax. Its denominator is a sum over all keys, and its numerically stable form subtracts the maximum over all keys. Both seem to need the whole row before any output can be computed.

The **online softmax** removes this obstacle. It processes keys block by block, keeping three running values per query row, and corrects earlier work whenever it learns something new.

## The recurrence

For one query row, process key blocks j = 1 to n, where s_j are this block's scores:

```
m_new = max(m, max(s_j))                          # running maximum
l     = l * exp(m - m_new) + sum(exp(s_j - m_new))  # running denominator
acc   = acc * exp(m - m_new) + exp(s_j - m_new) @ V_j  # running output
m     = m_new
# after the last block
out = acc / l
```

- m starts at minus infinity, l at 0 and acc at zeros.
- Each time the maximum increases, the factor exp(m − m_new), which is at most 1, scales down everything accumulated so far, so it is expressed relative to the new maximum.

## Why it is exact

Softmax does not change when the same constant is subtracted from every score: exp(s − c) ÷ Σ exp(s − c) is the same for any c. The running values are always expressed relative to the current maximum. When the maximum changes from m to m_new, multiplying by exp(m − m_new) converts the earlier terms to the new reference exactly. After the last block, l is the full denominator and acc is the full weighted sum, both relative to the true maximum. The result equals standard attention up to floating-point rounding.

<div class="ex">
<div class="lab">Worked example</div>

**the recurrence on four scores**

Scores 1, 3 for block 1 and 2, 5 for block 2. Values are 1-dimensional: 10, 20 for block 1 and 30, 40 for block 2.

Block 1: m = 3. l = e^(−2) + e^0 ≈ 1.135. acc = 0.135 × 10 + 1 × 20 ≈ 21.35.

Block 2: m_new = 5. Rescale by e^(3 − 5) ≈ 0.135: l ≈ 0.154, acc ≈ 2.89. Add the new block: l += e^(−3) + e^0 ≈ 1.050, so l ≈ 1.204. acc += 0.050 × 30 + 1 × 40 = 41.49, so acc ≈ 44.38.

out = 44.38 ÷ 1.204 ≈ 36.9. Computing softmax over all four scores directly gives the same 36.9.

</div>

## Tiling attention with it

~~~dg-flow Tiled attention for one block of queries.
Load a Q tile -> *Loop over K and V tiles: scores, update m, l and acc -> Normalize by l -> Write the output once
~~~

A program handles a tile of, say, 128 queries:

1. Load the Q tile into registers or shared memory.
2. For each K and V tile of, say, 64 keys: compute the 128 × 64 score tile with the tensor cores, apply the scale and causal mask, update m and l, compute the exponentials and accumulate acc with another tensor core multiply.
3. Divide acc by l and write the 128 output rows.

The working set is one Q tile, one K tile, one V tile, one score tile and the running statistics. For head size 128 in BF16, that is well under 100 KB, which fits in an SM's shared memory and registers.

## What it saves

- **HBM traffic** falls from about 4 × S² per head to roughly the size of Q, K, V and the output, times a small factor for re-reading K and V once per query tile.
- **Memory footprint** falls from O(S²) to O(S): only the inputs, the output and the running statistics.
- **Kernel launches** fall from three or more to one.

## Causal masking skips work too

With a causal mask, a query tile at positions 0 to 127 needs only the first key tiles, and tiles entirely above the diagonal can be skipped. This halves the work for prefill. Only tiles on the diagonal need element-wise masking.

## Deep dive: why the recurrence is exact

After processing any set of blocks, the running values satisfy two invariants:

⟦M772⟧

Rescaling by ⟦m773⟧ converts both sums from the old reference ⟦m774⟧ to the new one ⟦m775⟧ without changing their meaning. After all blocks, ⟦m776⟧, which is exactly softmax-weighted ⟦m777⟧. Subtracting the running maximum keeps every exponent at or below 0, so nothing overflows, even in BF16 inputs with FP32 accumulators.

~~~dg-raw
%%grid Tiles of the score matrix for a causal prefill of 8 query blocks × 8 key blocks. Each cell is a 128 × 128 tile.
cols = k1, k2, k3, k4, k5, k6, k7, k8
key F = full tile, no mask needed | c3
key D = diagonal tile, masked inside | c1
key X = skipped entirely | line
q1 | D X X X X X X X
q2 | F D X X X X X X
q3 | F F D X X X X X
q4 | F F F D X X X X
q5 | F F F F D X X X
q6 | F F F F F D X X
q7 | F F F F F F D X
q8 | F F F F F F F D
%%
~~~

Of 64 tiles, 28 are skipped, 8 need masking, and 28 run unmasked. Causal prefill does ⟦m778⟧ of the full square's work, approaching 50% as the number of tiles grows.

<div class="take">
<div class="lab">Key takeaway</div>

online softmax lets attention stream over key blocks with a running maximum, denominator and output, rescaling earlier work whenever the maximum grows. It is exact, and it removes the S × S matrix from memory entirely.

</div>
