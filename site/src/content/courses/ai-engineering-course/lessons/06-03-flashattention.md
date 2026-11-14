---
id: "06.03"
course: "ai-engineering-course"
module: "06"
slug: "flashattention"
title: "FlashAttention"
keyIdeas: ["On GPUs, attention is limited by memory traffic between slow HBM and fast on-chip SRAM, not by arithmetic.", "Standard attention writes the full T x T score matrix to HBM, reads it back for softmax, writes again, reads again for the V multiply.", "FlashAttention tiles Q, K and V into blocks that fit in SRAM and computes attention block by block, never storing the full matrix.", "Softmax is done incrementally with the online softmax trick: keep a running max and running sum, rescale as new blocks arrive.", "For the backward pass it recomputes attention instead of storing it, trading cheap FLOPs for expensive memory.", "Result: exact attention, memory linear in sequence length, and 2-4x speedups; FlashAttention-2 and -3 add better GPU utilisation."]
question: "FlashAttention does the same maths as standard attention. Where does its speedup come from, and why does it also save memory?"
answer: "The speedup comes from cutting reads and writes to slow GPU HBM: attention is memory-bound, and standard attention writes and re-reads the full T x T score and softmax matrices. FlashAttention tiles Q, K, V into SRAM-sized blocks, computes scores, an online (running max/sum) softmax and the output block by block, and writes only the final output. Since the T x T matrix is never materialised (and is recomputed in the backward pass instead of stored), memory is linear rather than quadratic in sequence length."
sources: [{"title": "Dao et al.: FlashAttention (2022)", "url": "https://arxiv.org/abs/2205.14135", "note": "IO-aware exact attention with tiling."}, {"title": "Dao: FlashAttention-2 (2023)", "url": "https://arxiv.org/abs/2307.08691", "note": "Better parallelism and work partitioning, about 2x faster again."}, {"title": "Horace He: Making Deep Learning Go Brrrr From First Principles", "url": "https://horace.io/brrr_intro.html", "note": "Compute-bound vs memory-bound, the background you need."}]
---

## The real bottleneck

Horace He's "Making Deep Learning Go Brrrr" explains the key fact: modern GPUs can do arithmetic far faster than they can move data from their main memory (HBM). An operation that reads and writes a lot of data but does little maths per byte is **memory-bound**, and speeding up the maths doesn't help.

A GPU has two kinds of memory:

| Memory | Size | Bandwidth |
| --- | --- | --- |
| HBM (main GPU memory) | tens of GB | a few TB/s |
| SRAM (on-chip, per SM) | ~100-200 KB per SM | roughly an order of magnitude faster |

## What standard attention does

1. Compute S = Q K^T and write the T x T matrix to HBM.
2. Read S, compute softmax P, write P to HBM.
3. Read P and V, compute output O, write O.

For long sequences the T x T matrices are huge, and most of the time goes to these reads and writes, not the maths. Memory use also grows with T^2.

## The FlashAttention idea

Dao et al. made attention **IO-aware**: never write S or P to HBM at all.

~~~dg-flow FlashAttention keeps each tile in fast on-chip memory.
Q block + K, V blocks (HBM) -(load tile)-> *SRAM: scores, partial softmax, partial output -(update running max and sum)-> Next K, V block -(after last block)-> Write O to HBM
~~~

- **Tiling**: split Q, K and V into blocks small enough for SRAM. For each Q block, loop over K/V blocks.
- **Online softmax**: softmax needs the max and sum over a whole row, which isn't known until all blocks are seen. Keep a running max and running sum, and rescale the partial output when a new block raises the max. The final result is exactly standard softmax.
- **Recomputation**: in the backward pass, recompute attention scores from Q, K, V instead of storing the T x T matrix.

## Results

- **Exact**: not an approximation; outputs match standard attention up to floating-point rounding.
- **Memory**: linear in sequence length instead of quadratic, which made long-context training practical.
- **Speed**: 2-4x faster attention in the original paper. FlashAttention-2 (2023) roughly doubled that again with better parallelism across thread blocks and less non-matmul work. FlashAttention-3 targets Hopper GPU features (asynchrony, FP8).

It's built into PyTorch's scaled_dot_product_attention and into serving engines, so you usually get it without asking.

## Tiles instead of a full matrix

Standard attention writes the full n × n score matrix to GPU main memory (HBM), reads it back for softmax, writes the probabilities, and reads them again to multiply by V. FlashAttention processes queries and keys in tiles small enough to fit in on-chip SRAM and never writes the full matrix. Step through it below.

~~~dg-raw
%%widget flash Tiled attention: which blocks are in fast memory, and how much memory traffic is saved.
~~~

The memory hierarchy is what makes this matter. On an A100, HBM is 40 or 80 GB at about 1.5 to 2 TB/s; on-chip SRAM is about 20 MB in total at roughly 19 TB/s (figures from the FlashAttention paper). Attention does few arithmetic operations per byte moved, so it was limited by HBM traffic, not by math.

## The online softmax trick

Softmax needs the maximum and the sum over the whole row, which seems to require the full row first. FlashAttention keeps a running maximum m and a running sum l per query row:

```
for each key block j:
    s      = q · K_j / sqrt(d)
    m_new  = max(m, max(s))
    l      = l * exp(m - m_new) + sum(exp(s - m_new))
    out    = out * exp(m - m_new) + exp(s - m_new) · V_j
    m      = m_new
final output = out / l
```

When a new block raises the maximum, earlier partial results are rescaled by exp(m_old − m_new). The final answer is exactly the same as standard attention, not an approximation.

## What changed in practice

- Memory for attention became linear in sequence length, which made 32k to 128k training contexts affordable.
- Dao et al. reported 2 to 4× faster attention and up to about 3× faster GPT-2 training end to end in the first paper (2022). FlashAttention-2 and -3 improved GPU utilisation further, with version 3 targeting H100 features such as FP8.
- In the backward pass, the attention matrix is recomputed from Q, K and V instead of stored. Recomputing is cheaper than reading it back from HBM.

<div class="take">
<div class="lab">Key takeaway</div>

attention is memory-bound, not compute-bound. FlashAttention computes exact attention tile by tile in on-chip SRAM with an online softmax, so the T x T matrix never touches main memory.

</div>
