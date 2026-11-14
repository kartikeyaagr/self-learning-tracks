---
id: "13.03"
course: "inference-eng-course"
module: "13"
slug: "coalescing-tiling-reductions"
title: "Coalescing, Tiling & Reductions"
keyIdeas: ["Coalescing: lay out data so a block's loads are contiguous.", "Tiling: load a tile into fast memory once, reuse it for many outputs.", "Bigger tiles mean more reuse but more shared memory and registers, lower occupancy.", "Reductions (sum, max) combine values within a tile, then across tiles.", "Softmax and norms are row reductions; keep one row per program if it fits."]
question: "Explain why tiling a matrix multiply reduces HBM traffic."
answer: "Without tiling each output element reads a full row of A and column of B from HBM, so the same data is read many times. With tiling, a block loads a tile of A and a tile of B into shared memory or registers once and uses each loaded element for many output elements in its tile. Traffic drops roughly by the tile size factor, raising arithmetic intensity."
sources: [{"title": "Triton tutorial: Matrix multiplication", "url": "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html", "note": "Tiled GEMM in Triton."}, {"title": "CUDA C++ Best Practices Guide", "url": "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/", "note": "Memory optimizations."}]
tags: ["triton", "kernels"]
---

## Tiling a matrix multiply

For C = A × B, with A of shape [M, K] and B of shape [K, N], a naive kernel computes each output element by reading a full row of A and a full column of B. Every element of A is then read N times from HBM. Tiling fixes this.

Each program computes one tile of C, of size BM × BN. It loops over K in chunks of BK. In each iteration it loads a BM × BK tile of A and a BK × BN tile of B into fast memory, multiplies them, and adds the result to an accumulator.

```
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k in range(0, K, BK):
    a = tl.load(a_ptrs, mask=..., other=0.0)    # [BM, BK]
    b = tl.load(b_ptrs, mask=..., other=0.0)    # [BK, BN]
    acc += tl.dot(a, b)                         # tensor cores
    a_ptrs += BK * stride_ak
    b_ptrs += BK * stride_bk
tl.store(c_ptrs, acc.to(tl.bfloat16), mask=...)
```

Each element of A loaded in a tile is used BN times, and each element of B is used BM times, all from registers and shared memory. The accumulator stays in FP32 for accuracy and is converted to BF16 only when written.

## How much tiling saves

For a 4,096 × 4,096 × 4,096 matmul in BF16:

- Naive: each of the 16.8 million outputs reads 4,096 elements of A and of B. That is about 275 GB of reads, far more than the data itself.
- Tiled with 128 × 128 tiles: A is read N ÷ BN = 32 times and B is read M ÷ BM = 32 times. That is 32 × 32 MB × 2 ≈ 2.1 GB, more than 100 times less traffic.

The L2 cache cuts the real number further when neighbouring tiles share inputs. Good kernels order their tiles so that tiles running at the same time reuse the same parts of A and B from L2.

## Choosing tile sizes

| Tile (BM × BN) | Reuse | Resources per program | Typical use |
| --- | --- | --- | --- |
| 64 × 64 | moderate | low | good occupancy, medium matrices |
| 128 × 128 or 128 × 256 | high | high | large prefill matmuls |
| 16 × N or 32 × N | low on the M side | low | decode matmuls with few tokens |

Decode is the tricky case. With M = 8 tokens, a 128-row tile wastes most of its rows. Decode matmuls use small BM, and often split the K dimension across programs so that enough programs exist to fill all SMs ("split-K"), then add the partial results.

## Reductions

Many inference kernels reduce along a row: softmax, RMSNorm, sums. The pattern:

- **Row fits in one block**: load the whole row, compute the statistic with `tl.max` or `tl.sum`, apply it, store. One load and one store per element.
- **Row too large**: loop over chunks, keeping running statistics. For softmax this means a running maximum and a running sum that is rescaled whenever the maximum changes, the same **online softmax** that FlashAttention uses.

Always accumulate in FP32. Summing 4,096 BF16 values in BF16 loses several digits of precision.

## Coalescing in practice

Load along the contiguous dimension. For a row-major [M, K] matrix, the K dimension is contiguous, so a tile's loads should run along K within each row. In Triton, build pointer blocks so that the last dimension of the block corresponds to the contiguous dimension in memory:

```
offs_m = pid_m * BM + tl.arange(0, BM)
offs_k = tl.arange(0, BK)
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
```

With `stride_ak = 1`, each row of the block is contiguous, and the compiler can issue wide, coalesced loads.

<div class="ex">
<div class="lab">Worked example</div>

**a transposed weight**

A team stored weights as [N, K] and computed x @ W.T. Their Triton matmul loaded B tiles with the contiguous dimension along K, which matched the transposed layout, and reached 78% of peak.

A refactor changed the layout to [K, N] without changing the kernel's pointer setup. The loads became strided, and the kernel dropped to 31% of peak.

The lesson: layout and kernel must agree. Check strides whenever a layout changes.

</div>

## Deep dive: how tiling raises intensity

A matmul tile of ⟦m726⟧ outputs, stepping through ⟦m727⟧ in chunks of ⟦m728⟧, loads ⟦m729⟧ values and does ⟦m730⟧ FLOPs per step:

⟦M724⟧

For square tiles of size ⟦m731⟧ in BF16, ⟦m732⟧ FLOP per byte from HBM (L2 reuse raises the effective value further).

~~~dg-raw
%%chart Intensity from HBM against square tile size (BF16, ignoring L2 reuse).
x = 16, 32, 64, 128, 256
logx = 1
logy = 1
xlab = Tile size T
ylab = FLOP per byte
Tile intensity: 8, 16, 32, 64, 128
H100 ridge*: 295, 295, 295, 295, 295
%%
~~~

The tile must fit in shared memory, with several pipeline stages to overlap loading and compute:

⟦M725⟧

128 × 128 tiles with ⟦m733⟧ and 3 stages need ⟦m734⟧ KB: comfortably inside 228 KB, which is why 128 × 128 is a common choice on H100.

<div class="take">
<div class="lab">Key takeaway</div>

tiling turns repeated HBM reads into reuse from fast memory, and its tile sizes must match the shape: large tiles for prefill, small tiles and split-K for decode. Reductions follow the same pattern, one pass with FP32 running statistics, and loads must run along the contiguous dimension.

</div>
