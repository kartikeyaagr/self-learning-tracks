---
id: "31.06"
course: "inference-eng-course"
module: "31"
slug: "convolution-and-ml-primitives"
title: "Convolution and ML primitives"
keyIdeas: ["Most ML kernels outside matmul are **memory-bound**; the goal is to read each input once and write each output once, which usually means **fusing** several framework ops into one kernel.", "A 1-D convolution with a $K$-tap filter reuses each input $K$ times; a shared-memory tile of $B$ outputs needs $B + K - 1$ inputs (the **halo**), and the filter sits in `__constant__` memory, which broadcasts to a warp reading the same tap.", "Deep-learning convolutions with many channels are run as **implicit GEMM** (im2col without materialising it), FFT or Winograd; cuDNN picks per shape, which is what `torch.backends.cudnn.benchmark` tunes.", "Row softmax maps a warp or block to each row and does two reductions (max, then sum of $e^{x-m}$); the **online** form keeps a running max and rescales the running sum, fusing both into one pass.", "LayerNorm and RMSNorm are per-row reductions in FP32 followed by an elementwise scale; fusing the residual add and saving mean and $1/\\sigma$ for the backward pass are the standard optimisations."]
question: "A softmax kernel for attention scores of shape (batch x heads x 2048 queries) x 2048 keys in FP16 uses one thread per row and three passes over each row. Profiling shows 15% of peak bandwidth. Redesign it and estimate the best achievable time if the score tensor has 2^28 elements on a 2 TB/s GPU."
answer: "One thread per row makes a warp read 32 different rows at the same column: uncoalesced, with 32 sectors per request. Three passes also triple the reads. Assign one warp per row instead: each lane takes elements lane, lane+32, ... (or uses 16-byte vector loads of 8 FP16 values for coalescing), keeps its 64 values in registers, computes the max and the sum of exponentials with warp-shuffle reductions (or a single online-softmax reduction carrying (max, sum) pairs), then writes the normalised values from registers. The row is read once and written once. Traffic is 2^28 elements x 2 bytes x 2 (read + write) = 1 GiB, so the floor is about 1.07 GB / 2 TB/s = 0.54 ms; a good kernel reaches roughly 85 to 90% of peak, around 0.6 ms. Better still is to avoid materialising the scores at all by fusing softmax into attention, as FlashAttention does."
sources: [{"title": "Chetlur et al., cuDNN: Efficient Primitives for Deep Learning (2014)", "url": "https://arxiv.org/abs/1410.0759", "note": "How convolution is lowered to an implicit GEMM, and the performance arguments for it."}, {"title": "Milakov and Gimelshein, Online Normalizer Calculation for Softmax (2018)", "url": "https://arxiv.org/abs/1805.02867", "note": "The single-pass (max, sum) recurrence and measured speedups."}, {"title": "Triton tutorial: Layer Normalization", "url": "https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html", "note": "Forward and backward LayerNorm kernels, including the parallel reduction for weight gradients."}, {"title": "Lavin and Gray, Fast Algorithms for Convolutional Neural Networks (2015)", "url": "https://arxiv.org/abs/1509.09308", "note": "Winograd minimal filtering for 3x3 convolutions."}]
practice: ["1d-convolution", "2d-convolution-2", "softmax-2", "rms-normalization", "layernorm-forward"]
tags: ["convolution", "softmax", "kernel fusion"]
tools: ["CUDA", "cuDNN"]
---

## The common shape of ML kernels

Outside matrix multiplies and attention, most work in a network is elementwise (activations, residual adds, dropout) or row-wise (softmax, normalisation, cross-entropy). These kernels do a handful of FLOPs per element, so they are bandwidth-bound (31.02) and their speed is set by bytes moved. A GELU on 64 MB of FP16 activations reads 64 MB and writes 64 MB; at 2 TB/s that is 64 microseconds whatever the arithmetic. Running GELU, bias add and dropout as three separate kernels moves 384 MB; fused, 128 MB. Compilers (torch.compile, Triton, covered in "torch.compile & Kernel Fusion") do much of this fusion automatically; custom CUDA kernels are written when they cannot.

## 1-D and 2-D convolution

A 1-D convolution with a filter of $K = 2r + 1$ taps computes $y_i = \sum_{k=-r}^{r} w_k\, x_{i+k}$. One thread per output works and is coalesced. Two improvements:

- **Constant memory for the filter.** `__constant__ float w[K]` lives in a small cached space that broadcasts a value to all threads of a warp reading the same address in one transaction, exactly the pattern here (all threads read tap $k$ at the same time).
- **Shared-memory tiling with a halo.** Each input is read by $K$ different threads. A block producing $B$ outputs loads the $B + 2r$ inputs it needs into shared memory once (the extra $2r$ are the halo, zero-filled at array edges), synchronises, then computes from shared memory.

~~~dg-flow Tiled convolution: each block loads its tile plus a halo once, then every output reads its neighbourhood from shared memory.
Global input -(load tile + halo, coalesced)-> *Shared tile -(__syncthreads)-> K-tap sums per thread -(coalesced store)-> Output
~~~

In 2-D, a $B \times B$ output tile needs a $(B + K - 1)^2$ input tile. For $B = 16$ and $K = 5$, that is 400 loads for 256 outputs which each need 25 inputs: global traffic drops from 6400 to 400 element loads, a 16x reduction. Larger filters raise reuse; tiny filters (3 x 3) make the halo a large fraction of the tile.

Convolutions in CNNs, with $C_{in}$ input and $C_{out}$ output channels, are a different regime: they have the reuse structure of matrix multiply. cuDNN lowers them to **implicit GEMM**: conceptually, unfold each $K \times K \times C_{in}$ input patch into a column (im2col), giving a matrix of size $(C_{in}K^2) \times (H_{out}W_{out})$ that multiplies the $C_{out} \times (C_{in}K^2)$ filter matrix. The "implicit" version computes patch addresses on the fly instead of materialising the $K^2$-times-larger matrix, then runs on Tensor Cores. For some shapes FFT or Winograd transforms (which trade multiplies for adds on 3 x 3 filters) are faster; cuDNN benchmarks candidates per shape when `cudnn.benchmark` is on.

## Softmax

Row softmax $y_j = e^{x_j - m} / \sum_k e^{x_k - m}$ with $m = \max_k x_k$ (subtracting the max prevents overflow). The standard kernel maps one warp to each row for rows up to a few thousand elements, or one block for longer rows:

1. Each thread scans its strided slice of the row for a local max; a warp-shuffle reduction gives $m$.
2. Each thread sums $e^{x_j - m}$ over its slice; a second reduction gives the denominator $d$.
3. Each thread writes $e^{x_j - m}/d$.

If the row fits in registers (each of 32 lanes holding, say, 32 values for a 1024-wide row), the input is read once and the kernel is optimal. Otherwise the naive version reads the row three times. The **online softmax** (Milakov and Gimelshein) fuses steps 1 and 2: keep a running max $m$ and running sum $d$, and on each new value $x$ update $m' = \max(m, x)$, $d' = d\, e^{m - m'} + e^{x - m'}$. Pairs $(m, d)$ from different threads combine the same way, so it fits a tree reduction. That cuts the passes to two, and it is the key step that lets FlashAttention compute softmax tile by tile ("Tiled Attention & Online Softmax" covers it in detail).

```cpp
__device__ void online_combine(float& m, float& d, float m2, float d2) {
    float mn = fmaxf(m, m2);
    d = d * __expf(m - mn) + d2 * __expf(m2 - mn);
    m = mn;
}
```

## LayerNorm and RMSNorm

LayerNorm normalises each row: $y = \gamma \odot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$. The kernel is a row reduction for $\mu$ and $\sigma^2$ followed by an elementwise pass. Practical points:

- Accumulate in FP32 even for FP16/BF16 inputs; use Welford or a two-level (count, mean, M2) combine for the variance to avoid cancellation.
- Save $\mu$ and $1/\sigma$ per row for the backward pass rather than recomputing them.
- Fuse the residual add that precedes the norm in a transformer block: read $x$ and the residual, write the sum (needed later) and the normalised output in one kernel.
- RMSNorm drops the mean: $y = \gamma \odot x / \sqrt{\frac{1}{d}\sum x^2 + \epsilon}$, one reduction instead of two.

The backward pass is harder than the forward: the gradients for $\gamma$ and $\beta$ are column sums over all rows, a reduction in the other direction. Deterministic implementations write per-block partial column sums and reduce them in a second kernel.

<div class="ex">
<div class="lab">Worked example</div>

**bytes for a fused norm**

A transformer layer with hidden size 4096 and 8192 tokens in BF16 holds 64 MiB per activation tensor. Unfused: residual add (read 2, write 1), then LayerNorm (read 1, write 1) is 5 x 64 MiB = 320 MiB of traffic. Fused add+norm reads the two inputs and writes the sum and the output: 4 x 64 MiB = 256 MiB, and one kernel launch instead of two. At 3 TB/s that is about 90 microseconds against 112, a 20% saving on a kernel that runs twice per layer.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

treat convolution, softmax and normalisation as data-movement problems: tile with halos for small filters, lean on implicit GEMM for channel-heavy convolutions, use one warp or block per row with shuffle reductions, and fuse neighbouring ops so each tensor crosses HBM once.

</div>
