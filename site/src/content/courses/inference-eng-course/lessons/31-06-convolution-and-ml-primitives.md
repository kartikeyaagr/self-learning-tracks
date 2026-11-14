---
id: "31.06"
course: "inference-eng-course"
module: "31"
slug: "convolution-and-ml-primitives"
title: "Convolution and ML primitives"
keyIdeas: ["Most ML kernels outside matrix multiply are **memory-bound**: their time is bytes moved divided by bandwidth. The goal is to read each input once and write each output once, which usually means **fusing** several framework operations into one kernel.", "A 1-D convolution with a $K$-tap filter reuses each input $K$ times. A shared-memory tile of $B$ outputs needs $B + K - 1$ inputs (the extra ones are the **halo**), and the filter sits in `__constant__` memory, which broadcasts one value to a whole warp reading the same tap.", "Convolutions with many channels have matrix-multiply reuse. cuDNN runs them as **implicit GEMM** (im2col without building the matrix), FFT or Winograd, picking per shape; `torch.backends.cudnn.benchmark` makes PyTorch time the options.", "Row softmax gives each row to a warp or block and needs a max and a sum. The **online** form keeps a running max and rescales the running sum by $e^{m - m'}$ when the max rises, so both reductions happen in one pass and partial results merge in any order.", "LayerNorm and RMSNorm are per-row reductions done in FP32 (with Welford's method for the variance) followed by an elementwise scale. Fusing the residual add and saving the mean and $1/\\sigma$ for the backward pass are the standard optimisations."]
question: "A softmax kernel for attention scores of shape (batch x heads x 2048 queries) x 2048 keys in FP16 uses one thread per row and three passes over each row. Profiling shows 15% of peak bandwidth. Redesign it and estimate the best achievable time if the score tensor has 2^28 elements on a 2 TB/s GPU."
answer: "Two problems explain the 15%. First, one thread per row means the 32 lanes of a warp read 32 different rows at the same column, so every warp load touches 32 different sectors and uses only 2 bytes of each 32-byte sector: badly uncoalesced. Second, three passes (max, sum of exponentials, write) read each row three times. Redesign: assign one warp per row. Lane $\\ell$ reads columns $\\ell, \\ell + 32, \\dots$ (or better, uses 16-byte vector loads of 8 FP16 values so each lane reads a contiguous chunk), so every load is coalesced. With 2048 columns each lane holds 64 values, which fit in registers. Compute the max and the sum of exponentials either with two warp-shuffle reductions over the register-resident values or, more simply, with one online-softmax reduction that carries (max, sum) pairs, accumulating in FP32 and guarding the merge against fully masked rows ($-\\infty - (-\\infty)$). Then write $e^{x - m}/d$ from registers. Each row is now read once and written once. Traffic is $2^{28}$ elements x 2 bytes x 2 (read and write) = 1 GiB, so the floor is about $1.07 \\times 10^9 / 2 \\times 10^{12} \\approx 0.54$ ms; a good kernel reaches roughly 85 to 90% of peak, around 0.6 ms, versus about 3.6 ms implied by 15% of peak. The better fix is to avoid materialising the score tensor at all by fusing softmax into attention with the online rescaling, as FlashAttention does; then this 1 GiB of traffic disappears entirely."
sources: [{"title": "Chetlur et al., cuDNN: Efficient Primitives for Deep Learning (2014)", "url": "https://arxiv.org/abs/1410.0759", "note": "How convolution is lowered to an implicit GEMM, and the performance arguments for it."}, {"title": "Milakov and Gimelshein, Online Normalizer Calculation for Softmax (2018)", "url": "https://arxiv.org/abs/1805.02867", "note": "The single-pass (max, sum) recurrence and measured speedups."}, {"title": "Triton tutorial: Layer Normalization", "url": "https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html", "note": "Forward and backward LayerNorm kernels, including the parallel reduction for weight gradients."}, {"title": "Lavin and Gray, Fast Algorithms for Convolutional Neural Networks (2015)", "url": "https://arxiv.org/abs/1509.09308", "note": "Winograd minimal filtering for 3x3 convolutions."}]
practice: ["1d-convolution", "2d-convolution-2", "softmax-2", "rms-normalization", "layernorm-forward"]
tags: ["convolution", "softmax", "kernel fusion"]
tools: ["CUDA", "cuDNN"]
---

## What you will learn

- Why most ML kernels outside matrix multiply are limited by bytes moved, and how **fusion** cuts those bytes.
- How to write a tiled convolution with a shared-memory **halo** and the filter in constant memory.
- Why many-channel convolutions are really matrix multiplies, and what **implicit GEMM**, FFT and Winograd do.
- How row softmax maps onto one warp per row, and how the **online softmax** merges max and sum into one pass.
- How LayerNorm and RMSNorm kernels are built, and why they accumulate in FP32 with Welford's method.

*Before you start:* you need coalescing and effective bandwidth (31.02), shared-memory tiles and `__syncthreads()` (31.03), and warp-shuffle reductions (31.04).

Lesson 31.05 used atomics to let many threads update shared counters. This lesson adds no new hardware: it assembles coalesced loads, shared tiles and warp reductions into the kernels a network runs between its matrix multiplies. The deciding question is always how many times each byte crosses HBM.

## The idea in one picture

Take one transformer activation: 8192 tokens by hidden size 4096 in BF16, so $8192 \times 4096 \times 2$ bytes $= 64$ MiB. A feed-forward block applies a bias, a GELU and dropout to it. Written as three PyTorch lines, that is three kernels:

| Kernel | Reads | Writes | Traffic
| --- | --- | --- | --- |
| `h = x + bias` | 64 MiB | 64 MiB | 128 MiB
| `h = gelu(h)` | 64 MiB | 64 MiB | 128 MiB
| `h = dropout(h)` | 64 MiB | 64 MiB | 128 MiB
| one fused kernel | 64 MiB | 64 MiB | 128 MiB (instead of 384)

(Bias and dropout mask are small and ignored.) At 10 to 20 operations per element, the arithmetic finishes long before the data streams, so time follows the traffic column.

~~~dg-fig 31.06-fusion.svg
A model from spec-sheet bandwidth, not a measurement. Left: HBM traffic for a three-kernel GELU chain versus one fused kernel, and for a residual add plus LayerNorm versus a fused add+norm. Right: the time those bytes need at 2.0 and 3.35 TB/s. Fusing the GELU chain removes two thirds of the time; fusing add+norm removes one fifth.
~~~

At 2.0 TB/s the chain needs at least 201 µs; fused, 67 µs, with the same arithmetic. The fused kernel simply keeps intermediates in registers.

Two terms for the whole lesson:

- A kernel is **memory-bound** when its run time is set by bytes moved, not operations. Its **arithmetic intensity** (FLOPs per byte) is far below the GPU's peak-FLOPs-to-bandwidth ratio.
- **Fusion** means doing several operations in one kernel so intermediates stay on chip. `torch.compile` and Triton do much of this automatically ("torch.compile & Kernel Fusion"); you write CUDA when the fusion needs a reduction or layout they cannot handle.

## Step 1: count bytes, not FLOPs

For a memory-bound kernel the best possible time is

$$t_{\min} = \frac{\text{bytes read} + \text{bytes written}}{\text{peak bandwidth}},$$

where bytes count each input once and each output once, and peak bandwidth is the spec-sheet HBM figure: about 2.0 TB/s on an A100 80 GB, 3.35 TB/s on an H100 SXM. Good kernels reach 80 to 90% of it (31.02).

<div class="ex">
<div class="lab">Worked example</div>

**bytes for a fused residual add and LayerNorm**

Hidden size 4096, 8192 tokens, BF16: 64 MiB per activation tensor.

Unfused: the residual add reads two tensors and writes one ($3 \times 64$ MiB), then LayerNorm reads the sum and writes its output ($2 \times 64$ MiB). Total $5 \times 64 = 320$ MiB.

Fused: read $x$ and the residual, write the sum (the next residual needs it) and the normalised output. Total $4 \times 64 = 256$ MiB, and one launch instead of two.

At 3 TB/s: $320 \times 2^{20} / (3 \times 10^{12}) \approx 112$ µs versus $256 \times 2^{20} / (3 \times 10^{12}) \approx 89$ µs, a 20% saving on a kernel that runs twice per layer.

</div>

## Step 2: a convolution with a halo

A 1-D convolution with $K = 2r + 1$ taps computes

$$y_i = \sum_{k=-r}^{r} w_k\, x_{i+k},$$

where $x$ is the input, $w_k$ the filter weight at offset $k$, $r$ the **radius** (taps on each side of the centre) and $y_i$ output $i$. (Strictly this is cross-correlation; DL libraries call it convolution.)

One thread per output gives coalesced loads, but each input is read by $K$ threads: $K$ global loads per output. Two changes fix that.

**The filter goes in constant memory**, declared `__constant__ float w[K]`. This 64 KB read-only space has its own cache and broadcasts a value to the whole warp when all 32 lanes read the same address, which is exactly what happens at loop step $k$. If lanes read different constant addresses, the reads serialise.

**The input goes through a shared tile with a halo.** A block producing $B$ outputs needs inputs from $r$ before its first output to $r$ after its last: $B + 2r$ values. The extra $2r$ are the **halo**. The block loads them once, calls `__syncthreads()`, and each thread reads its $K$ neighbours from shared memory. Halo cells beyond the array ends are zero, the usual "same" padding.

~~~dg-fig 31.06-halo.svg
Left: a tile for B = 8 outputs of a 5-tap filter: 12 inputs including 2 halo cells per side, and the 5 that output y3 reads. Right: global loads per output for 2-D B × B tiles, (B + K − 1)² / B². Small tiles and large filters pay heavily for the halo; at B = 32 every filter is within 41% of the ideal of one.
~~~

In 1-D the count is easy. For 1024 outputs and $K = 5$ the naive kernel issues 5120 loads. With $B = 256$, four blocks load 260 each, minus 4 halo cells that fall off the ends: 1036 loads, about 1.01 per output.

In 2-D a $B \times B$ output tile needs a $(B + K - 1)^2$ input tile.

<div class="ex">
<div class="lab">Worked example</div>

**how much a 2-D tile saves**

Output tile $16 \times 16$, filter $5 \times 5$: $B = 16$, $K = 5$.

Naive: $256$ outputs times $25$ inputs each $= 6400$ global loads.

Tiled: one $(16 + 5 - 1)^2 = 400$-element input tile.

Saving: $6400 / 400 = 16\times$, or $400/256 = 1.56$ loads per output.

With a $3 \times 3$ filter: $18^2 = 324$ loads against $256 \times 9 = 2304$, only $7\times$. Small filters reuse each input less, and the halo is a bigger share of the tile.

</div>

## Step 3: many channels turn convolution into matrix multiply

A CNN layer is a different animal. Each output pixel combines a $K \times K$ patch from all $C_{in}$ input channels, and there are $C_{out}$ filters. Each input is reused $K^2 C_{out}$ times and each weight at every output pixel: matrix-multiply reuse.

The bridge is **im2col**: copy each $K \times K \times C_{in}$ patch into one column of a matrix with $C_{in}K^2$ rows and $H_{out}W_{out}$ columns. Multiply by the filters written as a $C_{out} \times C_{in}K^2$ matrix, and the product is the whole output.

~~~dg-fig 31.06-im2col.svg
im2col on a 4 × 4 input with a 3 × 3 filter. Each of the four patches becomes a column (patch 1 is outlined); the small number under each pixel counts the patches that contain it. The filter as a row times this matrix gives the 2 × 2 output, here [−8, −8, −8, −8]. Centre pixels are copied four times.
~~~

The catch is duplication. For a $64 \times 56 \times 56$ input with a $3 \times 3$ filter and padding 1, the im2col matrix is $576 \times 3136$, nine times the input. **Implicit GEMM**, the approach cuDNN uses, runs a tiled matrix multiply (31.03) that computes each tile's patch addresses while loading into shared memory, so the big matrix never exists in HBM.

With $C_{out} = 128$ that GEMM does $2 \times 128 \times 576 \times 3136 \approx 462$ million FLOPs on about 1.35 MB of FP16 data (0.40 MB input, 0.15 MB weights, 0.80 MB output): roughly 340 FLOPs per byte. The A100's ridge point is about 150 (312 TFLOPS FP16 Tensor Core over 2.0 TB/s, spec sheet values), so this layer is compute-bound, unlike everything else here.

Two other algorithms compete. **FFT convolution** turns the convolution into an elementwise product of spectra and suits large filters. **Winograd** (Lavin and Gray) transforms small tiles to save multiplies: $F(m \times m, r \times r)$ needs $(m + r - 1)^2$ multiplies per output tile instead of $m^2 r^2$, so $F(2 \times 2, 3 \times 3)$ uses 16 instead of 36. Larger tiles save more but round worse, which limits them in FP16. cuDNN keeps several algorithms; with `torch.backends.cudnn.benchmark = True`, PyTorch times them on each new shape and caches the fastest.

## Step 4: softmax, one warp per row

Row softmax is

$$y_j = \frac{e^{x_j - m}}{d}, \qquad m = \max_k x_k, \qquad d = \sum_k e^{x_k - m},$$

where $x_j$ is score $j$ of the row, $m$ the row maximum and $d$ the normaliser. Subtracting $m$ keeps every exponent at most 0, so nothing overflows (FP16 overflows above about $e^{11}$).

Each row goes to one warp (up to a few thousand columns) or one block. Lane $\ell$ takes columns $\ell, \ell + 32, \dots$, so loads are coalesced:

1. Each lane finds its slice's max; a shuffle reduction gives $m$.
2. Each lane sums $e^{x_j - m}$; a second reduction gives $d$.
3. Each lane writes $e^{x_j - m}/d$.

A row that fits in registers (2048 columns is 64 values per lane) is read once; otherwise three times.

The **online softmax** (Milakov and Gimelshein) merges steps 1 and 2. Keep a running max $m$ and a running sum $d$ measured relative to the current max. For each new value $x$:

$$m' = \max(m, x), \qquad d' = d\, e^{m - m'} + e^{x - m'}.$$

The factor $e^{m - m'}$ rescales the old sum when the max rises, and equals 1 otherwise. Two partial states merge the same way, $m = \max(m_1, m_2)$, $d = d_1 e^{m_1 - m} + d_2 e^{m_2 - m}$, and the merge is associative, so it fits a shuffle tree. Steps 1 and 2 become one pass.

<div class="ex">
<div class="lab">Worked example</div>

**the online softmax by hand**

Row $x = (2, 1, 5, 3)$; start at $m = -\infty$, $d = 0$.

Read 2: $m = 2$, $d = e^0 = 1$.

Read 1: $m = 2$, $d = 1 + e^{-1} = 1.3679$.

Read 5: the max jumps, so rescale: $d = 1.3679 \times e^{2-5} + e^0 = 1.3679 \times 0.0498 + 1 = 1.0681$, $m = 5$.

Read 3: $d = 1.0681 + e^{-2} = 1.0681 + 0.1353 = 1.2034$.

Two-pass check: $e^{-3} + e^{-4} + e^0 + e^{-2} = 0.0498 + 0.0183 + 1 + 0.1353 = 1.2034$. Same answer in one pass.

</div>

~~~dg-fig 31.06-online.svg
Online softmax over eight values. Left: the running max steps up at elements 3 and 6. Right: the running denominator drops at exactly those steps, because the old sum is rescaled to the new max, and ends at 1.5814, the two-pass value.
~~~

This rescaling is what lets FlashAttention process keys one tile at a time ("Tiled Attention & Online Softmax").

## Step 5: LayerNorm and RMSNorm

LayerNorm normalises each row of length $D$:

$$y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,$$

where $\mu$ and $\sigma^2$ are the row mean and variance, $\epsilon$ (say $10^{-5}$) prevents division by zero, $\gamma$ and $\beta$ are learned per-column scale and shift, and $\odot$ is elementwise product. The kernel shape is softmax's: a warp or block per row, a reduction, an elementwise pass. **RMSNorm** drops the mean and shift, $y = \gamma \odot x / \sqrt{\tfrac{1}{D}\sum_j x_j^2 + \epsilon}$: one reduction instead of two.

What makes it production quality:

- **Accumulate in FP32**, even for FP16 or BF16 data.
- **Use Welford's method** for the variance rather than $E[x^2] - E[x]^2$, which subtracts two large, nearly equal numbers. Welford keeps a running (count, mean, M2), where M2 is the sum of squared deviations; two triples merge exactly, so it fits a warp reduction.
- **Save $\mu$ and $1/\sigma$ per row** for the backward pass: 8 bytes per row instead of re-reading $2D$ bytes.
- **Fuse the residual add** before the norm, as in the step 1 example.

~~~dg-fig 31.06-variance.svg
CPU float32 simulation of one thread accumulating a 1024-value row (unit-variance noise plus a constant mean). At a mean of 300 the one-pass formula is 6% off; at 3000 its error is ten times the true variance. Welford stays within about 1e-5.
~~~

Large means are realistic (LLM activations have a few outsized channels), and the one-pass formula can even go negative, giving NaN after the square root.

The backward pass reduces the other way: the gradients for $\gamma$ and $\beta$ are column sums over all rows. Deterministic kernels write per-block partial sums and reduce them in a second kernel, instead of float atomics (31.05), whose order and rounding vary between runs.

## The kernels in code

The tiled 1-D convolution: one thread per output, tile plus halo in shared memory, filter in constant memory.

```cpp
#define R 2                          // radius: K = 2R + 1 = 5 taps
#define B 256                        // outputs per block = threads per block
__constant__ float w[2 * R + 1];     // set with cudaMemcpyToSymbol

__global__ void conv1d_tiled(const float* x, float* y, int n) {
    __shared__ float tile[B + 2 * R];
    int t = threadIdx.x, i = blockIdx.x * B + t;     // i: this thread's output
    tile[t + R] = (i < n) ? x[i] : 0.f;              // main part, coalesced
    if (t < R) {                                     // first R threads load both halos
        int left = i - R, right = i + B;
        tile[t]         = (left  >= 0) ? x[left]  : 0.f;
        tile[t + B + R] = (right <  n) ? x[right] : 0.f;
    }
    __syncthreads();
    if (i < n) {
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < 2 * R + 1; ++k) acc += w[k] * tile[t + k];  // w[k] broadcast
        y[i] = acc;
    }
}
```

Softmax with one warp per row and the online merge. The guard matters: a lane with no elements (rows shorter than 32) holds $m = -\infty$, and $-\infty - (-\infty)$ is NaN.

```cpp
__device__ void online_combine(float& m, float& d, float m2, float d2) {
    float mn = fmaxf(m, m2);
    if (mn == -INFINITY) return;                     // both empty
    d = d * __expf(m - mn) + d2 * __expf(m2 - mn);
    m = mn;
}

__global__ void softmax_warp_row(const float* x, float* y, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;                         // whole warp exits together
    const float* r = x + (size_t)row * cols;
    float m = -INFINITY, d = 0.f;
    for (int j = lane; j < cols; j += 32) online_combine(m, d, r[j], 1.f);   // one pass
    for (int o = 16; o > 0; o >>= 1)                 // merge the 32 lanes' states
        online_combine(m, d, __shfl_xor_sync(0xffffffff, m, o), __shfl_xor_sync(0xffffffff, d, o));
    for (int j = lane; j < cols; j += 32) y[(size_t)row * cols + j] = __expf(r[j] - m) / d;
}
```

Check the maths on the CPU before writing CUDA. Here, im2col plus one matrix multiply must reproduce `conv2d`:

```python
import torch, torch.nn.functional as F

X, W = torch.randn(1, 64, 56, 56), torch.randn(128, 64, 3, 3)
cols = F.unfold(X, kernel_size=3, padding=1)            # im2col: (1, 576, 3136)
out = (W.reshape(128, -1) @ cols).reshape(1, 128, 56, 56)
print(cols.numel() / X.numel(), torch.allclose(out, F.conv2d(X, W, padding=1), atol=1e-3))
# 9.0 True
```

## Checking the result

- **Correctness against the framework** on awkward inputs: lengths not a multiple of the block, rows shorter than 32, fully masked ($-\infty$) rows. Tolerance about `1e-5` relative in FP32, `1e-2` in BF16.
- **Speed as effective bandwidth** (31.02): required bytes over measured time. A $2^{28}$-element FP16 score tensor read once and written once is $2^{28} \times 2 \times 2$ bytes $= 1$ GiB, so at 2 TB/s the floor is about 0.54 ms.
- **The profiler for the reason.** Row and elementwise kernels should show memory throughput near peak and about 4 sectors per request for 4-byte loads; an implicit-GEMM conv should show high Tensor Core utilisation instead.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Softmax or norm at 10 to 20% of peak | one thread per row: a warp reads 32 rows | one warp or block per row, lanes stride by 32
| Long-row softmax slow | row exceeds registers and is read three times | online (max, sum) pass, then a write pass
| NaN for short or fully masked rows | $-\infty - (-\infty)$ in the merge | guard the merge; define fully masked output
| LayerNorm NaN or garbage on some rows | one-pass $E[x^2] - E[x]^2$ with a large mean | FP32 and Welford
| Conv wrong only at block edges | halo not loaded, or read before `__syncthreads()` | load both halos, then synchronise
| Constant-memory weights slow | lanes read different constant addresses | constant memory only for warp-uniform values
| $\gamma$ gradients differ run to run | column sums with float atomics | partial sums plus a second reduction

## Going deeper

**Where fusion stops.** Extra elementwise math hides under memory latency, so fusion is limited by structure, not arithmetic: ops over the same index space fuse freely, a reduction forces a barrier, and fusing into a matrix multiply needs a GEMM **epilogue** (typically bias, activation, residual).

**Convolution regimes.** Depthwise convolutions ($C_{out} = C_{in}$, one filter per channel) have little reuse and are memory-bound like the tiled kernel. $1 \times 1$ convolutions are plain GEMMs with no duplication. Many-channel $3 \times 3$ convolutions are the compute-bound implicit GEMM case. The regime tells you whether to count bytes or FLOPs.

**Why the merges work.** Online softmax and Welford are both a state, an identity ($(-\infty, 0)$; count 0) and an associative merge. Any such merge can run as a shuffle tree, a block reduction or across tiles in any grouping, with the same result up to rounding. That is what lets FlashAttention fuse softmax into attention and lets a norm handle rows longer than one block.

**Softmax backward.** $\partial L/\partial x_j = y_j (g_j - \sum_k g_k y_k)$ with $g$ the incoming gradient: one row reduction and one elementwise pass, the same warp-per-row shape.

## Key terms

| Term | Meaning
| --- | --- |
| Memory-bound | Run time set by bytes moved, not arithmetic
| Kernel fusion | Several operations in one kernel so intermediates never touch HBM
| Halo | Extra input cells around a tile needed by its edge outputs; $2r$ in 1-D
| Constant memory | Small cached read-only space that broadcasts one address to a whole warp
| im2col | Copying each convolution patch into a matrix column so convolution becomes GEMM
| Implicit GEMM | Tiled GEMM that computes patch addresses on the fly instead of building im2col
| Online softmax | One-pass softmax carrying a running max and a rescaled running sum
| Welford's method | Stable running mean and variance with an exact merge rule

<div class="take">
<div class="lab">Key takeaway</div>

treat convolution, softmax and normalisation as data-movement problems: tile with halos and constant-memory filters for small convolutions, use implicit GEMM for channel-heavy ones, give each row to a warp or block carrying (max, sum) or Welford states through shuffle reductions, and fuse neighbours so each tensor crosses HBM once.

</div>
