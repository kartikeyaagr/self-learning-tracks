---
id: "31.03"
course: "inference-eng-course"
module: "31"
slug: "shared-memory-and-tiling"
title: "Shared memory and tiling"
keyIdeas: ["**Shared memory** is a small on-chip scratchpad per SM (up to about 164 KB on A100, 228 KB on H100), allocated per block and shared by its threads. Nothing enters it unless your code puts it there, and it is far faster than HBM.", "The naive matmul kernel loads 8 bytes per 2 FLOPs (0.25 FLOP/byte), so it is badly memory-bound. **Tiling** loads $T \\times T$ tiles of $A$ and $B$ once per phase and reuses each value $T$ times, raising intensity to $T/4$ FLOP/byte (8 at $T = 32$).", "Each phase needs **two** `__syncthreads()`: one after loading so nobody reads a half-filled tile, and one after computing so nobody overwrites a tile others still read. Missing barriers give intermittent wrong answers; never put a barrier in a branch only some threads take.", "Shared memory has 32 **banks** of 4-byte words; bank $= (\\text{address}/4) \\bmod 32$. Lanes hitting different words in one bank serialise. A column read of a `[32][32]` tile is a 32-way conflict; padding to `[32][33]` makes it conflict-free.", "Shared-memory tiling alone reaches only about a tenth of cuBLAS; fast GEMMs add register tiling, vectorised loads, double buffering and Tensor Cores, which is why production code calls cuBLAS, CUTLASS or Triton."]
question: "You tile a matrix transpose with 32 x 32 blocks: each thread reads `in[y][x]` into `tile[ty][tx]`, synchronises, then writes `out[x'][y'] = tile[tx][ty]` with the output indices swapped so the global write is coalesced. It is faster than the naive transpose but still well below copy bandwidth. Why, and what one-line change fixes it?"
answer: "The global reads and writes are now coalesced, so global memory is no longer the problem; shared memory is. The read `tile[tx][ty]` is a column access: the 32 threads of a warp have consecutive `tx` and the same `ty`, so lane $\\ell$ reads word $32\\ell + ty$. Every one of those words satisfies $(32\\ell + ty) \\bmod 32 = ty$, so all 32 lanes hit the same bank with different words. That is a 32-way bank conflict: the warp's shared-memory read is replayed 32 times, and every warp in the kernel pays it. The fix is one line: declare the tile as `__shared__ float tile[32][33]`. Now lane $\\ell$ reads word $33\\ell + ty$, which is $\\ell + ty$ modulo 32, so the 32 lanes hit 32 different banks and the read completes in one pass. The row write `tile[ty][tx]` stays conflict-free (consecutive words), and the only cost is 32 unused floats per tile. Any odd row width would work, since the conflict degree for a column read is gcd(width, 32). With the padding, the transpose typically runs close to the speed of a plain device-to-device copy; Nsight Compute's shared-memory bank-conflict counter should drop to about zero, which confirms the diagnosis."
sources: [{"title": "NVIDIA blog: Using Shared Memory in CUDA C/C++", "url": "https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/", "note": "Shared memory declaration, synchronisation and bank conflicts with a worked example."}, {"title": "NVIDIA blog: An Efficient Matrix Transpose in CUDA C/C++", "url": "https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/", "note": "The coalesced transpose and the one-column padding trick, with measured bandwidths."}, {"title": "Simon Boehm, How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance", "url": "https://siboehm.com/articles/22/CUDA-MMM", "note": "Step-by-step from naive to tiled to register-tiled kernels, each step benchmarked."}]
practice: ["tiled-matrix-multiplication", "shared-memory-bank-conflict-analyzer", "matrix-multiplication", "tiled-transpose"]
tags: ["shared memory", "tiling", "bank conflicts"]
tools: ["CUDA"]
---

## What you will learn

- What **shared memory** is and why a kernel author manages it by hand.
- Why the naive matmul kernel is starved for bytes, and how **tiling** reuses each loaded value $T$ times.
- How to write the tiled matmul kernel, including its two `__syncthreads()` barriers and zero-filled edges.
- How to predict a shared-memory **bank conflict** with one line of arithmetic, and why one padding column fixes the classic case.
- How far plain tiling gets you, and what fast libraries add on top.

*Before you start:* threads, blocks and warps (31.01) and coalesced global loads (31.02). Arithmetic intensity and the roofline are recapped below.

## From coalescing to reuse

Lesson 31.02 made each global load efficient. This lesson fixes how *often* you load. Matrix multiply is the standard example: it has huge potential reuse, and the naive kernel throws nearly all of it away.

Shared memory is the tool. It is a small, fast scratchpad on each streaming multiprocessor (SM): up to about 164 KB per SM on an A100 and about 228 KB on an H100. A block allocates a piece, all its threads can read and write it, and it vanishes when the block ends. Unlike L1 and L2, nothing goes in unless your code puts it there. More work for you, but you decide exactly what stays on-chip. (It shares physical storage with L1; the split is configurable.) Typical published latencies, quoted from memory, are a few tens of cycles for shared memory against several hundred for HBM.

## The idea in one picture

Take $C = AB$ with $A$ and $B$ both $8 \times 8$, and one thread per output element: 64 threads.

In the **naive kernel** each thread reads its whole row of $A$ (8 floats) and its whole column of $B$ (8 floats) from global memory. That is $64 \times 16 = 1024$ global loads, although $A$ and $B$ together hold only 128 distinct numbers. Every number is fetched 8 times, once by each thread that needs it.

In the **tiled kernel** we cut $C$ into $4 \times 4$ tiles (so $T = 4$) and give each tile to a block of 16 threads. The block walks along the shared dimension in two **phases**. In each phase it copies one $4 \times 4$ tile of $A$ and one of $B$ into shared memory, then every thread does its 4 multiply-adds reading only shared memory.

~~~dg-fig 31.03-tiles.svg
The block owning C tile (0, 1) needs the top strip of A and the right strip of B, brought in one 4 x 4 piece per phase. Counted in a CPU simulation: 1024 global loads naive, 256 tiled.
~~~

| Kernel | Global loads | Bytes (FP32) | FLOPs | FLOP per byte
| --- | --- | --- | --- | --- |
| Naive | 1024 | 4096 | 1024 | 0.25
| Tiled, $T = 4$ | 256 | 1024 | 1024 | 1.0

Same arithmetic, a quarter of the traffic. **Tiling** means: load a block of data once, keep it on-chip, and use it many times before discarding it. The last column, **arithmetic intensity**, decides whether a kernel waits on memory or on arithmetic.

## Step 1: count the naive kernel's appetite

With $A$ of size $M \times K$ and $B$ of size $K \times N$, a matmul does $2MNK$ FLOPs (a multiply and an add per term) on only $MK + KN$ inputs: each element of $A$ is used $N$ times, each of $B$ $M$ times. The naive kernel fetches a fresh copy every time:

```cpp
// one thread per C[row][col]
float acc = 0.f;
for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;
```

Each loop iteration does 2 FLOPs and loads two 4-byte floats, 8 bytes. Intensity is $2/8 = 0.25$ FLOP/byte.

Now the ceiling. A kernel cannot run faster than $\min(\text{peak FLOP/s},\ \text{bandwidth} \times \text{intensity})$; this is the **roofline**. With A100 spec sheet values (19.5 TFLOPS FP32 without Tensor Cores, about 2 TB/s HBM on the 80 GB part) the naive kernel is capped at $2 \times 0.25 = 0.5$ TFLOPS if every load really went to HBM, under 3% of peak. Caches catch some repeats, but the naive kernel still lands at a few percent of cuBLAS.

~~~dg-fig 31.03-roofline.svg
Roofline from spec-sheet values, not measurements. Tiling slides the kernel right along the memory-bound slope; at T = 32 it sits at 8 FLOP/byte, just short of the ridge (about 9.8).
~~~

## Step 2: decide what one block shares

All $T$ threads in a row of an output tile need the same row of $A$; all $T$ threads in a column need the same column of $B$. So a block computing a $T \times T$ tile of $C$ needs a $T \times K$ strip of $A$ and a $K \times T$ strip of $B$, and each value is useful to $T$ threads.

The strips are too long to fit on-chip for a large $K$, so we cut them into $T \times T$ pieces and process one pair per phase. Per phase the block loads $T^2$ floats of $A$ and $T^2$ of $B$ (one of each per thread), waits until all loads are done, lets each thread do $T$ multiply-adds from shared memory, and waits again before the next phase overwrites the tiles.

The intensity follows directly. Per phase the block moves $2T^2$ floats, which is $8T^2$ bytes, and does $T^2$ threads $\times\ T$ multiply-adds $\times\ 2 = 2T^3$ FLOPs:

$$\text{intensity} = \frac{2T^3}{8T^2} = \frac{T}{4}\ \text{FLOP/byte (FP32)}.$$

Here $T$ is the tile edge in elements, the 2 counts multiply plus add, and the 8 is two floats of 4 bytes. Number check: $T = 4$ gives 1.0, matching the table above; $T = 32$ gives 8, a 32x improvement over the naive 0.25.

~~~dg-fig 31.03-traffic.svg
Left: in a CPU simulation of a 256 x 256 matmul, global traffic falls from 128 MiB at T = 1 to 4 MiB at T = 32, exactly naive / T. Right: shared memory per block stays small, but one thread per output means T² threads, and the 1024-thread limit caps this design at T = 32.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the traffic for a 4096 x 4096 matmul, by hand**

Sizes: $M = N = K = 4096$. FLOPs: $2 \times 4096^3 = 2 \times 68{,}719{,}476{,}736 \approx 1.374 \times 10^{11}$.

Naive traffic: each of the $4096^2 = 16{,}777{,}216$ threads loads $2 \times 4096 = 8192$ floats, so $16{,}777{,}216 \times 8192 \times 4$ bytes $= 549{,}755{,}813{,}888$ bytes, about 550 GB. Check: $1.374 \times 10^{11} / 5.50 \times 10^{11} = 0.25$ FLOP/byte.

Tiled with $T = 32$: divide by 32, giving $17{,}179{,}869{,}184$ bytes, about 17.2 GB. Intensity $= 8$ FLOP/byte.

Time spent just moving bytes at 2 TB/s: naive $550 / 2000 = 0.275$ s; tiled $17.2 / 2000 \approx 0.0086$ s. Time for the FLOPs at 19.5 TFLOPS: $1.374 \times 10^{11} / 1.95 \times 10^{13} \approx 0.0070$ s.

Reading: the naive kernel's memory time is 39 times its compute time, hopelessly memory-bound. The tiled kernel's memory time (8.6 ms) is close to its compute time (7.0 ms), which is what "near the ridge" means.

</div>

## Step 3: write the kernel

```cpp
#define T 32
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[T][T], Bs[T][T];              // 2 x 4 KB, per block
    int row = blockIdx.y * T + threadIdx.y;           // which C element this thread owns
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += T) {               // one iteration = one phase
        As[threadIdx.y][threadIdx.x] = (row < M && k0 + threadIdx.x < K) ? A[row * K + k0 + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (k0 + threadIdx.y < K && col < N) ? B[(k0 + threadIdx.y) * N + col] : 0.f;
        __syncthreads();                              // barrier 1: tiles fully loaded
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                              // barrier 2: everyone done reading
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
// launch: dim3 block(T, T); dim3 grid((N + T - 1) / T, (M + T - 1) / T);
```

Reading it line by line:

- `__shared__` arrays live in the SM's shared memory, one copy per block: $2 \times 32 \times 32 \times 4 = 8192$ bytes here.
- Each thread loads one element of each tile. Both loads are coalesced: consecutive `threadIdx.x` read consecutive addresses along a row (31.02).
- Out-of-range positions get zero, which adds nothing to the dot product, so any matrix size works.
- The inner loop touches only shared memory; each loaded value is read by 32 threads.
- `__syncthreads()` is a **barrier**: no thread passes it until all threads of the block reach it, and shared-memory writes before it are visible to everyone after it.

## Step 4: understand the two barriers

Both barriers protect against warps running at different speeds: a 1024-thread block is 32 warps, interleaved freely by the scheduler.

- **Without barrier 1**, a fast warp starts multiplying while a slow warp has not yet stored its row of `Bs`. The fast warp reads leftovers from the previous phase (or uninitialised memory in phase 0).
- **Without barrier 2**, a fast warp finishes its math, loops around and stores the *next* phase's values into `Bs` while a slow warp is still reading the current ones.

Here the threads of one warp share `threadIdx.y`, so a warp is one row of the tile. Each warp reads only its own row of `As`, which it wrote itself, but it reads *every* row of `Bs`. That is why the damage shows up in whole rows.

~~~dg-fig 31.03-race.svg
CPU simulation of one 32 x 32 block, K = 64 (two phases), with five lagging warps. Without the second barrier they read B-tile rows that faster warps already replaced, so exactly their five rows (160 outputs) are wrong.
~~~

On real hardware the lag is timing-dependent, so the bug may show up once in a thousand runs; `compute-sanitizer --tool racecheck` finds it reliably. Also, every thread must reach the same `__syncthreads()`: never put one inside a branch only some threads take, such as `if (row < M)`. That is why the kernel zero-fills instead of skipping threads.

## Step 5: banks, the second half of the story

To serve a warp's 32 requests in parallel, shared memory is split into 32 **banks**, each delivering one 4-byte word per cycle. Consecutive words go to consecutive banks:

$$\text{bank} = \left(\frac{\text{byte address}}{4}\right) \bmod 32.$$

A warp's access completes in one pass if its 32 threads hit 32 different banks, or if several of them read the *same* word (that word is **broadcast**). If $d$ threads hit *different* words in the same bank, it is a **$d$-way bank conflict**: the access is replayed $d$ times.

| Access by a warp (float tile `s[32][32]`) | Banks touched | Cost
| --- | --- | --- |
| Row: `s[r][lane]` | 32 distinct | 1 pass
| Same element: `s[r][k]` for all lanes | 1 word, broadcast | 1 pass
| Column: `s[lane][c]` | all in bank `c` | 32 passes (32-way conflict)
| Column on padded `s[32][33]`: `s[lane][c]` | 32 distinct | 1 pass

~~~dg-fig 31.03-banks.svg
Left: the bank each lane hits for three patterns (column c = 5). Right: column-read cost by row width, computed by brute force; it equals gcd(width, 32), so any odd width is conflict-free.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why one extra column removes the conflict**

A tile `float s[32][32]` stores row $r$, column $c$ at word $32r + c$. A warp reads column $c = 5$: lane $\ell$ reads word $32\ell + 5$.

Lanes 0 to 3 read words 5, 37, 69, 101, and each is 5 modulo 32. All 32 lanes land in bank 5 with different words: 32 passes.

Declare `float s[32][33]` instead. Now the element sits at word $33r + c$, so lane $\ell$ reads word $33\ell + 5$, and $33\ell + 5 \equiv \ell + 5 \pmod{32}$.

Lanes 0 to 3 read words 5, 38, 71, 104, banks 5, 6, 7, 8. All 32 lanes hit 32 different banks: one pass. The cost is 32 unused floats (128 bytes) per tile.

</div>

In the matmul kernel a warp has fixed `threadIdx.y`, so `As[threadIdx.y][k]` is a broadcast and `Bs[k][threadIdx.x]` a row access: no conflicts.

The **tiled transpose** is where conflicts bite. To make both the global read and the global write coalesced, a block reads a $32 \times 32$ tile row by row into `tile[ty][tx]`, synchronises, and writes it out reading `tile[tx][ty]`: a column. Declared `[32][32]`, that read is exactly the 32-way conflict from the example above; declared `[32][33]`, it is conflict-free.

## Shared memory and banks in code

Both ideas can be modelled exactly on a CPU: the first function mimics the tiled kernel and counts global loads, the second counts the passes a warp needs.

```python
import numpy as np

def tiled_matmul(A, B, T):
    # CPU model of the tiled kernel: one (i, j) loop step = one thread block
    M, K = A.shape; N = B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype); gloads = 0
    for i in range(0, M, T):                   # blockIdx.y
        for j in range(0, N, T):               # blockIdx.x
            acc = np.zeros((T, T), dtype=A.dtype)
            for k0 in range(0, K, T):          # phases
                As = np.zeros((T, T), A.dtype); Bs = np.zeros((T, T), A.dtype)
                a = A[i:i+T, k0:k0+T]; b = B[k0:k0+T, j:j+T]
                As[:a.shape[0], :a.shape[1]] = a   # zero-fill the ragged edge
                Bs[:b.shape[0], :b.shape[1]] = b
                gloads += a.size + b.size
                acc += As @ Bs                 # T multiply-adds per thread, from "shared"
            C[i:i+T, j:j+T] = acc[:min(T, M-i), :min(T, N-j)]
    return C, gloads

def bank_passes(word_addrs):
    # passes a warp needs: max number of DISTINCT words that land in one bank
    banks = np.asarray(word_addrs) % 32
    return max(len(set(np.asarray(word_addrs)[banks == b])) for b in range(32))

rng = np.random.default_rng(0)
A = rng.standard_normal((100, 70)).astype(np.float32)
B = rng.standard_normal((70, 90)).astype(np.float32)
naive = 100 * 90 * 2 * 70                      # 1,260,000 loads
for T in (8, 16, 32):
    C, g = tiled_matmul(A, B, T)
    print(T, np.abs(C - A @ B).max() < 1e-4, g, round(naive / g, 1))
# 8  True 165900  7.6
# 16 True  86100 14.6
# 32 True  46200 27.3

lane = np.arange(32)
print(bank_passes(lane * 32 + 5))      # column of s[32][32]    -> 32
print(bank_passes(lane * 33 + 5))      # column of s[32][33]    -> 1
print(bank_passes(2 * lane))           # stride-2 floats        -> 2
```

On these awkward sizes the saving is below $T$ (27.3x at $T = 32$): $M = 100$ needs $\lceil 100/32 \rceil = 4$ block rows rather than 3.125, so the strips of $B$ are read once more than ideal. Ragged edges always cost a little.

## Checking that tiling worked

On a real GPU, measure rather than guess:

- **Correctness**: compare with `torch.matmul` using a tolerance (reordered FP32 sums differ in the last bits), on sizes that are *not* multiples of $T$, and run racecheck once.
- **Throughput**: achieved FLOP/s is $2MNK / t$; compare it with the roofline cap for your intensity.
- **Profiler**: Nsight Compute reports bank conflicts and DRAM bytes; after tiling, DRAM bytes should drop by roughly $T$.

How far does this go? In Simon Boehm's published worklog (RTX A6000, FP32), the naive kernel reached 1.3% of cuBLAS, coalesced loads 8.5%, shared-memory tiling 12.8%. Several outputs per thread lifted it to 68.7%, and vectorised loads plus warp tiling to 93.7%.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Wrong values that change between runs | a missing `__syncthreads()` | two barriers per phase; check with racecheck
| Kernel hangs or behaves unpredictably | `__syncthreads()` inside a branch not all threads take | move the barrier out; zero-fill instead of skipping threads
| Wrong results only for sizes not divisible by $T$ | no bounds checks on the tile loads | load 0 for out-of-range elements, guard the final store
| Tiled transpose far below copy bandwidth | 32-way bank conflict on the column read | declare the tile `[32][33]`
| Launch fails with too many resources | more than 48 KB of static shared memory, or more than 1024 threads per block | dynamic shared memory with an opt-in attribute; fewer threads, more outputs each
| Faster kernel, but fewer blocks resident | every KB of shared memory per block reduces how many blocks fit on an SM | size tiles with occupancy in mind (31.07)

Beyond 48 KB, use `extern __shared__ float s[];`, pass the byte count as the third launch parameter, and opt in with `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.

## Going deeper

**Register tiling.** This lesson's kernel still does two shared-memory loads per multiply-add, so it becomes limited by shared-memory bandwidth and instruction issue. Fast kernels give each thread a small block of outputs, say $8 \times 8$: load 8 values of `As` and 8 of `Bs` into registers, do 64 multiply-adds. That is tiling one level down, and it lifts the $T^2 \le 1024$ limit: 256 threads can own a $128 \times 128$ tile.

**Double buffering.** While multiplying tile $p$, the block can already fetch tile $p+1$ into a second buffer. Ampere's `cp.async` copies global to shared memory without passing through registers; Hopper's Tensor Memory Accelerator (TMA) copies whole tiles. This hides global latency at the cost of twice the shared memory.

**Swizzling instead of padding.** Padding wastes space and breaks alignment for vector loads, so libraries such as CUTLASS instead XOR the column index with bits of the row index. Same goal as `[32][33]`: rows and columns both spread over all banks.

**Tensor Cores and lower precision.** In BF16 each element is 2 bytes, doubling a tile's intensity, while Tensor Cores raise the compute roof by roughly an order of magnitude and push the ridge far right. So a hand-written tiled kernel is a learning tool; production code calls cuBLAS, CUTLASS or Triton.

## Key terms

| Term | Meaning
| --- | --- |
| Shared memory | Fast on-chip memory per SM, allocated per block and visible to all its threads; filled only by your code
| Tiling | Loading a block of data once into faster memory and reusing it many times before discarding it
| Arithmetic intensity | FLOPs performed per byte moved from global memory; $T/4$ for FP32 tiled matmul
| `__syncthreads()` | Block-wide barrier; all threads wait there and see each other's shared-memory writes afterwards
| Bank | One of 32 slices of shared memory; word $w$ lives in bank $w \bmod 32$
| Bank conflict | Several lanes of a warp hit different words in one bank, so the access is replayed
| Padding | An unused extra column (`[32][33]`) that spreads column reads across banks
| Register tiling | Each thread computes a small block of outputs, reusing shared-memory loads from registers

<div class="take">
<div class="lab">Key takeaway</div>

tile so that each byte fetched from global memory is reused many times from shared memory (intensity rises from 0.25 to $T/4$ FLOP/byte), fence each phase with two barriers, and lay out shared arrays so a warp never hits one bank with different addresses.

</div>
