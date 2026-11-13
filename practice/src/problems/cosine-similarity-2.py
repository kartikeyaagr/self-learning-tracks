import numpy as np

META = dict(title='Pairwise cosine similarity: precomputed norms and a coalescing layout',
            tags=['GPU Programming', 'Linear Algebra'], diff='Medium', runtime='cuda', tol=[1e-5, 1e-6],
            lessons=['ieng:13.03', 'ieng:13.01'],
            related=['l2-normalization', 'matrix-multiplication', 'cosine-similarity'])

STATEMENT = r"""
Retrieval scores a batch of queries against a set of documents with the cosine-similarity matrix

$$S_{ij} = \frac{a_i \cdot b_j}{\max\!\left(\lVert a_i \rVert_2 \, \lVert b_j \rVert_2,\ \varepsilon\right)}, \qquad \varepsilon = 10^{-8}$$

for rows $a_i$ of $A$ ($m \times d$) and $b_j$ of $B$ ($n \times d$). The clamp makes the similarity 0 whenever either vector is zero.

## Two decisions this problem is about
1. **Do not recompute norms.** A naive kernel where thread $(i, j)$ also computes $\lVert a_i \rVert$ and $\lVert b_j \rVert$ repeats each norm $n$ (or $m$) times. Compute all $m + n$ norms first with a small kernel (`row_norms`, one thread per row, blocks of 64), then run the main kernel.
2. **Pick the layout that coalesces.** In the main kernel, block `(16, 16)` with `j, i = cuda.grid(2)` (x → $j$). Threads of a warp then share $i$ and have consecutive $j$. Reading `B[j, k]` would make them touch addresses $d$ floats apart; reading from the transposed copy `Bt = B.T` (shape $(d, n)$, made C-contiguous on the host with `np.ascontiguousarray(B.T)`) as `Bt[k, j]` makes them read consecutive floats. `A[i, k]` is the same address for the whole half-warp: a broadcast.

Write `cosine_similarity_matrix(A, B)`.

## Inputs and output
- `A`: `float32` `(m, d)`, `B`: `float32` `(n, d)`, with $m, n, d \ge 1$. Rows may be all zero.
- Return `S`: `float32` `(m, n)`. Accumulate dot products and norms in `float32`. Compared with relative tolerance $10^{-5}$, absolute $10^{-6}$.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import math
import numpy as np
from numba import cuda, float32

EPS = 1e-8

def cosine_similarity_matrix(A, B):
    """A: float32 (m, d), B: float32 (n, d). Returns float32 (m, n)."""
    m, n = A.shape[0], B.shape[0]
    d_A, d_B = cuda.to_device(np.ascontiguousarray(A)), cuda.to_device(np.ascontiguousarray(B))
    d_S = cuda.device_array((m, n), dtype=np.float32)
    grid = ((n + 15) // 16, (m + 15) // 16)
    naive_kernel[grid, (16, 16)](d_A, d_B, d_S)
    return d_S.copy_to_host()

@cuda.jit
def naive_kernel(A, B, S):
    j, i = cuda.grid(2)
    if i < S.shape[0] and j < S.shape[1]:
        dot = float32(0.0)
        for k in range(A.shape[1]):
            dot += A[i, k] * B[j, k]     # TODO: uncoalesced reads of B; use a transposed copy
        S[i, j] = dot                    # TODO: divide by the clamped norm product (precomputed)
'''

SOLUTION = '''import math
import numpy as np
from numba import cuda, float32

EPS = 1e-8

def cosine_similarity_matrix(A, B):
    m, n = A.shape[0], B.shape[0]
    d_A = cuda.to_device(np.ascontiguousarray(A))
    d_B = cuda.to_device(np.ascontiguousarray(B))
    d_Bt = cuda.to_device(np.ascontiguousarray(B.T))       # (d, n): column j of Bt is b_j
    na = cuda.device_array(m, dtype=np.float32)
    nb = cuda.device_array(n, dtype=np.float32)
    row_norms[(m + 63) // 64, 64](d_A, na)                  # m + n norms, computed once
    row_norms[(n + 63) // 64, 64](d_B, nb)
    d_S = cuda.device_array((m, n), dtype=np.float32)
    grid = ((n + 15) // 16, (m + 15) // 16)
    cosine_kernel[grid, (16, 16)](d_A, d_Bt, na, nb, d_S)
    return d_S.copy_to_host()

@cuda.jit
def row_norms(X, out):
    r = cuda.grid(1)
    if r < X.shape[0]:
        s = float32(0.0)
        for k in range(X.shape[1]):
            s += X[r, k] * X[r, k]
        out[r] = math.sqrt(s)

@cuda.jit
def cosine_kernel(A, Bt, na, nb, S):
    j, i = cuda.grid(2)                                   # x -> j: a warp shares i, varies j
    if i < S.shape[0] and j < S.shape[1]:
        dot = float32(0.0)
        for k in range(A.shape[1]):
            dot += A[i, k] * Bt[k, j]                     # A: broadcast, Bt: consecutive j, coalesced
        S[i, j] = dot / max(na[i] * nb[j], float32(EPS))  # zero vector -> 0 / eps = 0
'''

HINTS = [
    'Write `row_norms(X, out)`: one thread per row, `out[r] = sqrt(sum_k X[r, k]**2)`. Launch it once for `A` and once for `B`.',
    'On the host, `np.ascontiguousarray(B.T)` gives a `(d, n)` array whose column `j` is $b_j$. In the main kernel, read `Bt[k, j]` instead of `B[j, k]`.',
    'Finish with `S[i, j] = dot / max(na[i] * nb[j], EPS)`. Keep `j, i = cuda.grid(2)` so x indexes columns of `S`; that is what makes both the `Bt` reads and the `S` writes contiguous across a warp.',
]

_r = np.random.default_rng(55)
_f = lambda *s: _r.standard_normal(s).astype(np.float32)
_zA = _f(3, 4)
_zA[1] = 0.0
TESTS = [
    (np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32), np.array([[0.0, 2.0], [3.0, 0.0], [-1.0, -1.0]], dtype=np.float32)),
    (np.array([[1.0, 2.0, 3.0]], dtype=np.float32), np.array([[2.0, 4.0, 6.0], [-1.0, -2.0, -3.0]], dtype=np.float32)),   # +1 and -1
    (_zA, _f(5, 4)),                                                   # a zero query row -> zeros
    (_f(1, 1), np.array([[2.0], [-3.0], [0.0]], dtype=np.float32)),    # d = 1: signs only
    (_f(17, 8), _f(33, 8)),                                            # partial tiles in both directions
    ((_f(10, 300) * 50).astype(np.float32), _f(20, 300)),
    (_f(40, 16), _f(48, 16)),
]

EXPLANATION = r"""
## Why it works
$S_{ij}$ is the dot product of $a_i$ and $b_j$ divided by the product of their norms, and the norms do not depend on the pair. Computing them in a separate pass costs $O((m+n)d)$ instead of the $O(mnd)$ extra work of recomputing them in every thread. Reading $b_j$ from the transposed copy changes only the address arithmetic, not the values.

## Coalescing, concretely
A warp of the main kernel holds 16 consecutive $j$ for one $i$ (and 16 for $i+1$). At loop step $k$:
- `B[j, k]` for 16 consecutive $j$: addresses $4d$ bytes apart. With $d = 300$ that is 16 separate memory transactions.
- `Bt[k, j]`: 16 consecutive floats, one 64-byte segment.
- `A[i, k]`: one address, broadcast.
The transpose costs one extra pass over $B$ ($8nd$ bytes), which is repaid as soon as $m$ is more than a handful of queries.

## Better still
The main loop is a matrix multiplication $A B^\top$ followed by an element-wise scale, so the fast version is a tiled (or tensor-core) GEMM with the normalisation fused into its epilogue: compute $S = (A / \lVert A \rVert)(B / \lVert B \rVert)^\top$, or scale each output tile by `1 / (na[i] * nb[j])` before writing. Vector databases go one step further and store document vectors pre-normalised, so retrieval is a plain matrix product.

## Common bugs
- `max(na * nb, eps)` vs `max(na, eps) * max(nb, eps)`: different values for tiny norms; follow the specification.
- Swapping `i` and `j` in `cuda.grid(2)`: still correct, but the reads of `Bt` and the writes of `S` are no longer coalesced.
- Passing `B.T` straight to `cuda.to_device`: it is a non-contiguous view; make a contiguous copy first.

## Interview follow-ups
- For $m = 1$ query and $n = 10^7$ documents, is this kernel compute-bound or memory-bound? What would you change?
- How does storing document vectors in `float16` or `int8` change the bandwidth cost, and what does it do to the accuracy of the ranking?
"""
