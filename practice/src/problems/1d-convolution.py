import numpy as np

META = dict(title="1-D 'same' convolution: one thread per output", tags=['GPU Programming', 'Convolutional Networks'],
            diff='Easy', runtime='cuda', tol=[1e-5, 1e-6], lessons=['ieng:13.01', 'ieng:13.03'],
            related=['2d-convolution-2', 'vector-addition', '2d-convolution'])

STATEMENT = r"""
Deep-learning "convolution" is really cross-correlation (the kernel is not flipped). With an odd kernel length $K$ and zero padding of $h = (K-1)/2$ on both sides, the output has the same length as the input:

$$y_i = \sum_{k=0}^{K-1} w_k \, x_{i + k - h}, \qquad x_j = 0 \text{ for } j < 0 \text{ or } j \ge n$$

This is the layer behind 1-D audio and sequence models (and the short causal convolution in Mamba-style blocks, which pads only on the left).

Write `conv1d_same(x, w)`.

## Requirements
- One thread per output element: blocks of **128 threads**, grid $\lceil n / 128 \rceil$, `i = cuda.grid(1)`, threads with `i >= n` do nothing.
- Padding is **implicit**: do not build a padded copy of `x`. Skip (or treat as zero) taps whose input index falls outside `[0, n)`.
- Accumulate in a local `float32` variable and write `y[i]` once.

## Inputs and output
- `x`: `float32`, shape `(n,)`, `n >= 1`. `w`: `float32`, shape `(K,)` with $K$ odd and $K \ge 1$; $K$ may be larger than $n$.
- Return `y`: `float32`, shape `(n,)`. Compared with relative tolerance $10^{-5}$, absolute $10^{-6}$.

## Example
`x = [1, 2, 3, 4]`, `w = [1, 0, -1]` gives `y = [0 - 2, 1 - 3, 2 - 4, 3 - 0] = [-2, -2, -2, 3]`.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda, float32

TPB = 128

def conv1d_same(x, w):
    """x: float32 (n,), w: float32 (K,), K odd. Returns float32 (n,)."""
    n = x.shape[0]
    d_x, d_w = cuda.to_device(x), cuda.to_device(w)
    d_y = cuda.device_array(n, dtype=np.float32)
    conv1d_kernel[(n + TPB - 1) // TPB, TPB](d_x, d_w, d_y)
    return d_y.copy_to_host()

@cuda.jit
def conv1d_kernel(x, w, y):
    i = cuda.grid(1)
    if i < x.shape[0]:
        acc = float32(0.0)
        for k in range(w.shape[0]):
            acc += w[k] * x[i + k]   # TODO: centre the window and handle the edges
        y[i] = acc
'''

SOLUTION = '''import numpy as np
from numba import cuda, float32

TPB = 128

def conv1d_same(x, w):
    n = x.shape[0]
    d_x, d_w = cuda.to_device(x), cuda.to_device(w)
    d_y = cuda.device_array(n, dtype=np.float32)
    conv1d_kernel[(n + TPB - 1) // TPB, TPB](d_x, d_w, d_y)
    return d_y.copy_to_host()

@cuda.jit
def conv1d_kernel(x, w, y):
    i = cuda.grid(1)
    n, K = x.shape[0], w.shape[0]
    if i >= n:
        return
    h = (K - 1) // 2                   # half-width: zero padding on each side
    acc = float32(0.0)
    for k in range(K):
        j = i + k - h                  # input index this tap reads
        if 0 <= j < n:                 # implicit zero padding
            acc += w[k] * x[j]
    y[i] = acc
'''

HINTS = [
    'The tap $k$ of output $i$ reads input index `j = i + k - h` with `h = (K - 1) // 2`. Check it against the example: for `i = 0`, `k = 0` reads `j = -1`, which is padding.',
    'Guard each tap with `if 0 <= j < n:` and add nothing otherwise. That is exactly zero padding, without allocating a padded array.',
    'Keep the bounds check on `i` too: the last block usually has threads past the end.',
]

_r = np.random.default_rng(51)
TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), np.array([1.0, 0.0, -1.0], dtype=np.float32)),
    (np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)),
    (np.array([2.0, -3.0, 5.0], dtype=np.float32), np.array([4.0], dtype=np.float32)),            # K = 1: scaling
    (np.array([1.0, 2.0], dtype=np.float32), np.arange(1, 8, dtype=np.float32)),                  # K > n
    (np.arange(200, dtype=np.float32), np.array([0.25, 0.5, 0.25], dtype=np.float32)),            # 2 blocks
    (_r.standard_normal(129).astype(np.float32), _r.standard_normal(5).astype(np.float32)),
    (_r.standard_normal(1000).astype(np.float32), _r.standard_normal(9).astype(np.float32)),
]

EXPLANATION = r"""
## Why it works
Each output is an independent weighted sum of a window of $K$ inputs centred at $i$, so one thread per output needs no communication. The index arithmetic $j = i + k - h$ places tap $h$ (the centre) on $x_i$; out-of-range taps are skipped, which equals multiplying by an implicit zero.

## Cost and reuse
Each thread reads $K$ inputs and $K$ weights for $2K$ FLOPs. Neighbouring threads read almost the same inputs: thread $i$ and thread $i+1$ share $K-1$ of them. Caches catch much of that, but the systematic fix is the 2-D convolution problem's approach: a block loads its input **tile plus halo** (128 + K - 1 values) into shared memory once, then every thread reads from there. The weights, read identically by every thread, are a textbook use of **constant memory** (`cuda.const.array_like` in Numba), whose cache broadcasts one address to a whole warp in a single cycle.

## Common bugs
- Off-by-$h$ indexing (`x[i + k]`, as in the starter): the output is shifted and the right edge reads past the end.
- Flipping the kernel (`w[K - 1 - k]`): that is mathematical convolution, not what deep-learning frameworks compute.
- Building a padded copy on the host: correct, but extra memory traffic and an allocation per call.

## Interview follow-ups
- How does the indexing change for a causal convolution (output $i$ may only see $x_{\le i}$)? And for stride 2 or dilation 2?
- For $K = 4$ and $n = 10^6$, is this kernel compute-bound or memory-bound? What about $K = 255$?
"""
