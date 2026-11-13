import numpy as np

META = dict(title='Adaptive average pooling', tags=['Convolutional Networks'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['average-pool-2d', 'max-pool-2d', 'roi-align'])

STATEMENT = r"""
Adaptive pooling fixes the **output** size instead of the window size, so a classifier head can accept images of any resolution. The input is split into `out_h x out_w` bins whose edges are rounded outward, which means neighbouring bins may overlap when the sizes don't divide evenly.

Write `adaptive_avg_pool2d(x, out_h, out_w)`.

## Bins
Output row `i` (for `0 <= i < out_h`) averages input rows `start_i, ..., end_i - 1` with

$$\text{start}_i = \left\lfloor \frac{i H}{\text{out\_h}} \right\rfloor, \qquad \text{end}_i = \left\lceil \frac{(i+1) H}{\text{out\_h}} \right\rceil$$

computed exactly in integers. Columns use the same rule with `W` and `out_w`. `out[c, i, j]` is the mean of `x[c, start_i:end_i, start_j:end_j]`.

## Input / output
- `x`: float array `(C, H, W)`; `out_h, out_w` are ints with `1 <= out_h` and `1 <= out_w`. They may be **larger** than `H` or `W`, in which case some input rows are repeated across bins.
- Return a `float64` array `(C, out_h, out_w)`.

## Example
`H = 5`, `out_h = 3` gives row bins `[0, 2)`, `[1, 4)`, `[3, 5)`.
"""

STARTER = '''import numpy as np

def adaptive_avg_pool2d(x, out_h, out_w):
    """x: (C, H, W). Return (C, out_h, out_w) averages over floor/ceil bins."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _bins(n, out):
    i = np.arange(out)
    return (i * n) // out, -((-(i + 1) * n) // out)       # floor and ceil in exact integer arithmetic

def adaptive_avg_pool2d(x, out_h, out_w):
    x = np.asarray(x, dtype=np.float64)
    C, H, W = x.shape
    # A 2-D prefix sum with a zero border turns every rectangle sum into 4 lookups.
    S = np.zeros((C, H + 1, W + 1))
    S[:, 1:, 1:] = x.cumsum(1).cumsum(2)
    r0, r1 = _bins(H, out_h)
    c0, c1 = _bins(W, out_w)
    R0, R1, C0, C1 = r0[:, None], r1[:, None], c0[None, :], c1[None, :]
    total = S[:, R1, C1] - S[:, R0, C1] - S[:, R1, C0] + S[:, R0, C0]
    return total / ((R1 - R0) * (C1 - C0))
'''

HINTS = [
    'Work out the start and end index of every bin along each axis first; they are two small integer arrays.',
    'Avoid float division for the bin edges: `floor(a / b)` is `a // b` and `ceil(a / b)` is `-((-a) // b)` for integers.',
    'Either loop over the `out_h * out_w` bins and take `.mean(axis=(1, 2))` of each slice, or use a 2-D cumulative sum so each bin sum is `S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]`.',
]

_r = np.random.default_rng(20)
TESTS = [
    (np.arange(16, dtype=float).reshape(1, 4, 4), 2, 2),
    (np.arange(5, dtype=float).reshape(1, 5, 1), 3, 1),
    (np.arange(6, dtype=float).reshape(1, 2, 3), 4, 5),
    (_r.normal(size=(3, 7, 9)), 3, 4),
    (_r.normal(size=(2, 6, 6)), 1, 1),
    (_r.normal(size=(1, 5, 7)), 5, 7),
    (_r.normal(size=(2, 13, 10)), 6, 3),
]

EXPLANATION = r"""
## Why floor and ceil
Bin $i$ ideally covers the continuous interval $[iH/n, (i+1)H/n)$. Rounding the start down and the end up guarantees every input row is covered by at least one bin, at the cost of overlaps when $n$ doesn't divide $H$. When it does divide, the bins tile the input exactly and the result equals ordinary average pooling with $k = s = H/n$ (test 1).

## Summed-area table
After `S[:, 1:, 1:] = x.cumsum(1).cumsum(2)`, the sum over rows $[r_0, r_1)$ and columns $[c_0, c_1)$ is $S_{r_1 c_1} - S_{r_0 c_1} - S_{r_1 c_0} + S_{r_0 c_0}$. Broadcasting the four corner index arrays evaluates all bins at once: $O(CHW)$ to build the table plus $O(C \cdot \text{out\_h} \cdot \text{out\_w})$ for the lookups, independent of bin size. Large prefix sums can lose a little precision in float32; float64 is ample here.

## Common bugs
- Computing edges with floats, e.g. `int(np.ceil((i + 1) * H / n))`: harmless here but fragile in general, and it's the kind of thing that differs across implementations.
- Using equal-width bins `H // n`, which silently drops the last rows.
- Forgetting the zero row/column in the prefix sum, which shifts every rectangle by one.

## Interview follow-ups
- Implement adaptive **max** pooling. Can you still use a prefix trick? (Not with sums; use a sparse table or just loop over bins.)
- Why does a ResNet end with adaptive average pooling to `1 x 1`? (Any input resolution then gives the same feature size for the linear classifier.)
"""
