import numpy as np

META = dict(title='Average pooling with padding', tags=['Convolutional Networks'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['max-pool-2d', 'adaptive-average-pool-2d'])

STATEMENT = r"""
Average pooling replaces each window with its mean. When the input is zero-padded, there's a choice at the border: divide by the full window size (the padding counts as zeros) or only by the number of **real** pixels in the window.

Write `avg_pool2d(x, k, stride, padding=0, count_include_pad=True)`.

## Input
- `x`: float array `(C, H, W)`.
- `k`: window side (int). `stride`: int `>= 1`. `padding`: int `>= 0` zeros added on every side, with `padding <= k // 2`.
- `count_include_pad`: if `True`, every window is divided by `k * k`. If `False`, it is divided by how many of its cells lie inside the original (unpadded) image.

## Output
`float64` array `(C, H_out, W_out)` with

$$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$$

(same for $W$). Window $(i, j)$ covers padded rows $i s, \dots, i s + k - 1$ and columns $j s, \dots, j s + k - 1$. Assume $H + 2p \ge k$ and $W + 2p \ge k$.

## Example
`x = [[[1, 2], [3, 4]]]`, `k = 2`, `stride = 2`, `padding = 1`: the four windows each contain one real pixel, giving `[[0.25, 0.5], [0.75, 1.0]]` with `count_include_pad=True` and `[[1, 2], [3, 4]]` with `False`.
"""

STARTER = '''import numpy as np

def avg_pool2d(x, k, stride, padding=0, count_include_pad=True):
    """x: (C, H, W). Return (C, H_out, W_out) window means over a zero-padded input."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def _window_sums(a, k, s):
    return sliding_window_view(a, (k, k), axis=(-2, -1))[..., ::s, ::s, :, :].sum(axis=(-2, -1))

def avg_pool2d(x, k, stride, padding=0, count_include_pad=True):
    x = np.asarray(x, dtype=np.float64)
    p = padding
    sums = _window_sums(np.pad(x, ((0, 0), (p, p), (p, p))), k, stride)
    if count_include_pad:
        return sums / (k * k)
    # Pool a mask of ones the same way: each window's sum is its count of real pixels.
    ones = np.pad(np.ones(x.shape[1:]), p)
    return sums / _window_sums(ones, k, stride)
'''

HINTS = [
    'Pad first, then this is ordinary window sums. The only difference between the two modes is the divisor.',
    'For `count_include_pad=False`, the divisor of a window is the number of real pixels it covers. That number is itself a window sum, of something simple.',
    'Pad an array of ones of shape `(H, W)` with zeros and pool it with the same windows: those sums are the counts to divide by.',
]

_r = np.random.default_rng(19)
x1 = np.array([[[1.0, 2.0], [3.0, 4.0]]])
TESTS = [
    (x1, 2, 2, 1, True),
    (x1, 2, 2, 1, False),
    (np.arange(16, dtype=float).reshape(1, 4, 4), 2, 2),
    (np.arange(25, dtype=float).reshape(1, 5, 5), 3, 1, 1, False),
    (_r.normal(size=(2, 6, 7)), 3, 2, 1, True),
    (_r.normal(size=(2, 6, 7)), 3, 2, 1, False),
    (_r.normal(size=(3, 5, 4)), 2, 1),
]

EXPLANATION = r"""
## Counting real pixels with a mask
A window's number of real pixels is the sum over the window of an indicator that is 1 inside the image and 0 in the padding. Pooling that padded mask with the same windows gives every divisor in one vectorised call. Interior windows get $k^2$; edge and corner windows get less. Because `padding <= k // 2`, every window contains at least one real pixel, so no divisor is 0.

## Which mode to use
`count_include_pad=True` biases border outputs toward 0, as if the image fades to black. `False` gives an unbiased mean of the visible pixels. Frameworks differ in their defaults, so a model ported from one to another can shift slightly near the borders.

## Complexity
$O(C \cdot H_{out} W_{out} k^2)$. A summed-area table (2-D cumulative sum) brings any window sum to $O(1)$, so the whole pool becomes $O(C H W)$ regardless of $k$.

## Common bugs
- Dividing padded windows by $k^2$ when the caller asked for real-pixel counts, or vice versa.
- Padding the channel axis as well.
- Computing the counts with a closed formula per border case: easy to get wrong at corners, and pointless once the mask trick is available.

## Interview follow-ups
- Implement the window sums with a summed-area table.
- Global average pooling is a special case of which parameters? (`k = H = W`, `stride = 1`, `padding = 0`.)
"""
