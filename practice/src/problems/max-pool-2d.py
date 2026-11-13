import numpy as np

META = dict(title='2-D max pooling with stride', tags=['Convolutional Networks'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['average-pool-2d', 'adaptive-average-pool-2d'])

STATEMENT = r"""
Max pooling slides a window over each channel of a feature map and keeps the largest value in each window. It shrinks the map and makes the response tolerant to small shifts.

Write `max_pool2d(x, kernel, stride=None)`.

## Input
- `x`: float array of shape `(C, H, W)`.
- `kernel`: an int `k` (a `k x k` window) or a pair `(kh, kw)`.
- `stride`: `None` (use the kernel size, so windows don't overlap), an int, or a pair `(sh, sw)`.
- No padding. Windows that would run past the bottom or right edge are dropped.

## Output
A `float64` array of shape `(C, H_out, W_out)` with

$$H_{out} = \left\lfloor \frac{H - k_h}{s_h} \right\rfloor + 1, \qquad W_{out} = \left\lfloor \frac{W - k_w}{s_w} \right\rfloor + 1$$

and `out[c, i, j] = max(x[c, i*sh : i*sh + kh, j*sw : j*sw + kw])`. You may assume `kh <= H` and `kw <= W`.
"""

STARTER = '''import numpy as np

def max_pool2d(x, kernel, stride=None):
    """x: (C, H, W). Return (C, H_out, W_out) window maxima, no padding."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)

def max_pool2d(x, kernel, stride=None):
    x = np.asarray(x, dtype=np.float64)
    kh, kw = _pair(kernel)
    sh, sw = _pair(kernel if stride is None else stride)
    # Every kh x kw window at stride 1, as a view: (C, H-kh+1, W-kw+1, kh, kw). Subsample by the stride.
    win = sliding_window_view(x, (kh, kw), axis=(1, 2))[:, ::sh, ::sw]
    return win.max(axis=(-2, -1))
'''

HINTS = [
    'Every output value is the max over one window. The only real work is getting the window positions right.',
    '`numpy.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))` gives all stride-1 windows without copying. Keep every `sh`-th and `sw`-th one.',
    'Normalise `kernel` and `stride` to pairs, take `view[:, ::sh, ::sw]`, then `.max(axis=(-2, -1))`.',
]

_r = np.random.default_rng(18)
TESTS = [
    (np.arange(16, dtype=float).reshape(1, 4, 4), 2),
    (np.array([[[1.0, 5.0, 2.0], [3.0, 0.0, 4.0], [6.0, 1.0, 1.0]]]), 2, 1),
    (np.arange(30, dtype=float).reshape(1, 5, 6), (2, 3)),
    (-np.arange(25, dtype=float).reshape(1, 5, 5), 2),
    (_r.normal(size=(3, 7, 7)), 3, 2),
    (_r.normal(size=(2, 6, 9)), (3, 2), (1, 3)),
    (np.array([[[2.0]]]), 1),
]

EXPLANATION = r"""
## Windows as a view
`sliding_window_view` returns a strided view whose last two axes index inside a window. Slicing the window-position axes with `::s` implements the stride. Nothing is copied until the reduction, so this is as fast as NumPy gets without writing a loop. The floor in the size formula is automatic: positions whose window would overflow simply don't exist in the view.

## Complexity
$O(C \cdot H_{out} W_{out} \cdot k_h k_w)$ time; the view itself is free.

## Common bugs
- Off-by-one output sizes: forgetting the `+ 1` or using ceil.
- Reshape tricks like `x.reshape(C, H//k, k, W//k, k).max((2, 4))` only work when the stride equals the kernel and the size divides evenly (test 3 breaks the second assumption on rows).
- Pooling over the channel axis by passing the wrong `axis` to the window view.

## Backward pass
The gradient flows only to the argmax of each window, so frameworks store those indices in the forward pass. With overlapping windows (stride < kernel) one input can receive gradient from several outputs and the contributions add up.

## Interview follow-ups
- Return the flat argmax index of each window as well, with a tie rule (first in row-major order).
- When would you replace max pooling with a strided convolution? (When you want the downsampling to be learned, as in most modern architectures.)
"""
