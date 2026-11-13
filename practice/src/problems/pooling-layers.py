import numpy as np

META = dict(title='Max and average pooling', tags=['Convolutional Networks', 'Computer Vision'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['2d-convolution', 'vgg-convolution-block'])

STATEMENT = r"""
Pooling shrinks a feature map by summarising each $k \times k$ window with one number. Write `pool2d(x, k, stride=None, mode='max')`.

## Inputs
- `x`: float array `(N, C, H, W)`.
- `k`: window size (square), `stride`: step between windows; `None` means `stride = k` (non-overlapping windows).
- `mode`: `'max'` or `'avg'`.

## Output
- Shape `(N, C, H_out, W_out)` with $H_\text{out} = \lfloor (H - k)/s \rfloor + 1$ and the same for $W$ (no padding; windows that would run past the edge are dropped).
- Output pixel $(i, j)$ summarises `x[..., i*s : i*s + k, j*s : j*s + k]`, separately per sample and channel.
- `k <= H` and `k <= W` in every test.
"""

STARTER = '''import numpy as np

def pool2d(x, k, stride=None, mode='max'):
    """x: (N, C, H, W). Return (N, C, H_out, W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pool2d(x, k, stride=None, mode='max'):
    x = np.asarray(x, dtype=float)
    s = k if stride is None else stride
    # (N, C, H-k+1, W-k+1, k, k) view of every window, then keep every s-th start position
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    return win.max(axis=(-2, -1)) if mode == 'max' else win.mean(axis=(-2, -1))
'''

HINTS = [
    'Output size first: start positions are $0, s, 2s, \\dots$ as long as the window still fits, which gives $\\lfloor (H-k)/s \\rfloor + 1$.',
    '`np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))` returns every $k \\times k$ window at stride 1, as two extra trailing axes, without copying.',
    'Slice that view with `[:, :, ::s, ::s]` to apply the stride, then reduce the last two axes with `max` or `mean`.',
]

_r = np.random.default_rng(29)
TESTS = [
    (np.arange(16.0).reshape(1, 1, 4, 4), 2),
    (np.arange(16.0).reshape(1, 1, 4, 4), 2, None, 'avg'),
    (np.arange(25.0).reshape(1, 1, 5, 5), 3, 1, 'max'),
    (np.arange(25.0).reshape(1, 1, 5, 5), 2, None, 'max'),
    (_r.normal(size=(2, 3, 7, 6)), 3, 2, 'max'),
    (_r.normal(size=(2, 3, 7, 6)), 3, 2, 'avg'),
    (-np.abs(_r.normal(size=(1, 2, 4, 4))), 4, None, 'max'),
    (_r.normal(size=(1, 1, 3, 8)), 1, 2, 'avg'),
]

EXPLANATION = r"""
## What pooling buys
- **Downsampling**: a 2x2/stride-2 pool cuts the activations by 4x, so later layers are cheaper and each unit sees a larger part of the image.
- **Small translation invariance**: a max pool's output doesn't change if the strongest response moves within its window.

## Window views
`sliding_window_view` builds a strided view: no data is copied until the reduction. It produces every stride-1 window, and slicing with `::s` keeps the ones that start on the stride grid. With `k = H` (test 7) it becomes global pooling.

## Complexity
$O(N C H_\text{out} W_\text{out} k^2)$ time; the view itself costs $O(1)$ extra memory.

## Common bugs
- Using `ceil` instead of `floor` for the output size (PyTorch's `ceil_mode=True` is a different convention).
- Reshaping `(H, W)` to `(H/k, k, W/k, k)` only works when $s = k$ and $k$ divides $H$; test 4 has a 5x5 input with $k = 2$.
- Initialising a running max with 0, which is wrong for all-negative inputs (test 7).

## Interview follow-ups
- What is the backward pass of max pooling? (Route the gradient to the arg-max of each window; overlapping windows add.)
- Why have many modern architectures replaced pooling with strided convolutions, and where is global average pooling still standard?
"""
