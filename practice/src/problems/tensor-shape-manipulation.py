import numpy as np

META = dict(title='Pixel shuffle with view and permute', tags=['Computer Vision', 'Convolutional Networks'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:03.04', 'air-fanout-v1:03.03'],
            related=['window-partition-and-reverse', 'transposed-convolution', 'tensor-factory'])

STATEMENT = r"""
Super-resolution networks upsample by predicting extra channels and then rearranging them into space. A conv outputs `C * r * r` channels at low resolution; **pixel shuffle** turns every group of $r^2$ channels into an $r \times r$ block of pixels.

Write `pixel_shuffle(x, r)`:

- `x`: tensor `(B, C * r * r, H, W)`, any float dtype.
- `r`: positive int, the upscale factor. The channel count is always divisible by $r^2$.
- Return a tensor `(B, C, H * r, W * r)` with

$$\text{out}[b,\ c,\ h r + i,\ w r + j] = x[b,\ c r^2 + i r + j,\ h,\ w], \qquad 0 \le i, j < r.$$

So within each group of $r^2$ channels, the channel offset $i r + j$ picks the row $i$ and column $j$ inside the block. This is the same layout as `torch.nn.functional.pixel_shuffle`, but don't call it: use `reshape`/`view`, `permute` and `reshape` again.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def pixel_shuffle(x, r):
    """x: (B, C*r*r, H, W). Return (B, C, H*r, W*r)."""
    B, Cr2, H, W = x.shape
    return x.reshape(B, Cr2 // (r * r), H * r, W * r)
'''

SOLUTION = '''import torch

def pixel_shuffle(x, r):
    B, Cr2, H, W = x.shape
    C = Cr2 // (r * r)
    x = x.reshape(B, C, r, r, H, W)          # split channels into (c, i, j)
    x = x.permute(0, 1, 4, 2, 5, 3)          # (B, C, H, i, W, j): i sits next to h, j next to w
    return x.reshape(B, C, H * r, W * r)     # merge (h, i) -> h*r + i and (w, j) -> w*r + j
'''

HINTS = [
    'Split the channel axis into three: `(C, r, r)`. Now you have named axes `b, c, i, j, h, w`.',
    'In the output, row index is `h * r + i`, so `h` must sit immediately before `i`. Likewise `w` before `j`. Which `permute` puts the axes in the order `b, c, h, i, w, j`?',
    'After `permute` the tensor isn\'t contiguous, so finish with `reshape(B, C, H * r, W * r)` (a `view` would raise).',
]

_r = np.random.default_rng(7)
TESTS = [
    (np.arange(4.0).reshape(1, 4, 1, 1), 2),
    (np.arange(8.0).reshape(1, 4, 1, 2), 2),
    (_r.normal(size=(2, 9, 2, 3)), 3),
    (_r.normal(size=(1, 3, 4, 4)), 1),
    (_r.normal(size=(3, 8, 3, 2)), 2),
    (np.arange(2 * 16 * 2 * 2, dtype=float).reshape(2, 16, 2, 2), 4),
    (_r.normal(size=(2, 12, 1, 5)), 2),
]

EXPLANATION = r"""
## Why it works
`reshape` never moves data; it only reinterprets the flat order. Splitting `C*r*r` into `(C, r, r)` is valid because channel index $c r^2 + i r + j$ is exactly row-major order over `(c, i, j)`. The output index `h*r + i` is row-major over `(h, i)`, so the job is to bring the axes into the order `(b, c, h, i, w, j)` with one `permute` and then merge neighbours. Merging adjacent axes with `reshape` is always the row-major combination of their indices.

## view vs reshape vs contiguous
`permute` returns a strided view with no copy. `view` requires a stride pattern compatible with the new shape and raises after a permute; `reshape` copies when it has to. In a hot loop, knowing which call copies matters: the final `reshape` here is the only copy.

## Complexity
One copy of the tensor, $O(B C r^2 H W)$. The inverse (pixel unshuffle, or space-to-depth) is the same three steps in reverse, and so is ViT patchification.

## Common bugs
- Permuting to `(b, c, i, h, j, w)`: produces the right shape but tiles the image as $r \times r$ copies of a low-resolution picture.
- Splitting channels as `(r, r, C)`: that's a different, also legitimate, layout (the channel-last convention some libraries use), so weights trained one way break the other.

## Interview follow-ups
- Why does pixel shuffle avoid the checkerboard artefacts that transposed convolutions can produce?
- Write pixel unshuffle and show that `unshuffle(shuffle(x)) == x`.
"""
