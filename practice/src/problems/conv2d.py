import numpy as np

META = dict(title='Grouped conv2d forward and backward with unfold and fold', tags=['Convolutional Networks', 'Neural Networks'],
            diff='Hard', runtime='torch', lessons=['air-fanout-v1:04.02'],
            related=['2d-convolution', 'multichannel-2d-convolution', 'depthwise-separable-convolution'])

STATEMENT = r"""
Most convolution layers can be written as one big matrix multiply: `F.unfold` copies every receptive field into a column (im2col), and `F.fold` scatters columns back, summing overlaps. Use the pair to write a grouped 2-D convolution **and its backward pass by hand**.

Write `conv2d_unfold(x, w, b, stride, padding, groups, gy)`:

- `x`: float64 `(N, C, H, W)`.
- `w`: float64 `(O, C // groups, kh, kw)`, the same layout as `nn.Conv2d.weight`. `b`: float64 `(O,)`.
- `stride`, `padding`: ints, applied to both spatial axes (zero padding). Dilation is 1.
- `groups`: int dividing both `C` and `O`. Input channels `[g*C/G, (g+1)*C/G)` feed only output channels `[g*O/G, (g+1)*O/G)`.
- `gy`: float64 `(N, O, Ho, Wo)`, the upstream gradient $\partial L / \partial y$, where $H_o = \lfloor (H + 2p - k_h)/s \rfloor + 1$ (same for $W_o$).

Return a tuple `(y, dx, dw, db)` with the shapes of the output, `x`, `w` and `b`: the forward output (identical to `F.conv2d(x, w, b, stride, padding, groups=groups)`) and the gradients of $L = \sum y \odot g_y$.

Don't call `F.conv2d`, `conv_transpose2d`, `torch.nn.grad` or `backward()`; derive the backward with `unfold`, `fold` and matrix products. Rows or columns of `x` that no window covers (when the stride doesn't divide evenly) get zero gradient.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
import torch.nn.functional as F

def conv2d_unfold(x, w, b, stride, padding, groups, gy):
    """Return (y, dx, dw, db) for a grouped conv2d, using F.unfold / F.fold."""
    N, C, H, W = x.shape
    O, Cg, kh, kw = w.shape
    cols = F.unfold(x, (kh, kw), padding=padding, stride=stride)    # (N, C*kh*kw, L)
    # TODO: split cols and w by group, multiply, reshape to (N, O, Ho, Wo); then the backward pass
    raise NotImplementedError
'''

SOLUTION = '''import torch
import torch.nn.functional as F

def conv2d_unfold(x, w, b, stride, padding, groups, gy):
    N, C, H, W = x.shape
    O, Cg, kh, kw = w.shape
    G, Og = groups, O // groups
    Ho = (H + 2 * padding - kh) // stride + 1
    Wo = (W + 2 * padding - kw) // stride + 1

    # im2col: rows are (channel, ki, kj) with channel outermost, so a group's rows are contiguous
    cols = F.unfold(x, (kh, kw), padding=padding, stride=stride)    # (N, C*kh*kw, L)
    cols = cols.view(N, G, Cg * kh * kw, -1)                         # (N, G, K, L)
    wm = w.reshape(G, Og, Cg * kh * kw)                              # (G, Og, K)

    y = torch.einsum("gok,ngkl->ngol", wm, cols).reshape(N, O, Ho, Wo) + b.view(1, O, 1, 1)

    g = gy.reshape(N, G, Og, Ho * Wo)                                # (N, G, Og, L)
    db = gy.sum(dim=(0, 2, 3))
    dw = torch.einsum("ngol,ngkl->gok", g, cols).reshape(w.shape)   # sum over batch and positions
    dcols = torch.einsum("gok,ngol->ngkl", wm, g).reshape(N, C * kh * kw, -1)
    dx = F.fold(dcols, (H, W), (kh, kw), padding=padding, stride=stride)   # col2im: overlaps add up
    return y, dx, dw, db
'''

HINTS = [
    '`F.unfold(x, (kh, kw), padding=p, stride=s)` returns `(N, C*kh*kw, Ho*Wo)` with the channel index outermost, matching `w.reshape(O, -1)`. Without groups the forward is `w.reshape(O, -1) @ cols`.',
    'For groups, view the columns as `(N, G, C/G*kh*kw, L)` and the weight as `(G, O/G, C/G*kh*kw)`, then use one `einsum` or a batched matmul.',
    'Backward of $Y = W_m\\,\\text{cols}$: $dW_m = g\\,\\text{cols}^\\top$ summed over the batch, $d\\text{cols} = W_m^\\top g$, and `F.fold` (the adjoint of `unfold`) maps `dcols` back to `dx`, adding overlapping windows.',
]

_r = np.random.default_rng(11)

def _case(n, c, h, w_, o, k, s, p, g):
    x = _r.normal(size=(n, c, h, w_))
    w = _r.normal(size=(o, c // g, k, k))
    ho, wo = (h + 2 * p - k) // s + 1, (w_ + 2 * p - k) // s + 1
    return x, w, _r.normal(size=o), s, p, g, _r.normal(size=(n, o, ho, wo))

TESTS = [
    (np.arange(9.0).reshape(1, 1, 3, 3), np.ones((1, 1, 2, 2)), np.array([0.5]), 1, 0, 1, np.ones((1, 1, 2, 2))),
    (np.arange(16.0).reshape(1, 2, 2, 4), np.array([[[[1.0]], [[-1.0]]], [[[2.0]], [[0.0]]]]), np.zeros(2), 2, 0, 1,
     np.array([[[[1.0, 2.0]]], [[[0.0, 1.0]]]]).reshape(1, 2, 1, 2)),
    _case(2, 3, 5, 5, 4, 3, 1, 1, 1),
    _case(1, 4, 6, 7, 6, 3, 2, 1, 2),
    _case(2, 4, 5, 5, 4, 3, 1, 1, 4),
    _case(1, 2, 7, 6, 2, 2, 3, 0, 1),
    _case(1, 6, 4, 4, 3, 1, 1, 0, 3),
    _case(3, 2, 6, 6, 4, 5, 1, 2, 2),
]

EXPLANATION = r"""
## Convolution is a matrix multiply in disguise
Each output position is a dot product between the flattened kernel and one flattened receptive field. `unfold` lays the receptive fields out as columns, so the whole layer becomes $Y = W_m \cdot \text{cols}$ with $W_m$ of shape `(O, C*kh*kw)`. Groups just make it block diagonal: group $g$'s output channels see only group $g$'s rows of `cols`, which is one batched matmul over the group axis.

## The backward pass
Because the forward is linear in both $W$ and `cols`, the matmul rules apply directly:

- $dW_m = \sum_n g_n \,\text{cols}_n^\top$ (sum over the batch and over all positions, since the kernel is shared).
- $d\text{cols} = W_m^\top g$.
- `unfold` is a linear map that copies pixels; its adjoint `fold` sums each column entry back into the pixel it came from. Pixels shared by overlapping windows collect several contributions, padding positions are dropped, and pixels no window covers get 0.
- $db$ is the sum of $g_y$ over batch and space.

## Complexity
Time $O(N \cdot O \cdot (C/G) \cdot k_h k_w \cdot H_o W_o)$ for each of the three products. Memory is the catch: `cols` holds $k_h k_w$ copies of the input, which is why cuDNN and implicit-GEMM kernels build tiles on the fly instead of materialising im2col.

## Common bugs
- Reshaping `cols` as `(N, K, G, L)`: the channel index is outermost in `unfold`, so the group axis must come first.
- Forgetting that `dw` sums over the batch.
- Calling `F.fold` without the original `(H, W)`: when the stride doesn't divide evenly the output size isn't recoverable from the columns.

## Interview follow-ups
- Show that the backward-data pass of a stride-1 convolution is a convolution of $g_y$ with the flipped kernel, and what changes with stride 2.
- How large is `cols` for a 3x3 conv on a `(64, 256, 56, 56)` float32 activation, and how do Winograd or FFT convolutions avoid it?
"""
