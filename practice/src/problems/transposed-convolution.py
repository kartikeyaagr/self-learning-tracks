import numpy as np

META = dict(title='Transposed convolution (learned upsampling)', tags=['Convolutional Networks', 'U-Net'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:04.02'], related=['2d-convolution', 'u-net-skip-connection'])

STATEMENT = r"""
A transposed convolution runs a convolution "backwards": every **input** pixel stamps a scaled copy of the kernel onto the output, and overlapping stamps add. With stride 2 it doubles the resolution, which is how decoders and U-Nets upsample.

Write `conv_transpose2d(x, W, b, stride=1, padding=0, output_padding=0)`.

## Shapes (PyTorch layout)
- `x`: `(N, C_in, H, W_in)`; `W`: `(C_in, C_out, k_h, k_w)`; note the input channels come **first**; `b`: `(C_out,)`.

## Definition
1. Start a zero buffer $Z$ of shape `(N, C_out, (H-1)s + k_h + op, (W_in-1)s + k_w + op)` where $s$ = `stride`, $op$ = `output_padding`.
2. For every input position $(i, j)$: $Z[n, :, is : is + k_h,\; js : js + k_w] \mathrel{+}= \sum_c x_{n,c,i,j}\, W_{c,:,:,:}$.
3. Crop $p$ = `padding` rows and columns from the **start** of each spatial axis and keep
$$H_\text{out} = (H - 1)s - 2p + k_h + op$$
rows (likewise columns), then add the bias to every pixel.

This matches `torch.nn.functional.conv_transpose2d` with dilation 1. In the tests $op < s$ and $H_\text{out}, W_\text{out} \ge 1$.

## Output
Float array `(N, C_out, H_out, W_out)`.
"""

STARTER = '''import numpy as np

def conv_transpose2d(x, W, b, stride=1, padding=0, output_padding=0):
    """x: (N, Cin, H, W); W: (Cin, Cout, kh, kw); b: (Cout,). Return (N, Cout, H_out, W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def conv_transpose2d(x, W, b, stride=1, padding=0, output_padding=0):
    x, W = np.asarray(x, dtype=float), np.asarray(W, dtype=float)
    N, Cin, H, Wd = x.shape
    _, Cout, kh, kw = W.shape
    s, p, op = stride, padding, output_padding
    Z = np.zeros((N, Cout, (H - 1) * s + kh + op, (Wd - 1) * s + kw + op))
    for u in range(kh):
        for v in range(kw):
            # tap (u, v) of every input pixel lands on a strided grid starting at (u, v)
            Z[:, :, u: u + s * (H - 1) + 1: s, v: v + s * (Wd - 1) + 1: s] += np.einsum('nchw,co->nohw', x, W[:, :, u, v])
    Ho, Wo = (H - 1) * s - 2 * p + kh + op, (Wd - 1) * s - 2 * p + kw + op
    return Z[:, :, p: p + Ho, p: p + Wo] + np.asarray(b, dtype=float)[None, :, None, None]
'''

HINTS = [
    'Think scatter, not gather: input pixel $(i, j)$ adds `x[n, :, i, j]`-weighted kernels into the output block starting at $(is, js)$.',
    'As with an ordinary convolution, loop over the kernel taps $(u, v)$ instead: tap $(u, v)$ of every input pixel lands at output $(is + u, js + v)$, a strided slice `Z[:, :, u : u + s*(H-1) + 1 : s, ...]`.',
    'Contract channels with `einsum("nchw,co->nohw", x, W[:, :, u, v])` and add into that slice. Allocate the buffer with the extra `output_padding` rows, then crop `[p : p + H_out]`.',
]

_r = np.random.default_rng(37)
TESTS = [
    (np.array([[[[1.0, 2.0]]]]), np.array([[[[1.0, 1.0], [1.0, 1.0]]]]), np.array([0.0])),
    (np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), np.ones((1, 1, 2, 2)), np.zeros(1), 2),
    (np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), np.ones((1, 1, 3, 3)), np.zeros(1), 2, 1),
    (_r.normal(size=(2, 3, 4, 4)), _r.normal(size=(3, 2, 2, 2)), _r.normal(size=2), 2),
    (_r.normal(size=(1, 2, 3, 5)), _r.normal(size=(2, 4, 3, 3)), _r.normal(size=4), 2, 1, 1),
    (_r.normal(size=(2, 2, 4, 3)), _r.normal(size=(2, 3, 3, 2)), _r.normal(size=3), 1, 1),
    (_r.normal(size=(1, 3, 3, 3)), _r.normal(size=(3, 2, 4, 4)), _r.normal(size=2), 3, 1, 2),
    (_r.normal(size=(1, 1, 1, 1)), _r.normal(size=(1, 2, 3, 3)), _r.normal(size=2)),
]

EXPLANATION = r"""
## Why "transposed"
Flatten the input and output. An ordinary convolution is a sparse matrix $C$ with $y = Cx$. The layer you just wrote computes $C^\top z$ for a convolution $C$ with the same kernel, stride and padding: that's why it is also the **gradient** of a convolution with respect to its input, and why its output size formula is the conv formula solved for the input size.

## Checkerboards
With $k = 2$, $s = 2$ (test 4) stamps tile the output with no overlap. With $k = 3$, $s = 2$ alternate rows receive one or two contributions (test 3: the middle row and column sum two inputs), which produces the familiar checkerboard artefacts in generated images. Common fixes: choose $k$ divisible by $s$, or use nearest/bilinear upsampling followed by an ordinary conv.

## output_padding
Several input sizes map to the same conv output size because of the floor. `output_padding` picks which one the transposed conv returns, by adding extra rows at the bottom/right.

## Complexity
$O(N C_\text{in} C_\text{out} k_h k_w H W)$, the same as the forward conv it transposes.

## Common bugs
- Using the conv weight layout `(C_out, C_in, ...)` instead of `(C_in, C_out, ...)`.
- Cropping `padding` from only one side, or adding `output_padding` to both sides.
- Implementing it as "zero-insert then conv" but forgetting to flip the kernel; the two views agree only with the flip.

## Interview follow-ups
- Show that conv_transpose2d is the adjoint: $\langle \mathrm{conv}(x), z\rangle = \langle x, \mathrm{conv}^\top(z)\rangle$ for zero biases.
- When would you choose a transposed convolution over upsample-plus-conv in a decoder?
"""
