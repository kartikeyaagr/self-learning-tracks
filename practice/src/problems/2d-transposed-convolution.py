import numpy as np

META = dict(title='Transposed convolution (scatter form)', tags=['Convolutional Networks', 'U-Net'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['multichannel-2d-convolution', 'dilated-2d-convolution', 'bilinear-resize'])

STATEMENT = r"""
A transposed convolution ("deconvolution") is the learnable upsampler of decoders and U-Nets. The easiest way to think about it is as a **scatter**: every input pixel stamps a scaled copy of the kernel into the output, the stamps are placed `stride` pixels apart, and overlapping stamps add up. `padding` then trims that many pixels from each side of the result.

Write `conv_transpose2d(x, w, b=None, stride=1, padding=0, output_padding=0)`.

## Definition
- `x`: float array `(C_in, H, W)`; `w`: float array `(C_in, C_out, kh, kw)` (note: input channels first); `b`: `(C_out,)` or `None`.
- Allocate a zero buffer `Z` of shape `(C_out, (H-1)*s + kh + op, (W-1)*s + kw + op)`, where `s = stride`, `op = output_padding`.
- For every input position `(i, j)`, channel `c` and kernel offset `(a, e)`:

$$Z[o,\; i s + a,\; j s + e] \mathrel{+}= x[c, i, j]\; w[c, o, a, e]$$

- The output is `Z[:, p : p + H_out, p : p + W_out]` plus the bias per output channel, with $p$ = `padding` and

$$H_{out} = (H - 1)s - 2p + k_h + \text{op}$$

(same for $W$). The extra `op` rows/columns therefore come from the bottom/right of the buffer, which receive no stamps beyond the kernel's reach.

## Constraints and output
`0 <= output_padding < stride`, `padding >= 0`, and `H_out, W_out >= 1`. Return a `float64` array `(C_out, H_out, W_out)`.
"""

STARTER = '''import numpy as np

def conv_transpose2d(x, w, b=None, stride=1, padding=0, output_padding=0):
    """x: (C_in, H, W), w: (C_in, C_out, kh, kw). Return (C_out, H_out, W_out)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def conv_transpose2d(x, w, b=None, stride=1, padding=0, output_padding=0):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    C_in, H, W = x.shape
    _, C_out, kh, kw = w.shape
    s, p, op = stride, padding, output_padding
    Z = np.zeros((C_out, (H - 1) * s + kh + op, (W - 1) * s + kw + op))
    # One kernel tap at a time: tap (a, e) of every stamp lands on a stride-s grid offset by (a, e).
    for a in range(kh):
        for e in range(kw):
            contrib = np.einsum('chw,co->ohw', x, w[:, :, a, e])           # (C_out, H, W)
            Z[:, a:a + (H - 1) * s + 1:s, e:e + (W - 1) * s + 1:s] += contrib
    H_out = (H - 1) * s - 2 * p + kh + op
    W_out = (W - 1) * s - 2 * p + kw + op
    y = Z[:, p:p + H_out, p:p + W_out]
    if b is not None:
        y = y + np.asarray(b, dtype=np.float64)[:, None, None]
    return y
'''

HINTS = [
    'Write the four-deep loop over `(i, j, a, e)` first if it helps: it is exactly the scatter in the statement. Then remove the loops over pixels.',
    'Fix one kernel offset `(a, e)`. The input pixels\' contributions for that tap land on the strided grid `Z[:, a::s, e::s]` (cut to `H` by `W` positions), and the channel mixing is a `(C_in, C_out)` matrix product.',
    'For each tap: `Z[:, a:a+(H-1)*s+1:s, e:e+(W-1)*s+1:s] += np.einsum("chw,co->ohw", x, w[:, :, a, e])`. Size the buffer with the `+ op`, then crop `[p:p+H_out]` and add the bias.',
]

_r = np.random.default_rng(23)
TESTS = [
    (np.array([[[1.0, 2.0]]]), np.ones((1, 1, 2, 2)), None, 2),
    (np.array([[[1.0, 2.0], [3.0, 4.0]]]), np.ones((1, 1, 3, 3)), None, 1),
    (np.array([[[1.0, 2.0], [3.0, 4.0]]]), np.ones((1, 1, 3, 3)), np.array([10.0]), 2, 1, 1),
    (_r.normal(size=(2, 3, 4)), _r.normal(size=(2, 3, 4, 4)), _r.normal(size=3), 2, 1),
    (_r.normal(size=(3, 5, 5)), _r.normal(size=(3, 2, 3, 3)), None, 2, 1, 1),
    (_r.normal(size=(1, 4, 3)), _r.normal(size=(1, 2, 2, 3)), _r.normal(size=2), 3, 0, 2),
    (_r.normal(size=(2, 3, 3)), _r.normal(size=(2, 1, 1, 1)), None, 1),
    (_r.normal(size=(2, 6, 4)), _r.normal(size=(2, 2, 5, 5)), np.array([0.5, -0.5]), 1, 2),
]

EXPLANATION = r"""
## Why it is called "transposed"
A convolution with stride $s$ is a linear map $y = A x$ from a big image to a smaller one. The scatter above computes $A^\top x$: each output of the forward conv "sends back" its value along the same weights it used to gather. That's why the backward pass of a strided convolution (with respect to its input) is a transposed convolution, and why `padding` and `output_padding` are defined to make the shapes the exact inverse of a forward conv with the same parameters.

## Why output_padding exists
A forward conv with stride 2 maps both $H = 7$ and $H = 8$ to the same output size, so the inverse shape is ambiguous. `output_padding` picks the larger one by adding rows at the bottom/right.

## Complexity
$O(C_{in} C_{out} k_h k_w H W)$, the same as the forward conv it transposes. The per-tap loop writes to non-overlapping positions within one tap, so the `+=` on a strided slice is safe.

## Common bugs
- Weight layout: transposed convs store `(C_in, C_out, kh, kw)`, the opposite of forward convs.
- Cropping `padding` from only one side, or applying padding to the **input** as a forward conv would.
- Checkerboard artefacts in generated images are not a bug in your code: when $k$ isn't a multiple of $s$, some output pixels receive more stamps than others. Resize-then-convolve avoids them.

## Interview follow-ups
- Verify numerically that this equals the gradient of `conv2d` with respect to its input, using $\langle A x, y\rangle = \langle x, A^\top y \rangle$.
- Implement it instead as "insert $s-1$ zeros between input pixels, pad by $k-1-p$, and run a regular convolution with the flipped, channel-swapped kernel".
"""
