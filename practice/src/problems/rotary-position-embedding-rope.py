import numpy as np

META = dict(title='Rotary position embeddings, both layouts', tags=['Transformers', 'Attention', 'Embeddings'], diff='Medium',
            runtime='numpy', lessons=['aieng:04.01', 'ieng:13.06'],
            related=['sinusoidal-positional-encoding', 'scaled-dot-product-attention', 'multi-head-attention'])

STATEMENT = r"""
RoPE encodes position by **rotating** pairs of query/key features by an angle proportional to the token's position. Pair $i$ ($i = 0, \dots, d/2 - 1$) of a token at position $m$ is rotated by $m\,\theta_i$ with

$$\theta_i = \text{base}^{-2i/d}, \qquad \begin{pmatrix} u' \\ v' \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}\begin{pmatrix} u \\ v \end{pmatrix}.$$

Implementations disagree on **which** features form pair $i$:

| `layout` | pair $i$ is $(u, v) =$ |
|---|---|
| `'interleaved'` (original paper, GPT-J) | features $(2i,\; 2i + 1)$ |
| `'half'` (GPT-NeoX, Llama in Hugging Face) | features $(i,\; i + d/2)$ |

Write `rope(x, positions, base=10000.0, layout='interleaved')`.

## Shapes
- `x`: float array `(..., T, d)` with $d$ even: the last two axes are sequence and features (leading axes such as batch and heads are left alone).
- `positions`: int array `(T,)`, the position $m$ of each token (not necessarily $0..T-1$, e.g. for a KV cache).

## Output
Float array shaped like `x`.
"""

STARTER = '''import numpy as np

def rope(x, positions, base=10000.0, layout='interleaved'):
    """x: (..., T, d), d even; positions: (T,). Return rotated x."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rope(x, positions, base=10000.0, layout='interleaved'):
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)                       # (d/2,): exponent 2i/d
    ang = np.asarray(positions, dtype=float)[:, None] * theta        # (T, d/2)
    cos, sin = np.cos(ang), np.sin(ang)
    if layout == 'interleaved':
        u, v = x[..., 0::2], x[..., 1::2]
    else:
        u, v = x[..., : d // 2], x[..., d // 2:]
    ru, rv = u * cos - v * sin, u * sin + v * cos                    # broadcast over leading axes
    out = np.empty_like(x)
    if layout == 'interleaved':
        out[..., 0::2], out[..., 1::2] = ru, rv
    else:
        out[..., : d // 2], out[..., d // 2:] = ru, rv
    return out
'''

HINTS = [
    'Build the angle table once: `theta = base ** (-np.arange(0, d, 2) / d)` and `ang = positions[:, None] * theta`, shape `(T, d/2)`. It broadcasts against `(..., T, d/2)`.',
    'Pick the two halves of each pair by slicing: `x[..., 0::2]` and `x[..., 1::2]` for interleaved, `x[..., :d//2]` and `x[..., d//2:]` for half.',
    'Rotate with $u\' = u\\cos - v\\sin$, $v\' = u\\sin + v\\cos$ and write the results back into the same slots they came from.',
]

_r = np.random.default_rng(83)
TESTS = [
    (np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([0, 1]), 10000.0),
    (np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([2]), 100.0, 'half'),
    (_r.normal(size=(5, 8)), np.arange(5)),
    (_r.normal(size=(5, 8)), np.arange(5), 10000.0, 'half'),
    (_r.normal(size=(2, 3, 4, 6)), np.array([7, 8, 9, 10]), 500.0),
    (_r.normal(size=(2, 3, 4, 6)), np.array([7, 8, 9, 10]), 500.0, 'half'),
    (_r.normal(size=(3, 4)), np.array([0, 0, 0])),
    (_r.normal(size=(1, 2)), np.array([100000])),
]

EXPLANATION = r"""
## Why rotations encode relative position
Rotations compose: $R(m\theta)^\top R(n\theta) = R((n - m)\theta)$. So for a query at position $m$ and a key at position $n$,
$$\langle R_m q, R_n k\rangle = \langle q, R_{n-m} k\rangle,$$
the attention score depends only on the **offset** $n - m$, while the vectors themselves carry absolute positions. Unlike additive sinusoidal embeddings, nothing is added to the token content, and the rotation preserves norms.

## Frequencies
$\theta_0 = 1$ rotates fast (a full turn every $2\pi$ tokens), $\theta_{d/2-1} \approx 1/\text{base}$ rotates very slowly. Context-extension tricks (position interpolation, NTK scaling, YaRN) all work by rescaling positions or the base so that long offsets map into the range of angles seen in training.

## Layouts
Both layouts are the same operation on a permuted feature order. They are not interchangeable at load time: weights trained with one must be permuted (the Q/K projection rows) to work with the other, which is a real source of bugs when converting checkpoints.

## Complexity
$O(T d)$ per head, applied to queries and keys only (not values).

## Common bugs
- Using $\theta_i = \text{base}^{-i/d}$ (missing the factor 2).
- Rotating with the token index $0..T-1$ instead of the given positions during cached decoding.
- Applying RoPE to the values.

## Interview follow-ups
- Prove the relative-position property above from the rotation matrices.
- Why does RoPE make attention scores decay (on average) with distance?
"""
