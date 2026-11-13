import numpy as np

META = dict(title='2-D sine-cosine position embeddings', tags=['Transformers', 'Computer Vision'], diff='Medium', runtime='numpy',
            lessons=['aieng:04.01'], related=['sinusoidal-positional-encoding', 'vit-patchify'])

STATEMENT = r"""
A fixed alternative to learned ViT position embeddings: encode a patch's **row** with half of the channels and its **column** with the other half, each half using 1-D sine-cosine features at geometrically spaced frequencies.

Write `sincos_2d(h, w, dim, temperature=10000.0, cls_token=False)`.

## Frequencies
`dim` is a multiple of 4. Let `q = dim // 4` and

$$\omega_k = \frac{1}{\text{temperature}^{\,k/q}}, \qquad k = 0, \dots, q - 1$$

## Embedding of grid position (i, j)
Row `i` (`0 <= i < h`) and column `j` (`0 <= j < w`) give the length-`dim` vector made of four blocks of `q` values, in this order:

$$\big[\sin(i\,\omega_k)\big]_k,\;\big[\cos(i\,\omega_k)\big]_k,\;\big[\sin(j\,\omega_k)\big]_k,\;\big[\cos(j\,\omega_k)\big]_k$$

## Output
A `float64` array `(h * w, dim)` with positions in **row-major** order (position `(i, j)` is row `i * w + j`). If `cls_token` is `True`, prepend one row of zeros, giving `(h * w + 1, dim)`.
"""

STARTER = '''import numpy as np

def sincos_2d(h, w, dim, temperature=10000.0, cls_token=False):
    """Return (h*w [+1], dim): [sin(i w), cos(i w), sin(j w), cos(j w)] per row-major grid position."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sincos_1d(pos, omega):
    angles = pos[:, None] * omega[None, :]                 # (P, q)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

def sincos_2d(h, w, dim, temperature=10000.0, cls_token=False):
    q = dim // 4
    omega = 1.0 / temperature ** (np.arange(q) / q)
    ii, jj = np.meshgrid(np.arange(h, dtype=float), np.arange(w, dtype=float), indexing='ij')
    emb = np.concatenate([_sincos_1d(ii.ravel(), omega), _sincos_1d(jj.ravel(), omega)], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros((1, dim)), emb], axis=0)
    return emb
'''

HINTS = [
    'Write a 1-D helper that maps a vector of positions to `[sin(pos * omega), cos(pos * omega)]`, shape `(P, 2q)`. The 2-D embedding is that helper applied to row indices and to column indices, side by side.',
    'Row-major order means you need, for every token, its row index `i` and column index `j`: `np.meshgrid(range(h), range(w), indexing="ij")` then `.ravel()` both.',
    '`omega = 1 / temperature ** (np.arange(q) / q)`; concatenate `helper(i)` and `helper(j)` along axis 1; optionally stack a zero row on top.',
]

TESTS = [
    (1, 2, 4),
    (2, 2, 8, 100.0),
    (3, 4, 16),
    (2, 3, 12, 10000.0, True),
    (14, 14, 32),
    (1, 1, 4, 10000.0, True),
    (5, 2, 8, 50.0),
]

EXPLANATION = r"""
## Why split the channels
A 2-D position has two independent coordinates. Giving each its own block of channels makes the embedding of $(i, j)$ the concatenation of a row code and a column code, so dot products between embeddings decompose into a row-similarity plus a column-similarity. Within each block, $\sin(i\omega_k)$ and $\cos(i\omega_k)$ form a rotation, so a fixed offset in $i$ corresponds to a fixed linear transform of the embedding: attention can learn relative positions.

## Frequencies
$\omega_k$ ranges from 1 down to nearly $1/\text{temperature}$. High frequencies tell neighbours apart; low frequencies vary slowly and encode coarse location without repeating over the grid.

## Fixed vs learned
Sine-cosine embeddings need no parameters, extend to any grid size, and work about as well as learned ones for ViTs at standard resolutions; MAE uses them for this reason. The class token gets a zero vector because it has no spatial position.

## Complexity
$O(hw \cdot \text{dim})$ time and memory.

## Common bugs
- Interleaving sin and cos (`sin, cos, sin, cos, ...`) when the convention is blocks, or vice versa. Both are valid; the order just has to match the weights.
- Using column-major token order, which transposes the grid relative to the patch tokens.
- Computing `temperature ** (2k / dim)` from the 1-D formula: in 2-D each block has only `dim / 4` frequencies.

## Interview follow-ups
- How does this differ from 2-D RoPE? (RoPE rotates the query and key vectors instead of adding to the input, making attention scores depend only on relative offsets.)
- Show that the dot product of two embeddings depends only on $i - i'$ and $j - j'$.
"""
