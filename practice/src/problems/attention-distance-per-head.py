import numpy as np

META = dict(title='How far each attention head looks', tags=['Attention', 'Computer Vision'], diff='Easy', runtime='numpy',
            related=['scaled-dot-product-attention'])

STATEMENT = r"""
For a vision transformer, the **mean attention distance** of a head is the average spatial distance between a query patch and the patches it attends to, weighted by the attention.

Write `attention_distance(attn, grid, patch)`:

- `attn (heads, N, N)` holds attention weights over the $N = $ `grid`$^2$ patch tokens only (no class token); each row sums to 1.
- Patch $i$ sits at row $\lfloor i / g\rfloor$, column $i \bmod g$. Distance between patches is Euclidean in pixels: grid distance times `patch`.
- For each head: $\frac{1}{N}\sum_i \sum_j a_{ij}\, d(i, j)$.

Return a list with one value per head.
"""

STARTER = '''import numpy as np

def attention_distance(attn, grid, patch):
    """Attention-weighted mean distance in pixels, per head."""
    a = np.asarray(attn, float)
    idx = np.arange(grid * grid)
    d = np.abs(idx[:, None] - idx[None, :]) * patch
    return (a * d).sum((1, 2)).tolist()
'''

SOLUTION = '''import numpy as np

def attention_distance(attn, grid, patch):
    a = np.asarray(attn, float)
    idx = np.arange(grid * grid)
    rc = np.stack([idx // grid, idx % grid], 1).astype(float)
    d = np.linalg.norm(rc[:, None] - rc[None, :], axis=2) * patch
    return ((a * d).sum(2).mean(1)).tolist()
'''

HINTS = [
    'Turn token indices into (row, column) positions first; index differences are not distances on a 2-D grid.',
    'Sum over keys with the attention weights, then average over queries.',
]

def _local(g, w):
    idx = np.arange(g * g); rc = np.stack([idx // g, idx % g], 1)
    D = np.linalg.norm(rc[:, None] - rc[None, :], axis=2)
    A = np.exp(-w * D)
    return A / A.sum(1, keepdims=True)

TESTS = [
    (np.eye(4)[None], 2, 16),
    (np.full((1, 4, 4), 0.25), 2, 16),
    (np.stack([_local(4, 3.0), _local(4, 0.5), np.full((16, 16), 1 / 16)]), 4, 16),
    (np.stack([_local(3, 10.0)]), 3, 8),
    (np.random.default_rng(1).dirichlet(np.ones(9), size=(2, 9)), 3, 14),
]

EXPLANATION = r"""
## What it reveals
Dosovitskiy et al. (2020) plotted this for every head of a trained ViT. In early layers some heads attend locally (small distance, like a convolution) and others attend across the whole image; deeper layers become almost entirely global. A convolution can only grow its receptive field slowly with depth, while a ViT head can look anywhere from the first layer.

## Reading the numbers
A head attending only to itself has distance 0. Uniform attention over a $g \times g$ grid gives the average distance between two random cells, about $0.52\,g$ patches for large $g$. Values in between show how local a head is.

## Interview follow-ups
- Why do ViTs need more data than CNNs to train well from scratch, and how does this measurement relate to that?
"""
