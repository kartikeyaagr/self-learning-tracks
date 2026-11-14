import numpy as np

META = dict(title='Siamese contrastive loss with a margin', tags=['Loss Functions', 'Embeddings'], diff='Medium', runtime='numpy',
            related=['triplet-loss', 'info-nce-loss', 'cosine-similarity'])

STATEMENT = r"""
A Siamese network embeds two inputs with shared weights; the loss pulls similar pairs together and pushes dissimilar pairs at least a margin apart. Write `contrastive_loss(z1, z2, same, margin)`.

- `z1`, `z2`: `(n, d)` embeddings of the two sides of each pair; `same`: array of 0/1 (1 = similar pair).
- Distance $d_i = \lVert z_{1,i} - z_{2,i} \rVert_2$.
- Per-pair loss: $\ell_i = s_i \, d_i^2 + (1 - s_i)\max(0, m - d_i)^2$.
- Return `(loss, grad_z1)` where `loss` is the mean of $\ell_i$ and `grad_z1` is $\partial\,\text{loss} / \partial z_1$ (shape `(n, d)`).

For the gradient of the dissimilar term when $d_i = 0$, use 0.
"""

STARTER = '''import numpy as np

def contrastive_loss(z1, z2, same, margin):
    """Return (mean_loss, gradient_wrt_z1)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def contrastive_loss(z1, z2, same, margin):
    z1, z2, s = np.asarray(z1, dtype=float), np.asarray(z2, dtype=float), np.asarray(same, dtype=float)
    n = len(s)
    diff = z1 - z2
    d = np.sqrt((diff ** 2).sum(axis=1))
    hinge = np.maximum(0.0, margin - d)
    loss = (s * d ** 2 + (1 - s) * hinge ** 2).mean()
    unit = np.divide(diff, d[:, None], out=np.zeros_like(diff), where=d[:, None] > 0)
    g = s[:, None] * 2 * diff - (1 - s)[:, None] * 2 * hinge[:, None] * unit
    return float(loss), g / n
'''

HINTS = [
    'For similar pairs, d^2 = sum of squared differences, whose gradient with respect to z1 is 2 (z1 - z2).',
    'For dissimilar pairs inside the margin, the loss is (m - d)^2; by the chain rule its gradient is -2 (m - d) times the unit vector (z1 - z2) / d.',
    'Divide the gradient by n because the loss is a mean.',
]

_r = np.random.default_rng(14)
TESTS = [
    (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[3.0, 4.0], [1.0, 0.5]]), np.array([1, 0]), 1.0),
    (np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([0]), 2.0),
    (_r.normal(size=(6, 4)), _r.normal(size=(6, 4)), np.array([1, 0, 1, 0, 0, 1]), 3.0),
    (_r.normal(size=(5, 3)), _r.normal(size=(5, 3)), np.array([0, 0, 0, 0, 0]), 0.5),
    (np.array([[1.0, 1.0]]), np.array([[1.0, 2.0]]), np.array([1]), 1.0),
]

EXPLANATION = r"""
## What the margin does
Similar pairs are always pulled together. Dissimilar pairs are pushed apart only while they are closer than the margin $m$; beyond it they contribute nothing, so the model does not waste effort separating pairs that are already clearly different.

## Collapse
Without the dissimilar term, mapping every input to the same point gives zero loss. The margin term prevents this by charging for dissimilar pairs that sit too close.

## Choosing pairs
Most random dissimilar pairs are already far apart and give zero gradient. Mining hard negatives (dissimilar pairs that are close) speeds up learning, but very hard negatives can be label noise.

## Interview follow-ups
- How does this compare with triplet loss and with InfoNCE, which compare against many negatives at once?
"""
