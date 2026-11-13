import numpy as np

META = dict(title='Symmetric CLIP contrastive loss with a learned temperature', tags=['Embeddings', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['aieng:02.02'],
            related=['clip-cosine-retrieval', 'siglip-pairwise-loss', 'in-batch-softmax-logq'])

STATEMENT = r"""
CLIP trains an image encoder and a text encoder together on a batch of $B$ matching (image, caption) pairs. Every image should be most similar to its own caption among the $B$ captions, and every caption to its own image. That is two softmax classification problems over the same similarity matrix.

Write `clip_loss(img, txt, log_scale)` and return a dict.

## Inputs
- `img`, `txt`: float64 `(B, D)` embeddings; row $i$ of each belongs to pair $i$. Rows are non-zero.
- `log_scale`: float, the learned log-temperature. The logit scale is $s = \min(e^{\text{log\_scale}}, 100)$.

## Computation
1. L2-normalise each row of `img` and `txt`.
2. `logits` $= s \cdot \hat I \hat T^\top$, shape `(B, B)`; entry $(i, j)$ scores image $i$ against caption $j$.
3. Image-to-text loss: cross-entropy of each **row** against target $i$. Text-to-image loss: cross-entropy of each **column** against target $j$. Each is averaged over the batch.
4. `loss` $= \tfrac12(\mathcal{L}_{i\to t} + \mathcal{L}_{t\to i})$.

Use a numerically stable log-softmax.

## Output
| key | value |
|---|---|
| `"loss"` | float |
| `"logits"` | float64 `(B, B)` |
| `"grad_logits"` | float64 `(B, B)`, $\partial\,\text{loss} / \partial\,\text{logits}$ |
| `"i2t_acc"`, `"t2i_acc"` | floats, fraction of rows (columns) whose argmax is the diagonal entry (on an exact tie, the lowest index wins) |
"""

STARTER = '''import numpy as np

def clip_loss(img, txt, log_scale):
    """Return a dict: loss, logits (B, B), grad_logits (B, B), i2t_acc, t2i_acc."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _log_softmax(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))

def clip_loss(img, txt, log_scale):
    I = img / np.linalg.norm(img, axis=1, keepdims=True)
    T = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    s = min(np.exp(log_scale), 100.0)                 # clamp keeps the softmax from collapsing early in training
    logits = s * I @ T.T
    B = len(logits)
    lp_rows = _log_softmax(logits, axis=1)            # image -> which caption
    lp_cols = _log_softmax(logits, axis=0)            # caption -> which image
    loss = -0.5 * (np.trace(lp_rows) + np.trace(lp_cols)) / B
    eye = np.eye(B)
    # d CE / d logits = softmax - onehot, for each direction, each averaged over B and halved.
    grad = ((np.exp(lp_rows) - eye) + (np.exp(lp_cols) - eye)) / (2 * B)
    ar = np.arange(B)
    return {"loss": float(loss), "logits": logits, "grad_logits": grad,
            "i2t_acc": float(np.mean(logits.argmax(axis=1) == ar)),
            "t2i_acc": float(np.mean(logits.argmax(axis=0) == ar))}
'''

HINTS = [
    'Both losses use the same `(B, B)` matrix: image-to-text is a softmax over each **row**, text-to-image a softmax over each **column**. The targets are the diagonal in both cases.',
    'With a stable log-softmax along `axis=1` and `axis=0`, each loss is minus the mean of the diagonal: `-np.trace(logp) / B`.',
    'The gradient of softmax cross-entropy is `softmax - onehot`. Add the row-softmax and column-softmax versions, each minus the identity, and divide by $2B$.',
]

_r = np.random.default_rng(51)
_x = _r.normal(size=(6, 8))
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.0),
    (np.array([[3.0, 0.0], [0.0, 2.0]]), np.array([[0.0, 1.0], [5.0, 0.0]]), np.log(10.0)),
    (_x, _x + 0.3 * _r.normal(size=(6, 8)), np.log(1 / 0.07)),
    (_r.normal(size=(5, 4)), _r.normal(size=(5, 4)), 2.0),
    (_x, _x + 0.3 * _r.normal(size=(6, 8)), 10.0),
    (np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 1.0),
    (_r.normal(size=(16, 32)), _r.normal(size=(16, 32)), np.log(50.0)),
]

EXPLANATION = r"""
## Why two directions
Image-to-text alone would let the caption embeddings collapse toward a few hubs that every image likes; adding text-to-image forces every caption to also pick out its own image. Both are classification over in-batch negatives: the other $B - 1$ items in the batch serve as wrong answers, which is why CLIP batches are huge (32k). A bigger batch means harder negatives and a tighter bound on the mutual information.

## The temperature
Cosine similarities live in $[-1, 1]$, far too narrow a range for a sharp softmax. The learned scale $s$ stretches them; CLIP initialises it at $1/0.07 \approx 14.3$ (test 3) and clamps it at 100 (test 5 asks for $e^{10}$ and must use 100). Without the clamp the scale can run away and destabilise training.

## Gradient
For one row, $\partial\,\mathrm{CE} / \partial z_j = \operatorname{softmax}(z)_j - [j = i]$. Each logit appears in one row loss and one column loss, each weighted $\frac{1}{2B}$. The gradient's rows and columns each sum to zero.

## Complexity
$O(B^2 D)$ for the similarity matrix, which dominates.

## Common bugs
- Forgetting to normalise, so the loss can be lowered by inflating norms instead of aligning directions.
- Using the same softmax axis for both directions.
- Averaging the two losses with weights 1 instead of $\frac12$.

## Interview follow-ups
- Large-batch CLIP training shards the batch over many GPUs. What has to be all-gathered to compute this loss exactly, and what is the communication cost?
- How does SigLIP's pairwise sigmoid loss avoid the global softmax normalisation, and why does that help at small batch sizes?
"""
