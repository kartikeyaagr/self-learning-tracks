import numpy as np

META = dict(title='SigLIP pairwise sigmoid loss and its gradients', tags=['Embeddings', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['aieng:02.02'],
            related=['symmetric-clip-contrastive-loss', 'sigmoid-focal-loss', 'clip-cosine-retrieval'])

STATEMENT = r"""
SigLIP replaces CLIP's two softmaxes with an independent **binary** decision for every (image, text) pair in the batch: the $B$ matching pairs are positives, the other $B^2 - B$ combinations negatives. No normalisation across the batch is needed, so each pair's term can be computed on its own.

Write `siglip_loss(img, txt, t_prime, bias)` and return a dict.

## Inputs
- `img`, `txt`: float64 `(B, D)`, non-zero rows; row $i$ of each belongs to pair $i$.
- `t_prime`: float, the log-temperature; the scale is $t = e^{t'}$ (no clamp).
- `bias`: float, a learned logit offset $b$.

## Computation
1. L2-normalise rows: $\hat x_i$, $\hat y_j$. Let $c_{ij} = \hat x_i \cdot \hat y_j$.
2. Logits $z_{ij} = t\, c_{ij} + b$; labels $\ell_{ij} = +1$ if $i = j$, else $-1$.
3. $$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B}\sum_{j=1}^{B} \log \sigma(\ell_{ij} z_{ij})$$
   (divided by $B$, not $B^2$). Compute $\log\sigma$ stably: $\log\sigma(u) = -\log(1 + e^{-u})$, e.g. with `np.logaddexp(0, -u)`.

## Output
| key | value |
|---|---|
| `"loss"` | float |
| `"logits"` | float64 `(B, B)` |
| `"grad_logits"` | float64 `(B, B)`, $\partial\mathcal{L}/\partial z_{ij} = -\frac1B\, \ell_{ij}\, \sigma(-\ell_{ij} z_{ij})$ |
| `"grad_bias"` | float, $\partial\mathcal{L}/\partial b$ |
| `"grad_t_prime"` | float, $\partial\mathcal{L}/\partial t'$ |
"""

STARTER = '''import numpy as np

def siglip_loss(img, txt, t_prime, bias):
    """Return a dict: loss, logits, grad_logits, grad_bias, grad_t_prime."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))            # stable for large |u|

def siglip_loss(img, txt, t_prime, bias):
    X = img / np.linalg.norm(img, axis=1, keepdims=True)
    Y = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    cos = X @ Y.T
    t = np.exp(t_prime)
    z = t * cos + bias
    B = len(z)
    labels = 2.0 * np.eye(B) - 1.0                   # +1 on the diagonal, -1 elsewhere
    loss = np.logaddexp(0.0, -labels * z).sum() / B  # -log sigmoid(l z) = log(1 + exp(-l z))
    g = -labels * _sigmoid(-labels * z) / B          # dL/dz for every pair
    return {"loss": float(loss), "logits": z, "grad_logits": g,
            "grad_bias": float(g.sum()),             # dz/db = 1 for every pair
            "grad_t_prime": float((g * t * cos).sum())}   # dz/dt' = t * cos (chain rule through exp)
'''

HINTS = [
    'Build the label matrix as `2 * np.eye(B) - 1`. Then the whole loss is one expression over the `(B, B)` matrix of $\\ell_{ij} z_{ij}$.',
    '$-\\log\\sigma(u) = \\log(1 + e^{-u})$ = `np.logaddexp(0, -u)`, which never overflows. Its derivative in $u$ is $-\\sigma(-u)$.',
    'Chain rule: $\\partial z_{ij}/\\partial b = 1$ and $\\partial z_{ij}/\\partial t\' = e^{t\'} c_{ij}$, so both scalar gradients are sums of `grad_logits` weighted by those factors.',
]

_r = np.random.default_rng(61)
_x = _r.normal(size=(6, 8))
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), 0.0, 0.0),
    (np.array([[2.0, 0.0], [1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 3.0]]), np.log(10.0), -10.0),
    (_x, _x + 0.3 * _r.normal(size=(6, 8)), np.log(10.0), -10.0),
    (_r.normal(size=(5, 4)), _r.normal(size=(5, 4)), 1.0, 0.5),
    (_x, _x + 0.3 * _r.normal(size=(6, 8)), 7.0, -500.0),
    (np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]]), 2.0, 1.0),
    (_r.normal(size=(12, 16)), _r.normal(size=(12, 16)), 2.5, -3.0),
]

EXPLANATION = r"""
## Why a sigmoid instead of a softmax
The softmax in CLIP couples every pair in the batch: each logit's gradient depends on the whole row and column. Across devices that needs the full similarity matrix, and the loss behaves poorly at small batch sizes. With independent sigmoids, each device can compute its block of pairs and only swap embeddings, and the loss works well even with modest batches.

## Why the bias starts at $-10$
With $B$ positives and $B^2 - B$ negatives, almost every pair is a negative. Starting at $b = 0$ makes the model spend its first steps learning "everything is a negative", with huge gradients from the $B^2$ negative terms. Initialising $b = -10$ (and $t = 10$, tests 2 and 3) starts the prior at "unlikely to match", so early training focuses on the positives.

## Numerics
Test 5 has logits around $-500$: $e^{500}$ overflows, but `logaddexp` and the `tanh` form of the sigmoid don't. The loss for a negative pair with logit $-500$ is essentially 0, and for a positive one about 500.

## Complexity
$O(B^2 D)$ for the similarities; the loss and gradients are $O(B^2)$.

## Common bugs
- Dividing by $B^2$ (that rescales the effective learning rate of everything).
- Writing $\log(\sigma(u))$ as `np.log(1 / (1 + np.exp(-u)))`, which overflows for large negative $u$.
- Forgetting the factor $t$ in the temperature gradient: $\partial z/\partial t' = t\,c$, not $c$.

## Interview follow-ups
- How would you compute this loss across $k$ devices without ever materialising the full $B \times B$ matrix on one device?
- Why does SigLIP need the learned bias while CLIP doesn't?
"""
