import numpy as np

META = dict(title='Capstone: train and evaluate a tiny CLIP', tags=['Embeddings', 'Training'], diff='Hard', runtime='numpy',
            related=['cosine-similarity-matrix', 'symmetric-clip-contrastive-loss', 'clip-cosine-retrieval'])

STATEMENT = r"""
Train the projection heads of a tiny CLIP for one step and measure whether retrieval improved.

Image features `Xi (n, a)` and text features `Xt (n, b)` come from frozen encoders; row `i` of each describes the same pair. The trainable heads are `Wi (a, d)` and `Wt (b, d)`.

Write `clip_step(Xi, Xt, Wi, Wt, tau, lr)`:

1. $u = \text{normalise}(X_i W_i)$, $v = \text{normalise}(X_t W_t)$ (unit rows). Logits $Z = u v^\top / \tau$.
2. Loss $L = \tfrac12(\text{CE}_{rows}(Z) + \text{CE}_{cols}(Z))$, each averaged over the `n` pairs, with the diagonal as the targets.
3. Backpropagate through the normalisation: if $z = y / \lVert y \rVert$ then $\partial L/\partial y = (g - z\,(z\cdot g)) / \lVert y \rVert$ with $g = \partial L/\partial z$.
4. One SGD step on `Wi` and `Wt`.

Return `(loss_before, loss_after, r1_before, r1_after)`, where `r1` is image-to-text recall@1: the fraction of rows of $Z$ whose maximum is on the diagonal (ties go to the lowest index, as `argmax` does).
"""

STARTER = '''import numpy as np

def clip_step(Xi, Xt, Wi, Wt, tau, lr):
    """One SGD step on CLIP projection heads, with loss and recall@1 before and after."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _norm(Y):
    r = np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / r, r

def _eval(Xi, Xt, Wi, Wt, tau):
    u, ru = _norm(Xi @ Wi); v, rv = _norm(Xt @ Wt)
    Z = u @ v.T / tau
    n = len(Z)
    Pr = np.exp(Z - Z.max(1, keepdims=True)); Pr /= Pr.sum(1, keepdims=True)
    Pc = np.exp(Z - Z.max(0, keepdims=True)); Pc /= Pc.sum(0, keepdims=True)
    d = np.arange(n)
    loss = float(0.5 * (-np.log(Pr[d, d]).mean() - np.log(Pc[d, d]).mean()))
    r1 = float((Z.argmax(1) == d).mean())
    return loss, r1, u, ru, v, rv, Pr, Pc

def clip_step(Xi, Xt, Wi, Wt, tau, lr):
    Xi, Xt, Wi, Wt = (np.asarray(a, float) for a in (Xi, Xt, Wi, Wt))
    loss0, r10, u, ru, v, rv, Pr, Pc = _eval(Xi, Xt, Wi, Wt, tau)
    n = len(u)
    dZ = 0.5 * ((Pr - np.eye(n)) + (Pc - np.eye(n))) / n
    gu = dZ @ v / tau
    gv = dZ.T @ u / tau
    dyu = (gu - u * (u * gu).sum(1, keepdims=True)) / ru
    dyv = (gv - v * (v * gv).sum(1, keepdims=True)) / rv
    Wi2 = Wi - lr * Xi.T @ dyu
    Wt2 = Wt - lr * Xt.T @ dyv
    loss1, r11 = _eval(Xi, Xt, Wi2, Wt2, tau)[:2]
    return loss0, loss1, r10, r11
'''

HINTS = [
    'The gradient of the symmetric loss with respect to Z is half of (row softmax minus identity) plus half of (column softmax minus identity), divided by n.',
    'Gradient through normalisation removes the component along z: only changes in direction matter, not in length.',
    'Check the gradient on a tiny case with finite differences before trusting the full step.',
]

def _case(seed, n, a, b, d, tau, lr):
    g = np.random.default_rng(seed)
    S = g.normal(size=(n, 3))
    Xi = S @ g.normal(size=(3, a)) + 0.3 * g.normal(size=(n, a))
    Xt = S @ g.normal(size=(3, b)) + 0.3 * g.normal(size=(n, b))
    return (Xi, Xt, g.normal(size=(a, d)) / np.sqrt(a), g.normal(size=(b, d)) / np.sqrt(b), tau, lr)

TESTS = [
    _case(1, 6, 5, 4, 3, 0.5, 0.5),
    _case(2, 8, 6, 6, 4, 0.1, 0.05),
    _case(3, 16, 10, 8, 6, 0.2, 0.5),
    _case(4, 4, 3, 3, 2, 1.0, 1.0),
    ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], 0.5, 0.5),
]

EXPLANATION = r"""
## What was built
The steps gave a scaled similarity matrix, the symmetric contrastive loss, recall@K and zero-shot classification. This capstone joins them: the loss trains the heads, and recall@1 on the same pairs checks that the step helped.

## The gradient
Every image competes against every caption and every caption against every image, so $\partial L/\partial Z$ is a dense $n \times n$ matrix. Each pair pushes its own entry up and pulls the others down, weighted by how much probability they currently take. That is why CLIP needs large batches: more negatives give a sharper signal.

## Normalisation
CLIP compares directions only. The projection through $(I - zz^\top)/\lVert y\rVert$ removes any change in length, so the heads cannot reduce the loss by growing their outputs. The temperature does that job instead, and real CLIP learns $\log(1/\tau)$ with a cap at 100.

## The last test
The text head starts with its axes swapped, so recall@1 is 0. A single step is not enough to rotate it the right way, which is a reminder that a falling loss can precede any change in the ranking metric.

## Interview follow-ups
- Recall@1 measured on the training pairs is optimistic. How would you set up a held-out retrieval evaluation, and what makes it hard to keep the test captions out of a web-scale training set?
"""
