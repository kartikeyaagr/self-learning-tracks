import numpy as np

META = dict(title='GloVe: co-occurrence counts and weighted least squares', tags=['Embeddings', 'NLP'], diff='Hard', runtime='numpy',
            lessons=['aieng:02.01'], related=['word2vec', 'low-rank-approximation', 'vanilla-gradient-descent'])

STATEMENT = r"""
GloVe learns word vectors by fitting dot products to the logarithm of co-occurrence counts. Write `glove(tokens, V, window, W, Wt, b, bt, lr, steps, x_max=100.0, alpha=0.75)`.

## 1. Co-occurrence matrix
$X$ is a float array `(V, V)` of zeros. For every pair of positions $i \ne j$ in `tokens` with $|i - j| \le$ `window`, add $1 / |i - j|$ to `X[tokens[i], tokens[j]]`. (Each unordered pair of positions is visited in both directions, so $X$ is symmetric.)

## 2. Objective
Over the pairs $(i, j)$ with $X_{ij} > 0$:
$$J = \sum_{X_{ij} > 0} f(X_{ij}) \big( w_i \cdot \tilde w_j + b_i + \tilde b_j - \ln X_{ij} \big)^2, \qquad f(x) = \min\!\Big(1, \big(x / x_{\max}\big)^{\alpha}\Big)$$
where $w_i$ is row $i$ of `W` `(V, d)`, $\tilde w_j$ is row $j$ of `Wt` `(V, d)`, and `b`, `bt` are bias vectors `(V,)`.

## 3. Training
Run `steps` iterations of full-batch gradient descent on $J$: compute all four gradients at the current parameters, then update every parameter with `p -= lr * grad`.

## Output
`(X, losses, vectors)`:
- `X`: the co-occurrence matrix.
- `losses`: a list of `steps + 1` floats, $J$ before each step and once more after the last step.
- `vectors`: the final `W + Wt`, the usual word representation.

Do not modify the input arrays.
"""

STARTER = '''import numpy as np

def glove(tokens, V, window, W, Wt, b, bt, lr, steps, x_max=100.0, alpha=0.75):
    """Return (X, losses list of length steps+1, final W + Wt)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def glove(tokens, V, window, W, Wt, b, bt, lr, steps, x_max=100.0, alpha=0.75):
    W, Wt, b, bt = (np.array(a, dtype=float) for a in (W, Wt, b, bt))   # np.array copies
    X = np.zeros((V, V))
    for i in range(len(tokens)):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j != i:
                X[tokens[i], tokens[j]] += 1.0 / abs(i - j)
    nz = X > 0
    logX = np.log(np.where(nz, X, 1.0))                      # log only where defined
    f = np.where(nz, np.minimum(1.0, (X / x_max) ** alpha), 0.0)   # weight 0 drops the X == 0 terms

    def residual():
        return W @ Wt.T + b[:, None] + bt[None, :] - logX

    losses = []
    for _ in range(steps):
        r = residual()
        losses.append(float(np.sum(f * r * r)))
        g = 2.0 * f * r                                      # dJ/d(prediction_ij)
        gW, gWt = g @ Wt, g.T @ W
        gb, gbt = g.sum(axis=1), g.sum(axis=0)
        W -= lr * gW; Wt -= lr * gWt; b -= lr * gb; bt -= lr * gbt
    r = residual()
    losses.append(float(np.sum(f * r * r)))
    return X, losses, W + Wt
'''

HINTS = [
    'Build $X$ with two loops over positions. Then precompute, once, a weight matrix `f` that is 0 where $X_{ij} = 0$, and `log X` with a safe placeholder where $X_{ij} = 0$.',
    'With weights zero outside the observed pairs, the whole objective is dense matrix algebra: residual `R = W @ Wt.T + b[:, None] + bt[None, :] - logX`, loss `sum(f * R**2)`.',
    'Let `G = 2 * f * R`. Then the gradients are `G @ Wt` for `W`, `G.T @ W` for `Wt`, `G.sum(1)` for `b` and `G.sum(0)` for `bt`. Compute all four before updating any parameter.',
]

_r = np.random.default_rng(13)
_V, _d = 8, 3
_toks = _r.integers(0, _V, size=40).tolist()
_init = lambda s: (_r.normal(scale=s, size=(_V, _d)), _r.normal(scale=s, size=(_V, _d)), np.zeros(_V), np.zeros(_V))
TESTS = [
    ([0, 1, 0], 2, 1, np.array([[0.1], [0.2]]), np.array([[0.3], [-0.1]]), np.zeros(2), np.zeros(2), 0.1, 2),
    ([0, 1, 2, 1], 3, 2, np.eye(3)[:, :2], np.eye(3)[:, :2], np.zeros(3), np.zeros(3), 0.05, 3, 2.0, 1.0),
    ([4], 5, 3, np.ones((5, 2)), np.ones((5, 2)), np.zeros(5), np.zeros(5), 0.1, 2),
    ([1, 1, 1, 1], 2, 2, np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2), np.ones(2), 0.1, 0),
    (_toks, _V, 3, *_init(0.1), 0.05, 25),
    (_toks, _V, 5, *_init(0.5), 0.02, 15, 3.0, 0.5),
]

EXPLANATION = r"""
## Why it works
If $w_i \cdot \tilde w_j + b_i + \tilde b_j \approx \ln X_{ij}$, then differences of vectors encode ratios of co-occurrence probabilities, which is what makes analogies like *ice : solid :: steam : gas* appear as vector offsets. The weight $f$ does two jobs: pairs that never co-occur get weight 0 (their log is $-\infty$), and very frequent pairs are capped at 1 so that *the, of* does not dominate. Rare pairs, whose counts are noisy, get small weights.

## Gradients in matrix form
Write $R$ for the residual matrix and $G = 2 f \odot R$. Since prediction $P = W \tilde W^\top + b\mathbf{1}^\top + \mathbf{1}\tilde b^\top$, the chain rule gives $\nabla_W J = G \tilde W$, $\nabla_{\tilde W} J = G^\top W$, $\nabla_b J = G\mathbf{1}$ and $\nabla_{\tilde b} J = G^\top \mathbf{1}$. The dense form is fine for a tiny vocabulary; real GloVe iterates over the non-zero entries only, with AdaGrad.

## Why return W + Wt
$X$ is symmetric, so $W$ and $\tilde W$ play the same role and differ only by their random initialisation. Summing them averages out some of that noise and was found to help slightly.

## Complexity
Building $X$ is $O(T \cdot \text{window})$. Each dense step costs $O(V^2 d)$; the sparse version costs $O(\text{nnz}(X) \cdot d)$.

## Common bugs
- Taking `np.log(X)` everywhere, which produces `-inf` and then `nan` once multiplied by a zero weight ($0 \cdot \infty$).
- Counting each pair of positions only once, which halves $X$ and breaks symmetry.
- Updating `W` before computing the gradient for `Wt`.

## Interview follow-ups
- How is GloVe related to factorizing the PMI matrix, and to skip-gram with negative sampling?
- Why does GloVe need far fewer passes over the corpus than word2vec? (It trains on the aggregated count matrix, not on individual tokens.)
"""
