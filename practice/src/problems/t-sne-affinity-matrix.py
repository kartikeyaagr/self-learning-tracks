import numpy as np

META = dict(title='t-SNE input affinities with a perplexity search', tags=['Probability', 'Embeddings'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:01.11'], related=['softmax-rows', 'pca-projection'])

STATEMENT = r"""
t-SNE starts by turning distances in the input space into probabilities. Each point $i$ gets its own Gaussian bandwidth, set so that its neighbour distribution has a chosen **perplexity**.

With squared Euclidean distances $D_{ij}$ and precision $\beta_i$,

$$p_{j\mid i} = \frac{\exp(-\beta_i D_{ij})}{\sum_{k\ne i}\exp(-\beta_i D_{ik})}\ \ (j \ne i), \qquad p_{i\mid i} = 0 .$$

Choose each $\beta_i > 0$ so that the entropy of row $i$ (in **nats**) equals $\log(\text{perplexity})$:

$$H_i = -\sum_{j\ne i} p_{j\mid i}\log p_{j\mid i} = \log(\text{perplexity}).$$

$H_i$ decreases monotonically as $\beta_i$ grows, so the solution is unique. Finally symmetrise:

$$P_{ij} = \frac{p_{j\mid i} + p_{i\mid j}}{2n}.$$

Write `tsne_affinities(X, perplexity)` that returns $P$.

## Input / output
- `X`: float `(n, d)` with $n \ge 3$ and no duplicate rows; $1 < \text{perplexity} < n - 1$.
- Return a float array `(n, n)`: symmetric, zero diagonal, summing to 1.
- Solve each $\beta_i$ to (near) machine precision, for example by bisection on $\beta$ or on $\log\beta$. Rows whose distances are large must not underflow to `0/0`.
"""

STARTER = '''import numpy as np

def tsne_affinities(X, perplexity):
    """Symmetric t-SNE affinity matrix P (n, n) with per-point bandwidths matched to perplexity."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def row_probs(d, beta):
    """Conditional probabilities for one row (self excluded) and their entropy in nats."""
    e = np.exp(-beta * d)
    s = e.sum()
    p = e / s
    return p, np.log(s) + beta * (d * p).sum()            # H = log s + beta * E[d], avoids 0 * log 0

def tsne_affinities(X, perplexity):
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    target = np.log(perplexity)
    cond = np.zeros((n, n))
    for i in range(n):
        others = np.arange(n) != i
        d = D[i, others]
        d = d - d.min()                                   # shifting distances leaves p unchanged, avoids underflow
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(200):                              # bisection: expand until bracketed, then halve
            _, H = row_probs(d, beta)
            if H > target:                                # too flat: sharpen
                lo = beta
                beta = beta * 2.0 if hi == np.inf else 0.5 * (lo + hi)
            else:
                hi = beta
                beta = 0.5 * (lo + hi)
        cond[i, others] = row_probs(d, beta)[0]
    return (cond + cond.T) / (2.0 * n)
'''

HINTS = [
    'Handle one row at a time. Subtract the row\'s smallest (non-self) distance before exponentiating: the shift cancels in the normalisation and keeps the largest term at $e^0 = 1$.',
    'Entropy is easy from the unnormalised weights: with $s = \\sum_j e^{-\\beta d_j}$, $H = \\log s + \\beta\\,\\sum_j p_j d_j$. If $H$ is above the target, the distribution is too flat and $\\beta$ must grow.',
    'Bisection: keep `lo = 0, hi = inf`. If $H$ is too high set `lo = beta` and double (or bisect once `hi` is finite); otherwise set `hi = beta` and bisect. A couple of hundred iterations reaches machine precision. Then $P = (C + C^\\top)/(2n)$.',
]

_r = np.random.default_rng(107)
_X = np.vstack([_r.normal(0, 1, size=(12, 3)), _r.normal(5, 1, size=(8, 3))])
TESTS = [
    (np.array([[0.0], [1.0], [3.0], [6.0]]), 2.0),
    (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0], [-1.0, 1.5]]), 3.0),
    (np.array([[0.0], [1.0], [2.5]]), 1.5),
    (np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [103.0]]), 2.5),
    (np.array([[0.0, 0.0], [1e3, 0.0], [0.0, 2e3], [3e3, 1e3], [5e3, 5e3]]), 2.0),
    (_X, 5.0),
    (_X * 0.01, 10.0),
]

EXPLANATION = r"""
## Why a per-point bandwidth
Density varies across the data. A single global $\sigma$ would give points in dense regions dozens of effective neighbours and points in sparse regions almost none. Fixing the perplexity, $2^{H}$ in bits or $e^{H}$ in nats, fixes the effective number of neighbours for every point: each row is roughly as informative as a uniform distribution over that many points.

## Why the entropy formula
$-\log p_j = \beta d_j + \log s$, so $H = \sum_j p_j(\beta d_j + \log s) = \log s + \beta\,\mathbb E_p[d]$. This never evaluates $\log 0$, even when some $p_j$ underflow (test 4 has points 100 units apart). Differentiating gives $dH/d\beta = -\beta\,\mathrm{Var}_p(d) \le 0$, which is why the search is monotone and bisection works.

## Why symmetrise
Conditional affinities aren't symmetric: an outlier $i$ has $p_{j\mid i}$ spread over its few distant neighbours, but $p_{i\mid j}$ is tiny for all of them. Averaging and dividing by $2n$ gives a joint distribution where every point has $\sum_j P_{ij} > \frac{1}{2n}$, so outliers still pull on the embedding.

## Complexity
$O(n^2 d)$ for distances plus $O(n^2 \cdot \text{iterations})$ for the searches. Barnes-Hut t-SNE only computes affinities for about $3\times$ perplexity nearest neighbours per point.

## Common bugs
- Including $j = i$ in the sum ($D_{ii} = 0$ dominates and the entropy target becomes unreachable).
- Using $\log_2$ for the target with natural-log entropy, or vice versa.
- Forgetting the $2n$ in the symmetrisation, so $P$ sums to $n$ instead of 1.

## Interview follow-ups
- Why does t-SNE use a Student-t kernel in the low-dimensional space but a Gaussian in the input space? (The heavy tail fixes the crowding problem.)
- What does perplexity control in the final picture, and why shouldn't you read distances between clusters in a t-SNE plot?
"""
