import numpy as np

META = dict(title='One epoch of skip-gram with negative sampling', tags=['Embeddings', 'NLP'], diff='Hard', runtime='numpy',
            lessons=['dlcore:06.04'],
            related=['word2vec-training-pairs', 'word2vec', 'embedding-similarity', 'fasttext'])

STATEMENT = r"""
Put the word2vec pipeline together: build training pairs from a corpus, draw negatives from the noise distribution, and run plain SGD over the pairs one at a time, exactly as the original implementation does.

Write `sgns_epoch(corpus, W_in, W_out, t, u, b, u_neg, lr)` returning `(W_in, W_out, mean_loss)` (new arrays; don't modify the inputs).

## Pairs and noise (same rules as the data-pipeline step)
- Counts and frequencies over the full corpus; position $j$ is discarded if `u[j]` $< \max(0, 1 - \sqrt{t/f})$; the $m$-th kept token uses window radius `b[m]`; contexts in order $m - b_m .. m + b_m$, skipping $m$.
- Noise $P_n(w) \propto \text{count}(w)^{3/4}$ over the $V$ rows of `W_in`.

## Negatives
`u_neg` has shape `(R, K)` with $R \ge$ the number of pairs; pair number $i$ uses row $i$. Each uniform $\upsilon$ becomes the word `np.searchsorted(np.cumsum(noise), υ, side='right')` (inverse-CDF sampling). Negatives may repeat or equal the context; use them as drawn.

## SGD, pair by pair in order
For centre $c$, context $o$ and negatives $n_1..n_K$, with $v = W_{in}[c]$ and $u_w = W_{out}[w]$ taken **before** this pair's update:
$$\ell = -\log\sigma(u_o^\top v) - \sum_k \log\sigma(-u_{n_k}^\top v)$$
Gradients: $g_o = \sigma(u_o^\top v) - 1$, $g_k = \sigma(u_{n_k}^\top v)$; $\nabla_v = g_o u_o + \sum_k g_k u_{n_k}$, $\nabla_{u_o} = g_o v$, $\nabla_{u_{n_k}} = g_k v$ (accumulate if a word appears more than once). Subtract `lr` times each gradient. `mean_loss` is the mean of $\ell$ over pairs (0.0 if there are none). Compute $\log\sigma$ stably.
"""

STARTER = '''import numpy as np

def sgns_epoch(corpus, W_in, W_out, t, u, b, u_neg, lr):
    """Return (W_in, W_out, mean_loss)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _log_sig(z):
    return -np.logaddexp(0.0, -z)

def _pairs(corpus, V, t, u, b):
    counts = np.bincount(corpus, minlength=V).astype(float)
    f = counts[corpus] / len(corpus)
    kept = corpus[np.asarray(u) >= np.maximum(0.0, 1.0 - np.sqrt(t / f))]
    out = []
    for m, c in enumerate(kept):
        r = int(b[m])
        out += [(int(c), int(kept[j])) for j in range(max(0, m - r), min(len(kept), m + r + 1)) if j != m]
    noise = counts ** 0.75
    return out, noise / noise.sum()

def sgns_epoch(corpus, W_in, W_out, t, u, b, u_neg, lr):
    corpus = np.asarray(corpus)
    Win, Wout = np.array(W_in, dtype=float), np.array(W_out, dtype=float)
    pairs, noise = _pairs(corpus, Win.shape[0], t, u, b)
    cdf = np.cumsum(noise)
    losses = []
    for i, (c, o) in enumerate(pairs):
        negs = np.minimum(np.searchsorted(cdf, u_neg[i], side='right'), len(noise) - 1)
        v = Win[c].copy()
        so = Wout[o] @ v
        sn = Wout[negs] @ v
        losses.append(float(-_log_sig(so) - _log_sig(-sn).sum()))
        go = 1 / (1 + np.exp(-so)) - 1
        gn = 1 / (1 + np.exp(-sn))
        grad_v = go * Wout[o] + gn @ Wout[negs]
        Wout[o] -= lr * go * v
        np.subtract.at(Wout, negs, lr * gn[:, None] * v[None, :])
        Win[c] -= lr * grad_v
    return Win, Wout, float(np.mean(losses)) if losses else 0.0
'''

HINTS = [
    'Reuse the pair builder from the data-pipeline step, then loop over pairs. Read $v$, $u_o$ and the negatives\' rows **before** writing any update for that pair.',
    'Turn each row of `u_neg` into word ids with `np.searchsorted(np.cumsum(noise), row, side="right")`.',
    'Repeated negatives must each contribute: `np.subtract.at(W_out, negs, ...)` accumulates, while `W_out[negs] -= ...` silently keeps only one update per word.',
]

_r = np.random.default_rng(2013)
_corp = np.minimum(_r.zipf(1.5, size=40) - 1, 7)
def _w(V, D):
    return _r.normal(size=(V, D)) * 0.3, _r.normal(size=(V, D)) * 0.3
TESTS = [
    (np.array([0, 1, 2, 1, 0]), *_w(3, 2), 1.0, np.zeros(5), np.ones(5, dtype=int), _r.random((10, 2)), 0.1),
    (np.array([0, 1, 2, 1, 0]), *_w(3, 2), 1.0, np.zeros(5), np.full(5, 2), _r.random((20, 3)), 0.5),
    (_corp, *_w(8, 4), 0.05, _r.random(40), _r.integers(1, 3, size=40), _r.random((200, 3)), 0.05),
    (_corp, *_w(8, 4), 1.0, _r.random(40), _r.integers(1, 4, size=40), _r.random((300, 5)), 0.025),
    (np.array([3, 3, 3]), *_w(4, 2), 1e-6, np.array([0.1, 0.2, 0.3]), np.ones(3, dtype=int), _r.random((6, 2)), 0.1),
]

EXPLANATION = r"""
## What one update does
The positive term pulls the context's output vector and the centre's input vector together; each negative term pushes a noise word's output vector away from the centre. Because only $K + 1$ output rows and one input row are touched per pair, an update costs $O(KD)$ instead of the $O(VD)$ of a full softmax, which is the whole point of negative sampling.

## Two embedding tables
`W_in` (the vectors usually published) and `W_out` play different roles: a word's input vector is trained to predict its contexts, its output vector to be predicted. Some work averages the two; GloVe-style models sum them.

## Sequential SGD
The original implementation updates after every pair with a learning rate that decays linearly to near zero over training, from many threads without locks (Hogwild). Doing the updates in order with values read before each update reproduces the single-threaded behaviour.

## Complexity
$O(MKD)$ for $M$ pairs.

## Common bugs
- Updating $v$ first and then using the new $v$ for the output-vector gradients.
- Fancy-index assignment with repeated negatives (loses updates).
- `log(sigmoid(z))` underflowing to `-inf` for very negative $z$.

## Interview follow-ups
- Show that SGNS implicitly factorises a shifted PMI matrix.
- Why is the noise distribution's exponent a hyperparameter of the learned geometry, not just of training speed?
"""
