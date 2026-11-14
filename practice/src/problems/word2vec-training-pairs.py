import numpy as np

META = dict(title='word2vec data pipeline: subsampling, dynamic windows, noise distribution', tags=['Embeddings', 'NLP'],
            diff='Medium', runtime='numpy', lessons=['dlcore:06.04'],
            related=['word2vec', 'uniform-negative-sampling', 'vocabulary-building'])

STATEMENT = r"""
Before any gradient step, word2vec turns a corpus into (centre, context) training pairs and builds the distribution that negative samples are drawn from. Three tricks from the paper shape that data:

1. **Subsampling frequent words**: token $w$ with corpus frequency $f(w)$ (count / corpus length) is discarded with probability $P(w) = \max\big(0,\ 1 - \sqrt{t / f(w)}\big)$.
2. **Dynamic window**: each kept centre position $i$ uses its own window radius $b_i \in [1, \text{window}]$, so nearer words are paired more often on average.
3. **Noise distribution**: negatives are drawn from the unigram distribution raised to the $3/4$ power, $P_n(w) \propto \text{count}(w)^{3/4}$.

Write `w2v_pairs(corpus, V, t, u, b)` returning `(pairs, noise)`.

## Rules
- `corpus`: int array `(L,)` with ids in $[0, V)$. Counts and frequencies are computed over the full corpus.
- `u`: `(L,)` uniform draws; position $j$ is discarded when `u[j] < P(corpus[j])`. Windows act on the sequence of **kept** tokens.
- `b`: `(L,)` ints; the $m$-th kept token (0-based) uses radius `b[m]`.
- `pairs`: int array `(M, 2)` of `(centre, context)` ids: for each kept position $m$ in order, the contexts at kept positions $m-b_m, \dots, m-1, m+1, \dots, m+b_m$ (clipped to the sequence) in that order. If there are none, return shape `(0, 2)`.
- `noise`: `(V,)` probabilities; words with count 0 get probability 0.
"""

STARTER = '''import numpy as np

def w2v_pairs(corpus, V, t, u, b):
    """Return (pairs (M, 2) int, noise (V,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def w2v_pairs(corpus, V, t, u, b):
    corpus = np.asarray(corpus)
    counts = np.bincount(corpus, minlength=V).astype(float)
    f = counts[corpus] / len(corpus)
    p_drop = np.maximum(0.0, 1.0 - np.sqrt(t / f))
    kept = corpus[np.asarray(u) >= p_drop]
    pairs = []
    for m, c in enumerate(kept):
        r = int(b[m])
        for j in range(max(0, m - r), min(len(kept), m + r + 1)):
            if j != m:
                pairs.append((int(c), int(kept[j])))
    noise = counts ** 0.75
    noise /= noise.sum()
    return np.array(pairs, dtype=int).reshape(-1, 2), noise
'''

HINTS = [
    'Compute counts with `np.bincount`, then each position\'s discard probability from its word\'s frequency. Filter the corpus once; windows are taken over what is left.',
    'Loop over kept positions $m$ and take contexts $j$ from $m - b_m$ to $m + b_m$, skipping $j = m$ and anything outside the kept sequence.',
    '`count ** 0.75` normalised to sum to 1 is the noise distribution; a zero count stays at zero.',
]

_r = np.random.default_rng(2013)
_zipf = np.minimum(_r.zipf(1.6, size=60) - 1, 11)
TESTS = [
    (np.array([0, 1, 2, 3, 4]), 5, 1.0, np.zeros(5), np.ones(5, dtype=int)),
    (np.array([0, 1, 2, 3, 4]), 6, 1.0, np.zeros(5), np.array([2, 1, 2, 1, 2])),
    (np.array([0, 0, 0, 0, 1, 2, 0, 0, 3, 0]), 4, 0.1, _r.random(10), _r.integers(1, 3, size=10)),
    (_zipf, 12, 0.05, _r.random(60), _r.integers(1, 4, size=60)),
    (_zipf, 12, 1e-3, _r.random(60), _r.integers(1, 6, size=60)),
    (np.array([5, 5, 5, 2]), 7, 1e-6, np.array([0.9999, 0.5, 0.99995, 0.1]), np.array([1, 2, 1, 1])),
]

EXPLANATION = r"""
## Why each trick
- **Subsampling**: words like "the" occur in a huge fraction of windows but say little about their neighbours. Dropping most of their occurrences speeds training several-fold and improves the vectors of rarer words, which now see more informative contexts. With $t = 10^{-5}$ only words with frequency above $t$ are ever dropped.
- **Dynamic windows**: drawing $b_i$ uniformly from $1..\text{window}$ means a context at distance $d$ is used with probability $(\text{window} - d + 1)/\text{window}$, a cheap way to weight near context more.
- **The 3/4 power** flattens the unigram distribution: frequent words are still sampled more, but rare words get more negative examples than their raw frequency would give. The paper found it beat both the raw unigram and the uniform distribution.

## Subsampling before windowing
Because windows act on the kept sequence, removing a frequent word effectively widens the window around it. This is how the reference implementation behaves.

## Complexity
$O(L + M)$ where $M \le 2\,\text{window}\,L$ is the number of pairs.

## Common bugs
- Computing frequencies over the kept tokens rather than the full corpus.
- Using the radius of the context position instead of the centre.
- Forgetting to clip the discard probability at 0 for rare words (negative probabilities are harmless here only by luck).

## Interview follow-ups
- The released C code uses a slightly different keep probability, $(\sqrt{f/t} + 1)\,t/f$. How does it differ for very frequent and for rare words?
- How would you sample negatives from `noise` in $O(1)$ each?
"""
