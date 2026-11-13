import numpy as np

META = dict(title='SIF sentence embeddings', tags=['Embeddings', 'NLP'], diff='Medium', runtime='numpy',
            lessons=['aieng:02.02'], related=['embedding-similarity', 'svd-components', 'masked-mean-pooling'])

STATEMENT = r"""
Averaging word vectors is a surprisingly strong sentence embedding, but frequent words dominate the average and all averages share a common direction. *Smooth inverse frequency* (SIF) fixes both: down-weight frequent words, then remove the shared direction.

Write `sif(E, sentences, p, a=1e-3, remove_pc=True)`.

## Inputs
- `E`: float array `(V, d)` of word vectors.
- `sentences`: a list of sentences, each a list of word ids in `[0, V)`. A sentence may be empty.
- `p`: float array `(V,)`, the unigram probability of each word.
- `a`: smoothing constant $> 0$.

## Steps
1. For each sentence $s$ with tokens $w_1, \dots, w_n$ (repeats count separately):
$$v_s = \frac{1}{n} \sum_{i=1}^{n} \frac{a}{a + p_{w_i}}\, E_{w_i}$$
An empty sentence gets the zero vector.
2. Stack the $v_s$ into $S$ of shape `(len(sentences), d)`.
3. If `remove_pc` is true and $S$ is not all zeros: let $u$ be the first right singular vector of $S$ (the top row of $V^\top$ in $S = U \Sigma V^\top$, **without** centering $S$). Replace $S$ by $S - S u u^\top$.

Return $S$ as a float array `(len(sentences), d)`. The tests always have a unique largest singular value.
"""

STARTER = '''import numpy as np

def sif(E, sentences, p, a=1e-3, remove_pc=True):
    """Return (n_sentences, d) SIF embeddings."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sif(E, sentences, p, a=1e-3, remove_pc=True):
    E, p = np.asarray(E, dtype=float), np.asarray(p, dtype=float)
    weight = a / (a + p)                                   # rare words -> close to 1, frequent words -> small
    S = np.zeros((len(sentences), E.shape[1]))
    for i, s in enumerate(sentences):
        if len(s):
            ids = np.asarray(s)
            S[i] = (weight[ids, None] * E[ids]).mean(axis=0)
    if remove_pc and np.any(S):
        u = np.linalg.svd(S, full_matrices=False)[2][0]    # top right singular vector; its sign does not matter
        S = S - np.outer(S @ u, u)
    return S
'''

HINTS = [
    'Compute the per-word weights `a / (a + p)` once as a vector, then each sentence is a weighted mean of rows `E[ids]`. Indexing with a list of ids keeps repeated words.',
    '`np.linalg.svd(S, full_matrices=False)` returns `(U, s, Vt)`. The first right singular vector is `Vt[0]`.',
    'Removing the component is $S - (Su)u^\\top$, i.e. `S - np.outer(S @ u, u)`. Flipping the sign of `u` gives the same result, so you do not need to canonicalize it. Skip this when S is all zeros.',
]

_r = np.random.default_rng(17)
_V, _d = 30, 8
_E = _r.normal(size=(_V, _d)) + 0.5                       # a shared offset, like real embeddings have
_p = _r.dirichlet(np.ones(_V))
_sents = [_r.integers(0, _V, size=n).tolist() for n in (5, 8, 3, 12, 6, 1)]
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]]), [[0, 1], [2]], np.array([0.1, 0.1, 0.8]), 0.1, False),
    (np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]]), [[0, 1], [2], [0, 0, 2]], np.array([0.1, 0.1, 0.8]), 0.1),
    (np.array([[1.0, 2.0]]), [[]], np.array([1.0])),
    (np.array([[1.0, 2.0], [3.0, -1.0]]), [[], [0], []], np.array([0.5, 0.5]), 0.5, False),
    (_E, _sents, _p),
    (_E, _sents, _p, 1e-2, False),
    (_E, _sents + [[], [3, 3, 3]], _p, 1e-4),
]

EXPLANATION = r"""
## Why it works
The weight $a/(a + p_w)$ is close to 1 for rare words and small for words like *the* whose probability is far above $a$. That is a smooth version of removing stop words, and it comes from a generative model where each word is produced either from the sentence's meaning or from a frequency-driven background. The background also adds a direction shared by almost every sentence (it is visible as the large first singular value). Projecting that direction out makes the remaining vectors differ in what they actually say, so cosine similarities between sentences become more informative.

## Sign of the singular vector
SVD returns $u$ or $-u$ depending on the LAPACK build. The projection $u u^\top$ is the same for both, so the output is well defined without any sign fixing.

## Complexity
$O(N d)$ for the weighted averages over $N$ tokens, and $O(n d \min(n, d))$ for the SVD of $n$ sentence vectors. In practice $u$ is computed once on a large reference set and reused for every new sentence.

## Common bugs
- Dividing by the sum of weights instead of the token count, which undoes the down-weighting.
- Using the first **left** singular vector (`U[:, 0]`), which has length `n`, not `d`.
- Centering $S$ first. That turns step 3 into PCA, which removes a different direction.

## Interview follow-ups
- Why do trained sentence encoders (contrastive fine-tuning) beat SIF, and when is SIF still a good choice? (Word order and context; SIF needs no training and is fast.)
- How would you apply the same correction to a new sentence at query time? (Keep the reference $u$ and subtract $(v \cdot u)u$.)
"""
