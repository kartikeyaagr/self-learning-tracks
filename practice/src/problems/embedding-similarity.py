import numpy as np

META = dict(title='Nearest neighbours in an embedding table', tags=['Embeddings'], diff='Easy', runtime='numpy',
            lessons=['aieng:02.02', 'air-fanout-v1:02.01'], related=['cosine-similarity-matrix', 'sentence-embeddings', 'word2vec'])

STATEMENT = r"""
Word embeddings place similar words close together. Write `nearest(E, words, query, k)` that returns the `k` words most similar to `query` by cosine similarity:

$$\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert\, \lVert v \rVert}$$

## Inputs
- `E`: float array `(V, d)`, row `i` is the embedding of `words[i]`.
- `words`: list of `V` distinct strings.
- `query`: a string in `words`.
- `k`: int $\ge 0$.

## Rules
- The query word itself is never returned.
- If either vector has norm 0, their similarity is defined as 0.
- Sort by similarity, highest first. Equal similarities are ordered by row index, smallest first.
- If fewer than `k` other words exist, return all of them.

## Output
A list of `(word, similarity)` tuples with `similarity` a Python float.
"""

STARTER = '''import numpy as np

def nearest(E, words, query, k):
    """Return the k most cosine-similar words to query as [(word, sim), ...]."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def nearest(E, words, query, k):
    E = np.asarray(E, dtype=float)
    q = words.index(query)
    norms = np.linalg.norm(E, axis=1)
    denom = norms * norms[q]
    sims = np.divide(E @ E[q], denom, out=np.zeros(len(E)), where=denom > 0)   # zero vector -> 0
    order = [i for i in np.argsort(-sims, kind="stable") if i != q]            # stable: ties keep index order
    return [(words[i], float(sims[i])) for i in order[:k]]
'''

HINTS = [
    'You need one similarity per row: a single matrix-vector product `E @ E[q]` divided by the product of norms.',
    'Use `np.divide(..., where=denom > 0, out=np.zeros(...))` so zero vectors get similarity 0 instead of `nan`.',
    'Sort with `np.argsort(-sims, kind="stable")`: negating gives descending order, and a stable sort keeps equal values in index order. Then drop the query index and take `k`.',
]

_r = np.random.default_rng(5)
_V = [f"w{i}" for i in range(12)]
_E = _r.normal(size=(12, 6))
TESTS = [
    (np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]), ["king", "queen", "apple", "pawn"], "king", 2),
    (np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [1.0, -1.0], [2.0, 2.0]]), ["a", "b", "c", "d", "e"], "a", 4),
    (np.array([[1.0, 0.0]]), ["solo"], "solo", 3),
    (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), ["zero", "x", "y"], "zero", 2),
    (np.array([[1.0, 2.0], [3.0, 1.0]]), ["p", "q"], "q", 0),
    (_E, _V, "w3", 5),
    (_E * 1e-100, _V, "w7", 3),
    (np.vstack([_E, _E[2] * 4.0]), _V + ["dup"], "w2", 3),
]

EXPLANATION = r"""
## Why cosine
Embedding length often tracks word frequency rather than meaning, so the raw dot product favours frequent words. Dividing by both norms compares directions only. If you normalize every row once up front, cosine becomes a plain dot product and the whole table can be searched with one matrix product, which is what vector databases do.

## Ties and zero vectors
In test 2 the rows `b` and `e` are identical, so their similarities are bit-for-bit equal and come out in row order because the sort is stable. `c` (a zero vector) and `d` (orthogonal) both score exactly 0 and are ordered the same way. A zero vector has no direction, so its similarity is defined as 0 instead of `0/0`.

Ties between vectors that are only *mathematically* parallel, like `(1, 1)` and `(3, 3)`, are not safe: rounding can make one cosine `0.9999999999999998` and the other `1.0`. If tie order matters to you, round the scores or compare with a tolerance.

## Complexity
$O(Vd)$ for the similarities and $O(V \log V)$ for the sort. With `np.argpartition` you can get the top $k$ in $O(V)$ and sort only those $k$.

## Common bugs
- Not excluding the query, so it always comes back first with similarity 1.
- Normalizing only the table rows and forgetting the query (or the other way round). Cosine is scale-invariant, so test 7, which scales every vector by $10^{-100}$, must give the same similarities as test 6's table would.
- `np.argsort(sims)[::-1]`, which reverses the tie order too.

## Interview follow-ups
- How do you answer analogy queries like *king - man + woman*? (Normalize, form the target vector, and search excluding the three input words.)
- Exact search is $O(Vd)$ per query. How do approximate indexes (IVF, HNSW, product quantization) trade recall for speed?
"""
