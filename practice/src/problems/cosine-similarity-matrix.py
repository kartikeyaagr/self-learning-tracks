import numpy as np

META = dict(title='Pairwise cosine similarity', tags=['Linear Algebra', 'Embeddings'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.01'], related=['scaled-dot-product-attention', 'precision-recall-at-k'])

STATEMENT = r"""
Cosine similarity compares directions and ignores lengths:

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert\, \lVert b \rVert}$$

Write `cosine_matrix(A, B)` that returns every pairwise similarity between the rows of `A` (shape `(n, d)`) and the rows of `B` (shape `(m, d)`), as an `(n, m)` array.

## Requirements
- No Python loops: normalise the rows, then use one matrix product.
- If either vector in a pair has zero length, its similarity is defined as `0.0`.
"""

STARTER = '''import numpy as np

def cosine_matrix(A, B):
    """A: (n, d), B: (m, d). Return S with S[i, j] = cos(A[i], B[j]), shape (n, m)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cosine_matrix(A, B):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    def unit(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return np.divide(X, n, out=np.zeros_like(X), where=n > 0)   # zero rows stay zero
    return unit(A) @ unit(B).T
'''

HINTS = [
    'Divide every row by its own norm first. Then each similarity is just a dot product.',
    'With unit rows, all n·m dot products are one matrix product: `An @ Bn.T`.',
    'For zero rows, `np.divide(X, n, out=np.zeros_like(X), where=n > 0)` leaves them at 0, so their dot products are 0 with no 0/0.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])),
    (np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([[3.0, 3.0]])),
    (np.array([[0.0, 0.0], [1.0, 2.0]]), np.array([[1.0, 2.0], [0.0, 0.0]])),
    (_r.normal(size=(4, 8)), _r.normal(size=(3, 8))),
    (_r.normal(size=(1, 16)) * 1e6, _r.normal(size=(5, 16)) * 1e-6),
    (np.eye(3), np.eye(3)),
]

EXPLANATION = r"""
## Normalise first, multiply once
$\cos(a,b) = \hat a \cdot \hat b$ with $\hat a = a/\lVert a\rVert$. Normalising costs $O((n+m)d)$, and the product $\hat A \hat B^\top$ costs $O(nmd)$ but runs as one BLAS call. That call is what makes retrieval over millions of embeddings practical.

## Edge cases
- Zero vectors have no direction, so the similarity is undefined. Returning 0 (orthogonal to everything) is the usual convention, and it keeps `nan` out of downstream top-k.
- Scale invariance: multiplying a row by $10^6$ must not change its similarities. Test 5 checks this.

## Where this shows up
Embedding search (RAG, recommendations), contrastive losses (CLIP uses a temperature-scaled cosine matrix as logits), and deduplication. Vector databases often store pre-normalised vectors so that inner product equals cosine.

## Interview follow-ups
- When is cosine better than Euclidean distance? (When magnitude encodes something you want to ignore, like document length.)
- For unit vectors, $\lVert a-b\rVert^2 = 2 - 2\cos(a,b)$. So the nearest neighbour by L2 is the same as by cosine.
"""
