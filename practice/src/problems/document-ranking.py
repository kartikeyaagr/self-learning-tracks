import numpy as np

META = dict(title='Hybrid retrieval: BM25, dense scores and rank fusion', tags=['NLP', 'Embeddings'], diff='Hard', runtime='numpy',
            lessons=['sdsd:09.05', 'sdsd:09.13'], related=['tf-idf', 'embedding-similarity', 'precision-recall-at-k'])

STATEMENT = r"""
Search systems often combine a lexical ranker (exact word matches) with a dense ranker (embedding similarity). Their scores live on different scales, so instead of adding scores you fuse **ranks**.

Write `hybrid_rank(docs, query, doc_emb, query_emb, k1=1.5, b=0.75, k_rrf=60, top_k=3)` that returns `(bm25, top_docs, top_scores)`.

## 1. BM25
For $N$ documents (token lists), $\text{avgdl}$ the mean document length, $\text{df}(t)$ the number of documents containing $t$ and $\text{tf}(t, d)$ its count in document $d$:
$$\mathrm{idf}(t) = \ln\Big(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\Big)$$
$$\text{bm25}(d) = \sum_{t \in \text{query}} \mathrm{idf}(t) \cdot \frac{\text{tf}(t, d)\,(k_1 + 1)}{\text{tf}(t, d) + k_1\big(1 - b + b\,|d| / \text{avgdl}\big)}$$
The sum runs over the query tokens **with repetition** (a token that appears twice in the query counts twice). If $\text{avgdl} = 0$, every score is 0.

## 2. Dense
The dense score of document $d$ is the cosine similarity between `doc_emb[d]` and `query_emb`, or 0 if either has norm 0.

## 3. Ranks and fusion
For each of the two score lists, rank the documents by score, highest first, ties by smaller document index; the best document has rank 1. The reciprocal-rank-fusion score is
$$\text{rrf}(d) = \frac{1}{k_{\text{rrf}} + \text{rank}_{\text{bm25}}(d)} + \frac{1}{k_{\text{rrf}} + \text{rank}_{\text{dense}}(d)}$$

## Output
- `bm25`: float array `(N,)` of BM25 scores.
- `top_docs`: list of the `min(top_k, N)` document indices with the highest rrf score, highest first, ties by smaller index.
- `top_scores`: list of their rrf scores (floats).
"""

STARTER = '''import numpy as np

def hybrid_rank(docs, query, doc_emb, query_emb, k1=1.5, b=0.75, k_rrf=60, top_k=3):
    """Return (bm25 scores (N,), fused top document ids, their RRF scores)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from collections import Counter

def ranks(scores):
    """1-based rank of each document: higher score first, ties to the smaller index."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r

def hybrid_rank(docs, query, doc_emb, query_emb, k1=1.5, b=0.75, k_rrf=60, top_k=3):
    N = len(docs)
    lens = np.array([len(d) for d in docs], dtype=float)
    avgdl = lens.mean()
    tfs = [Counter(d) for d in docs]
    bm25 = np.zeros(N)
    if avgdl > 0:
        norm = k1 * (1 - b + b * lens / avgdl)              # per-document length normalization
        for t in query:                                     # repeated query terms add again
            tf = np.array([c[t] for c in tfs], dtype=float)
            df = np.count_nonzero(tf)
            idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
            bm25 += idf * tf * (k1 + 1) / (tf + norm)

    E, q = np.asarray(doc_emb, dtype=float), np.asarray(query_emb, dtype=float)
    denom = np.linalg.norm(E, axis=1) * np.linalg.norm(q)
    dense = np.divide(E @ q, denom, out=np.zeros(N), where=denom > 0)

    rrf = 1.0 / (k_rrf + ranks(bm25)) + 1.0 / (k_rrf + ranks(dense))
    top = np.argsort(-rrf, kind="stable")[:top_k]
    return bm25, [int(i) for i in top], [float(rrf[i]) for i in top]
'''

HINTS = [
    'Precompute a `Counter` per document and the per-document length term $k_1(1 - b + b|d|/\\text{avgdl})$. Then loop over query tokens and add each term\'s contribution to all documents at once with NumPy.',
    'Turn scores into ranks with a stable argsort of `-scores`: `order = np.argsort(-s, kind="stable")`, then `rank[order] = 1..N`.',
    'RRF only needs the two rank arrays: `1/(k + r1) + 1/(k + r2)`. Sort that again with a stable argsort for the final order. Watch the zero-norm case in the cosine.',
]

_r = np.random.default_rng(67)
_W = ["gpu", "kernel", "memory", "cache", "token", "attention", "batch", "latency", "the", "a"]
_docs = [[_W[k] for k in _r.integers(0, len(_W), size=n)] for n in (12, 5, 20, 8, 15, 3, 9)]
_E = _r.normal(size=(7, 6))
TESTS = [
    ([["cat", "sat"], ["dog", "sat", "down"], ["cat", "cat", "cat"]], ["cat"],
     np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([0.0, 1.0]), 1.5, 0.75, 60, 3),
    ([["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"]], ["a", "a", "c"],
     np.eye(4)[:, :2], np.array([1.0, 1.0]), 1.2, 0.5, 1, 4),
    ([[], []], ["x"], np.zeros((2, 3)), np.ones(3)),
    ([["x"]], ["x"], np.ones((1, 2)), np.zeros(2), 1.5, 0.75, 60, 5),
    ([["the", "cache"], ["the", "the", "cache", "miss"]], ["cache", "miss"], np.array([[1.0, 0.0], [0.0, 1.0]]),
     np.array([1.0, 0.0]), 1.5, 1.0, 60, 2),
    (_docs, ["gpu", "memory", "latency"], _E, _r.normal(size=6)),
    (_docs, ["attention", "attention", "the", "zzz"], _E, _E[2] + 0.1 * _r.normal(size=6), 0.9, 0.4, 10, 5),
]

EXPLANATION = r"""
## Why BM25 looks like this
- **idf** rewards rare query terms; this form never goes negative, unlike the original Robertson-Spärck Jones weight, which matters for terms that appear in most documents.
- **Saturation**: $\frac{\text{tf}(k_1 + 1)}{\text{tf} + k_1}$ rises quickly and levels off at $k_1 + 1$, so the tenth occurrence adds little. Raw tf-idf grows linearly.
- **Length normalization** with $b$: long documents contain more words by chance, so their tf is discounted. $b = 0$ turns it off and $b = 1$ normalizes fully (test 5).

## Why fuse ranks
BM25 scores are unbounded and depend on the query; cosine similarities lie in $[-1, 1]$. Adding them would let one system dominate arbitrarily. RRF uses only positions: a document ranked high by either system gets a large share, and $k_{\text{rrf}}$ (60 by convention) controls how quickly credit decays with rank. It needs no training and is surprisingly hard to beat.

## Ties
Rank-based scores tie exactly and often. In test 5, document 0 is ranked 2nd by BM25 and 1st by the dense scores, and document 1 the other way round, so both get $1/62 + 1/61$ and $1/61 + 1/62$: the same float, because floating-point addition is commutative. The index tie-break then decides.

## Complexity
BM25 is $O(|q| \cdot N)$ here; real engines walk the posting lists of an inverted index so only documents containing a query term are touched. Dense scoring is $O(Nd)$ exact, or sublinear with an ANN index. Fusion is $O(N \log N)$.

## Common bugs
- Using the number of query terms that match instead of summing over query tokens.
- Ranking with an unstable sort, or using 0-based ranks in the RRF formula.
- Normalizing document length by the maximum length instead of the average.

## Interview follow-ups
- When does lexical search beat dense retrieval? (Rare identifiers, product codes, names, out-of-domain vocabulary.)
- How would you evaluate the hybrid against each component? (Recall@k and nDCG on labelled queries; then add a cross-encoder re-ranker on the fused top 100.)
"""
