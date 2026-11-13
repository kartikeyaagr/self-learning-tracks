import numpy as np

META = dict(title='TF-IDF with smoothing and L2 normalization', tags=['NLP', 'Feature Engineering'], diff='Medium', runtime='numpy',
            lessons=['sdsd:09.04'], related=['bag-of-words', 'document-ranking', 'cosine-similarity-matrix'])

STATEMENT = r"""
TF-IDF reweights word counts so that words appearing in every document (*the*, *of*) count for little and distinctive words count for a lot.

Write `tfidf(docs, sublinear_tf=False, norm=True)` that returns `(vocab, X)`.

## Definitions
For $N$ documents, vocabulary word $t$ and document $d$:
- $\mathrm{tf}(t, d)$ = number of times $t$ occurs in $d$. With `sublinear_tf=True`, use $1 + \ln \mathrm{tf}$ instead when $\mathrm{tf} > 0$ (and 0 when $\mathrm{tf} = 0$).
- $\mathrm{df}(t)$ = number of documents that contain $t$ at least once.
- Smoothed inverse document frequency:
$$\mathrm{idf}(t) = \ln\frac{1 + N}{1 + \mathrm{df}(t)} + 1$$
- $X[d, t] = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t)$.
- If `norm` is true, divide each row by its Euclidean norm. A row of all zeros stays all zeros.

## Inputs and output
- `docs`: a non-empty list of documents, each a list of token strings.
- `vocab`: all distinct tokens, sorted ascending; it fixes the columns.
- `X`: float array `(N, len(vocab))`.
"""

STARTER = '''import numpy as np

def tfidf(docs, sublinear_tf=False, norm=True):
    """Return (sorted vocab, X float (n_docs, n_vocab))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def tfidf(docs, sublinear_tf=False, norm=True):
    vocab = sorted({t for d in docs for t in d})
    col = {t: j for j, t in enumerate(vocab)}
    tf = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for t in d:
            tf[i, col[t]] += 1
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1          # +1 inside: no division by 0; +1 outside: keep df=N words
    if sublinear_tf:
        tf = np.where(tf > 0, 1 + np.log(np.maximum(tf, 1)), 0.0)   # maximum() avoids log(0) warnings
    X = tf * idf
    if norm:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        X = np.divide(X, n, out=np.zeros_like(X), where=n > 0)
    return vocab, X
'''

HINTS = [
    'Build the raw count matrix first (rows = documents, columns = sorted vocabulary). Document frequency is then `(tf > 0).sum(axis=0)`.',
    'idf is one vector of length V; broadcasting `tf * idf` scales every column at once. For sublinear tf, transform only the non-zero counts.',
    'Normalize rows with `np.linalg.norm(X, axis=1, keepdims=True)` and a guarded division so empty documents stay zero instead of `nan`.',
]

_r = np.random.default_rng(21)
_w = ["gpu", "cpu", "token", "batch", "loss", "the", "a", "grad"]
_docs = [[_w[k] for k in _r.integers(0, 8, size=n)] for n in (10, 4, 0, 7, 15, 3)]
TESTS = [
    ([["the", "cat"], ["the", "dog", "dog"]],),
    ([["a", "a", "a", "b"], ["b"]], True, False),
    ([["x"]],),
    ([[], ["y", "y"]],),
    ([["same", "same"], ["same"]],),
    ([["a", "b"], ["b", "c"], ["c", "a"]], False, False),
    (_docs,),
    (_docs, True, True),
]

EXPLANATION = r"""
## Why it works
$\mathrm{idf}$ is large for rare words and close to 1 for words in every document, so multiplying counts by it emphasises what makes a document different. The two smoothing terms come from pretending there is one extra document that contains every word once: no division by zero, and a word in all $N$ documents still gets weight $\ln 1 + 1 = 1$ instead of vanishing. L2-normalizing makes the dot product of two rows their cosine similarity, so long documents don't win just by being long.

## Sublinear tf
The tenth occurrence of a word says much less than the first. $1 + \ln \mathrm{tf}$ grows slowly: counts 1, 10, 100 become 1, 3.3, 5.6.

## Complexity
$O(N_{\text{tokens}})$ to count and $O(NV)$ for the dense matrix operations; real implementations keep $X$ sparse.

## Common bugs
- Computing df as the total count of a word rather than the number of documents containing it.
- Using `log10` or a different idf variant; every library has its own, so read the definition.
- Normalizing columns instead of rows.
- Fitting idf on the test set. At inference time reuse the training vocabulary and idf.

## Interview follow-ups
- How does BM25 improve on TF-IDF for search? (Saturating tf with $k_1$ and length normalization with $b$.)
- Why can TF-IDF + logistic regression still beat a fine-tuned transformer on some text classification tasks? (Small data, long documents, keyword-driven labels, and far lower cost.)
"""
