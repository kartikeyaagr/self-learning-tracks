import numpy as np

META = dict(title='Bag-of-words document-term matrix', tags=['NLP', 'Feature Engineering'], diff='Easy', runtime='numpy',
            lessons=['sdsd:09.02'], related=['tf-idf', 'vocabulary-building', 'text-classification'])

STATEMENT = r"""
A bag-of-words model forgets word order and represents each document by how often each vocabulary word appears in it.

Write `bag_of_words(docs, vocab=None, binary=False)` that returns `(vocab, X)`.

## Inputs
- `docs`: a list of documents, each a list of token strings.
- `vocab`: `None`, or a list of distinct token strings that fixes the columns and their order.
- `binary`: if true, record presence (1) instead of counts.

## Output
- `vocab`: the list of column tokens. When the argument is `None`, it is every distinct token in `docs`, sorted in ascending string order.
- `X`: `np.int64` array of shape `(len(docs), len(vocab))`. `X[d, j]` is how many times `vocab[j]` occurs in document `d` (or 1/0 when `binary`). Tokens that are not in `vocab` are ignored.
"""

STARTER = '''import numpy as np

def bag_of_words(docs, vocab=None, binary=False):
    """Return (vocab list, X int64 (n_docs, n_vocab))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bag_of_words(docs, vocab=None, binary=False):
    if vocab is None:
        vocab = sorted({t for d in docs for t in d})
    col = {t: j for j, t in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.int64)
    for i, d in enumerate(docs):
        for t in d:
            j = col.get(t)
            if j is not None:
                X[i, j] += 1
    if binary:
        X = (X > 0).astype(np.int64)
    return list(vocab), X
'''

HINTS = [
    'First decide the columns: either the given `vocab`, or `sorted(set(...))` of every token in every document.',
    'Build a dict from token to column index, allocate a zero matrix, and add 1 for each token you find in the dict.',
    'For `binary=True`, compute the counts and then `(X > 0).astype(np.int64)`. Keep the dtype `int64` in both cases.',
]

_r = np.random.default_rng(11)
_words = ["data", "model", "train", "test", "loss", "token", "layer", "batch"]
_docs = [[_words[k] for k in _r.integers(0, 8, size=n)] for n in (6, 0, 9, 3, 12)]
TESTS = [
    ([["the", "cat", "sat"], ["the", "the", "dog"]],),
    ([["a", "b", "a"], ["c"]], ["c", "a", "z"]),
    ([["solo"]],),
    ([[], []],),
    ([["x", "x", "y"], ["y", "z", "z", "z"]], None, True),
    ([["Apple", "apple", "APPLE"], ["b"]],),
    (_docs,),
    (_docs, ["token", "loss", "missing"], True),
]

EXPLANATION = r"""
## Why it works
Fixing the vocabulary first gives every document the same coordinate system, so documents become comparable vectors. A dict lookup per token makes the whole matrix one pass over the corpus. Ignoring unknown tokens is the standard behaviour at inference time: a vocabulary fitted on training data cannot grow when new words appear.

## Complexity
$O(N)$ for $N$ tokens to fill, plus $O(V \log V)$ to sort the vocabulary, plus $O(DV)$ memory for the dense matrix. Real corpora have $V$ in the hundreds of thousands and most entries are zero, so libraries store a sparse matrix (CSR) instead.

## Common bugs
- Using `set(...)` without sorting, so the column order depends on hash seeds and changes between runs.
- `vocab.index(t)` inside the loop, which makes it $O(NV)$.
- Case: `Apple` and `apple` are different tokens unless you normalize first (test 6).

## Interview follow-ups
- What does bag-of-words lose, and how do n-gram features get some of it back? (Order; bigrams like *not good* recover local negation.)
- How does the hashing trick avoid storing a vocabulary at all, and what does it cost? (Columns are `hash(token) % m`; collisions merge unrelated words, and you cannot map a column back to a word.)
"""
