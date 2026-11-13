import numpy as np

META = dict(title='fastText vectors from hashed character n-grams', tags=['Embeddings', 'NLP'], diff='Hard', runtime='numpy',
            lessons=['aieng:02.01'], related=['word2vec', 'embedding-similarity', 'subword-tokenization'])

STATEMENT = r"""
fastText represents a word as the sum of its own vector and the vectors of its character n-grams. Because n-grams are shared, it can build a vector even for a word it never saw. To keep memory bounded, n-grams are not stored in a dictionary: each one is hashed into one of `bucket` rows.

Write `fasttext_vectors(queries, vocab, E, minn, maxn, bucket)` that returns `(ids, vectors)`.

## Rows of E
`E` is a float array of shape `(len(vocab) + bucket, d)`. Row `t` for `t < len(vocab)` is the vector of the whole word `vocab[t]`. Row `len(vocab) + h` is n-gram bucket `h`.

## Ids for a word w
1. If `w` is in `vocab`, its index comes first.
2. Let `s = "<" + w + ">"`. For `n` from `minn` to `maxn` inclusive, and for each start `i` from left to right, take the character n-gram `g = s[i:i+n]` (characters are Python code points). Skip `g` if it equals `s` itself. Every occurrence counts, so a repeated n-gram contributes twice.
3. The row for `g` is `len(vocab) + fnv1a(g) % bucket`, where `fnv1a` is the 32-bit FNV-1a hash of `g.encode("utf-8")`:

```
h = 2166136261
for byte in data:
    h = h ^ byte
    h = (h * 16777619) % 2**32
```

## Output
- `ids`: a list with one list of row indices (Python ints) per query, in the order above.
- `vectors`: float array `(len(queries), d)`, row $q$ is the **mean** of `E[ids[q]]`, or zeros if `ids[q]` is empty.
"""

STARTER = '''import numpy as np

def fasttext_vectors(queries, vocab, E, minn, maxn, bucket):
    """Return (list of row-id lists, (len(queries), d) mean vectors)."""
    raise NotImplementedError

def fnv1a(data):
    """32-bit FNV-1a hash of a bytes object."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def fnv1a(data):
    h = 2166136261
    for byte in data:
        h = ((h ^ byte) * 16777619) % 2**32       # Python ints don't overflow, so reduce explicitly
    return h

def fasttext_vectors(queries, vocab, E, minn, maxn, bucket):
    E = np.asarray(E, dtype=float)
    index = {w: t for t, w in enumerate(vocab)}
    all_ids, out = [], np.zeros((len(queries), E.shape[1]))
    for q, w in enumerate(queries):
        ids = [index[w]] if w in index else []
        s = "<" + w + ">"                            # boundary markers: prefixes and suffixes get their own n-grams
        for n in range(minn, maxn + 1):
            for i in range(len(s) - n + 1):
                g = s[i:i + n]
                if g != s:
                    ids.append(len(vocab) + fnv1a(g.encode("utf-8")) % bucket)
        if ids:
            out[q] = E[ids].mean(axis=0)
        all_ids.append(ids)
    return all_ids, out
'''

HINTS = [
    'Write the hash first and test it on `b"a"`: FNV-1a gives 3826002220 for that byte. Python ints are unbounded, so apply `% 2**32` after each multiplication.',
    'For each query, start the id list with the word index (if known), then loop `n` over `minn..maxn` and `i` over every start position in `"<" + w + ">"`, skipping the n-gram that equals the whole bracketed word.',
    'Offset every bucket by `len(vocab)` so n-gram rows never collide with word rows. The vector is `E[ids].mean(axis=0)`; an unknown word that is too short for any n-gram gets zeros.',
]

_r = np.random.default_rng(31)
_vocab = ["where", "here", "her", "the", "naïve"]
_E = _r.normal(size=(len(_vocab) + 40, 5))
TESTS = [
    (["her"], ["her"], np.arange(2 * 4, dtype=float).reshape(4, 2), 3, 3, 3),
    (["ab", "zz"], ["ab"], np.eye(5)[:, :3], 2, 2, 4),
    (["he"], _vocab, _E, 1, 2, 40),
    ([""], _vocab, _E, 3, 6, 40),
    (["where", "wherever", "there"], _vocab, _E, 3, 6, 40),
    (["aaaa", "naïve", "naive"], _vocab, _E, 2, 4, 40),
    (["x"], ["y"], np.ones((1 + 7, 2)), 5, 6, 7),
]

EXPLANATION = r"""
## Why it works
Words with shared morphology share n-grams: *where*, *wherever* and *there* all contain `her`, `ere` and `re>`. Averaging their vectors gives an unseen word like *wherever* a meaningful representation, and rare words borrow strength from common pieces. The `<` and `>` markers make prefixes and suffixes distinct from the same letters in the middle of a word (`<re` vs `re>` vs `ere`).

## Why hashing
A language has millions of distinct n-grams. Hashing them into a fixed number of buckets bounds memory, at the cost of collisions: unrelated n-grams can share a row. With a few million buckets, collisions are rare enough not to matter. FNV-1a is used because it is fast and mixes bits well for short strings.

## Bytes vs characters
n-grams are taken over characters but hashed over their UTF-8 bytes, so `ï` contributes two bytes to the hash (test 6). Mixing these up gives different buckets for every non-ASCII n-gram.

## Complexity
A word of length $L$ has $O(L \cdot (\text{maxn} - \text{minn} + 1))$ n-grams, each hashed in $O(\text{maxn})$, plus $O(d)$ per row to average.

## Common bugs
- Forgetting `% 2**32` (Python never overflows, so the hash grows without bound).
- Leaving out the boundary markers, or skipping the whole-word n-gram when it is shorter than `maxn` but not equal to `s`.
- Indexing n-gram rows without the `len(vocab)` offset.
- Deduplicating repeated n-grams (`aaaa` has `aa` three times plus `<a` and `a>`).

## Interview follow-ups
- Why does fastText handle morphologically rich languages (Finnish, Turkish) better than word2vec?
- What happens to similarity between two unknown words that share no n-grams with any known word?
"""
