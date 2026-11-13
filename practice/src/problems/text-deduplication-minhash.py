META = dict(title='Near-duplicate detection with MinHash and LSH bands', tags=['NLP', 'Data Processing'], diff='Medium', runtime='python',
            related=['text-normalization', 'bag-of-words'])

STATEMENT = r"""
Web-scale training sets are full of near-duplicate documents. MinHash compresses each document's set of shingles into a short signature whose agreement rate estimates Jaccard similarity, and LSH banding finds candidate pairs without comparing every pair.

Write `minhash_lsh(docs, k, a, b, bands)` that returns `(signatures, candidates)`.

## Shingles
Lower-case each document and split it with `str.split()` into words (every document has at least one word). The shingles are the **set** of strings `" ".join(words[i:i+k])` for every start `i` with `i + k <= len(words)`. A document with fewer than `k` words has the single shingle `" ".join(words)`.

## Signatures
Map each shingle to an int with `x = zlib.crc32(shingle.encode("utf-8"))`. With $P = 2^{31} - 1$ and the lists `a`, `b` of equal length $n$, hash function $i$ is $h_i(x) = (a_i x + b_i) \bmod P$. The signature of a document is the list of $n$ ints $\min_{x} h_i(x)$ over its shingles.

## Bands
$n$ is divisible by `bands`; with $r = n / \text{bands}$, band $j$ is the slice `sig[j*r:(j+1)*r]`. Two documents are a **candidate pair** if at least one band is identical in both.

## Output
- `signatures`: list of signature lists, one per document.
- `candidates`: list of tuples `(p, q, estimate, jaccard)` for every candidate pair with `p < q`, sorted by `(p, q)`. `estimate` is the fraction of the $n$ positions where the two signatures are equal, and `jaccard` the exact Jaccard similarity $|S_p \cap S_q| / |S_p \cup S_q|$ of the shingle sets.
"""

STARTER = '''import zlib

def minhash_lsh(docs, k, a, b, bands):
    """Return (signatures, sorted list of (p, q, estimated_jaccard, true_jaccard))."""
    raise NotImplementedError
'''

SOLUTION = '''import zlib
from collections import defaultdict

P = 2**31 - 1

def shingles(doc, k):
    w = doc.lower().split()
    if len(w) < k:
        return {" ".join(w)}
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}

def minhash_lsh(docs, k, a, b, bands):
    sets = [shingles(d, k) for d in docs]
    sigs = []
    for s in sets:
        xs = [zlib.crc32(g.encode("utf-8")) for g in s]
        sigs.append([min((ai * x + bi) % P for x in xs) for ai, bi in zip(a, b)])
    r = len(a) // bands
    buckets = defaultdict(set)                       # (band index, band values) -> documents
    for d, sig in enumerate(sigs):
        for j in range(bands):
            buckets[j, tuple(sig[j * r:(j + 1) * r])].add(d)
    pairs = {(p, q) for docs_in in buckets.values() for p in docs_in for q in docs_in if p < q}
    out = []
    for p, q in sorted(pairs):
        est = sum(x == y for x, y in zip(sigs[p], sigs[q])) / len(a)
        jac = len(sets[p] & sets[q]) / len(sets[p] | sets[q])
        out.append((p, q, est, jac))
    return sigs, out
'''

HINTS = [
    'Build each document\'s shingle **set** first (duplicates inside a document don\'t matter). Python ints are unbounded, so `(a * x + b) % P` is exact.',
    'The signature has one entry per hash function: for function `i`, the minimum of `h_i(x)` over the document\'s shingle hashes.',
    'For the bands, put every document into a dict keyed by `(band index, tuple of that band\'s values)`. Any bucket with two or more documents yields candidate pairs. Include the band index in the key so equal values in different bands don\'t collide.',
]

import random as _random
_g = _random.Random(61)
_a = [_g.randrange(1, 2**31 - 1) for _ in range(12)]
_b = [_g.randrange(0, 2**31 - 1) for _ in range(12)]
_base = "the quick brown fox jumps over the lazy dog near the river bank today"
_docs = [_base, _base.replace("lazy", "sleepy"), "an entirely different sentence about tokenizers and data",
         _base.upper(), _base + " and then it rained", "the quick brown fox"]
TESTS = [
    (["a b c d", "a b c e", "x y z"], 2, [3, 5, 7, 11], [1, 2, 3, 4], 2),
    (["a rose is a rose", "A ROSE is a rose"], 1, [2, 3], [0, 0], 1),
    (["one"], 3, [5], [9], 1),
    (["solo word", "word solo"], 3, _a[:4], _b[:4], 4),
    (_docs, 3, _a, _b, 4),
    (_docs, 2, _a, _b, 6),
    (_docs, 5, _a, _b, 2),
]

EXPLANATION = r"""
## Why it works
For a random hash function, the shingle with the smallest hash is equally likely to be any element of $S_p \cup S_q$, and the two minima are equal exactly when that element lies in $S_p \cap S_q$. So each signature position agrees with probability $J(S_p, S_q)$, and the fraction of agreeing positions is an unbiased estimate of Jaccard similarity. Here the functions $a x + b \bmod P$ only approximate random permutations, which is the usual practical choice.

## Why bands
With $b$ bands of $r$ rows, a pair with similarity $s$ becomes a candidate with probability $1 - (1 - s^r)^b$. This is an S-shaped curve: pairs well above the threshold $\approx (1/b)^{1/r}$ are almost always caught and pairs well below it almost never are. You then check only the candidates, instead of all $O(N^2)$ pairs.

## Complexity
Signatures cost $O(N \cdot |S| \cdot n)$. Bucketing is $O(N \cdot n)$ and candidate pairs are typically far fewer than $N^2$.

## Common bugs
- Using Python's built-in `hash()`, which is randomized per process for strings, so signatures change between runs.
- Treating shingles as a list, so a repeated phrase counts twice.
- Bucketing on the band values without the band index.
- Normalizing inconsistently (test 2 shows upper and lower case collapse to the same shingles).

## Interview follow-ups
- How would you pick `bands` and rows for a target threshold of 0.8 with 128 hash functions?
- After finding duplicate clusters, which document do you keep, and why does deduplication improve LLM training? (Less memorization of repeated text and more diverse tokens per compute.)
"""
