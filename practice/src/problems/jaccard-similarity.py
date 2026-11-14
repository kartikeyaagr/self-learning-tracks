META = dict(title='Jaccard similarity for sets and shingles', tags=['NLP', 'Recommender Systems'], diff='Easy', runtime='python')

STATEMENT = r"""
Write `jaccard(a, b, k=None)`.

- If `k` is `None`, `a` and `b` are iterables of items; return $|A \cap B| / |A \cup B|$ over their sets.
- If `k` is an integer, `a` and `b` are strings: lowercase them, collapse runs of whitespace to one space, strip the ends, and compare their sets of character `k`-shingles (all substrings of length `k`). A string shorter than `k` has one shingle, the string itself, unless it is empty.
- Two empty sets have similarity `1.0`.
"""

STARTER = '''def jaccard(a, b, k=None):
    """Jaccard similarity of two sets, or of two strings' k-shingles."""
    return len(set(a) & set(b)) / len(set(a) | set(b))
'''

SOLUTION = '''def _shingles(s, k):
    s = " ".join(s.lower().split())
    if not s:
        return set()
    if len(s) < k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}

def jaccard(a, b, k=None):
    A, B = (set(a), set(b)) if k is None else (_shingles(a, k), _shingles(b, k))
    union = A | B
    return 1.0 if not union else len(A & B) / len(union)
'''

HINTS = [
    "' '.join(s.split()) collapses whitespace and strips the ends in one step.",
    'Handle the empty union before dividing.',
]

TESTS = [
    ([1, 2, 3], [2, 3, 4], None),
    ([], [], None),
    (["a", "a", "b"], ["b"], None),
    ("The cat sat", "the  cat sat ", 3),
    ("abcde", "abxde", 2),
    ("ab", "abc", 3),
    ("", "abc", 2),
    ("night", "nacht", 2),
]

EXPLANATION = r"""
## Where it is used
Jaccard on shingles is the standard near-duplicate test for web pages and training corpora; Jaccard on item sets compares users by what they bought. It ignores counts and order, which is what you want for "do these share content".

## Scaling it
Comparing all pairs is quadratic. MinHash estimates Jaccard from short signatures, because the probability that two sets share a minimum hash equals their Jaccard similarity, and locality-sensitive hashing then finds candidate pairs without all-pairs comparison.

## Choosing k
Small $k$ makes unrelated texts look similar (they share common letter pairs); large $k$ makes one typo change many shingles. Word-level shingles of 5 or so are common for deduplication of documents.

## Interview follow-ups
- Why is Jaccard a poor similarity for users with very different activity levels?
"""
