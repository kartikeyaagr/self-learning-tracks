META = dict(title='Stable exact deduplication of a text corpus', tags=['Data Processing', 'NLP'], diff='Easy',
            runtime='python', lessons=['air-fanout-v1:10.03', 'air-fanout-v1:11.12'],
            related=['text-deduplication-minhash', 'hashed-dataset-split', 'temperature-based-data-mixtures'])

STATEMENT = r"""
Web crawls contain the same page many times: mirrors, boilerplate, reposts. Before near-duplicate detection, pipelines remove **exact** duplicates after a light normalisation. The pass must be **stable**: the first occurrence survives and the corpus order is preserved, so reruns produce identical shards.

Write `exact_dedup(docs)` that returns the tuple `(kept, dup_of)`.

## Normalisation
The key of a document is: strip leading and trailing whitespace, collapse every run of whitespace (as `str.split()` sees it) to a single space, and lower-case. Two documents are duplicates iff their keys are equal.

## Rules
- A document whose key is empty is dropped and gets `dup_of = -1`.
- Otherwise the first document with a given key is **kept** (`dup_of[i] = i`); every later one gets `dup_of[i] =` the index of that first document.

## Output
- `kept`: list of the kept indices, in increasing order.
- `dup_of`: list of ints, one per input document.

Run in one pass over the corpus, using a hash of the key (`hashlib.sha256` of the UTF-8 bytes) as the lookup key, the way a pipeline that can't keep whole documents in memory would.
"""

STARTER = '''import hashlib

def exact_dedup(docs):
    """Return (kept indices, dup_of list)."""
    raise NotImplementedError
'''

SOLUTION = '''import hashlib

def exact_dedup(docs):
    first = {}                                 # digest -> index of first occurrence
    kept, dup_of = [], []
    for i, doc in enumerate(docs):
        key = " ".join(doc.split()).lower()    # split() with no argument strips and collapses all whitespace
        if not key:
            dup_of.append(-1)
            continue
        h = hashlib.sha256(key.encode("utf-8")).digest()   # 32 bytes per document instead of the text
        if h in first:
            dup_of.append(first[h])
        else:
            first[h] = i
            kept.append(i)
            dup_of.append(i)
    return kept, dup_of
'''

HINTS = [
    '`" ".join(doc.split())` does the whole whitespace normalisation: `split()` without arguments drops leading/trailing whitespace and splits on any run of spaces, tabs or newlines.',
    'Keep a dict from digest to the index of the first document that produced it. Check it before inserting so the first occurrence wins.',
    'Handle empty keys before hashing: they are neither kept nor duplicates of anything.',
]

TESTS = [
    (['Hello world', 'hello   WORLD', 'goodbye'],),
    (['a', '', '  ', 'A', 'b', 'a '],),
    (['The cat\tsat.\n', 'the cat sat.', 'The cat sat!', 'THE  CAT  SAT.', 'cat sat.'],),
    ([],),
    (['x'] * 5,),
    (['Ünïcode', 'ünïcode', 'unicode', 'ÜNÏCODE', 'straße', 'STRASSE'],),
    ([f'doc {i % 7} body' if i % 3 else f'  Doc {i % 7}\nbody  ' for i in range(30)],),
]

EXPLANATION = r"""
## Why stable, why hashed
Stability (keep the first occurrence, preserve order) makes the pass deterministic: rerunning on the same input, or on shards processed in the same order, gives byte-identical output, which matters for reproducing a training run. Hashing the key means memory scales with the number of **distinct** documents times 32 bytes, not with their text; at billions of documents the digest set is sharded by hash prefix across machines. A 256-bit hash has a negligible collision probability at any realistic corpus size.

## How much normalisation
Lower-casing and whitespace collapse catch the most common trivial variants without merging documents that genuinely differ. Stronger normalisation (stripping punctuation, Unicode NFKC folding) catches more but risks false merges; anything fuzzier belongs in the near-duplicate stage (MinHash). Note that `str.lower()` doesn't fold `"ß"` into `"ss"` (test 6): `str.casefold()` would, a deliberate design choice either way.

## Complexity
$O(\text{total characters})$ time; $O(\text{distinct documents})$ memory.

## Common bugs
- `doc.strip().lower()` only: internal whitespace runs then differ (test 3).
- Keeping the **last** occurrence by overwriting the dict entry.
- Treating whitespace-only documents as one kept "empty" document.

## Interview follow-ups
- Documents arrive from 1,000 parallel workers. How do you make the dedup result independent of which worker finished first?
- Exact dedup at the document level leaves repeated paragraphs (cookie banners, licence text). How would you dedup at the line or paragraph level without destroying documents?
"""
