META = dict(title='One byte-pair-encoding merge', tags=['NLP', 'Language Models'], diff='Medium', runtime='python',
            lessons=['ieng:01.02'], related=['sinusoidal-positional-encoding'])

STATEMENT = r"""
BPE builds a subword vocabulary by repeatedly merging the most frequent pair of adjacent symbols. One training step:

1. Count every adjacent pair `(a, b)` across all words, weighting each word by its frequency.
2. Pick the most frequent pair. Break ties by the pair that sorts first (`("a", "b") < ("a", "c")`).
3. Replace every occurrence of that pair, scanning left to right without overlaps, with the joined symbol `a + b`.

Write `bpe_merge(words, counts)` and return `(pair, new_words)`.

## Inputs
- `words`: list of words, each a list of symbol strings, e.g. `[["l","o","w"], ["l","o","w","e","r"]]`.
- `counts`: list of ints, the frequency of each word (same length as `words`).

## Output
- `pair`: the merged pair as a tuple of two strings, or `None` if no word has two symbols.
- `new_words`: the words after the merge (a new list; unchanged if `pair` is `None`).

Non-overlapping means `["a","a","a"]` merged on `("a","a")` becomes `["aa","a"]`.
"""

STARTER = '''def bpe_merge(words, counts):
    """Return (best_pair or None, new_words)."""
    raise NotImplementedError
'''

SOLUTION = '''from collections import Counter

def bpe_merge(words, counts):
    pairs = Counter()
    for w, c in zip(words, counts):
        for a, b in zip(w, w[1:]):
            pairs[(a, b)] += c
    if not pairs:
        return None, [list(w) for w in words]
    best = min(pairs, key=lambda p: (-pairs[p], p))     # highest count, then smallest pair
    out = []
    for w in words:
        merged, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                merged.append(w[i] + w[i + 1]); i += 2      # consume both: no overlapping merges
            else:
                merged.append(w[i]); i += 1
        out.append(merged)
    return best, out
'''

HINTS = [
    '`zip(w, w[1:])` yields every adjacent pair of a word. Add the word\'s count for each one.',
    'For "highest count, then alphabetical", use `min(pairs, key=lambda p: (-pairs[p], p))`.',
    'Merge with an index-based while loop. On a match, emit the joined symbol and advance by 2, which is what prevents overlapping merges.',
]

TESTS = [
    ([['l', 'o', 'w'], ['l', 'o', 'w', 'e', 'r'], ['n', 'e', 'w']], [5, 2, 6]),
    ([['a', 'a', 'a']], [1]),
    ([['a', 'b'], ['b', 'a']], [1, 1]),
    ([['x']], [3]),
    ([['h', 'u', 'g'], ['p', 'u', 'g'], ['p', 'u', 'n'], ['b', 'u', 'n'], ['h', 'u', 'g', 's']], [10, 5, 12, 4, 5]),
    ([['t', 'h', 'e'], ['t', 'h', 'i', 's'], ['a', 't', 'h']], [3, 1, 2]),
    ([['a', 'a', 'a', 'a']], [2]),
]

EXPLANATION = r"""
## Why BPE
A word-level vocabulary can't spell unseen words, and a character-level one makes sequences long. BPE starts from characters (or bytes, in GPT-2) and greedily adds the merges that compress the corpus most, so common words become single tokens while rare words split into reusable pieces.

## Determinism matters
Training is a sequence of these steps, and every later merge depends on earlier ones. Without a fixed tie-break, two runs on the same corpus can produce different vocabularies and incompatible token IDs. Real tokenizers fix the order and save the ranked merge list.

## Efficiency
Recounting all pairs each step is $O(\text{corpus})$ per merge. Production trainers keep a pair-count index and update only the pairs that touch the merged positions, plus a heap to find the max.

## Interview follow-ups
- How do you **apply** a trained merge list to new text? (Repeatedly merge the adjacent pair with the lowest rank.)
- Byte-level BPE vs SentencePiece unigram: what are the trade-offs?
"""
