META = dict(title='Train BPE merges, then encode new words', tags=['NLP', 'Language Models'], diff='Hard', runtime='python',
            lessons=['ieng:01.02'], related=['bpe-merge-step', 'tokenizer-training', 'subword-tokenization'])

STATEMENT = r"""
A byte-pair-encoding tokenizer has two phases. **Training** learns an ordered list of merges from a corpus. **Encoding** applies those merges to any word, including words never seen in training.

Write `bpe(word_counts, num_merges, test_words)` that returns `(merges, encoded)`.

## Symbols
A word is split into its characters followed by the end-of-word marker `"</w>"`, so `"low"` starts as `["l", "o", "w", "</w>"]`. The marker lets the tokenizer tell a word-final `"st"` from one in the middle of a word.

## Training
`word_counts` is a dict from word (non-empty string) to frequency. Repeat at most `num_merges` times:
1. Count every adjacent pair of symbols over all words, weighting each word by its frequency.
2. If there are no pairs, stop early. Otherwise choose the pair with the highest count; ties go to the smallest pair in tuple order.
3. Replace every occurrence of that pair in every word by the concatenated symbol, scanning left to right without overlaps.
4. Append the pair (a tuple of two strings) to `merges`.

## Encoding a test word
Split it into characters plus `"</w>"`. Then repeat: among the adjacent pairs of the current symbols, find the one that appears **earliest** in `merges`. If no adjacent pair is in `merges`, stop. Otherwise replace every occurrence of that pair, left to right without overlaps, and repeat. Characters never seen in training simply stay as single symbols.

## Output
- `merges`: list of tuples, in the order they were learned.
- `encoded`: list of symbol lists, one per entry of `test_words`.
"""

STARTER = '''def bpe(word_counts, num_merges, test_words):
    """Return (list of merge pairs in order, list of encoded symbol lists)."""
    raise NotImplementedError
'''

SOLUTION = '''from collections import Counter

def merge(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1]); i += 2      # non-overlapping, left to right
        else:
            out.append(symbols[i]); i += 1
    return out

def bpe(word_counts, num_merges, test_words):
    words = {w: list(w) + ["</w>"] for w in word_counts}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for w, sym in words.items():
            for p in zip(sym, sym[1:]):
                pairs[p] += word_counts[w]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        words = {w: merge(sym, best) for w, sym in words.items()}
        merges.append(best)

    rank = {p: i for i, p in enumerate(merges)}
    encoded = []
    for w in test_words:
        sym = list(w) + ["</w>"]
        while True:
            known = [p for p in zip(sym, sym[1:]) if p in rank]
            if not known:
                break
            sym = merge(sym, min(known, key=rank.get))           # lowest rank = learned first
        encoded.append(sym)
    return merges, encoded
'''

HINTS = [
    'Write one helper `merge(symbols, pair)` that does the left-to-right, non-overlapping replacement. Both phases use it.',
    'Training: count weighted pairs with a `Counter`, choose with `min(pairs, key=lambda p: (-pairs[p], p))`, merge every word, record the pair, repeat.',
    'Encoding: build `rank = {pair: index}` from the merge list. While some adjacent pair has a rank, merge the one with the smallest rank. This replays the training order on a new word.',
]

_corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
_corpus2 = {"token": 12, "tokens": 7, "tokenizer": 4, "talk": 3, "taken": 2, "open": 5, "opener": 2}
TESTS = [
    (_corpus, 3, ["lowest", "new"]),
    (_corpus, 10, ["newer", "wider", "low"]),
    ({"ab": 1}, 0, ["ab"]),
    ({"a": 4}, 5, ["a", "aa", "b"]),
    ({"aaaa": 3}, 3, ["aaaaa", "aaa"]),
    (_corpus2, 12, ["tokenize", "openers", "taken", "ozone"]),
    (_corpus2, 40, ["token", "tokens", "talking"]),
]

EXPLANATION = r"""
## Why encoding replays the merge order
Training merged pairs in a specific order, and later merges were learned on text where earlier merges had already been applied. Applying the lowest-ranked available merge first reproduces that history on new text, so a word from the training set encodes exactly as it ended up during training. Merging in a different order (for example longest match) can produce pieces that never co-occurred in training.

## What the marker does
With `"</w>"`, `est</w>` (a word ending) and `est` (inside *estimate*) are different symbols. Byte-level BPE in GPT-2 drops the marker and instead attaches the preceding **space** to the next word (`Ġlow`), which serves the same purpose while keeping the round trip to raw text exact.

## Complexity
Training is $O(M \cdot S)$ for $M$ merges over $S$ corpus symbols (recounting each time). Encoding a word of length $L$ costs up to $O(L)$ rounds of $O(L)$ work. Fast implementations keep a priority queue of adjacent pairs and a cache of encoded words.

## Common bugs
- Choosing which pair to merge in a test word by how often it occurs in that word, instead of by its merge rank.
- Merging overlapping occurrences: `aaa` with `("a", "a")` must give `["aa", "a"]`.
- Breaking count ties by dict order, so different Python versions or corpora orders give different merges.
- Forgetting to stop when no pairs remain (test 4 has only one-character words).

## Interview follow-ups
- Why does byte-level BPE never produce an unknown token? (The base alphabet is all 256 bytes.)
- How would you make training incremental? (After a merge, update only the pair counts that involve the merged positions, and keep the pairs in a heap.)
"""
