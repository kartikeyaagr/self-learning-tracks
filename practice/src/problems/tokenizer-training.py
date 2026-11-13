META = dict(title='Train a WordPiece vocabulary', tags=['NLP', 'Language Models'], diff='Hard', runtime='python',
            lessons=['ieng:01.02'], related=['subword-tokenization', 'byte-pair-encoding', 'bpe-merge-step'])

STATEMENT = r"""
WordPiece training looks like BPE, but it does not merge the most **frequent** pair. It merges the pair whose parts occur together much more often than their individual frequencies would suggest:
$$\text{score}(a, b) = \frac{\text{count}(a, b)}{\text{count}(a) \cdot \text{count}(b)}$$

Write `train_wordpiece(word_counts, vocab_size)` that returns `(vocab, splits)`.

## Setup
- `word_counts`: dict from word to its corpus frequency (positive int). Words are non-empty.
- Each word starts split into characters; every character except the first gets the prefix `"##"`. So `"hug"` starts as `["h", "##u", "##g"]`.
- The initial `vocab` is the sorted list (ascending string order) of all distinct initial symbols.

## One training step
1. With the current splits, compute `count(s)` for every symbol `s` (the number of occurrences over all words, each word weighted by its frequency) and `count(a, b)` for every adjacent pair, weighted the same way.
2. If there are no pairs, stop. Otherwise pick the pair with the highest score; break ties by the smallest pair in tuple order `(a, b)`. Compare scores exactly, for example with `fractions.Fraction`.
3. The merged symbol is `a + b[2:]` if `b` starts with `"##"`, else `a + b`. For example `"h"` + `"##u"` gives `"hu"`, `"##u"` + `"##g"` gives `"##ug"`.
4. In every word, replace each occurrence of `a` followed by `b` with the merged symbol, scanning left to right without overlaps.
5. Append the merged symbol to `vocab` if it is not already there.

Repeat steps until `len(vocab) >= vocab_size` or there are no pairs.

## Output
- `vocab`: the list described above.
- `splits`: dict from each word to its final list of symbols.
"""

STARTER = '''from fractions import Fraction

def train_wordpiece(word_counts, vocab_size):
    """Return (vocab list, {word: final split})."""
    raise NotImplementedError
'''

SOLUTION = '''from collections import Counter
from fractions import Fraction

def train_wordpiece(word_counts, vocab_size):
    splits = {w: [w[0]] + ["##" + c for c in w[1:]] for w in word_counts}
    vocab = sorted({s for sp in splits.values() for s in sp})
    while len(vocab) < vocab_size:
        sym, pair = Counter(), Counter()
        for w, sp in splits.items():
            c = word_counts[w]
            for s in sp:
                sym[s] += c
            for ab in zip(sp, sp[1:]):
                pair[ab] += c
        if not pair:
            break
        # exact scores, then smallest pair on ties
        a, b = min(pair, key=lambda ab: (-Fraction(pair[ab], sym[ab[0]] * sym[ab[1]]), ab))
        new = a + (b[2:] if b.startswith("##") else b)
        for w, sp in splits.items():
            out, i = [], 0
            while i < len(sp):
                if i + 1 < len(sp) and sp[i] == a and sp[i + 1] == b:
                    out.append(new); i += 2
                else:
                    out.append(sp[i]); i += 1
            splits[w] = out
        if new not in vocab:
            vocab.append(new)
    return vocab, splits
'''

HINTS = [
    'Initialize the splits with `[w[0]] + ["##" + c for c in w[1:]]`, and the vocabulary as the sorted set of those symbols.',
    'Each step recounts symbols and adjacent pairs with two `Counter`s, weighting by word frequency. Score with `Fraction(count_ab, count_a * count_b)` and choose with `min(pairs, key=lambda p: (-score(p), p))`.',
    'The merge is the same left-to-right, non-overlapping scan as in BPE. Strip the `##` from the second symbol when gluing. Stop when the vocabulary is big enough or no word has two symbols left.',
]

TESTS = [
    ({"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}, 14),
    ({"aaab": 2, "ab": 5}, 6),
    ({"x": 3, "y": 1}, 10),
    ({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 5),
    ({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 22),
    ({"tokenize": 4, "tokenizer": 3, "token": 9, "tokens": 6, "organize": 2}, 26),
    ({"abab": 1, "ba": 1}, 100),
]

EXPLANATION = r"""
## Why the score
Frequency-based merging (BPE) would happily merge very common letters like `##e` + `##s` just because both are common. Dividing by the parts' individual counts measures association instead, like pointwise mutual information: the pair wins when its parts rarely appear apart. Merges therefore tend to build meaningful units first. This is the score popularised by the Hugging Face description of WordPiece training; the original Google implementation is not public, so treat it as one well-defined variant.

## Exact ties
Scores are ratios of integers. Two different pairs can have mathematically equal scores, for example $2/(4 \cdot 2)$ and $1/(2 \cdot 2)$. Floating division happens to round equal rationals identically, but comparing `Fraction`s removes any doubt, and the tuple tie-break makes the whole run deterministic.

## Complexity
Each step is $O(\text{total symbols})$ to recount and merge, for at most `vocab_size` steps. Production trainers update counts incrementally, as with BPE.

## Common bugs
- Not weighting by word frequency, or counting pairs once per word type.
- Keeping the `##` of the second piece, giving `h##u`.
- Picking the most frequent pair (that's BPE).
- Stopping after a fixed number of merges instead of at the target size; merges that recreate an existing symbol don't grow the vocabulary.

## Interview follow-ups
- At inference time WordPiece encodes with greedy longest-match, not by replaying merges. Why can that give a different split than the training-time one?
- How does the Unigram LM tokenizer (SentencePiece) go the other way, starting from a big vocabulary and pruning it?
"""
