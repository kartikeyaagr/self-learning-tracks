META = dict(title='WordPiece greedy longest-match encoding', tags=['NLP', 'Language Models'], diff='Medium', runtime='python',
            lessons=['ieng:01.02'], related=['tokenizer-training', 'byte-pair-encoding', 'bpe-merge-step'])

STATEMENT = r"""
WordPiece (used by BERT) splits each word into the longest pieces found in its vocabulary. Pieces that continue a word are stored with a `##` prefix, so `playing` might become `play`, `##ing`.

Write `wordpiece_encode(words, vocab, unk="[UNK]", max_chars=100)` that returns one flat list of tokens for all the words, in order.

## Algorithm, per word
1. If the word has more than `max_chars` characters, emit `unk` for the whole word.
2. Otherwise set `start = 0`. While `start < len(word)`:
   - Find the **largest** `end` with `start < end <= len(word)` such that the piece `word[start:end]` is in `vocab`, where the piece is prefixed with `"##"` whenever `start > 0`.
   - If there is none, the **whole word** becomes a single `unk` (pieces already found for this word are discarded).
   - Otherwise append the piece and set `start = end`.

## Inputs
- `words`: a list of non-empty strings (already split on whitespace and punctuation).
- `vocab`: a list or set of token strings.
"""

STARTER = '''def wordpiece_encode(words, vocab, unk="[UNK]", max_chars=100):
    """Return the flat list of WordPiece tokens for all words."""
    raise NotImplementedError
'''

SOLUTION = '''def wordpiece_encode(words, vocab, unk="[UNK]", max_chars=100):
    vocab = set(vocab)
    out = []
    for word in words:
        if len(word) > max_chars:
            out.append(unk)
            continue
        pieces, start = [], 0
        while start < len(word):
            for end in range(len(word), start, -1):          # longest candidate first
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in vocab:
                    pieces.append(piece)
                    start = end
                    break
            else:                                            # no piece fits: the whole word is unknown
                pieces = [unk]
                break
        out.extend(pieces)
    return out
'''

HINTS = [
    'Work one word at a time. From the current start position, try the longest remaining substring first and shrink it until you find a vocabulary entry.',
    'Only the first piece of a word is looked up bare. Every later piece is looked up as `"##" + piece`.',
    "Python's `for ... else` is handy here: the `else` runs when the loop finishes without `break`, which is exactly the case where nothing matched and the word becomes `unk`.",
]

_V = ["un", "##aff", "##able", "aff", "##a", "##b", "##le", "play", "##ing", "##er", "##s", "p", "##l", "##a", "##y",
      "the", "a", "##n", "run", "##ning", "##ne", "[UNK]", "x"]
TESTS = [
    (["unaffable", "playing"], _V),
    (["players", "plays", "plaything"], _V),
    ([], _V),
    (["x", "xx", "a"], _V),
    (["running", "runner", "run"], _V, "<unk>"),
    (["playing", "the"], _V, "[UNK]", 5),
    (["play", "pla", "unable", "unlike"], _V),
    (["abcd", "abc", "a"], ["ab", "a", "##bcd", "##c", "##d"]),
]

EXPLANATION = r"""
## Why greedy longest-match
Taking the longest known prefix keeps sequences short and reuses whole units like `play` before falling back to letters. The `##` marker records that a piece continues a word, so decoding can glue pieces back together without guessing where the spaces were.

Greedy is not always the segmentation with the fewest pieces. In test 8, `abcd` becomes `ab`, `##c`, `##d` (3 pieces), although `a`, `##bcd` (2 pieces) was available: grabbing `ab` first ruled it out. WordPiece accepts that; it is fast and deterministic.

## All or nothing
If any suffix cannot be matched, the whole word maps to `[UNK]`. Emitting the pieces found so far plus `[UNK]` would give the model a partial word that never appeared in training. That is why BERT vocabularies include every single character with and without `##`, so this almost never happens.

## Complexity
For a word of length $L$, each start position tries up to $L$ end positions and slicing/hashing costs up to $L$, so $O(L^3)$ in the worst case. Words are short, and `max_chars` bounds the damage. Production tokenizers use a trie to make each word $O(L)$ to $O(L^2)$.

## Interview follow-ups
- How does this differ from applying BPE merges? (BPE replays merges in training order; WordPiece only needs the final vocabulary and matches greedily.)
- How would you map a span of WordPiece tokens back to the original characters?
"""
