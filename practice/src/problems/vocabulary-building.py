META = dict(title='Build a vocabulary with specials, min count and a size cap', tags=['NLP', 'Data Processing'], diff='Medium', runtime='python',
            lessons=['ieng:01.02'], related=['bag-of-words', 'padding-and-truncation', 'subword-tokenization'])

STATEMENT = r"""
A word-level model needs a fixed vocabulary: a list of tokens whose positions are the ids. Write `build_vocab(corpus, min_freq=1, max_size=None, specials=("<pad>", "<unk>"))` that returns `(itos, encoded)`.

## Inputs
- `corpus`: a list of sentences, each a list of token strings.
- `min_freq`: a token is kept only if it occurs at least this many times in the corpus.
- `max_size`: `None` (no cap), or the maximum length of `itos`, **including** the specials. It is always at least `len(specials)`.
- `specials`: tokens that always come first, in this order. `"<unk>"` is always one of them.

## Building `itos`
1. Start with the specials.
2. Count every corpus token that is not a special.
3. Keep tokens with count $\ge$ `min_freq`, ordered by count (highest first), ties broken by the token string in ascending order.
4. Append them until `itos` has `max_size` entries.

## Encoding
`encoded` is the corpus with every token replaced by its id (its index in `itos`). Tokens not in the vocabulary map to the id of `"<unk>"`. A corpus token that equals a special maps to that special's id.
"""

STARTER = '''def build_vocab(corpus, min_freq=1, max_size=None, specials=("<pad>", "<unk>")):
    """Return (itos list, encoded corpus as lists of ids)."""
    raise NotImplementedError
'''

SOLUTION = '''from collections import Counter

def build_vocab(corpus, min_freq=1, max_size=None, specials=("<pad>", "<unk>")):
    itos = list(specials)
    counts = Counter(t for sent in corpus for t in sent if t not in specials)
    ranked = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
    room = None if max_size is None else max_size - len(itos)
    itos += ranked[:room]                          # slicing with None keeps everything
    stoi = {t: i for i, t in enumerate(itos)}
    unk = stoi["<unk>"]
    encoded = [[stoi.get(t, unk) for t in sent] for sent in corpus]
    return itos, encoded
'''

HINTS = [
    'Count with `collections.Counter`, skipping tokens that are already specials so they are never added twice.',
    'Sort the surviving tokens with `key=lambda t: (-counts[t], t)` for "most frequent first, then alphabetical", and cut the list so specials plus kept tokens fit in `max_size`.',
    'Build the reverse map `stoi = {t: i for i, t in enumerate(itos)}` once, then encode with `stoi.get(t, stoi["<unk>"])`.',
]

_r_words = ["the", "a", "cat", "dog", "sat", "ran", "on", "mat", "log", "fast", "slow", "big"]
_rng_seq = [[_r_words[(i * 7 + j * 3 + i * j) % len(_r_words)] for j in range(3 + i % 4)] for i in range(9)]
TESTS = [
    ([["the", "cat", "sat"], ["the", "dog", "sat"]],),
    ([["b", "a", "b", "c", "a", "b"]], 2),
    ([],),
    ([[], ["x"]], 1, 2),
    ([["hi", "<unk>", "hi"], ["<pad>", "yo"]], 1, None, ("<pad>", "<unk>", "<bos>")),
    ([["z", "y", "x", "z", "y", "z"]], 1, 4, ("<unk>",)),
    (_rng_seq, 2, 8),
    (_rng_seq, 1, None, ("<pad>", "<unk>", "<s>", "</s>")),
]

EXPLANATION = r"""
## Why it works
Specials go first so their ids are fixed no matter what corpus you train on: id 0 is padding, which many embedding layers treat specially (`padding_idx`). Sorting by `(-count, token)` makes the order fully deterministic, so two runs on the same corpus always give the same ids. The size cap then keeps the most useful tokens, and everything else falls back to `<unk>`.

## Complexity
Counting is $O(N)$ for $N$ corpus tokens, sorting is $O(V \log V)$ for $V$ distinct tokens, and encoding is $O(N)$ with a dict.

## Common bugs
- Adding a special again because it also appears in the text, which gives one token two ids.
- Sorting by count only. Python's sort is stable, so ties then follow the corpus order, which is a different (and fragile) specification.
- Forgetting that `max_size` includes the specials.
- Building the vocabulary on train + test data, which leaks test information into the model.

## Interview follow-ups
- How do you choose `min_freq` and `max_size`? (Look at coverage: the fraction of corpus tokens that are not `<unk>` as a function of vocabulary size. It follows Zipf's law, so a small vocabulary already covers most tokens.)
- Why did subword tokenizers replace this? (No `<unk>` at all, and rare words are split into pieces that share statistics with common ones.)
"""
