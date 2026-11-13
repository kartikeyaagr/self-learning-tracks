import numpy as np

META = dict(title='Pad and truncate a batch of token ids', tags=['NLP', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['ieng:01.02'], related=['create-arrays-from-lists', 'vocabulary-building', 'next-sentence-prediction'])

STATEMENT = r"""
Sequences in a batch have different lengths, but a model needs one rectangular array. Write `pad_batch(seqs, max_len=None, pad_id=0, padding="right", truncation="right")` that returns `(ids, mask)`.

## Inputs
- `seqs`: a non-empty list of lists of ints (a list may be empty).
- `max_len`: `None` or an int $\ge 0$.
- `pad_id`: the id written into padded positions.
- `padding`: `"right"` puts padding after the tokens, `"left"` before them.
- `truncation`: for a sequence longer than the target length, `"right"` keeps its **first** tokens, `"left"` keeps its **last** tokens.

## Target length
- `max_len is None`: $L$ is the length of the longest sequence (no truncation happens).
- Otherwise $L$ = `max_len` exactly: shorter sequences are padded up to it and longer ones truncated.

## Output
- `ids`: `np.int64` array of shape `(len(seqs), L)`.
- `mask`: boolean array of the same shape, `True` at real tokens and `False` at padding.
"""

STARTER = '''import numpy as np

def pad_batch(seqs, max_len=None, pad_id=0, padding="right", truncation="right"):
    """Return (ids int64 (B, L), mask bool (B, L))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pad_batch(seqs, max_len=None, pad_id=0, padding="right", truncation="right"):
    L = max(len(s) for s in seqs) if max_len is None else max_len
    ids = np.full((len(seqs), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        if len(s) > L:
            s = s[:L] if truncation == "right" else s[len(s) - L:]   # not s[-L:]: that breaks when L == 0
        n = len(s)
        cols = slice(0, n) if padding == "right" else slice(L - n, L)
        ids[i, cols] = s
        mask[i, cols] = True
    return ids, mask
'''

HINTS = [
    'Start from an array filled with `pad_id` of the final shape, and an all-False mask. Then copy each (possibly truncated) sequence into its row.',
    'Right truncation keeps `s[:L]`; left truncation keeps the last `L` tokens. Right padding writes into columns `0:n`; left padding into `L-n:L`.',
    'Careful with `s[-L:]` for left truncation: when `L == 0` it is `s[0:]`, the whole list. Use `s[len(s) - L:]` instead.',
]

_r = np.random.default_rng(3)
_seqs = [_r.integers(1, 500, size=k).tolist() for k in (7, 2, 12, 0, 5)]
TESTS = [
    ([[5, 6, 7], [8]],),
    ([[1, 2, 3, 4, 5], [9, 9]], 3, 0, "left", "left"),
    ([[], []],),
    ([[4, 4], [], [7]], None, -1, "left"),
    ([[1, 2, 3]], 0),
    ([[1, 2, 3], [4, 5, 6, 7, 8, 9]], 4, 0, "right", "left"),
    (_seqs, 8, 3),
    (_seqs, None, 0, "left", "right"),
]

EXPLANATION = r"""
## Why it works
Pre-filling with `pad_id` means only real tokens have to be written. The mask is built in the same step, so it can never disagree with the ids. Truncation happens before placement, which keeps the two choices (which tokens survive, and where they go) independent.

## Which side to pad?
- Encoders (BERT) usually pad on the right; the attention mask hides the padding.
- Decoder-only models generating in a batch pad on the **left**, so every row's last real token sits in the final column, where the next token is appended.
- Left truncation keeps the most recent context, which is what a chat model wants when a conversation runs too long.

## Complexity
$O(B \cdot L)$ to allocate and fill the arrays.

## Common bugs
- `s[-L:]` when `L == 0` returns the whole sequence.
- Building the mask from `ids != pad_id`. That breaks as soon as a real token has the pad id (test 4 pads with `-1`; a real tokenizer's pad id can also appear in the data).
- Returning Python lists, or an int32 array on some platforms, when the caller expects `int64`.

## Interview follow-ups
- Why is padding to the longest sequence in the batch (dynamic padding) faster than padding everything to the model's maximum length? How does sorting by length into buckets help further?
- How do "packed" sequences avoid padding altogether, and what must the attention mask do so packed documents don't attend to each other?
"""
