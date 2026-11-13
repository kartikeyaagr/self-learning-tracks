import numpy as np

META = dict(title='Mask tokens without corrupting labels', tags=['NLP', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.05'], related=['fancy-indexing', 'basic-indexing'])

STATEMENT = r"""
Masked language modelling trains a model to recover hidden tokens. From a batch of token ids you build two arrays:

- `inputs`: the ids, with the chosen positions replaced by a special `mask_id`.
- `labels`: the **original** id at each chosen position, and `ignore` everywhere else (so the loss skips those positions).

Write `mask_tokens(ids, mask, mask_id, ignore=-100)` that returns `(inputs, labels)`.

## Input
- `ids`: integer array of any shape (typically `(B, T)`).
- `mask`: boolean array of the same shape; `True` marks the positions to hide.
- `mask_id`, `ignore`: Python ints.

## Output
Two integer arrays, both the same shape as `ids`. The caller's `ids` array must not be changed.

## The trap
`inputs = ids` does not copy anything; both names point to the same buffer. If you then write `inputs[mask] = mask_id` and build `labels` from `ids` afterwards, the labels contain `mask_id` instead of the real tokens.
"""

STARTER = '''import numpy as np

def mask_tokens(ids, mask, mask_id, ignore=-100):
    """Return (inputs, labels): ids with masked positions set to mask_id, and original ids at masked positions else ignore."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mask_tokens(ids, mask, mask_id, ignore=-100):
    ids, mask = np.asarray(ids), np.asarray(mask, dtype=bool)
    labels = np.where(mask, ids, ignore)      # read the originals before anything is overwritten
    inputs = ids.copy()                       # an explicit copy; plain assignment would alias ids
    inputs[mask] = mask_id
    return inputs, labels
'''

HINTS = [
    'In NumPy, `b = a` and most slices `b = a[1:]` share memory with `a`. Writing through `b` changes `a`. Which of your arrays must stay untouched?',
    'Build `labels` first, from the untouched ids: `np.where(mask, ids, ignore)` creates a brand-new array.',
    'Then `inputs = ids.copy()` and `inputs[mask] = mask_id`. Without `.copy()`, the masked write would also land in `ids`.',
]

_r = np.random.default_rng(29)
_ids = _r.integers(5, 1000, size=(3, 8))
TESTS = [
    (np.array([[11, 12, 13, 14]]), np.array([[False, True, False, True]]), 4),
    (np.array([7, 8, 9]), np.array([True, True, True]), 0, -1),
    (np.array([[1, 2], [3, 4]]), np.zeros((2, 2), dtype=bool), 103),
    (np.array([5]), np.array([True]), 5),
    (np.array([[2**40, 3], [4, 2**40]]), np.array([[True, False], [False, True]]), 1),
    (_ids, _r.random(size=(3, 8)) < 0.15, 103),
    (_ids, _r.random(size=(3, 8)) < 0.5, 103, -1),
]

EXPLANATION = r"""
## Why it works
`np.where(mask, ids, ignore)` allocates a new array, so `labels` captures the original tokens before any write happens. `ids.copy()` gives `inputs` its own buffer, so the boolean assignment `inputs[mask] = mask_id` cannot leak into `ids` or `labels`.

## Views vs copies in NumPy
- **Views** (shared memory): plain assignment `b = a`, basic slices `a[1:3]`, `a[::2]`, `reshape` (when possible), `transpose`, `.T`, `ravel` (when possible).
- **Copies** (new memory): fancy indexing `a[[0, 2]]`, boolean indexing `a[mask]`, arithmetic results, `np.where`, `.copy()`, `flatten`.
- When unsure, check `np.shares_memory(a, b)`.

Note the asymmetry: **reading** `a[mask]` returns a copy, but **assigning** `a[mask] = v` writes into `a` in place.

## Complexity
$O(N)$ time and memory for two outputs of $N$ elements.

## Common bugs
- `inputs = ids; inputs[mask] = mask_id`, which mutates the caller's batch. The next epoch then sees ids that are already masked.
- Building labels after masking, so they hold `mask_id`.
- `labels = ids.copy(); labels[~mask] = ignore` is correct, but only if it runs before the masking write.

## Interview follow-ups
- BERT masks 15% of tokens, then replaces 80% of those with `[MASK]`, 10% with a random token and leaves 10% unchanged. Why? (So the model cannot learn that only `[MASK]` positions matter, and so fine-tuning, which never shows `[MASK]`, looks like pre-training.)
- When is mutating in place the right choice? (When memory is tight and you own the buffer, for example `x *= scale` inside a training loop. Name it clearly or take an `out=` argument.)
"""
