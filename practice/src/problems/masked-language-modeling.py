import numpy as np

META = dict(title='BERT-style 80/10/10 token corruption', tags=['Language Models', 'NLP'], diff='Easy', runtime='numpy',
            lessons=['aieng:03.02'], related=['mutation-trap', 'token-classification', 'next-sentence-prediction'])

STATEMENT = r"""
Masked language modelling hides some input tokens and trains the model to recover them. BERT does not always show `[MASK]` at the chosen positions: of the chosen tokens, 80% become `[MASK]`, 10% become a random token, and 10% are left unchanged.

Write `bert_corrupt(ids, special, u_select, u_action, random_ids, mask_id, p=0.15, ignore=-100)` that returns `(inputs, labels)`. The randomness is passed in, so the result is fully determined.

## Inputs
All arrays have the same shape as `ids` (any shape, typically `(B, T)`).
- `ids`: int array of token ids.
- `special`: bool array, `True` at tokens that may never be chosen (`[CLS]`, `[SEP]`, padding).
- `u_select`, `u_action`: float arrays of uniform numbers in $[0, 1)$.
- `random_ids`: int array, the replacement used where a random token is needed.
- `mask_id`, `p`, `ignore`: scalars.

## Rules
1. A position is **chosen** if `u_select < p` and it is not special.
2. At chosen positions: if `u_action < 0.8` the input becomes `mask_id`; else if `u_action < 0.9` it becomes `random_ids` at that position; otherwise it keeps its original id.
3. `labels` holds the original id at chosen positions and `ignore` everywhere else.

Return two int arrays shaped like `ids`, and leave `ids` unchanged.
"""

STARTER = '''import numpy as np

def bert_corrupt(ids, special, u_select, u_action, random_ids, mask_id, p=0.15, ignore=-100):
    """Return (corrupted inputs, labels)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bert_corrupt(ids, special, u_select, u_action, random_ids, mask_id, p=0.15, ignore=-100):
    ids = np.asarray(ids)
    chosen = (np.asarray(u_select) < p) & ~np.asarray(special, dtype=bool)
    to_mask = chosen & (u_action < 0.8)
    to_random = chosen & (u_action >= 0.8) & (u_action < 0.9)    # the remaining 10% keep their token
    inputs = np.where(to_mask, mask_id, np.where(to_random, random_ids, ids))   # new array: ids untouched
    labels = np.where(chosen, ids, ignore)
    return inputs, labels
'''

HINTS = [
    'Turn every rule into a boolean array first: `chosen`, then `to_mask` and `to_random` as subsets of `chosen`.',
    'The last 10% needs no mask of its own: wherever neither replacement applies, the input is simply the original id.',
    'Nested `np.where` builds the inputs in one expression and never writes into `ids`. The labels depend only on `chosen`, not on which of the three actions happened.',
]

_r = np.random.default_rng(41)
_ids = _r.integers(5, 1000, size=(3, 10))
_ids[:, 0] = 101
_ids[:, -1] = 102
_sp = np.zeros((3, 10), dtype=bool)
_sp[:, [0, -1]] = True
TESTS = [
    (np.array([101, 7, 8, 9, 102]), np.array([True, False, False, False, True]),
     np.array([0.0, 0.1, 0.5, 0.05, 0.0]), np.array([0.5, 0.3, 0.3, 0.95, 0.5]), np.array([1, 2, 3, 4, 5]), 103),
    (np.array([[11, 12, 13]]), np.zeros((1, 3), dtype=bool), np.array([[0.1, 0.1, 0.1]]),
     np.array([[0.79, 0.8, 0.9]]), np.array([[50, 51, 52]]), 103, 0.15, -1),
    (np.array([4, 5]), np.array([True, True]), np.zeros(2), np.zeros(2), np.array([9, 9]), 103),
    (np.array([4, 5, 6]), np.zeros(3, dtype=bool), np.array([0.15, 0.2, 0.99]), np.zeros(3), np.array([9, 9, 9]), 103),
    (np.array([4, 5, 6]), np.zeros(3, dtype=bool), np.array([0.3, 0.2, 0.99]), np.array([0.85, 0.5, 0.0]), np.array([9, 8, 7]), 0, 0.5),
    (_ids, _sp, _r.random((3, 10)), _r.random((3, 10)), _r.integers(5, 1000, size=(3, 10)), 103),
    (_ids, _sp, _r.random((3, 10)), _r.random((3, 10)), _r.integers(5, 1000, size=(3, 10)), 103, 0.6, -1),
]

EXPLANATION = r"""
## Why 80/10/10
If the model only ever saw `[MASK]` at positions it is graded on, it could learn to produce good representations only for `[MASK]` tokens, and fine-tuning text never contains `[MASK]`. Replacing some chosen tokens with random words and leaving some unchanged forces the model to keep a correct representation of **every** token, because any token might be a target. The random replacements are rare enough (1.5% of all tokens) not to damage language understanding much.

## Why labels don't depend on the action
All three kinds of chosen position are predicted, including the unchanged ones. Unchosen positions get `ignore` so cross-entropy skips them (PyTorch's `ignore_index=-100`).

## Boundaries
Comparisons are strict on the left: `u_select = 0.15` is **not** chosen with `p = 0.15` (test 4), and `u_action = 0.8` exactly goes to the random branch while `0.9` keeps the token (test 2).

## Complexity
A few vectorized passes: $O(N)$ time and memory.

## Common bugs
- Choosing `[CLS]`/`[SEP]`/padding positions.
- Drawing the 80/10/10 split over all tokens instead of only the chosen ones.
- Writing into `ids` in place, so the labels come out corrupted (see the related problem on aliasing).

## Interview follow-ups
- Why do later models (RoBERTa) re-sample the mask every epoch instead of fixing it once during preprocessing? (Dynamic masking shows each sentence with many different masks.)
- What do whole-word masking and span masking change, and why do they make the task harder in a useful way?
"""
