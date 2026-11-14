import numpy as np

META = dict(title='Frequency encoding learned on training data', tags=['Feature Engineering'], diff='Easy', runtime='python',
            related=['categorical-encoding', 'target-encoding', 'ordinal-encoding'])

STATEMENT = r"""
Frequency encoding replaces a category by how common it was in the **training** data. Write `frequency_encode(train, test, normalize=True)`:

- `train`, `test`: lists of category strings.
- Count each category in `train`. With `normalize=True`, the code is count divided by `len(train)`; otherwise the raw count.
- Encode both lists. Categories in `test` that never appear in `train` get 0.

Return a tuple `(train_codes, test_codes)` of lists (floats if normalised, ints otherwise).
"""

STARTER = '''def frequency_encode(train, test, normalize=True):
    """Return (train_codes, test_codes)."""
    raise NotImplementedError
'''

SOLUTION = '''from collections import Counter

def frequency_encode(train, test, normalize=True):
    cnt = Counter(train)
    n = len(train)
    f = (lambda c: cnt.get(c, 0) / n) if normalize else (lambda c: cnt.get(c, 0))
    return [f(c) for c in train], [f(c) for c in test]
'''

HINTS = [
    'Count categories in the training list only; `collections.Counter` does it in one line.',
    'Unseen test categories should map to 0, so use `.get(c, 0)`.',
    'Normalising divides by the number of training rows, not the number of distinct categories.',
]

TESTS = [
    (['a', 'b', 'a', 'c'], ['a', 'd']),
    (['a', 'b', 'a', 'c'], ['a', 'd'], False),
    (['x'] * 5 + ['y'] * 3 + ['z'] * 2, ['z', 'y', 'x', 'w']),
    (['only'], ['only', 'other'], False),
    (['p', 'q', 'r', 's'], []),
]

EXPLANATION = r"""
## What it captures
How common a category is often carries signal on its own: rare merchants are riskier, popular items are clicked more. Frequency encoding gives that in one numeric column, without the label leakage risk of target encoding.

## Limitations
Different categories with the same frequency get the same code, so the model cannot tell them apart. Frequencies also drift over time; recompute them on each training run and store them with the model.

## Why train only
Counting on train plus test leaks information about the test distribution into features, and at serving time you do not have future data to count.

## Interview follow-ups
- Would you use counts or log counts for a linear model?
"""
