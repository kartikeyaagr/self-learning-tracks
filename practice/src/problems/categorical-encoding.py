import numpy as np

META = dict(title='One-hot encoding with rare and unseen categories', tags=['Feature Engineering', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:11.12'], related=['missing-value-imputation', 'feature-scaling'])

STATEMENT = r"""
Most models need numbers, so a categorical column is expanded into one 0/1 indicator column per category. Two practical wrinkles: categories that appear only a handful of times in training aren't worth a column, and the test set can contain categories never seen in training.

Write `one_hot(train, test, min_count)` that learns the vocabulary from `train` and encodes both lists.

## Rules
1. The vocabulary is every distinct value in `train` that occurs **at least `min_count` times**, sorted ascending (normal string order).
2. Column $k$ of the output is 1 where the value equals `vocabulary[k]`, else 0.
3. A value not in the vocabulary (rare in training, or never seen) is encoded as an **all-zero** row.

## Input / output
- `train`: list of `n >= 1` strings; `test`: list of `m >= 0` strings; `min_count >= 1`.
- Return `(vocabulary, train_matrix, test_matrix)`: a Python list of strings, and int arrays of shapes `(n, K)` and `(m, K)` where $K$ = `len(vocabulary)` (which may be 0).
"""

STARTER = '''import numpy as np

def one_hot(train, test, min_count):
    """Return (sorted vocabulary of values with count >= min_count, train one-hot (n, K), test one-hot (m, K))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np
from collections import Counter

def one_hot(train, test, min_count):
    counts = Counter(train)
    vocab = sorted(v for v, c in counts.items() if c >= min_count)
    col = {v: k for k, v in enumerate(vocab)}

    def encode(values):
        M = np.zeros((len(values), len(vocab)), dtype=int)
        for i, v in enumerate(values):
            if v in col:                                    # rare and unseen values stay all-zero
                M[i, col[v]] = 1
        return M

    return vocab, encode(train), encode(test)
'''

HINTS = [
    'Count the training values first (`collections.Counter`), then keep and sort the ones that meet the threshold.',
    'Build a dict from category to column index so each lookup is O(1).',
    'Allocate an `(rows, K)` zero matrix with `dtype=int` and set a single 1 per row only when the value is in the dict. `np.zeros((m, 0))` is a valid shape for an empty vocabulary or empty test list.',
]

_r = np.random.default_rng(139)
_cities = ['oslo', 'lima', 'pune', 'kyiv', 'baku', 'nuuk']
_train = [_cities[i] for i in _r.choice(6, size=40, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])]
_test = [_cities[i] for i in _r.integers(0, 6, size=10)] + ['rome', 'oslo']
TESTS = [
    (['red', 'green', 'red', 'blue'], ['green', 'purple'], 1),
    (['a', 'b', 'a', 'c', 'a', 'b'], ['c', 'b', 'a'], 2),
    (['x', 'y'], ['y', 'y', 'x'], 1),
    (['x', 'y', 'z'], ['x', 'q'], 5),
    (['B', 'a', 'A', 'b'], ['a', 'B'], 1),
    (['same', 'same', 'same'], ['same', 'other'], 3),
    (_train, _test, 3),
]

EXPLANATION = r"""
## Why one-hot
Encoding categories as integers $0, 1, 2, \dots$ imposes an order and spacing that usually means nothing, and linear models would treat "category 2" as twice "category 1". One indicator per category lets the model learn an independent effect for each.

## Why a minimum count
A category seen twice in training gets a coefficient fitted on two rows: noisy, and a source of overfitting. Folding rare categories into the all-zero "baseline" row (or a dedicated "other" column) trades a little information for a lot of stability. It also bounds the width of the matrix.

## Unseen categories
Production data always contains new values. Crashing on them, or silently mapping them to some existing column, are both bugs. All-zero means "none of the known categories", which every downstream model can handle.

## Complexity
$O(n)$ to count, $O(K\log K)$ to sort, $O((n + m))$ to encode into a dense matrix. For large vocabularies use a sparse matrix: the dense one is $O((n+m)K)$ memory for $n + m$ ones.

## Common bugs
- Building the vocabulary from train **and** test (leakage, and the column set changes between runs).
- Using `pd.get_dummies` separately on train and test, which produces different columns.
- Case handling: `'A'` and `'a'` are different strings here (test 5), and uppercase sorts first.

## Interview follow-ups
- With 10 million user IDs, one-hot is infeasible. What are your options? (Feature hashing, target encoding with out-of-fold statistics, or learned embeddings.)
- Why drop one column for an unregularised linear model with an intercept? (The $K$ indicators sum to 1, which is collinear with the intercept.)
"""
