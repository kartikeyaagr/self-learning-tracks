import numpy as np

META = dict(title='Shuffled mini-batch generator', tags=['Training', 'Data Processing'], diff='Medium', runtime='numpy',
            related=['batch-vs-stochastic-vs-mini-batch-gd', 'balanced-dataloader', 'mini-batch-training-loop'])

STATEMENT = r"""
Write `minibatches(X, y, batch_size, epochs, seed, drop_last)` that returns the list of mini-batches a training loop would see.

For each epoch:
1. Draw a permutation with a generator created **once** as `rng = np.random.default_rng(seed)`: `perm = rng.permutation(n)` at the start of every epoch.
2. Walk through `perm` in order, `batch_size` indices at a time.
3. If `drop_last` is true, skip a final batch smaller than `batch_size`.

Return a list of `(X_batch, y_batch)` tuples across all epochs, in order. `X` has shape `(n, d)`, `y` shape `(n,)`.
"""

STARTER = '''import numpy as np

def minibatches(X, y, batch_size, epochs, seed, drop_last):
    """Return a list of (X_batch, y_batch) for all epochs."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def minibatches(X, y, batch_size, epochs, seed, drop_last):
    X, y = np.asarray(X), np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if drop_last and len(idx) < batch_size:
                break
            out.append((X[idx], y[idx]))
    return out
'''

HINTS = [
    'Shuffle indices, not the arrays themselves, then index X and y with the same slice of the permutation so rows stay paired.',
    'Create the generator once, outside the epoch loop. Re-seeding every epoch would repeat the same order each time.',
    'With drop_last, the only batch that can be short is the last one of each epoch.',
]

_X = np.arange(20, dtype=float).reshape(10, 2)
_y = np.arange(10)
TESTS = [
    (_X, _y, 4, 1, 0, False),
    (_X, _y, 4, 1, 0, True),
    (_X, _y, 3, 2, 42, False),
    (_X, _y, 10, 3, 7, True),
    (_X[:5], _y[:5], 2, 2, 1, True),
    (_X, _y, 12, 1, 3, False),
]

EXPLANATION = r"""
## Why shuffle every epoch
Stochastic gradient descent assumes each batch is a fresh random sample. A fixed order makes successive updates correlated (for example, all of one class at the end of the file), which slows and biases training. A new permutation per epoch also gives every example the same number of visits.

## Why drop the last batch
A tiny final batch has a noisy gradient and, with batch normalisation, unstable statistics. Dropping it costs at most `batch_size - 1` examples per epoch, and they are different examples each epoch because of the shuffle.

## Reproducibility
Seeding one generator at the start makes the whole sequence of epochs reproducible. Reseeding inside the loop is a common bug that silently repeats the same order.

## Complexity
$O(n)$ per epoch for the permutation and the gathers.

## Interview follow-ups
- How do distributed data loaders make sure each worker sees a different shard of each epoch?
- When would you sample with replacement instead of permuting?
"""
