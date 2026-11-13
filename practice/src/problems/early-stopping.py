META = dict(title='Early stopping with patience', tags=['Training'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:10.04'],
            related=['dropout', 'learning-rate-sweep'])

STATEMENT = r"""
Early stopping watches the validation loss and ends training once it has stopped improving for a while, then restores the best checkpoint.

Write `early_stopping(val_losses, patience, min_delta=0.0)` that replays a list of per-epoch validation losses and returns the tuple `(stop_epoch, best_epoch)` (0-based ints).

## Rules
Process epochs in order, starting with `best = inf`, `wait = 0`, `best_epoch = -1`:
1. Epoch `e` **improves** if `val_losses[e] < best - min_delta` (strictly). Then set `best = val_losses[e]`, `best_epoch = e`, `wait = 0`.
2. Otherwise `wait += 1`. If `wait` reaches `patience`, stop: return `(e, best_epoch)`.
3. If the list ends first, return `(len(val_losses) - 1, best_epoch)`.

## Inputs
- `val_losses`: non-empty list of floats. A `nan` loss never counts as an improvement.
- `patience`: int `>= 1`; `min_delta`: float `>= 0`.
- If no epoch ever improves (e.g. every loss is `nan`), `best_epoch` stays `-1`.
"""

STARTER = '''def early_stopping(val_losses, patience, min_delta=0.0):
    """Return (stop_epoch, best_epoch) as 0-based ints."""
    raise NotImplementedError
'''

SOLUTION = '''def early_stopping(val_losses, patience, min_delta=0.0):
    best, best_epoch, wait = float('inf'), -1, 0
    for e, loss in enumerate(val_losses):
        if loss < best - min_delta:          # False for nan, so nan just burns patience
            best, best_epoch, wait = loss, e, 0
        else:
            wait += 1
            if wait >= patience:
                return e, best_epoch
    return len(val_losses) - 1, best_epoch
'''

HINTS = [
    'You only need three variables: the best loss so far, the epoch it happened, and how many epochs since then.',
    'An epoch that ties the best (or improves by less than `min_delta`) is **not** an improvement, so it increments the counter.',
    'Comparisons with `nan` are always `False` in Python, so `loss < best - min_delta` already treats `nan` as no improvement.',
]

TESTS = [
    ([1.0, 0.8, 0.7, 0.72, 0.71, 0.73], 2),
    ([1.0, 0.9, 0.8, 0.7], 2),
    ([1.0, 0.75, 0.5, 0.375], 1, 0.25),
    ([0.5, 0.5, 0.5], 2),
    ([float('nan'), float('nan'), float('nan')], 5),
    ([2.0, float('nan'), 1.5, float('nan'), float('nan')], 2),
    ([3.0, 2.0, 2.5, 2.4, 1.0, 1.1, 1.2, 1.3], 3),
    ([0.3], 1),
    ([1.0, 0.99, 0.98, 0.97, 0.5], 3, 0.05),
]

EXPLANATION = r"""
## Why it works as regularisation
Training loss keeps falling, but past some point the model fits noise and validation loss rises. Stopping at the validation minimum limits how far the weights travel from their initialisation; for linear models with gradient descent, the number of steps acts much like the inverse of an L2 penalty strength.

## Patience and min_delta
Validation loss is noisy, so a single bad epoch shouldn't end training. `patience` tolerates that many non-improving epochs in a row. `min_delta` stops tiny, noise-level gains from resetting the counter (test 9: 0.99, 0.98, 0.97 are all within 0.05 of 1.0, so patience runs out before the 0.5 arrives). Test 3 checks the boundary: $0.75$ is not strictly less than $1.0 - 0.25$.

## Complexity
$O(E)$ time for $E$ epochs and $O(1)$ memory.

## Common bugs
- Using `<=` instead of `<`, so plateaus reset the counter forever (test 4).
- Not resetting `wait` on improvement.
- Returning the stop epoch as the model to use: you should restore the checkpoint from `best_epoch`.
- Letting `nan` become the best loss (`min` with `nan` behaves unpredictably).

## Interview follow-ups
- Is it valid to report the validation score at `best_epoch` as your model's performance? (No: you selected on it, so it's optimistically biased. Report a held-out test set.)
- With a learning-rate schedule, why might early stopping trigger too early? (Loss often plateaus right before a decay step and then drops; schedules and patience must be tuned together.)
"""
