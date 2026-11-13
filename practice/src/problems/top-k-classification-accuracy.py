import numpy as np

META = dict(title='Top-k accuracy with a tie rule', tags=['Metrics & Evaluation', 'Computer Vision'], diff='Easy', runtime='numpy',
            lessons=[], related=['precision-recall-at-k', 'clip-cosine-retrieval', 'mean-iou-for-segmentation'])

STATEMENT = r"""
ImageNet results are reported as **top-1** and **top-5** accuracy: a prediction counts as correct if the true class is among the `k` highest-scoring classes.

Sorting with `argsort` makes ties arbitrary, so this problem uses a rank-based definition that needs no sort:

$$\text{correct}_k(i) \iff \#\{\,c : s_{i,c} > s_{i, y_i}\,\} < k$$

The true class is in the top `k` if **fewer than `k` classes score strictly higher** than it. Classes tied with the true class do not count against it.

Write `topk_accuracy(scores, labels, ks)`.

## Input
- `scores`: float array `(N, C)` (logits or probabilities), `N >= 1`.
- `labels`: int array `(N,)` in `0..C-1`.
- `ks`: list of ints, each `>= 1` (values above `C` are allowed and give accuracy 1).

## Output
A list of Python floats, one per entry of `ks` in the same order: the fraction of the `N` examples that are correct for that `k`.
"""

STARTER = '''import numpy as np

def topk_accuracy(scores, labels, ks):
    """Return [accuracy@k for k in ks], where correct means fewer than k classes score strictly higher."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def topk_accuracy(scores, labels, ks):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    true = s[np.arange(len(y)), y][:, None]           # (N, 1) score of the true class
    rank = (s > true).sum(axis=1)                     # classes strictly ahead of it
    return [float((rank < k).mean()) for k in ks]
'''

HINTS = [
    'You don\'t need to sort. For each example, what single number decides whether the true class is in the top `k` for every `k` at once?',
    'Gather the true-class score of each row with `s[np.arange(N), labels]`, then count how many scores in the row are strictly greater.',
    'With `rank = (s > true[:, None]).sum(1)`, accuracy at `k` is `(rank < k).mean()`.',
]

_r = np.random.default_rng(40)
TESTS = [
    (np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]), np.array([2, 0]), [1, 2]),
    (np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 2.0]]), np.array([1, 2]), [1, 2, 3]),
    (np.array([[3.0, 2.0, 1.0, 0.0]]), np.array([3]), [1, 3, 4, 10]),
    (_r.normal(size=(50, 10)), _r.integers(0, 10, size=50), [1, 5]),
    (_r.normal(size=(20, 3)), _r.integers(0, 3, size=20), [3, 1, 2]),
    (np.round(_r.normal(size=(30, 6)), 0), _r.integers(0, 6, size=30), [1, 2, 3]),
]

EXPLANATION = r"""
## A rank instead of a sort
Top-$k$ correctness only depends on the true class's rank: the number of classes beating it. Computing that rank once per example answers every $k$, takes $O(NC)$ instead of $O(NC\log C)$, and gives a precise tie rule. Test 6 rounds the scores to integers so ties are common.

## About ties
This rule is optimistic: a true class tied with the top score counts as top-1 correct. Frameworks that use `topk` pick tied entries in an implementation-defined order, so they can disagree on the same scores. With real-valued logits exact ties are rare, but they do appear with quantised models or when scores saturate (e.g. probabilities that underflow to 0).

## Complexity
$O(NC)$ time, $O(NC)$ temporary memory for the comparison (loop over chunks of rows for huge $N$).

## Common bugs
- `argsort` and checking membership in the last `k` columns: slower, and ties are resolved arbitrarily.
- Using `>=` instead of `>`: then the true class counts itself and every example looks one rank worse.
- Averaging per-class accuracies when the metric is per-example.

## Interview follow-ups
- Compute per-class top-1 accuracy (balanced accuracy) with `np.bincount`.
- Why is top-5 the headline ImageNet metric alongside top-1? (Many images contain several valid objects or near-synonym classes.)
"""
