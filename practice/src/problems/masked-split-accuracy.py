import numpy as np

META = dict(title='Accuracy on one split of a graph', tags=['Classification', 'Metrics & Evaluation'], diff='Easy', runtime='numpy')

STATEMENT = r"""
In node classification the whole graph goes through the model at once, and boolean masks say which nodes belong to the training, validation and test splits.

Write `split_accuracy(logits, labels, mask)`:

- Predictions are the argmax of each row of `logits (n, C)` (lowest class on ties).
- Accuracy is computed over nodes where `mask` is true only.
- Return a dict with `accuracy` (0.0 if the mask is empty), `correct` and `n` (ints).
"""

STARTER = '''import numpy as np

def split_accuracy(logits, labels, mask):
    """Accuracy of argmax predictions on the masked nodes only."""
    pred = np.asarray(logits).argmax(1)
    ok = pred == np.asarray(labels)
    return {"accuracy": float(ok.mean()), "correct": int(ok.sum()), "n": len(ok)}
'''

SOLUTION = '''import numpy as np

def split_accuracy(logits, labels, mask):
    m = np.asarray(mask, bool)
    pred = np.asarray(logits, float).argmax(1)
    ok = (pred == np.asarray(labels))[m]
    n = int(m.sum())
    return {"accuracy": float(ok.mean()) if n else 0.0, "correct": int(ok.sum()), "n": n}
'''

HINTS = [
    'Select with the mask after computing predictions; the model saw every node, but only the masked ones are scored.',
    'An empty mask should give 0.0, not NaN.',
]

_r = np.random.default_rng(0)
_L = _r.normal(size=(20, 3))
_y = _r.integers(0, 3, 20)

TESTS = [
    ([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0]], [0, 1, 1], [True, True, True]),
    ([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0]], [0, 1, 1], [False, False, True]),
    ([[1.0, 1.0]], [1], [True]),
    (_L, _y, np.arange(20) < 12),
    (_L, _y, np.zeros(20, bool)),
]

EXPLANATION = r"""
## Transductive splits
In a citation graph every paper, labelled or not, is a node in the same graph, and the GCN uses all of them to compute every node's representation. The test papers' features and links are visible during training; only their labels are hidden. The mask is what keeps the labels out of the loss and out of the score.

## Why this matters for honest numbers
Standard benchmarks such as Cora use a small fixed training set (20 labelled nodes per class). Results move noticeably with a different random split, so reporting the mean over several splits and seeds is more reliable than one number on the fixed split.

## Interview follow-ups
- How would you evaluate a model that must classify papers added to the graph after training (the inductive setting)?
"""
