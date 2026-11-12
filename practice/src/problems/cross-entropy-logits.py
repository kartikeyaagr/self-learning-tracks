import numpy as np

META = dict(title='Cross-entropy from logits', tags=['Loss Functions', 'Classification'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:02.02'], related=['softmax-rows', 'softmax-jacobian'])

STATEMENT = r"""
Classifiers output raw logits $z \in \mathbb{R}^{N \times C}$. The cross-entropy loss for sample $n$ with true class $y_n$ is

$$\ell_n = -\log \mathrm{softmax}(z_n)_{y_n} = \mathrm{logsumexp}(z_n) - z_{n, y_n}$$

Write `cross_entropy(logits, labels, ignore_index=-100)` that returns the **mean** loss over samples whose label isn't `ignore_index`.

## Requirements
- `logits`: float array `(N, C)`; `labels`: int array `(N,)` with values in `[0, C)` or equal to `ignore_index`.
- Use the log-sum-exp form so huge logits don't overflow, and don't take `log` of a softmax that could be 0.
- If every label is ignored, return `0.0`.
- Return a Python `float`.
"""

STARTER = '''import numpy as np

def cross_entropy(logits, labels, ignore_index=-100):
    """Mean cross-entropy over non-ignored samples. Returns a float."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cross_entropy(logits, labels, ignore_index=-100):
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels)
    keep = labels != ignore_index
    if not keep.any():
        return 0.0
    z, y = logits[keep], labels[keep]
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))      # stable log-sum-exp per row
    return float(np.mean(lse - z[np.arange(len(y)), y]))
'''

HINTS = [
    'Drop the ignored rows first with a boolean mask, then compute the loss as if they were never there.',
    '$\\log\\sum_j e^{z_j} = m + \\log\\sum_j e^{z_j - m}$ with $m = \\max_j z_j$. This is the same trick as stable softmax.',
    'Pick each row\'s true-class logit with fancy indexing: `z[np.arange(len(y)), y]`.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([[2.0, 1.0, 0.1]]), np.array([0])),
    (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([0, 1])),
    (np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([1, 1])),
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([1, -100, 0])),
    (np.array([[1.0, 2.0]]), np.array([-100])),
    (_r.normal(size=(8, 5)) * 3, _r.integers(0, 5, size=8)),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 7]), 7),
]

EXPLANATION = r"""
## Why logits, not probabilities
$\log(\mathrm{softmax}(z)_y)$ computed as two steps underflows. A logit gap of ~750 already makes the softmax entry exactly 0.0 in float64, and $\log 0 = -\infty$. Writing it as $z_y - \mathrm{logsumexp}(z)$ never forms the tiny probability.

## Gradient (the reason this loss is everywhere)
$\partial \ell / \partial z = \mathrm{softmax}(z) - \mathrm{onehot}(y)$. It's bounded, cheap, and doesn't vanish when the model is confidently wrong. That's why frameworks ship `CrossEntropyLoss` as one fused op that takes logits.

## ignore_index
Padding tokens in language modelling get label `-100` so they contribute neither loss nor gradient. The mean is over **kept** samples. Dividing by $N$ instead would shrink the loss in proportion to the padding.

## Interview follow-ups
- Add label smoothing $\varepsilon$: the target becomes $(1-\varepsilon)\,\mathrm{onehot} + \varepsilon / C$. How does the loss change?
- Why is cross-entropy preferred over MSE on probabilities for classification?
"""
