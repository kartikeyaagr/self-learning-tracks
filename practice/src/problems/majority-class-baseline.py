import numpy as np

META = dict(title='Majority-class and stratified baselines', tags=['Classification', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['popularity-baseline-leave-last-out', 'log-loss-binary-cross-entropy', 'classification-report'])

STATEMENT = r"""
Before trusting a classifier, compare it with trivial baselines. Write `baselines(y_train, y_test)` for integer labels that returns a dict:

- `"majority_class"`: the most frequent training label (smallest label on ties);
- `"majority_accuracy"`: accuracy on `y_test` of always predicting that label;
- `"prior_log_loss"`: mean log loss (natural log) on `y_test` of predicting the training class frequencies for every row. Classes seen in test but not in train get probability `1e-15` before renormalising all class probabilities to sum to 1;
- `"expected_random_accuracy"`: expected accuracy of guessing labels at random in proportion to the training frequencies, $\sum_c p^{\text{train}}_c\, p^{\text{test}}_c$.

All values except the class are floats.
"""

STARTER = '''import numpy as np

def baselines(y_train, y_test):
    """Return the four baseline numbers described in the statement."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def baselines(y_train, y_test):
    ytr, yte = np.asarray(y_train), np.asarray(y_test)
    classes = np.union1d(ytr, yte)
    ptr = np.array([(ytr == c).mean() for c in classes])
    pte = np.array([(yte == c).mean() for c in classes])
    cls, cnt = np.unique(ytr, return_counts=True)
    maj = int(cls[np.argmax(cnt)])
    q = np.where(ptr == 0, 1e-15, ptr)
    q = q / q.sum()
    idx = np.searchsorted(classes, yte)
    return {"majority_class": maj, "majority_accuracy": float((yte == maj).mean()),
            "prior_log_loss": float(-np.log(q[idx]).mean()),
            "expected_random_accuracy": float((ptr * pte).sum())}
'''

HINTS = [
    '`np.unique(..., return_counts=True)` plus `argmax` gives the majority class; argmax returns the first (smallest) class on ties.',
    'The prior model predicts the same distribution for every row, so its log loss is the average of -log q[y] over test rows.',
    'Random guessing in proportion to training frequencies is correct with probability sum over classes of p_train * p_test.',
]

TESTS = [
    (np.array([0, 0, 0, 1]), np.array([0, 1, 0, 0])),
    (np.array([1, 2, 2, 1]), np.array([1, 1, 2])),
    (np.array([0, 0, 1]), np.array([2, 0])),
    (np.array([3, 3, 3, 3]), np.array([3, 3])),
    (np.array([0] * 90 + [1] * 10), np.array([0] * 95 + [1] * 5)),
]

EXPLANATION = r"""
## Why baselines
A model with 95% accuracy sounds good until you learn that 95% of test rows are negative. The majority baseline sets the floor for accuracy; the prior log loss (the label entropy) sets the floor for log loss. A useful model must beat both, by a margin larger than noise.

## The prior model
Predicting class frequencies is the best constant probabilistic prediction, and its log loss equals the cross-entropy between the test label distribution and the training one. If train and test priors differ, this number shows it immediately.

## Unseen classes
A class that never appears in training gets a near-zero probability, which makes the log loss very large. That is the honest answer: the model would be confidently wrong.

## Interview follow-ups
- What baseline would you use for a ranking or recommendation task? (Most popular items.)
"""
