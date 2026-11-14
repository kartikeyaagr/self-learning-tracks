import numpy as np

META = dict(title='Bernoulli naive Bayes log-likelihoods', tags=['Classification', 'Classic ML'], diff='Hard', runtime='numpy',
            related=['text-classification', 'gaussian-naive-bayes', 'bag-of-words'])

STATEMENT = r"""
Bernoulli naive Bayes models each binary feature as present or absent, independently given the class. Write `bernoulli_nb(X_train, y_train, X_test, alpha)`.

## Fit
For each class $c$ (in sorted order) with $n_c$ training rows:
- prior $\pi_c = n_c / n$;
- $\theta_{cj} = \dfrac{\text{count of rows of class } c \text{ with } x_j = 1 + \alpha}{n_c + 2\alpha}$ (Laplace smoothing).

## Predict
For each test row, the joint log-likelihood is
$$\log \pi_c + \sum_j \big[x_j \log\theta_{cj} + (1-x_j)\log(1-\theta_{cj})\big].$$
Note that **absent** features contribute too (unlike multinomial naive Bayes).

Return `(classes, jll, pred)`: the sorted class labels, the `(n_test, n_classes)` joint log-likelihood array, and predicted labels (the class with the largest value; the first one on ties).
"""

STARTER = '''import numpy as np

def bernoulli_nb(X_train, y_train, X_test, alpha):
    """Return (classes, joint_log_likelihood, predictions)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bernoulli_nb(X_train, y_train, X_test, alpha):
    X, y, T = (np.asarray(a, dtype=float) for a in (X_train, y_train, X_test))
    classes = np.unique(y)
    logp, logq, prior = [], [], []
    for c in classes:
        Xc = X[y == c]
        th = (Xc.sum(axis=0) + alpha) / (len(Xc) + 2 * alpha)
        logp.append(np.log(th)); logq.append(np.log1p(-th)); prior.append(np.log(len(Xc) / len(y)))
    logp, logq, prior = np.array(logp), np.array(logq), np.array(prior)
    jll = prior + T @ logp.T + (1 - T) @ logq.T
    return classes, jll, classes[np.argmax(jll, axis=1)]
'''

HINTS = [
    'Estimate one theta vector per class from the share of rows where each feature is 1, with alpha added to the count and 2 alpha to the denominator.',
    'Vectorise prediction: X_test @ log(theta).T handles present features, (1 - X_test) @ log(1 - theta).T handles absent ones.',
    'Add the log prior to every row, then take argmax across classes.',
]

_r = np.random.default_rng(21)
_X = (_r.random((30, 6)) < np.where(np.arange(30)[:, None] < 15, 0.8, 0.2)).astype(float)
_y = (np.arange(30) >= 15).astype(int)
TESTS = [
    (np.array([[1, 0], [1, 1], [0, 1], [0, 0]]), np.array([0, 0, 1, 1]), np.array([[1, 0], [0, 1]]), 1.0),
    (np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]]), np.array([1, 1, 2]), np.array([[0, 0, 0], [1, 1, 1]]), 0.5),
    (_X, _y, _X[:5], 1.0),
    (_X, _y, (_r.random((4, 6)) < 0.5).astype(float), 0.1),
    (np.array([[1], [1], [0]]), np.array([0, 1, 1]), np.array([[0], [1]]), 1.0),
]

EXPLANATION = r"""
## Bernoulli versus multinomial
Multinomial naive Bayes looks only at which words occur and how often. Bernoulli naive Bayes also counts the evidence from words that are absent: if "invoice" appears in 80% of a class's documents, a document without it is less likely to belong there. This helps with short texts and binary features, and hurts with long documents where absence is common and uninformative.

## Smoothing
Without $\alpha$, a feature never seen in a class gives $\theta = 0$ and $\log 0 = -\infty$, so one word can veto a class. Laplace smoothing adds $\alpha$ pseudo-counts to both outcomes.

## Why work in logs
Products of many probabilities underflow to zero; sums of logs do not.

## Complexity
Fit is $O(nd)$; prediction is one matrix product, $O(n_{\text{test}}\, d\, C)$.

## Interview follow-ups
- Why are naive Bayes probabilities usually poorly calibrated even when its accuracy is good?
"""
