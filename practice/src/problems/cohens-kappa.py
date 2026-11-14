import numpy as np

META = dict(title="Cohen's kappa, unweighted and quadratic-weighted", tags=['Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['normalized-confusion-matrix', 'classification-report', 'majority-class-baseline'])

STATEMENT = r"""
Cohen's kappa measures agreement between two raters (or a model and labels) beyond what chance would give. Write `kappa(a, b, labels, weights=None)`.

1. Build the $k \times k$ confusion matrix $O$ of counts (rows from `a`, columns from `b`, in the order of `labels`), and divide by the total so it sums to 1.
2. The chance matrix is $E = r\,c^\top$, the outer product of the row and column marginals of $O$.
3. Disagreement weights $w_{ij}$: unweighted (`None`) uses $w_{ij} = 0$ if $i = j$ else 1; `"quadratic"` uses $w_{ij} = (i - j)^2 / (k-1)^2$.
4. $\kappa = 1 - \dfrac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}$.

If the denominator is 0 (for example, both raters always use one identical class), return 1.0 when the numerator is also 0, else 0.0.
"""

STARTER = '''import numpy as np

def kappa(a, b, labels, weights=None):
    """Return Cohen's kappa as a float."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def kappa(a, b, labels, weights=None):
    pos = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    O = np.zeros((k, k))
    for x, y in zip(list(a), list(b)):
        O[pos[x], pos[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    i, j = np.indices((k, k))
    W = (i - j) ** 2 / (k - 1) ** 2 if weights == "quadratic" else (i != j).astype(float)
    num, den = (W * O).sum(), (W * E).sum()
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return float(1 - num / den)
'''

HINTS = [
    'Observed agreement comes from the normalised confusion matrix; chance agreement from the product of the two raters\' label frequencies.',
    'Written with weights, kappa compares weighted disagreement observed with weighted disagreement expected by chance.',
    'For quadratic weights, a disagreement of two levels costs four times as much as a disagreement of one level.',
]

TESTS = [
    (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), [0, 1]),
    (np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), [0, 1]),
    (np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 2, 2]), [0, 1, 2]),
    (np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 2, 2]), [0, 1, 2], "quadratic"),
    (np.array([1, 1, 1]), np.array([1, 1, 1]), [0, 1]),
    (np.array([1, 2, 3, 4, 5, 3]), np.array([1, 3, 3, 5, 4, 2]), [1, 2, 3, 4, 5], "quadratic"),
]

EXPLANATION = r"""
## Why correct for chance
If 90% of items are "fine", two raters who both say "fine" at random 90% of the time agree 82% of the time without looking. Accuracy-style agreement hides this; kappa subtracts it. A kappa of 0 means agreement at chance level, 1 means perfect agreement, and negative values mean systematic disagreement.

## Weighted kappa
For ordered labels (ratings 1 to 5), confusing 4 with 5 is milder than confusing 1 with 5. Quadratic weights encode that, and quadratic-weighted kappa is a common metric for ordinal prediction tasks.

## Use in labelling
Inter-annotator kappa tells you whether a labelling guideline is clear. Low agreement means the labels are noisy, which caps how well any model can do.

## Interview follow-ups
- Why can kappa be low even when raw agreement is high? (Very imbalanced classes.)
"""
