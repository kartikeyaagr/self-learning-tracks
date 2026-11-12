import numpy as np

META = dict(title="Bayes' rule for a screening test", tags=['Probability'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:10.01'], related=['conditional-probability', 'bayesian-updating'])

STATEMENT = r"""
A screening test for a condition $D$ is described by three numbers:

- the **prevalence** $P(D)$,
- the **sensitivity** $P(+ \mid D)$, the chance a true case tests positive,
- the **specificity** $P(- \mid \bar D)$, the chance a healthy person tests negative.

Bayes' rule turns these into what a patient actually wants to know:

$$P(D \mid +) = \frac{P(+ \mid D)\,P(D)}{P(+ \mid D)\,P(D) + P(+ \mid \bar D)\,P(\bar D)}$$

Write `screening(prevalence, sensitivity, specificity)` that returns the tuple `(p_pos, ppv, npv)` as Python floats:

- `p_pos` $= P(+)$, the fraction of everyone who tests positive,
- `ppv` $= P(D \mid +)$, the positive predictive value,
- `npv` $= P(\bar D \mid -)$, the negative predictive value.

## Requirements
- All three inputs are floats in $[0, 1]$. They are chosen so that $P(+) > 0$ and $P(-) > 0$.
"""

STARTER = '''def screening(prevalence, sensitivity, specificity):
    """Return (P(+), P(D | +), P(not D | -)) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''def screening(prevalence, sensitivity, specificity):
    tp = sensitivity * prevalence                    # P(+ and D)
    fp = (1 - specificity) * (1 - prevalence)        # P(+ and not D)
    tn = specificity * (1 - prevalence)              # P(- and not D)
    fn = (1 - sensitivity) * prevalence              # P(- and D)
    p_pos = tp + fp
    return float(p_pos), float(tp / p_pos), float(tn / (tn + fn))
'''

HINTS = [
    'Split the population into four groups: true positives, false positives, true negatives and false negatives. Each is a product of two of the inputs.',
    '$P(+)$ is the sum of the two positive groups. That sum is the denominator of Bayes\' rule (the law of total probability).',
    'PPV is true positives over all positives. NPV is true negatives over all negatives, $P(-) = 1 - P(+)$.',
]

TESTS = [
    (0.01, 0.99, 0.95),
    (0.5, 0.8, 0.8),
    (0.001, 1.0, 0.999),
    (0.3, 0.0, 0.9),
    (0.2, 1.0, 1.0),
    (0.9, 0.6, 0.7),
    (1e-6, 0.999, 0.9999),
]

EXPLANATION = r"""
## Why a 99% sensitive test can mislead
With prevalence 1%, sensitivity 99% and specificity 95%, out of 10,000 people about 99 are true positives and $0.05 \times 9900 = 495$ are false positives. So $P(D \mid +) = 99 / 594 \approx 17\%$. When the condition is rare, the healthy group is so large that even a small false-positive rate swamps the true cases. This is base-rate neglect, and it applies to fraud flags and anomaly alerts too.

## The general form
For hypotheses $H_1, \dots, H_K$ and evidence $E$:
$$P(H_k \mid E) = \frac{P(E \mid H_k)\,P(H_k)}{\sum_j P(E \mid H_j)\,P(H_j)}$$
Posterior is proportional to likelihood times prior, and the denominator just normalises.

## Complexity
$O(1)$.

## Common bugs
- Confusing $P(+ \mid D)$ with $P(D \mid +)$ (the prosecutor's fallacy).
- Using the specificity as the false-positive rate. The false-positive rate is $1 - \text{specificity}$.

## Interview follow-ups
- A patient tests positive twice on independent tests. What is the PPV now? (Use the first posterior as the new prior.)
- How do PPV and NPV relate to precision and to the classifier's decision threshold? (PPV is precision. Raising the threshold trades sensitivity for specificity, so PPV usually rises and NPV falls.)
"""
