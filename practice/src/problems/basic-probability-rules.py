import numpy as np

META = dict(title='Event probabilities on a finite sample space', tags=['Probability'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:10.01'], related=['conditional-probability', 'independence-testing'])

STATEMENT = r"""
A finite sample space has outcomes $0, 1, \dots, n-1$ with probabilities $p_0, \dots, p_{n-1}$. An **event** is a set of outcomes, and its probability is the sum of its outcomes' probabilities.

Given two events $A$ and $B$, write `event_probs(p, A, B)` that returns the tuple

$$\big(P(A),\; P(B),\; P(A \cup B),\; P(A \cap B),\; P(A^c)\big)$$

as five Python floats, where $A^c$ is the complement of $A$.

## Requirements
- `p`: 1-D float array of non-negative probabilities that sum to 1.
- `A`, `B`: Python lists of outcome indices in `[0, n)`. They may be empty, may overlap, and may contain the same index more than once. A repeated index still counts once: events are **sets**.
- Use boolean masks over the outcomes rather than looping over pairs.
"""

STARTER = '''import numpy as np

def event_probs(p, A, B):
    """Return (P(A), P(B), P(A or B), P(A and B), P(not A)) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def event_probs(p, A, B):
    p = np.asarray(p, dtype=float)
    a = np.zeros(len(p), dtype=bool)
    b = np.zeros(len(p), dtype=bool)
    a[A] = True                       # duplicates just set the same entry twice
    b[B] = True
    P = lambda mask: float(p[mask].sum())
    return P(a), P(b), P(a | b), P(a & b), P(~a)
'''

HINTS = [
    'Turn each event into a boolean mask of length $n$. Then union, intersection and complement are `|`, `&` and `~`.',
    '`mask[A] = True` works for an empty list and ignores repeats, which is exactly the set semantics you need.',
    "Don't compute $P(A) + P(B)$ by summing `p[A]` with the list as an index: a repeated index would be counted twice.",
]

_r = np.random.default_rng(17)
_p = _r.dirichlet(np.ones(12))
TESTS = [
    (np.array([0.1, 0.2, 0.3, 0.4]), [0, 1], [1, 2]),
    (np.full(6, 1 / 6), [1, 3, 5], [3, 4, 5]),
    (np.array([0.5, 0.5]), [], [0, 1]),
    (np.array([0.25, 0.25, 0.25, 0.25]), [2, 2, 2], [2]),
    (np.array([1.0]), [0], []),
    (np.array([0.0, 0.7, 0.0, 0.3]), [0, 2], [1, 3]),
    (_p, [0, 3, 4, 7, 11], [1, 3, 5, 7, 9, 11]),
]

EXPLANATION = r"""
## The rules this checks
- **Complement:** $P(A^c) = 1 - P(A)$, because $A$ and $A^c$ split the space.
- **Inclusion-exclusion:** $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. Adding $P(A)$ and $P(B)$ counts the overlap twice.
- **Monotonicity:** $P(A \cap B) \le \min(P(A), P(B)) \le \max(P(A), P(B)) \le P(A \cup B)$.

Computing each from masks independently and then checking these identities is a good habit when testing probability code.

## Complexity
$O(n + |A| + |B|)$ time and $O(n)$ memory for the masks.

## Common bugs
- Summing `p[A]` directly: NumPy fancy indexing keeps duplicates, so `[2, 2, 2]` counts outcome 2 three times.
- Treating disjoint as the same as independent. Disjoint events with positive probability are never independent: $P(A \cap B) = 0 \ne P(A)P(B)$.
- Computing the complement as `1 - P(A)` is fine mathematically, but can differ from the mask sum in the last bit. Both are within tolerance here.

## Interview follow-ups
- Generalise inclusion-exclusion to three events. How many terms does it have for $k$ events? ($2^k - 1$.)
- Roll two dice. What is the probability that at least one shows a 6? (Complement: $1 - (5/6)^2 = 11/36$.)
"""
