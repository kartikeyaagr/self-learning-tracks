import numpy as np

META = dict(title='Conditional distributions from a joint table', tags=['Probability'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:11.01', 'mlmaths:10.01'], related=['basic-probability-rules', 'bayes-theorem', 'independence-testing'])

STATEMENT = r"""
A two-way table holds the joint distribution of discrete variables $X$ (rows) and $Y$ (columns). It may be given as raw **counts**: only the proportions matter. Conditioning on $X = i$ keeps row $i$ and renormalises it:

$$P(Y = j \mid X = i) = \frac{P(X = i, Y = j)}{P(X = i)} = \frac{c_{ij}}{\sum_{j'} c_{ij'}}$$

Write `conditionals(table)` that returns the tuple `(p_x, y_given_x, x_given_y)`:

- `p_x`: shape `(m,)`, the marginal $P(X = i)$,
- `y_given_x`: shape `(m, k)`, row $i$ is the distribution of $Y$ given $X = i$ (each row sums to 1),
- `x_given_y`: shape `(m, k)`, column $j$ is the distribution of $X$ given $Y = j$ (each column sums to 1).

## Requirements
- `table`: float array of shape `(m, k)`, non-negative, with a positive total.
- If a row (or column) is all zeros, the conditional on it is undefined. Fill that row of `y_given_x` (or column of `x_given_y`) with zeros.
- Vectorised: no loops over rows or columns.
"""

STARTER = '''import numpy as np

def conditionals(table):
    """table: (m, k) non-negative counts or probabilities. Return (p_x, y_given_x, x_given_y)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def conditionals(table):
    t = np.asarray(table, dtype=float)
    rows = t.sum(axis=1, keepdims=True)
    cols = t.sum(axis=0, keepdims=True)
    # Divide only where the margin is positive; empty rows/columns stay 0.
    y_given_x = np.divide(t, rows, out=np.zeros_like(t), where=rows > 0)
    x_given_y = np.divide(t, cols, out=np.zeros_like(t), where=cols > 0)
    return rows[:, 0] / t.sum(), y_given_x, x_given_y
'''

HINTS = [
    'Normalising a row means dividing it by its own sum. The scale of the table cancels, so counts work as well as probabilities.',
    'Row sums with `keepdims=True` have shape `(m, 1)` and broadcast across columns. Column sums with `keepdims=True` have shape `(1, k)`.',
    '`np.divide(t, s, out=np.zeros_like(t), where=s > 0)` leaves zeros wherever the margin is 0, with no warnings and no `nan`.',
]

_r = np.random.default_rng(23)
TESTS = [
    (np.array([[10.0, 30.0], [20.0, 40.0]]),),
    (np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]]),),
    (np.array([[5.0, 0.0], [0.0, 0.0], [1.0, 4.0]]),),
    (np.array([[7.0]]),),
    (np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 0.0]]),),
    (_r.integers(0, 50, size=(4, 5)).astype(float),),
    (_r.dirichlet(np.ones(12)).reshape(3, 4),),
]

EXPLANATION = r"""
## Why it's just renormalising
Conditioning on $X = i$ throws away every outcome where $X \ne i$. What's left is row $i$, whose entries no longer sum to 1, so you rescale by the row total $P(X=i)$. Rows give $P(Y \mid X)$, columns give $P(X \mid Y)$, and the two are linked through Bayes' rule: $P(X=i \mid Y=j)\,P(Y=j) = P(Y=j \mid X=i)\,P(X=i)$.

## Complexity
$O(mk)$ time and memory.

## Common bugs
- Normalising by the grand total, which gives the joint, not the conditional.
- Dividing columns by row sums, from mixing up `axis=0` and `axis=1` or dropping `keepdims`.
- Producing `nan` for empty rows (`0 / 0`) and letting it spread into later sums.

## Interview follow-ups
- From a click log, how would you estimate $P(\text{click} \mid \text{position})$, and what bias does that estimate carry? (Count clicks and impressions per position. Higher positions get more attention, so it mixes relevance with position bias.)
- With very sparse counts, a row like `[1, 0, 0]` gives a conditional of exactly `[1, 0, 0]`. How would you smooth it? (Add-$\alpha$ / Laplace smoothing: add $\alpha$ to every cell before normalising.)
"""
