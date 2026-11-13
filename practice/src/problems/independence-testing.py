import numpy as np

META = dict(title='Are these two variables independent?', tags=['Probability'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:11.01'], related=['conditional-probability', 'chi-square-goodness-of-fit'])

STATEMENT = r"""
Discrete variables $X$ and $Y$ are **independent** when every cell of their joint table factors into the product of the marginals:

$$P(X = i, Y = j) = P(X = i)\,P(Y = j) \quad \text{for all } i, j$$

Write `independence_gap(table, tol=1e-9)` that returns the tuple `(product, gap, independent)`:

- `product`: shape `(m, k)`, the table $P(X=i)\,P(Y=j)$ you would see if the variables were independent,
- `gap`: a Python float, $\max_{i,j} \left|P(X=i, Y=j) - P(X=i)P(Y=j)\right|$,
- `independent`: a Python `bool`, `True` when `gap <= tol`.

## Requirements
- `table`: float array of shape `(m, k)`, non-negative with a positive total. It may hold counts, so normalise it to a joint distribution first.
- Use an outer product, not loops.
"""

STARTER = '''import numpy as np

def independence_gap(table, tol=1e-9):
    """Return (product_of_marginals (m, k), max_abs_gap float, independent bool)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def independence_gap(table, tol=1e-9):
    joint = np.asarray(table, dtype=float)
    joint = joint / joint.sum()
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))   # P(X=i) * P(Y=j)
    gap = float(np.abs(joint - product).max())
    return product, gap, bool(gap <= tol)
'''

HINTS = [
    'Normalise the table so it sums to 1. Then row sums are $P(X)$ and column sums are $P(Y)$.',
    '`np.outer(px, py)` builds the whole independent table in one call.',
    'Cast the results with `float(...)` and `bool(...)` so the types match exactly.',
]

_r = np.random.default_rng(31)
TESTS = [
    (np.array([[0.06, 0.14], [0.24, 0.56]]),),
    (np.array([[30.0, 10.0], [10.0, 30.0]]),),
    (np.outer([1.0, 2.0, 5.0], [3.0, 1.0, 1.0, 5.0]),),
    (np.array([[0.5, 0.0], [0.0, 0.5]]),),
    (np.array([[4.0]]),),
    (np.array([[2.0, 2.0], [2.0, 2.0]]), 0.0),
    (np.array([[10.0, 20.0], [30.0, 41.0]]), 0.01),
    (_r.integers(1, 30, size=(3, 4)).astype(float),),
]

EXPLANATION = r"""
## Why the outer product
Under independence, the joint table has rank 1: each row is the same distribution of $Y$, scaled by $P(X=i)$. The outer product of the marginals is the unique rank-1 table with those marginals, so any gap from it measures dependence.

Equivalently, $X$ and $Y$ are independent exactly when every row of the conditional table $P(Y \mid X = i)$ is the same.

## Exact versus sampled
On a true probability table, comparing with a tiny tolerance is the right check. On **sampled counts** the gap is never exactly 0, even for truly independent variables, because of sampling noise. There you need a test that knows how large the noise is: the chi-square test of independence compares $\sum (O - E)^2 / E$, with $E = n \cdot$ `product`, against $\chi^2$ with $(m-1)(k-1)$ degrees of freedom.

## Complexity
$O(mk)$.

## Common bugs
- Forgetting to normalise counts, so the "product" is off by a factor of the total.
- Using `==` on floats instead of a tolerance. `0.3 * 0.2` is not exactly `0.06` in floating point.
- Thinking zero correlation implies independence. It only rules out a linear relationship.

## Interview follow-ups
- $X$ uniform on $\{-1, 0, 1\}$ and $Y = X^2$. Show they're uncorrelated but dependent.
- How would you measure dependence strength instead of giving a yes/no answer? (Mutual information $\sum p \log \frac{p}{p_x p_y}$, or Cramér's V from the chi-square statistic.)
"""
