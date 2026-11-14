import numpy as np

META = dict(title='Pairwise interaction and crossed categorical features', tags=['Feature Engineering'], diff='Easy', runtime='python',
            related=['polynomial-features', 'categorical-encoding', 'frequency-encoding'])

STATEMENT = r"""
Write `add_interactions(rows, numeric_pairs, categorical_pairs)` for a list of dict rows.

- For each pair `(a, b)` in `numeric_pairs`, add key `f"{a}*{b}"` with value `row[a] * row[b]`.
- For each pair `(a, b)` in `categorical_pairs`, add key `f"{a}_x_{b}"` with the string `f"{row[a]}|{row[b]}"`. If either value is `None`, the crossed value is `None`.
- If a numeric value is `None`, the product is `None`.

Return a new list of new dicts (do not modify the input), each with the original keys first and the new keys in the order given.
"""

STARTER = '''def add_interactions(rows, numeric_pairs, categorical_pairs):
    """Return new rows with interaction features added."""
    raise NotImplementedError
'''

SOLUTION = '''def add_interactions(rows, numeric_pairs, categorical_pairs):
    out = []
    for r in rows:
        n = dict(r)
        for a, b in numeric_pairs:
            n[f"{a}*{b}"] = None if r[a] is None or r[b] is None else r[a] * r[b]
        for a, b in categorical_pairs:
            n[f"{a}_x_{b}"] = None if r[a] is None or r[b] is None else f"{r[a]}|{r[b]}"
        out.append(n)
    return out
'''

HINTS = [
    'Copy each row with `dict(row)` before adding keys, so the caller\'s data is unchanged.',
    'A crossed categorical is just a new category made by joining the two values with a separator.',
    'Propagate missing values: if either input is None, the interaction is None.',
]

TESTS = [
    ([{'price': 2.0, 'qty': 3, 'city': 'NY', 'device': 'ios'}], [('price', 'qty')], [('city', 'device')]),
    ([{'a': 1, 'b': None, 'c': 'x', 'd': 'y'}], [('a', 'b')], [('c', 'd')]),
    ([{'x': 2, 'y': 5}, {'x': -1, 'y': 4}], [('x', 'y'), ('x', 'x')], []),
    ([{'u': 'p', 'v': None}], [], [('u', 'v')]),
    ([], [('a', 'b')], []),
]

EXPLANATION = r"""
## Why interactions
A linear model adds feature effects. If the effect of price depends on quantity, or the click rate of a device depends on the country, the model needs a feature that represents the combination. Products capture numeric interactions; crossed categories capture "this pair of values" effects, a staple of wide linear models for ads and recommendations.

## Costs
Crosses of high-cardinality features explode the vocabulary (1,000 cities by 50 devices gives 50,000 values), so they are usually hashed into a fixed number of buckets and regularised. Tree models and neural networks learn many interactions on their own, which is one reason they often beat linear models on tabular data.

## Interview follow-ups
- How does a factorisation machine learn interactions without enumerating every cross?
"""
