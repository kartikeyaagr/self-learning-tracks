import numpy as np

META = dict(title='Ordinal encoding with a fixed order and unknown values', tags=['Feature Engineering'], diff='Easy', runtime='python',
            related=['categorical-encoding', 'frequency-encoding', 'target-encoding'])

STATEMENT = r"""
Some categories have a natural order, such as sizes or education levels. Write `ordinal_encode(values, order, unknown=-1)`:

- `values`: list of strings (may contain `None`).
- `order`: list of strings from lowest to highest, for example `["S", "M", "L"]`.
- Each value becomes its index in `order` (`"S"` to 0, `"M"` to 1, ...). Matching ignores case and surrounding whitespace.
- Values not in `order`, and `None`, become `unknown`.

Return a list of ints.
"""

STARTER = '''def ordinal_encode(values, order, unknown=-1):
    """Return a list of integer codes."""
    raise NotImplementedError
'''

SOLUTION = '''def ordinal_encode(values, order, unknown=-1):
    rank = {o.strip().lower(): i for i, o in enumerate(order)}
    return [unknown if v is None else rank.get(v.strip().lower(), unknown) for v in values]
'''

HINTS = [
    'Build a dictionary from normalised category name to its position in the order list once, then look each value up.',
    'Normalise both the order list and the values the same way: `strip()` and `lower()`.',
    'Use `dict.get(key, unknown)` so missing categories fall back to the unknown code.',
]

TESTS = [
    (['S', 'M', 'L', 'M'], ['S', 'M', 'L']),
    ([' small ', 'LARGE', 'medium'], ['small', 'medium', 'large']),
    (['XL', None, 'S'], ['S', 'M', 'L'], -1),
    (['high school', 'phd', 'bachelor', 'masters'], ['High School', 'Bachelor', 'Masters', 'PhD']),
    ([], ['a']),
    (['b', 'z', 'a'], ['a', 'b'], 99),
]

EXPLANATION = r"""
## When ordinal encoding is right
It keeps order in a single integer column, which suits ordered categories and tree models. For unordered categories (city, colour), it invents an order and a spacing that linear models will take literally; use one-hot or target encoding instead.

## Unknown values
New data contains categories the encoder never saw: typos, new product sizes, missing values. A fixed unknown code keeps the pipeline from crashing and lets the model learn what "unknown" means if it appears in training.

## Spacing
Codes 0, 1, 2 assume equal steps. If the steps are uneven (for example, income bands), consider mapping to representative numeric values instead.

## Interview follow-ups
- How would you handle an ordered category with a "not applicable" level?
"""
