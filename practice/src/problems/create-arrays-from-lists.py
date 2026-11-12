import numpy as np

META = dict(title='Pack ragged lists into a padded array', tags=['Data Processing', 'NLP'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.01'], related=['mutation-trap', 'zeros-and-ones'])

STATEMENT = r"""
Real data rarely arrives as a neat rectangle. Token sequences, purchase histories and sensor bursts come as Python lists of different lengths, but a NumPy array must be rectangular. The usual fix is to **pad** every row to the longest length.

Write `pad_rows(rows, fill=0.0)` that returns a tuple `(padded, lengths)`.

## Input
- `rows`: a Python list of lists of numbers (ints or floats). Rows may have different lengths, a row may be empty, and `rows` itself may be empty.
- `fill`: the number used for padding (it may be `nan`).

## Output
- `padded`: a `float64` array of shape `(len(rows), L)`, where `L` is the length of the longest row (`L = 0` when `rows` is empty or every row is empty). Row `i` holds `rows[i]` left-aligned; the remaining cells hold `fill`.
- `lengths`: an `int64` array of shape `(len(rows),)` with the original length of each row.

## Example
`rows = [[1, 2, 3], [4], [5, 6]]` gives

```
padded  = [[1., 2., 3.],
           [4., 0., 0.],
           [5., 6., 0.]]
lengths = [3, 1, 2]
```
"""

STARTER = '''import numpy as np

def pad_rows(rows, fill=0.0):
    """rows: list of lists of numbers. Return (padded float64 (n, L), lengths int64 (n,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pad_rows(rows, fill=0.0):
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    width = int(lengths.max()) if len(rows) else 0
    padded = np.full((len(rows), width), fill, dtype=np.float64)
    # mask[i, j] is True for the cells that hold real values. Boolean assignment fills
    # True cells in row-major order, which is exactly the order of the flattened rows.
    mask = np.arange(width) < lengths[:, None]
    padded[mask] = [v for r in rows for v in r]
    return padded, lengths
'''

HINTS = [
    '`np.array(rows)` fails (or builds an object array) when the rows have different lengths. Allocate the output first, then fill it.',
    '`np.full((n, L), fill, dtype=float)` creates the padded canvas. A plain loop that writes `padded[i, :len(r)] = r` already passes.',
    'Loop-free version: `mask = np.arange(L) < lengths[:, None]` marks the real cells, and `padded[mask] = flat_values` fills them in row-major order. Guard `lengths.max()` against an empty list.',
]

_r = np.random.default_rng(11)
_ragged = [_r.normal(size=int(k)).round(3).tolist() for k in _r.integers(0, 6, size=7)]
TESTS = [
    ([[1, 2, 3], [4], [5, 6]],),
    ([[1.5], [], [2, 3]], -1.0),
    ([[], []],),
    ([[7]],),
    ([[1e12, -1e12], [3]], float('nan')),
    (_ragged,),
    (_ragged, 99.0),
]

EXPLANATION = r"""
## Why it works
A NumPy array needs one fixed length per axis, so the ragged structure has to be made rectangular before NumPy can help. The output size is known up front (`n` rows, `L = max length` columns), so allocate it once with `np.full` and copy each row into its left-aligned slot. The `lengths` vector keeps the information that padding would otherwise destroy; downstream code turns it into a mask with `np.arange(L) < lengths[:, None]`.

The solution uses that same mask to fill the array without a Python loop over rows: boolean-mask assignment visits the `True` cells in row-major (C) order, which is exactly the order you get by concatenating the rows.

## Complexity
$O(nL)$ time and memory for the output, plus $O(\text{total elements})$ to flatten the input.

## Common bugs
- Calling `np.array(rows)` on ragged input. NumPy 2 raises `ValueError` rather than silently building an object array.
- `lengths.max()` on an empty array raises. Handle `rows == []` separately.
- Forgetting the dtype: `np.full((n, L), 0)` makes an integer array, and float values would then be truncated when written in.

## Interview follow-ups
- How would you pad on the **left** instead (as decoder-only language models often need for batched generation)? (Use the mask `np.arange(L) >= (L - lengths)[:, None]`.)
- Padding wastes compute when lengths vary wildly. What is bucketing, and how does sorting by length before batching reduce the waste?
"""
