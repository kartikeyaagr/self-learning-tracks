import numpy as np

META = dict(title='Hyperparameter grid: configs still to run (CROSS JOIN)', tags=['Data Processing', 'Training'],
            diff='Medium', runtime='sql', related=['self-join', 'subqueries'])

STATEMENT = r"""
A grid search tries every combination of a learning rate and a batch size. The candidate values and the finished runs are stored as:

| column | type |
| `lrs.lr` | REAL, never NULL, values may repeat |
| `batch_sizes.bs` | INTEGER, never NULL, values may repeat |
| `finished.lr` | REAL |
| `finished.bs` | INTEGER |

`finished` can contain the same configuration more than once (reruns) and configurations outside the current grid (from an older search); both are irrelevant here.

Write **one SQLite query** that returns the grid configurations `(lr, bs)` that have **not** been run yet.

## Output
- Columns, in this order: `lr`, `bs`.
- Each configuration at most once, even if a value is repeated in `lrs` or `batch_sizes`.
- Sort by `lr`, then `bs`, both ascending.
"""

STARTER = '''-- lrs(lr), batch_sizes(bs), finished(lr, bs)
SELECT lr, bs
FROM finished;
'''

SOLUTION = '''-- every grid point, minus the ones already run; EXCEPT also removes duplicates
SELECT l.lr, b.bs
FROM lrs AS l
CROSS JOIN batch_sizes AS b
EXCEPT
SELECT lr, bs FROM finished
ORDER BY lr, bs;
'''

HINTS = [
    'First build the full grid: every learning rate paired with every batch size. `CROSS JOIN` produces all pairs with no condition.',
    'Then remove the configurations that appear in `finished`. Either `NOT EXISTS (SELECT 1 FROM finished f WHERE f.lr = l.lr AND f.bs = b.bs)`, or a set difference.',
    '`SELECT l.lr, b.bs FROM lrs l CROSS JOIN batch_sizes b EXCEPT SELECT lr, bs FROM finished ORDER BY lr, bs`. `EXCEPT` returns distinct rows, which handles repeated grid values for free.',
]

_T = ('CREATE TABLE lrs (lr REAL);\nCREATE TABLE batch_sizes (bs INTEGER);\n'
      'CREATE TABLE finished (lr REAL, bs INTEGER);\n')
_rng = np.random.default_rng(2)
_L = ['0.1', '0.03', '0.01', '0.003', '0.001', '0.0003']
_B = ['16', '32', '64', '128', '256']
_fin = ','.join(f"({_rng.choice(_L + ['0.5'])},{_rng.choice(_B + ['8'])})" for _ in range(25))
TESTS = [
    _T + "INSERT INTO lrs VALUES (0.01),(0.001);\nINSERT INTO batch_sizes VALUES (32),(64);\n"
         "INSERT INTO finished VALUES (0.01,32);",
    _T + "INSERT INTO lrs VALUES (0.1),(0.1);\nINSERT INTO batch_sizes VALUES (8),(16),(8);\n"
         "INSERT INTO finished VALUES (0.1,16),(0.1,16),(0.5,8);",                      # repeats and stale runs
    _T + "INSERT INTO lrs VALUES (0.01);\nINSERT INTO batch_sizes VALUES (32);\nINSERT INTO finished VALUES (0.01,32);",
    _T + "INSERT INTO lrs VALUES (0.3),(0.03),(0.003);\nINSERT INTO batch_sizes VALUES (128);",   # nothing finished
    _T + "INSERT INTO lrs VALUES (0.01);\nINSERT INTO finished VALUES (0.01,32);",              # empty batch-size list
    _T + "",
    _T + f"INSERT INTO lrs VALUES {','.join(f'({v})' for v in _L)};\n"
         f"INSERT INTO batch_sizes VALUES {','.join(f'({v})' for v in _B)};\nINSERT INTO finished VALUES {_fin};",
]

EXPLANATION = r"""
## Why it works
`CROSS JOIN` returns the Cartesian product: with $L$ learning rates and $B$ batch sizes, $L \times B$ rows, every grid point. `EXCEPT` is set difference: it keeps the rows of the first query that do not appear in the second, and like `UNION` it returns **distinct** rows, so duplicate grid values and repeated reruns both disappear. The `ORDER BY` at the end applies to the whole compound query.

## The NOT EXISTS version
```
SELECT DISTINCT l.lr, b.bs
FROM lrs l CROSS JOIN batch_sizes b
WHERE NOT EXISTS (SELECT 1 FROM finished f WHERE f.lr = l.lr AND f.bs = b.bs)
ORDER BY l.lr, b.bs;
```
This is the same anti-join written explicitly. It needs `DISTINCT` because the cross join itself can produce duplicates (test 2).

## Complexity
The grid has $L \cdot B$ rows by construction, which is the point of a cross join and also its danger: crossing two million-row tables by accident produces $10^{12}$ rows. The difference step is a hash or index lookup per grid point.

## Common bugs
- Forgetting `DISTINCT` in the `NOT EXISTS` version.
- Writing `NOT IN (SELECT ...)` with two columns. SQLite supports row values, but if any `finished` value is NULL, `NOT IN` returns no rows at all.
- Comparing float learning rates that were computed rather than typed: `0.1 * 0.1` is not `0.01`. Storing them as text or as integer exponents avoids this.

## Interview follow-ups
- Add a third axis (weight decay). How does the row count grow, and why does random search become attractive?
- Given the finished runs and their accuracies, return for each learning rate the best batch size so far. This combines the grid with a `ROW_NUMBER()` per `lr`.
"""
