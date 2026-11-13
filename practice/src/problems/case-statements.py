import numpy as np

META = dict(title='Confidence bands with CASE (and a custom sort order)', tags=['Feature Engineering', 'Data Processing'],
            diff='Easy', runtime='sql', related=['where-clauses', 'conditional-aggregation'])

STATEMENT = r"""
A classifier's predictions are stored in the table `predictions`:

| column | type |
| `id` | INTEGER, unique |
| `score` | REAL in $[0, 1]$, or NULL if the model failed on the example |

For a review queue you want to bucket each prediction into a **band**:

| band | rule |
| `'high'` | `score >= 0.8` |
| `'medium'` | `0.5 <= score < 0.8` |
| `'low'` | `score < 0.5` |
| `'missing'` | `score` is NULL |

Write **one SQLite query** that returns every prediction with its band.

## Output
- Columns, in this order: `id`, `score`, `band`.
- Sort by band in the order `high`, `medium`, `low`, `missing` (not alphabetically), then by `id` ascending.
"""

STARTER = '''-- predictions(id, score)
SELECT id, score
FROM predictions
ORDER BY id;
'''

SOLUTION = '''SELECT id,
       score,
       CASE
         WHEN score IS NULL THEN 'missing'
         WHEN score >= 0.8  THEN 'high'      -- the first matching WHEN wins,
         WHEN score >= 0.5  THEN 'medium'    -- so each test can assume the ones above failed
         ELSE 'low'
       END AS band
FROM predictions
ORDER BY CASE
           WHEN score IS NULL THEN 4
           WHEN score >= 0.8  THEN 1
           WHEN score >= 0.5  THEN 2
           ELSE 3
         END,
         id;
'''

HINTS = [
    '`CASE WHEN cond THEN value ... ELSE value END` evaluates the conditions top to bottom and returns the first match.',
    'Order the branches from the most specific to the least: once `score >= 0.8` failed, `score >= 0.5` already means "between 0.5 and 0.8". Handle NULL explicitly, because `NULL >= 0.8` is not true and would fall into `ELSE`.',
    'For the custom sort, put a second `CASE` in `ORDER BY` that maps the bands to 1, 2, 3, 4, and add `id` as the tie-breaker.',
]

_T = 'CREATE TABLE predictions (id INTEGER PRIMARY KEY, score REAL);\n'
_rng = np.random.default_rng(131)
_rand = ','.join(f"({i},{'NULL' if _rng.random() < 0.15 else round(float(_rng.random()), 2)})" for i in _rng.permutation(np.arange(1, 41)))
TESTS = [
    _T + "INSERT INTO predictions VALUES (1,0.95),(2,0.10),(3,0.65),(4,NULL);",
    _T + "INSERT INTO predictions VALUES (5,0.3),(6,0.85),(7,0.4),(8,0.9);",
    _T + "INSERT INTO predictions VALUES (1,0.8),(2,0.5),(3,0.79999),(4,0.49999);",          # boundaries
    _T + "INSERT INTO predictions VALUES (1,0.0),(2,1.0),(3,NULL),(4,NULL);",
    _T + "INSERT INTO predictions VALUES (9,0.2);",
    _T + "",
    _T + f"INSERT INTO predictions VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
A searched `CASE` checks its `WHEN` conditions in order and returns the value of the first one that is true; if none is, it returns the `ELSE` value (or NULL without an `ELSE`). Putting the NULL test first and the thresholds in descending order means each branch can rely on the earlier ones having failed, so there is no need to write `score >= 0.5 AND score < 0.8`.

## Why NULL needs its own branch
`NULL >= 0.8` and `NULL >= 0.5` are both NULL, not true, so without the first branch a missing score would fall through to `ELSE 'low'`, silently labelling failures as low-confidence predictions.

## Custom sort order
`ORDER BY band` would sort alphabetically: high, low, medium, missing. A `CASE` in `ORDER BY` maps each band to a rank. You could also write `ORDER BY CASE band WHEN 'high' THEN 1 ... END`, since SQLite allows output aliases inside `ORDER BY` expressions, but repeating the conditions on `score` is portable.

## Complexity
One pass to compute the band and one sort: $O(N \log N)$.

## Interview follow-ups
- Count the predictions per band with one query (`GROUP BY` the same `CASE`, or conditional aggregation), and make sure bands with zero predictions still appear.
- The thresholds should come from a table `bands(name, lo, hi)` instead of the query. Join on `score >= lo AND score < hi`: what happens at the top boundary 1.0?
"""
