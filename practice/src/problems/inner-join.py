import numpy as np

META = dict(title='Attach model details to training runs (INNER JOIN)', tags=['Data Processing'], diff='Easy', runtime='sql',
            lessons=['sdsd:03.01'], related=['left-join', 'multiple-joins'])

STATEMENT = r"""
Training runs and model architectures live in separate tables:

| column | type |
| `runs.run_id` | INTEGER, unique |
| `runs.model_id` | INTEGER, refers to `models.model_id` (may point at a model that was deleted) |
| `runs.val_acc` | REAL |
| `models.model_id` | INTEGER, unique |
| `models.name` | TEXT |
| `models.params_m` | REAL, parameter count in millions |

Write **one SQLite query** that lists every run together with its model's name and size.

- Runs whose `model_id` has no matching row in `models` are left out.
- Models that were never trained do not appear.

## Output
- Columns, in this order: `run_id`, `model`, `params_m`, `val_acc` (`model` is `models.name`).
- Sort by `run_id` ascending.
"""

STARTER = '''-- runs(run_id, model_id, val_acc), models(model_id, name, params_m)
SELECT run_id, model_id, val_acc
FROM runs;
'''

SOLUTION = '''SELECT r.run_id, m.name AS model, m.params_m, r.val_acc
FROM runs AS r
JOIN models AS m ON m.model_id = r.model_id
ORDER BY r.run_id;
'''

HINTS = [
    'A join pairs each row of one table with the matching rows of another. Which column do the two tables share?',
    '`FROM runs r JOIN models m ON m.model_id = r.model_id`. Table aliases keep the column references short and unambiguous.',
    'A plain `JOIN` is an inner join: rows without a partner on the other side are dropped, which is exactly the rule for deleted models. Select `m.name AS model`.',
]

_T = ('CREATE TABLE models (model_id INTEGER PRIMARY KEY, name TEXT, params_m REAL);\n'
      'CREATE TABLE runs (run_id INTEGER PRIMARY KEY, model_id INTEGER, val_acc REAL);\n')
_M = "INSERT INTO models VALUES (1,'resnet18',11.7),(2,'resnet50',25.6),(3,'vit-b16',86.6),(4,'mobilenet',3.5);\n"
_rng = np.random.default_rng(41)
_rand = ','.join(f"({i},{int(_rng.integers(1, 7))},{round(float(_rng.uniform(0.5, 0.95)), 3)})"
                 for i in _rng.permutation(np.arange(100, 140)))
TESTS = [
    _T + _M + "INSERT INTO runs VALUES (1,1,0.71),(2,3,0.82),(3,1,0.74);",
    _T + _M + "INSERT INTO runs VALUES (10,2,0.76),(11,9,0.80),(12,NULL,0.60);",   # unknown and NULL model ids
    _T + _M + "INSERT INTO runs VALUES (5,4,0.65);",
    _T + _M + "INSERT INTO runs VALUES (1,7,0.5),(2,8,0.5);",                     # no run matches
    _T + "INSERT INTO models VALUES (1,'same',1.0),(2,'same',2.0);\nINSERT INTO runs VALUES (1,2,0.9),(2,1,0.8);",
    _T + _M,
    _T + _M + f"INSERT INTO runs VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`JOIN ... ON condition` forms every pair of a `runs` row and a `models` row for which the condition is true. Each run has at most one model (because `models.model_id` is unique), so every surviving run appears exactly once. A run whose `model_id` matches nothing, including NULL, which never equals anything, has no pair and disappears.

## Complexity
With an index on `models.model_id` (the primary key here) the database looks up each run's model: $O(N \log M)$. Without any usable index it falls back to a hash join or a nested loop.

## Common bugs
- Selecting `name` without a table prefix once both tables have a `name` column. Get into the habit of qualifying every column in a join.
- Joining on the wrong pair of columns, for example `ON m.model_id = r.run_id`. The query runs and returns nonsense.
- Using the old comma syntax `FROM runs, models` and forgetting the `WHERE` condition, which silently produces a cross join.

## Interview follow-ups
- How do you list the runs whose model is missing? That is an **anti-join**: `LEFT JOIN models m ... WHERE m.model_id IS NULL`, or `NOT EXISTS`.
- If `models.model_id` were not unique, what would happen to the row count, and how would you detect it? Compare `COUNT(*)` before and after the join.
"""
