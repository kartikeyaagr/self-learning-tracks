import numpy as np

META = dict(title='Training steps per experiment (SELECT with computed columns)', tags=['Data Processing'], diff='Easy',
            runtime='sql', related=['where-clauses', 'order-by'])

STATEMENT = r"""
The table `experiments` describes training runs that have been planned but not launched yet:

| column | type |
| `id` | INTEGER, unique |
| `model_name` | TEXT |
| `train_size` | INTEGER, number of training examples, at least 1 |
| `batch_size` | INTEGER, at least 1 |
| `epochs` | INTEGER, at least 1 |

Before launching, you want to know how many optimizer steps each run will take. One epoch walks the whole training set in batches of `batch_size`; the last batch may be smaller than the others but still counts as one step, so

$$\text{steps\_per\_epoch} = \left\lceil \frac{\text{train\_size}}{\text{batch\_size}} \right\rceil, \qquad \text{total\_steps} = \text{steps\_per\_epoch} \times \text{epochs}.$$

Write **one SQLite query** that returns one row per experiment.

## Output
- Columns, in this order and with these names: `id`, `model`, `steps_per_epoch`, `total_steps`.
- `model` is the `model_name` column under a new name.
- Both step counts are INTEGERs (exact, no floating point).
- Sort by `id` ascending.
"""

STARTER = '''-- experiments(id, model_name, train_size, batch_size, epochs)
SELECT id, model_name, train_size / batch_size
FROM experiments;
'''

SOLUTION = '''SELECT id,
       model_name AS model,
       -- integer ceiling division: adding batch_size - 1 rounds any partial batch up
       (train_size + batch_size - 1) / batch_size AS steps_per_epoch,
       ((train_size + batch_size - 1) / batch_size) * epochs AS total_steps
FROM experiments
ORDER BY id;
'''

HINTS = [
    '`SELECT expr AS name` both computes a column and names it. Every output column here needs an alias.',
    'Dividing two INTEGERs in SQLite gives an INTEGER, rounded toward zero, so `train_size / batch_size` drops the partial last batch.',
    'For positive integers, $\\lceil a/b \\rceil$ equals `(a + b - 1) / b` with integer division. Multiply that by `epochs` for `total_steps`.',
]

_T = ('CREATE TABLE experiments (id INTEGER PRIMARY KEY, model_name TEXT, train_size INTEGER, '
      'batch_size INTEGER, epochs INTEGER);\n')
_rng = np.random.default_rng(11)
_rand = ','.join(f"({i},'m{int(_rng.integers(0, 5))}',{int(_rng.integers(1, 200000))},"
                 f"{int(_rng.choice([1, 8, 32, 64, 100, 256]))},{int(_rng.integers(1, 50))})" for i in range(1, 31))
TESTS = [
    _T + "INSERT INTO experiments VALUES (1,'resnet18',1000,64,3),(2,'vit-tiny',512,128,10);",
    _T + "INSERT INTO experiments VALUES (7,'mlp',10,3,2),(3,'mlp',9,3,2);",
    _T + "INSERT INTO experiments VALUES (1,'tiny',5,32,4);",                       # batch larger than the data
    _T + "INSERT INTO experiments VALUES (1,'one',1,1,1),(2,'exact',4096,4096,1);",
    _T + "INSERT INTO experiments VALUES (1,'llm',2000000000,7,1000);",             # large values stay exact
    _T + "",
    _T + f"INSERT INTO experiments VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`SELECT` evaluates its expressions once per row and `AS` names the result. The only subtle part is the ceiling: SQLite's `/` between two INTEGERs is integer division that truncates toward zero. For positive $a, b$, adding $b-1$ before dividing pushes any remainder over the next multiple of $b$, so `(a + b - 1) / b` $= \lceil a/b \rceil$, with no floating point involved.

## Complexity
One pass over the table, $O(N)$, plus the sort for `ORDER BY id`. That sort is free here because `id` is the primary key.

## Common bugs
- Writing `train_size / batch_size`, which silently drops the last partial batch: 1000 examples at batch 64 is 16 steps, not 15.
- Using `CAST(train_size AS REAL) / batch_size` and then `ROUND`, which rounds 15.6 up but also rounds 15.2 down.
- Forgetting the aliases. The grader compares column names, and SQLite would otherwise name the column after the whole expression.
- Leaving out `ORDER BY`. SQL does not guarantee any row order without it.

## Interview follow-ups
- How would you also show the number of examples in the last batch? `train_size - (steps_per_epoch - 1) * batch_size`, or `train_size % batch_size` with a special case for 0.
- With `drop_last=True` the partial batch is skipped. Which expression gives the step count then, and what happens when `batch_size > train_size`?
"""
