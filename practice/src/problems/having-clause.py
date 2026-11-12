import numpy as np

META = dict(title='Models that are consistently good on the test split (HAVING)', tags=['Data Processing', 'Metrics & Evaluation'],
            diff='Medium', runtime='sql', related=['group-by', 'nested-aggregations'])

STATEMENT = r"""
A benchmark harness stores one row per model, dataset and split:

| column | type |
| `model` | TEXT |
| `dataset` | TEXT |
| `split` | TEXT, `'val'` or `'test'` |
| `accuracy` | REAL |

Each `(model, dataset, split)` combination appears at most once. Write **one SQLite query** that finds the models worth shipping, using **test-split rows only**:

- the model was evaluated on **at least 3** datasets (test split), and
- its mean test accuracy is **at least 0.75**.

Validation rows must not count toward either condition.

## Output
- Columns, in this order: `model`, `n_datasets`, `mean_acc`.
- `n_datasets` is the number of test-split datasets; `mean_acc` is the average test accuracy (a REAL).
- Sort by `mean_acc` descending, then `model` ascending.
"""

STARTER = '''-- evals(model, dataset, split, accuracy)
SELECT model, COUNT(*) AS n_datasets
FROM evals
GROUP BY model;
'''

SOLUTION = '''SELECT model,
       COUNT(*)      AS n_datasets,
       AVG(accuracy) AS mean_acc
FROM evals
WHERE split = 'test'              -- row filter: runs before grouping
GROUP BY model
HAVING COUNT(*) >= 3              -- group filter: runs after the aggregates exist
   AND AVG(accuracy) >= 0.75
ORDER BY mean_acc DESC, model;
'''

HINTS = [
    'Two kinds of filtering are needed: one on individual rows (the split) and one on whole groups (count and mean).',
    '`WHERE` filters rows before `GROUP BY`, so it cannot see aggregates. `HAVING` filters groups after `GROUP BY` and can use `COUNT(*)` and `AVG(...)`.',
    "`WHERE split = 'test' GROUP BY model HAVING COUNT(*) >= 3 AND AVG(accuracy) >= 0.75`. Because a (model, dataset, split) appears once, `COUNT(*)` counts datasets.",
]

_T = 'CREATE TABLE evals (model TEXT, dataset TEXT, split TEXT, accuracy REAL, UNIQUE (model, dataset, split));\n'
_rng = np.random.default_rng(29)
_rows = []
for m in ['alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta']:
    for d in _rng.choice(['mnist', 'cifar', 'svhn', 'stl', 'imnet'], size=int(_rng.integers(1, 6)), replace=False):
        for s in ['val', 'test']:
            if _rng.random() < 0.8:
                _rows.append(f"('{m}','{d}','{s}',{int(_rng.integers(8, 17)) / 16})")   # multiples of 1/16: exact sums
TESTS = [
    _T + "INSERT INTO evals VALUES ('a','d1','test',0.75),('a','d2','test',0.875),('a','d3','test',1.0),"
         "('b','d1','test',0.5),('b','d2','test',0.625),('b','d3','test',0.75);",
    _T + "INSERT INTO evals VALUES ('c','d1','test',0.9375),('c','d2','test',0.875),('c','d1','val',0.9375),"
         "('c','d2','val',0.875),('c','d3','val',0.9375);",                             # val rows don't count
    _T + "INSERT INTO evals VALUES ('x','d1','test',0.5),('x','d2','test',0.75),('x','d3','test',1.0),"
         "('y','d1','test',1.0),('y','d2','test',1.0),('y','d3','test',1.0),('y','d4','test',0.0);",   # mean exactly 0.75
    _T + "INSERT INTO evals VALUES ('p','d1','test',0.8125),('p','d2','test',0.8125),('p','d3','test',0.8125),"
         "('q','d1','test',0.8125),('q','d2','test',0.8125),('q','d3','test',0.8125);",   # tie on mean: by model
    _T + "INSERT INTO evals VALUES ('r','d1','test',0.25),('r','d2','test',1.0),('r','d3','test',1.0),"
         "('r','d1','val',1.0);",
    _T + "",
    _T + f"INSERT INTO evals VALUES {','.join(_rows)};",
]

EXPLANATION = r"""
## Why it works
A query runs in a fixed logical order: `FROM` → `WHERE` → `GROUP BY` → `HAVING` → `SELECT` → `ORDER BY`. The split filter is about rows, so it belongs in `WHERE`, and the validation rows are gone before any counting happens. The thresholds are about groups, so they belong in `HAVING`, which is evaluated once the per-model aggregates exist.

## Exact boundaries
"At least" means `>=`, and test 3 contains models whose mean is exactly 0.75. The test data uses multiples of $1/16$, which binary floating point represents exactly, so those sums are exact. Decimals such as 0.1 are not exact in binary (`0.1 + 0.2` is `0.30000000000000004`), so with arbitrary accuracies a mean that should equal the threshold can land a hair either side of it. Production code often compares against `threshold - 1e-9` or rounds first.

## Complexity
One scan with a hash or sort on `model`: $O(N)$ to $O(N \log N)$. `HAVING` costs nothing extra; it inspects the $G$ group results.

## Common bugs
- Putting `split = 'test'` in `HAVING`. SQLite accepts it but then evaluates `split` from an arbitrary row of the group, which is wrong; PostgreSQL rejects it.
- Writing `WHERE AVG(accuracy) >= 0.75`: an error, aggregates are not allowed in `WHERE`.
- Counting with `COUNT(DISTINCT dataset)` without the split filter, which counts datasets that only have validation results.

## Interview follow-ups
- Require the model to have a test result on **every** dataset in a `datasets` table. Compare `COUNT(*)` with `(SELECT COUNT(*) FROM datasets)` in `HAVING`: relational division.
- Also report how many models were rejected by each rule. Use conditional aggregation over a per-model subquery.
"""
