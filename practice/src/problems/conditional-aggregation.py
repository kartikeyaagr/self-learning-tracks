import numpy as np

META = dict(title='Confusion matrix per model (conditional aggregation)', tags=['Metrics & Evaluation', 'Data Processing'],
            diff='Medium', runtime='sql', related=['precision-recall-at-k', 'case-statements', 'group-by'])

STATEMENT = r"""
Several binary classifiers were evaluated on the same examples. The table `preds` holds one row per (model, example):

| column | type |
| `id` | INTEGER, unique |
| `model` | TEXT |
| `label` | INTEGER, the true class, 0 or 1 |
| `pred` | INTEGER, the predicted class, 0 or 1 |

Write **one SQLite query** that turns this long table into one row per model with its confusion matrix and two metrics:

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}.$$

## Output
- Columns, in this order: `model`, `tp`, `fp`, `fn`, `tn`, `precision`, `recall`.
- The four counts are INTEGERs (0 when a cell is empty).
- `precision` and `recall` are REALs, or NULL when their denominator is 0 (the model never predicted 1, or there are no positive labels).
- Sort by `model` ascending.
"""

STARTER = '''-- preds(id, model, label, pred)
SELECT model, COUNT(*) AS tp
FROM preds
GROUP BY model;
'''

SOLUTION = '''WITH cm AS (
  SELECT model,
         SUM(CASE WHEN label = 1 AND pred = 1 THEN 1 ELSE 0 END) AS tp,
         SUM(CASE WHEN label = 0 AND pred = 1 THEN 1 ELSE 0 END) AS fp,
         SUM(CASE WHEN label = 1 AND pred = 0 THEN 1 ELSE 0 END) AS fn,
         SUM(CASE WHEN label = 0 AND pred = 0 THEN 1 ELSE 0 END) AS tn
  FROM preds
  GROUP BY model
)
SELECT model, tp, fp, fn, tn,
       -- NULLIF turns a zero denominator into NULL, and x / NULL is NULL
       1.0 * tp / NULLIF(tp + fp, 0) AS precision,
       1.0 * tp / NULLIF(tp + fn, 0) AS recall
FROM cm
ORDER BY model;
'''

HINTS = [
    'Group by model. Each confusion-matrix cell is a count of the rows in that group matching one (label, pred) combination.',
    '`SUM(CASE WHEN label = 1 AND pred = 1 THEN 1 ELSE 0 END)` counts the true positives without a separate query per cell. Compute the four counts in a CTE, then the ratios from them.',
    'Divide as `1.0 * tp / NULLIF(tp + fp, 0)`: `1.0 *` avoids integer division, and `NULLIF(x, 0)` makes the result NULL instead of dividing by zero.',
]

_T = 'CREATE TABLE preds (id INTEGER PRIMARY KEY, model TEXT, label INTEGER, pred INTEGER);\n'
_rng = np.random.default_rng(43)
_rand = ','.join(f"({i},'{_rng.choice(['knn', 'logreg', 'svm', 'tree'])}',{int(_rng.integers(0, 2))},{int(_rng.integers(0, 2))})"
                 for i in range(1, 121))
TESTS = [
    _T + "INSERT INTO preds VALUES (1,'a',1,1),(2,'a',1,0),(3,'a',0,1),(4,'a',0,0),(5,'a',1,1);",
    _T + "INSERT INTO preds VALUES (1,'zero',1,0),(2,'zero',0,0),(3,'one',1,1),(4,'one',0,1);",   # zero precision denominator
    _T + "INSERT INTO preds VALUES (1,'neg',0,0),(2,'neg',0,1);",                                # no positive labels
    _T + "INSERT INTO preds VALUES (1,'perfect',1,1),(2,'perfect',0,0),(3,'perfect',1,1);",
    _T + "INSERT INTO preds VALUES (1,'b',0,0),(2,'a',1,1),(3,'c',1,0);",
    _T + "",
    _T + f"INSERT INTO preds VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`GROUP BY model` makes one row per model; inside each group, `SUM(CASE WHEN cond THEN 1 ELSE 0 END)` counts the rows that satisfy `cond`. Four such sums give the four cells in one scan, which is the core idea of conditional aggregation: pivoting a long table into columns. The CTE names the counts so the ratio expressions can reuse them.

## Division edge cases
- Integer division: `tp / (tp + fp)` with INTEGERs is 0 unless the model is perfect. `1.0 * tp` makes the numerator REAL.
- Zero denominators: `NULLIF(tp + fp, 0)` is NULL when the sum is 0, and any arithmetic with NULL is NULL. SQLite would also return NULL for `x / 0`, but most databases raise an error, so make the intent explicit.

## Shorter forms
SQLite comparisons return 0 or 1, so `SUM(label = 1 AND pred = 1)` works too, and SQLite 3.30+ supports the standard `COUNT(*) FILTER (WHERE label = 1 AND pred = 1)`. The `CASE` form is the most portable.

## Complexity
One pass over the rows, $O(N)$, regardless of how many cells you count.

## Interview follow-ups
- Add F1. With `precision` and `recall` both possibly NULL or 0, how do you define it? $F_1 = 2TP / (2TP + FP + FN)$ avoids the problem.
- The model outputs a score, not a class. Compute precision at thresholds 0.3, 0.5 and 0.7 in one query by cross-joining a small thresholds table and grouping by `(model, threshold)`.
"""
