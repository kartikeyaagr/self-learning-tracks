import numpy as np

META = dict(title='Confident mistakes on the validation split (WHERE)', tags=['Data Processing', 'Classification'], diff='Easy',
            runtime='sql', lessons=['sdsd:03.03'], related=['basic-select', 'case-statements'])

STATEMENT = r"""
A binary classifier has scored every example. The table `predictions` holds one row per example:

| column | type |
| `id` | INTEGER, unique |
| `split` | TEXT, one of `'train'`, `'val'`, `'test'` |
| `label` | INTEGER, the true class, 0 or 1 |
| `score` | REAL, the predicted probability of class 1, or NULL if scoring failed |

A **confident mistake** is an example where the model was sure and wrong:

- `label = 0` and `score >= 0.9`, or
- `label = 1` and `score <= 0.1`.

Both bounds are inclusive. Write **one SQLite query** that lists the confident mistakes **on the `'val'` split only**. Rows with a NULL `score` are never confident mistakes.

## Output
- Columns, in this order: `id`, `label`, `score`.
- Sort by `id` ascending.
"""

STARTER = '''-- predictions(id, split, label, score)
SELECT id, split
FROM predictions
WHERE split = 'val';
'''

SOLUTION = '''SELECT id, label, score
FROM predictions
WHERE split = 'val'
  -- the parentheses matter: AND binds tighter than OR
  AND ((label = 0 AND score >= 0.9) OR (label = 1 AND score <= 0.1))
ORDER BY id;
'''

HINTS = [
    'Combine three conditions: the split, and one of two "sure and wrong" cases.',
    '`AND` binds more tightly than `OR`. Without parentheses, `split = \'val\' AND a OR b` means `(split = \'val\' AND a) OR b`, which lets `b` rows from any split through.',
    "`WHERE split = 'val' AND ((label = 0 AND score >= 0.9) OR (label = 1 AND score <= 0.1))`. A comparison with NULL is never true, so NULL scores drop out on their own.",
]

_T = 'CREATE TABLE predictions (id INTEGER PRIMARY KEY, split TEXT, label INTEGER, score REAL);\n'
_rng = np.random.default_rng(23)
_rand = ','.join(
    f"({i},'{_rng.choice(['train', 'val', 'test'])}',{int(_rng.integers(0, 2))},"
    f"{'NULL' if _rng.random() < 0.1 else round(float(_rng.random()), 2)})" for i in range(1, 61))
TESTS = [
    _T + "INSERT INTO predictions VALUES (1,'val',0,0.95),(2,'val',1,0.97),(3,'val',1,0.04),(4,'val',0,0.30);",
    _T + "INSERT INTO predictions VALUES (1,'train',1,0.02),(2,'val',1,0.02),(3,'test',0,0.99);",
    _T + "INSERT INTO predictions VALUES (1,'val',0,0.9),(2,'val',1,0.1),(3,'val',0,0.89),(4,'val',1,0.11);",   # inclusive bounds
    _T + "INSERT INTO predictions VALUES (1,'val',0,NULL),(2,'val',1,NULL),(3,'val',1,0.0),(4,'val',0,1.0);",
    _T + "INSERT INTO predictions VALUES (5,'val',0,0.5),(6,'train',0,0.99);",              # no mistakes on val
    _T + "",
    _T + f"INSERT INTO predictions VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`WHERE` keeps a row only when its condition is **true**. The condition here is "on the val split" AND "one of two error patterns", so the two patterns must be grouped with parentheses before they are combined with the split test.

## Operator precedence
SQL evaluates `NOT` first, then `AND`, then `OR`. Without parentheses,

```
split = 'val' AND label = 0 AND score >= 0.9 OR label = 1 AND score <= 0.1
```

means `(val AND confident-false-positive) OR (confident-false-negative on any split)`. Test 2 has a training row that this version wrongly returns.

## NULLs
`NULL >= 0.9` is not false but NULL ("unknown"), and `WHERE` drops anything that is not true. So rows with a failed score fall out without an explicit `score IS NOT NULL`. Be careful with `NOT (...)`: `NOT NULL` is still NULL, so negated conditions also drop NULL rows, which is often not what people expect.

## Complexity
A full scan, $O(N)$. With an index on `(split, label, score)` the database could jump straight to the two score ranges inside the val split.

## Interview follow-ups
- Return the 20 most confidently wrong val examples, the worst first. Order by the distance from the true label, `ABS(label - score)` descending, then `id`.
- How would you write this as a single expression without OR? Hint: `ABS(label - score) >= 0.9`. Is it exactly equivalent at the boundaries in floating point?
"""
