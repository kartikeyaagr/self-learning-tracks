import numpy as np

META = dict(title='Benchmark podium with tied scores (RANK and DENSE_RANK)', tags=['Data Processing', 'Metrics & Evaluation'],
            diff='Medium', runtime='sql', related=['row-number', 'order-by'])

STATEMENT = r"""
The table `scores` holds one row per model per benchmark:

| column | type |
| `benchmark` | TEXT |
| `model` | TEXT, unique within a benchmark |
| `score` | REAL, higher is better; NULL if the run crashed |

Within each benchmark, models with equal scores share a place. Compute two rankings by `score` descending:

- `rank_pos`, competition ranking: tied models share a place and the next place is skipped (1, 2, 2, 4).
- `dense_pos`, dense ranking: tied models share a place and no place is skipped (1, 2, 2, 3).

Write **one SQLite query** that returns the **podium**: the rows whose `dense_pos` is 1, 2 or 3. Rows with a NULL score are ignored entirely (they get no rank and do not affect anyone else's).

## Output
- Columns, in this order: `benchmark`, `model`, `score`, `rank_pos`, `dense_pos`.
- Sort by `benchmark`, then `rank_pos`, then `model`, all ascending.
"""

STARTER = '''-- scores(benchmark, model, score)
SELECT benchmark, model, score
FROM scores
ORDER BY score DESC
LIMIT 3;
'''

SOLUTION = '''WITH ranked AS (
  SELECT benchmark, model, score,
         RANK()       OVER w AS rank_pos,
         DENSE_RANK() OVER w AS dense_pos
  FROM scores
  WHERE score IS NOT NULL                       -- crashed runs take no place
  WINDOW w AS (PARTITION BY benchmark ORDER BY score DESC)
)
SELECT benchmark, model, score, rank_pos, dense_pos
FROM ranked
WHERE dense_pos <= 3
ORDER BY benchmark, rank_pos, model;
'''

HINTS = [
    'Both rankings are window functions partitioned by benchmark and ordered by score descending. They differ only in what happens after a tie.',
    '`RANK()` skips places after ties, `DENSE_RANK()` does not. Drop NULL scores in `WHERE` before ranking, so they cannot take a place.',
    'The podium filter uses a window result, so rank in a CTE and filter `WHERE dense_pos <= 3` in the outer query. A named `WINDOW w AS (...)` saves repeating the window.',
]

_T = 'CREATE TABLE scores (benchmark TEXT, model TEXT, score REAL, UNIQUE (benchmark, model));\n'
_rng = np.random.default_rng(83)
_rand = ','.join(
    f"('{b}','m{j:02d}',{'NULL' if _rng.random() < 0.1 else _rng.choice(['0.5', '0.6', '0.7', '0.75', '0.8', '0.9'])})"
    for b in ['arc', 'gsm', 'mmlu'] for j in range(12))
TESTS = [
    _T + "INSERT INTO scores VALUES ('mmlu','a',0.81),('mmlu','b',0.77),('mmlu','c',0.77),('mmlu','d',0.70),('mmlu','e',0.65);",
    _T + "INSERT INTO scores VALUES ('gsm','x',0.9),('gsm','y',NULL),('arc','x',0.4),('arc','z',0.6);",
    _T + "INSERT INTO scores VALUES ('b','p',0.5),('b','q',0.5),('b','r',0.5),('b','s',0.4),('b','t',0.3),('b','u',0.2);",
    _T + "INSERT INTO scores VALUES ('b','only',0.1),('c','dead',NULL);",                  # a benchmark with only NULLs
    _T + "INSERT INTO scores VALUES ('h','a',-1.0),('h','b',-2.0),('h','c',-2.0),('h','d',-3.0),('h','e',-4.0);",
    _T + "",
    _T + f"INSERT INTO scores VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`RANK()` returns 1 + the number of rows strictly ahead of the current row; `DENSE_RANK()` returns 1 + the number of **distinct** scores strictly ahead. For scores 0.81, 0.77, 0.77, 0.70 that is 1, 2, 2, 4 and 1, 2, 2, 3. `PARTITION BY benchmark` restarts both for each benchmark. The `WHERE score IS NOT NULL` runs before the window, so a crashed run neither gets a place nor pushes anyone down.

The podium is defined on `dense_pos`, so in test 3 it contains three tied winners plus the next two distinct scores: five rows with `rank_pos` 1, 1, 1, 4, 5.

## Ties in the output order
Tied models share `rank_pos`, so `ORDER BY benchmark, rank_pos` alone would leave their order undefined; `model` breaks the tie. Always finish an `ORDER BY` with a unique key.

## Complexity
One sort per benchmark: $O(N \log N)$. Both rankings share the same window, so the database sorts once.

## Common bugs
- `ORDER BY score DESC LIMIT 3` (the starter), which ignores benchmarks and ties.
- Keeping NULL scores: in a descending sort SQLite puts NULLs last, but they would still get a rank and could land on the podium of a small benchmark.
- Filtering on `rank_pos <= 3`, which is a different question: in test 3 it would drop the fourth-best score.

## Interview follow-ups
- Express each model's position as a percentile within its benchmark. `PERCENT_RANK()` and `CUME_DIST()` do exactly this; how do they treat ties?
- Two scores of 0.7700000001 and 0.77 should tie. Rank on `ROUND(score, 4)` instead, and discuss why "equal" floats need a stated precision.
"""
