import numpy as np

META = dict(title='Sort a model leaderboard with tie-breakers (ORDER BY)', tags=['Data Processing'], diff='Easy',
            runtime='sql', related=['limit-and-offset', 'rank-and-dense-rank'])

STATEMENT = r"""
A competition leaderboard is stored in the table `leaderboard`:

| column | type |
| `team` | TEXT, unique |
| `accuracy` | REAL, never NULL |
| `latency_ms` | INTEGER, inference latency, or NULL if it was not measured |

Write **one SQLite query** that returns the leaderboard in display order:

1. Higher `accuracy` first.
2. Among equal accuracies, lower `latency_ms` first. A team whose latency was not measured (NULL) comes **after** every measured team with the same accuracy.
3. Remaining ties: `team` in alphabetical order (ascending).

## Output
- Columns, in this order: `team`, `accuracy`, `latency_ms`.
- One row per team, in the order above.
"""

STARTER = '''-- leaderboard(team, accuracy, latency_ms)
SELECT team, accuracy
FROM leaderboard
ORDER BY accuracy;
'''

SOLUTION = '''SELECT team, accuracy, latency_ms
FROM leaderboard
ORDER BY accuracy DESC,
         latency_ms IS NULL,   -- 0 for measured, 1 for NULL: measured teams first
         latency_ms,
         team;
'''

HINTS = [
    '`ORDER BY` takes a list of keys. The second key only matters when the first one ties, and so on.',
    'Each key has its own direction: `accuracy DESC`, then `latency_ms` ascending. Check where SQLite puts NULL in an ascending sort.',
    'SQLite treats NULL as smaller than every value, so it sorts first in ascending order. Add the key `latency_ms IS NULL` (0 or 1) before `latency_ms`, or write `latency_ms NULLS LAST`.',
]

_T = 'CREATE TABLE leaderboard (team TEXT PRIMARY KEY, accuracy REAL NOT NULL, latency_ms INTEGER);\n'
_rng = np.random.default_rng(5)
_rand = ','.join(
    f"('t{i:02d}',{_rng.choice([0.81, 0.85, 0.9, 0.93])},"
    f"{'NULL' if _rng.random() < 0.25 else int(_rng.choice([12, 20, 35, 50]))})" for i in _rng.permutation(40))
TESTS = [
    _T + "INSERT INTO leaderboard VALUES ('ants',0.91,40),('bees',0.95,80),('cats',0.91,25);",
    _T + "INSERT INTO leaderboard VALUES ('owl',0.88,NULL),('elk',0.88,30),('fox',0.92,NULL);",
    _T + "INSERT INTO leaderboard VALUES ('d',0.9,10),('b',0.9,10),('c',0.9,NULL),('a',0.9,NULL);",   # full ties
    _T + "INSERT INTO leaderboard VALUES ('solo',0.5,NULL);",
    _T + "INSERT INTO leaderboard VALUES ('x',0.99,1000),('y',0.1,1),('z',0.99,999);",
    _T + "",
    _T + f"INSERT INTO leaderboard VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`ORDER BY a DESC, b, c` sorts by `a`; rows that tie on `a` are ordered by `b`; rows tied on both by `c`. Because `team` is unique, the last key makes the order total, so the output is fully determined.

## Where NULLs go
SQLite (like SQL Server and MySQL) treats NULL as smaller than any value: first in `ASC`, last in `DESC`. PostgreSQL and Oracle do the opposite. To be explicit and portable:

- `ORDER BY latency_ms IS NULL, latency_ms` works everywhere. `IS NULL` is 0 or 1, so measured rows (0) come first.
- `ORDER BY latency_ms NULLS LAST` is standard SQL and works in SQLite 3.30+ and PostgreSQL.

Here `accuracy DESC` never meets a NULL because the column is `NOT NULL`, but `latency_ms` ascending would put unmeasured teams at the top of each accuracy group.

## Complexity
A sort, $O(N \log N)$. An index on `(accuracy DESC, latency_ms, team)` lets the database read rows already in order.

## Interview follow-ups
- Show only the top 10. Add `LIMIT 10`, and explain why the tie-breakers matter even more then: without them, which rows make the cut can change between runs.
- Accuracy is a float. Two teams at 0.9 and 0.9000000001 look tied on screen but are not. How would you treat them as tied? Round first: `ORDER BY ROUND(accuracy, 3) DESC, ...`.
"""
