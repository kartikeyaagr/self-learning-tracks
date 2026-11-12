import numpy as np

META = dict(title='Latency per model and region (GROUP BY)', tags=['Data Processing'], diff='Medium', runtime='sql',
            lessons=['sdsd:10.07'], related=['count-sum-avg', 'having-clause'])

STATEMENT = r"""
An inference service logs one row per request:

| column | type |
| `id` | INTEGER, unique |
| `model` | TEXT, never NULL |
| `region` | TEXT, may be NULL (region not reported) |
| `latency_ms` | INTEGER, never NULL |

Write **one SQLite query** that summarises the log per `(model, region)` pair. Requests with a NULL region form **one group per model** whose `region` is NULL.

## Output
- Columns, in this order: `model`, `region`, `n_requests`, `mean_latency`, `max_latency`, `share`.
- `n_requests`: requests in the group.
- `mean_latency`: the average latency in the group, as a REAL.
- `max_latency`: the largest latency in the group.
- `share`: the group's fraction of **that model's** requests, $n_{\text{group}} / n_{\text{model}}$, a REAL in $(0, 1]$.
- Sort by `model`, then `region`, ascending. The NULL region comes first within its model (SQLite's default for ascending order).
"""

STARTER = '''-- requests(id, model, region, latency_ms)
SELECT model, COUNT(*) AS n_requests
FROM requests
GROUP BY model;
'''

SOLUTION = '''SELECT model,
       region,
       COUNT(*)        AS n_requests,
       AVG(latency_ms) AS mean_latency,
       MAX(latency_ms) AS max_latency,
       -- a window over the grouped rows: total requests of the same model
       1.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY model) AS share
FROM requests
GROUP BY model, region
ORDER BY model, region;
'''

HINTS = [
    '`GROUP BY model, region` makes one output row per distinct pair, and SQL already puts all NULL regions of a model into one group.',
    '`COUNT(*)`, `AVG(...)` and `MAX(...)` summarise each group. For `share` you also need the model total, which is a sum over several groups.',
    'Window functions run after grouping, so `SUM(COUNT(*)) OVER (PARTITION BY model)` is the model total. Multiply by `1.0` first so the division is not an integer division. A join to a per-model subquery works too.',
]

_T = 'CREATE TABLE requests (id INTEGER PRIMARY KEY, model TEXT NOT NULL, region TEXT, latency_ms INTEGER NOT NULL);\n'
_rng = np.random.default_rng(13)
_rand = ','.join(
    f"({i},'{_rng.choice(['llama', 'mistral', 'phi'])}',"
    f"{_rng.choice([repr('eu'), repr('us'), repr('ap'), 'NULL'])},{int(_rng.integers(20, 2000))})" for i in range(1, 81))
TESTS = [
    _T + "INSERT INTO requests VALUES (1,'small','eu',40),(2,'small','eu',60),(3,'small','us',90),(4,'big','us',300);",
    _T + "INSERT INTO requests VALUES (1,'m','us',10),(2,'m',NULL,30),(3,'m',NULL,50),(4,'m','ap',20);",
    _T + "INSERT INTO requests VALUES (1,'a','eu',7);",
    _T + "INSERT INTO requests VALUES (1,'x',NULL,5),(2,'x',NULL,6),(3,'y',NULL,100);",       # only NULL regions
    _T + "INSERT INTO requests VALUES (1,'z','eu',1),(2,'z','eu',2),(3,'z','EU',3);",         # case matters
    _T + "",
    _T + f"INSERT INTO requests VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`GROUP BY model, region` partitions the rows by the pair and each aggregate collapses its group to one value. For grouping, SQL treats all NULLs as equal, so a model's unreported requests form one group (test 2). `AVG` returns a REAL even for INTEGER input.

The `share` column needs a total that spans several groups. Window functions are evaluated **after** `GROUP BY`, over the grouped rows, so `SUM(COUNT(*)) OVER (PARTITION BY model)` adds up the group counts of the same model. The equivalent without windows joins to a subquery:

```
JOIN (SELECT model, COUNT(*) AS n FROM requests GROUP BY model) t USING (model)
```

## Complexity
One sort or hash over $(model, region)$: $O(N \log N)$ or expected $O(N)$, plus a tiny window pass over the $G$ groups.

## Common bugs
- `COUNT(*) / SUM(COUNT(*)) OVER (...)` without the `1.0 *`: integer division gives 0 for every group except a model's only one.
- Selecting a column that is neither grouped nor aggregated. SQLite accepts it and returns a value from an arbitrary row; PostgreSQL rejects it. Don't rely on it.
- `GROUP BY model` only, which mixes regions together.

## Interview follow-ups
- The p95 latency per group. SQLite has no `PERCENTILE_CONT`; you can rank rows with `ROW_NUMBER()` inside each group and pick the one at position `CEIL(0.95 * n)`.
- How would you also include a "total" row per model (all regions)? `GROUP BY ... WITH ROLLUP` in some databases, or a `UNION ALL` of the per-model aggregate in SQLite.
"""
