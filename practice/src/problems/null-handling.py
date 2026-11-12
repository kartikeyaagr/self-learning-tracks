import numpy as np

META = dict(title='Forward-fill missing sensor readings in SQL', tags=['Data Processing', 'Time Series'], diff='Medium',
            runtime='sql', related=['running-totals', 'lag-and-lead', 'count-sum-avg'])

STATEMENT = r"""
Sensors report on a schedule, but some reports arrive empty. The table `readings`:

| column | type |
| `sensor` | TEXT |
| `ts` | INTEGER, the reporting time; unique per sensor |
| `value` | REAL, NULL if the report was empty |

Before feeding the series to a model you want to **forward-fill** it: each missing value is replaced by the most recent non-NULL value of the same sensor at an earlier time.

Write **one SQLite query** that returns every row with the filled value next to the original.

- `filled` equals `value` when `value` is not NULL.
- When `value` is NULL, `filled` is the latest earlier non-NULL `value` of the same sensor.
- If the sensor has no earlier non-NULL value, `filled` stays NULL. Never fill from another sensor.

## Output
- Columns, in this order: `sensor`, `ts`, `value`, `filled`.
- Sort by `sensor`, then `ts`, ascending.
"""

STARTER = '''-- readings(sensor, ts, value)
SELECT sensor, ts, value, COALESCE(value, 0) AS imputed
FROM readings
ORDER BY sensor, ts;
'''

SOLUTION = '''WITH grouped AS (
  SELECT sensor, ts, value,
         -- COUNT(value) skips NULLs, so it only steps up on rows that have a value:
         -- each non-NULL row starts a new group and the NULLs after it join that group
         COUNT(value) OVER (PARTITION BY sensor ORDER BY ts) AS grp
  FROM readings
)
SELECT sensor, ts, value,
       MAX(value) OVER (PARTITION BY sensor, grp) AS filled   -- the group's single non-NULL value
FROM grouped
ORDER BY sensor, ts;
'''

HINTS = [
    '`LAG(value)` only looks one row back, which fails when two NULLs are in a row. Think of the series as groups: a non-NULL reading followed by the NULLs after it.',
    'A running `COUNT(value)` (which skips NULLs) increases exactly at each non-NULL reading. Use it as a group number per sensor.',
    'In a CTE compute `COUNT(value) OVER (PARTITION BY sensor ORDER BY ts) AS grp`; then `MAX(value) OVER (PARTITION BY sensor, grp)` is the value that starts the group. Group 0 holds the leading NULLs and gets NULL.',
]

_T = 'CREATE TABLE readings (sensor TEXT, ts INTEGER, value REAL, UNIQUE (sensor, ts));\n'
_rng = np.random.default_rng(101)
_rand = ','.join(f"('{s}',{t},{'NULL' if _rng.random() < 0.4 else round(float(_rng.normal(20, 5)), 2)})"
                 for s in ['s1', 's2', 's3'] for t in _rng.permutation(np.arange(0, 300, 10)))
TESTS = [
    _T + "INSERT INTO readings VALUES ('a',1,20.5),('a',2,NULL),('a',3,NULL),('a',4,21.0),('a',5,NULL);",
    _T + "INSERT INTO readings VALUES ('b',1,NULL),('b',2,7.0),('c',1,3.0),('c',2,NULL);",       # leading NULL; no cross-sensor fill
    _T + "INSERT INTO readings VALUES ('d',30,NULL),('d',10,1.0),('d',20,NULL),('d',40,-2.5);",  # inserted out of order
    _T + "INSERT INTO readings VALUES ('e',1,NULL),('e',2,NULL);",                              # never any value
    _T + "INSERT INTO readings VALUES ('f',1,0.0),('f',2,NULL),('f',3,-0.5),('f',4,NULL);",     # zero is a real value
    _T + "",
    _T + f"INSERT INTO readings VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
Picture one sensor's series: `20.5, NULL, NULL, 21.0, NULL`. A running `COUNT(value)` ignores NULLs, so it produces `1, 1, 1, 2, 2`: it steps up exactly on each real reading and stays flat across the gaps. Rows with the same count form a group that starts with one non-NULL reading followed by the NULLs to fill. Inside a group there is exactly one non-NULL value, so `MAX(value)` (or `MIN`, or `FIRST_VALUE`) returns it, and every row in the group gets it. Leading NULLs have count 0, a group with no value at all, so they stay NULL.

## Why not COALESCE or LAG?
- `COALESCE(value, 0)` invents a reading of zero, which is a different imputation and usually a bad one for sensors.
- `COALESCE(value, LAG(value) OVER (...))` fills only the first NULL of a gap; the second one looks back at another NULL.
- SQLite has no `IGNORE NULLS` option for window functions. Warehouses such as Oracle, Snowflake and BigQuery support `LAST_VALUE(value IGNORE NULLS)`, which solves this directly.

## NULL rules used here
Aggregates and window aggregates skip NULLs (`COUNT(value)`, `MAX(value)`), whereas `COUNT(*)` counts rows. That difference is the whole trick.

## Complexity
Two window passes over each sensor's rows sorted by `ts`: $O(N \log N)$ for the sort, then linear.

## Interview follow-ups
- Fill only gaps of at most 3 readings and leave longer outages NULL. Also number the rows inside each group with `ROW_NUMBER() OVER (PARTITION BY sensor, grp ORDER BY ts)` and only fill when it is at most 4.
- Linear interpolation instead of forward-fill: you also need the **next** non-NULL value and its time. Run the same trick in reverse order (`ORDER BY ts DESC`).
"""
