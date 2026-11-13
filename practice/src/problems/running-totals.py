import numpy as np

META = dict(title='Cumulative GPU-hours per team (running totals over daily sums)', tags=['Data Processing', 'Time Series'],
            diff='Medium', runtime='sql', related=['sql-running-total', 'lag-and-lead', 'group-by'])

STATEMENT = r"""
A shared cluster logs every finished job:

| column | type |
| `job_id` | INTEGER, unique |
| `team` | TEXT |
| `day` | INTEGER, day of the quarter (1, 2, ...) |
| `gpu_hours` | REAL, greater than 0 |

Teams have a quarterly budget and want to see how fast they are burning it. Write **one SQLite query** that returns one row per `(team, day)` on which the team ran at least one job (days without jobs are **not** filled in):

| column | meaning |
| `team` | the team |
| `day` | the day |
| `day_hours` | total GPU-hours of the team's jobs that day |
| `cum_hours` | the team's total from its first day up to and including this day |
| `pct_of_team` | `cum_hours` as a percentage of the team's quarter total, $100 \cdot \text{cum} / \text{total}$ |

## Output
- Columns in the order of the table above; the last three are REALs.
- Sort by `team`, then `day`, ascending.
"""

STARTER = '''-- jobs(job_id, team, day, gpu_hours)
SELECT team, day, gpu_hours AS day_hours
FROM jobs
ORDER BY team, day;
'''

SOLUTION = '''SELECT team,
       day,
       SUM(gpu_hours) AS day_hours,
       -- windows run after GROUP BY, so they can aggregate the daily sums
       SUM(SUM(gpu_hours)) OVER (PARTITION BY team ORDER BY day) AS cum_hours,
       100.0 * SUM(SUM(gpu_hours)) OVER (PARTITION BY team ORDER BY day)
             / SUM(SUM(gpu_hours)) OVER (PARTITION BY team)       AS pct_of_team
FROM jobs
GROUP BY team, day
ORDER BY team, day;
'''

HINTS = [
    'Two steps: collapse the jobs to one row per (team, day), then accumulate those daily rows in order of day.',
    'Window functions are evaluated after `GROUP BY`, so `SUM(SUM(gpu_hours)) OVER (PARTITION BY team ORDER BY day)` is a running total of daily sums. A CTE with the daily sums followed by a plain window is equivalent.',
    'The team total is the same window without `ORDER BY`: `SUM(SUM(gpu_hours)) OVER (PARTITION BY team)`. Divide with `100.0 *` in front.',
]

_T = 'CREATE TABLE jobs (job_id INTEGER PRIMARY KEY, team TEXT, day INTEGER, gpu_hours REAL);\n'
_rng = np.random.default_rng(97)
_rand = ','.join(f"({i},'{_rng.choice(['infra', 'nlp', 'vision'])}',{int(_rng.integers(1, 15))},"
                 f"{int(_rng.integers(1, 200)) / 4})" for i in range(1, 91))
TESTS = [
    _T + "INSERT INTO jobs VALUES (1,'nlp',1,10.0),(2,'nlp',2,30.0),(3,'nlp',2,10.0),(4,'nlp',5,50.0);",
    _T + "INSERT INTO jobs VALUES (1,'cv',3,2.0),(2,'ml',1,8.0),(3,'cv',1,6.0),(4,'ml',4,8.0);",
    _T + "INSERT INTO jobs VALUES (1,'a',7,1.5),(2,'a',7,2.5),(3,'a',7,4.0);",                 # a single day
    _T + "INSERT INTO jobs VALUES (1,'big',1,1000000.0),(2,'big',2,0.25),(3,'big',90,1.0);",
    _T + "INSERT INTO jobs VALUES (5,'z',2,3.0),(4,'z',1,1.0),(3,'y',2,1.0),(2,'y',1,1.0),(1,'z',3,4.0);",
    _T + "",
    _T + f"INSERT INTO jobs VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
SQL evaluates `GROUP BY` before window functions. After grouping, each row is one team-day and `SUM(gpu_hours)` is its daily total. A window over those rows can aggregate the aggregate: `SUM(SUM(gpu_hours)) OVER (PARTITION BY team ORDER BY day)` adds the daily totals of the team up to the current day, and the same window without `ORDER BY` spans the whole partition, giving the team total for the percentage.

## Frames and ties
With `ORDER BY` and no frame clause, the default frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes all rows that tie with the current one on `day`. That would be a problem on the raw job rows (two jobs on day 2 would both show the total through both), but after grouping there is exactly one row per team-day, so `RANGE` and `ROWS` agree. Aggregating first is what makes the running total per **day** rather than per job.

## Complexity
A group-by and one sort of the $G$ daily rows per team: $O(N + G \log G)$.

## Common bugs
- Running the window over the raw jobs and then de-duplicating, which reports per-job intermediate totals.
- `SUM(gpu_hours) OVER (...)` without the inner `SUM` while also using `GROUP BY`: the window then sums one arbitrary job per day.
- Integer division in the percentage. Here `gpu_hours` is REAL, but writing `100.0 *` keeps it safe if the column ever becomes INTEGER.

## Interview follow-ups
- Fill the missing days with `day_hours = 0` so the burn chart has a point per day. Generate a day spine with a recursive CTE and left-join the daily sums onto it.
- Flag the first day each team crossed 80% of its total. Compute `pct_of_team` in a CTE, then take `MIN(day)` where it is at least 80, per team.
"""
