import numpy as np

META = dict(title='Weekly active users from timestamps (date functions)', tags=['Time Series', 'Data Processing'], diff='Medium',
            runtime='sql', lessons=['sdsd:10.06'], related=['group-by', 'running-totals'])

STATEMENT = r"""
The table `events` logs product usage:

| column | type |
| `id` | INTEGER, unique |
| `user_id` | INTEGER |
| `ts` | TEXT, UTC timestamp `'YYYY-MM-DD HH:MM:SS'` |

Write **one SQLite query** that reports **weekly active users**. Weeks start on **Monday**: every event belongs to the week that begins on the Monday on or before its date.

## Output
- Columns, in this order: `week_start`, `active_users`, `n_events`.
- `week_start` is that Monday as TEXT `'YYYY-MM-DD'`.
- `active_users` is the number of different users with at least one event that week; `n_events` counts the events.
- One row per week that has at least one event (weeks with none are not listed).
- Sort by `week_start` ascending.
"""

STARTER = '''-- events(id, user_id, ts)
SELECT date(ts) AS day, COUNT(*) AS n_events
FROM events
GROUP BY date(ts);
'''

SOLUTION = '''SELECT
  -- step back 6 days, then forward to the next Monday (weekday 1): lands on this week's Monday
  date(ts, '-6 days', 'weekday 1') AS week_start,
  COUNT(DISTINCT user_id)          AS active_users,
  COUNT(*)                         AS n_events
FROM events
GROUP BY week_start
ORDER BY week_start;
'''

HINTS = [
    'Map every timestamp to the Monday of its week, then `GROUP BY` that date. SQLite\'s `date(ts, modifier, ...)` applies modifiers left to right and drops the time of day.',
    "The modifier `'weekday 1'` moves a date **forward** to the next Monday, or leaves it alone if it already is one. You need the Monday on or **before** the date.",
    "`date(ts, '-6 days', 'weekday 1')`: going back 6 days first means moving forward to a Monday lands in the original week. Count users with `COUNT(DISTINCT user_id)`.",
]

_T = 'CREATE TABLE events (id INTEGER PRIMARY KEY, user_id INTEGER, ts TEXT);\n'
_rng = np.random.default_rng(139)
_base = np.datetime64('2024-12-20T00:00:00')
_rand = ','.join(
    f"({i},{int(_rng.integers(1, 15))},'{str(_base + np.timedelta64(int(_rng.integers(0, 40 * 86400)), 's')).replace('T', ' ')}')"
    for i in range(1, 121))
TESTS = [
    _T + "INSERT INTO events VALUES (1,1,'2024-03-04 09:00:00'),(2,2,'2024-03-06 18:30:00'),"
         "(3,1,'2024-03-10 23:59:59'),(4,1,'2024-03-11 00:00:00');",                  # Mon, Wed, Sun, next Mon
    _T + "INSERT INTO events VALUES (1,5,'2024-01-02 10:00:00'),(2,5,'2024-01-02 11:00:00'),(3,6,'2024-01-20 08:00:00');",
    _T + "INSERT INTO events VALUES (1,1,'2024-12-30 12:00:00'),(2,2,'2025-01-05 12:00:00'),(3,3,'2024-12-29 12:00:00');",  # across a year
    _T + "INSERT INTO events VALUES (1,1,'2024-02-29 00:00:00'),(2,1,'2024-02-26 00:00:00'),(3,1,'2024-03-03 23:00:00');",  # leap day
    _T + "INSERT INTO events VALUES (1,9,'2023-10-01 00:00:00');",                                               # a Sunday
    _T + "",
    _T + f"INSERT INTO events VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
SQLite stores dates as text and computes with them through `date()`, `datetime()`, `strftime()` and `julianday()`, each taking a value and a list of **modifiers** applied left to right. `'weekday N'` advances the date to the next day whose weekday is N (0 = Sunday, 1 = Monday), or does nothing if it already is. Subtracting 6 days first means the next Monday is never later than the original date:

- Monday 2024-03-04: back to Tuesday 02-27, forward to Monday 03-04.
- Sunday 2024-03-10: back to Monday 03-04, which stays.

`date()` also drops the time component, so `23:59:59` on Sunday still lands in the same week (test 1). Grouping by the result aggregates whole weeks, including across a year boundary (test 3) where `strftime('%W')` week numbers would reset.

## Complexity
One pass to compute the key and a hash or sort for the grouping: $O(N \log N)$. `COUNT(DISTINCT ...)` per group adds memory proportional to the number of users per week.

## Common bugs
- `date(ts, 'weekday 1')` alone, which moves Tuesday to **next** Monday: the wrong week.
- Grouping by `strftime('%Y-%W', ts)`. `%W` counts weeks from the first Monday of the year, so the days before it fall into week 00, and the last days of December and the first days of January end up in different groups although they are the same Monday-to-Sunday week.
- Forgetting time zones. The timestamps here are UTC; "weekly" for a user in Tokyo needs `date(ts, '+9 hours', ...)` first.

## Interview follow-ups
- Return a row for every week in the range, with 0 for weeks without events: generate weeks with a recursive CTE (`date(w, '+7 days')`) and left-join.
- Weekly **retention**: of the users active in week $w$, what fraction were active in week $w+1$? Self-join the per-user-week activity on `date(a.week_start, '+7 days') = b.week_start`.
"""
