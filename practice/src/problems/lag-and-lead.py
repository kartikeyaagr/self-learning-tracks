import numpy as np

META = dict(title='Sessionize a clickstream (LAG and LEAD)', tags=['Time Series', 'Data Processing'], diff='Hard', runtime='sql',
            related=['running-totals', 'null-handling', 'row-number'])

STATEMENT = r"""
The table `events` is a clickstream:

| column | type |
| `event_id` | INTEGER, unique |
| `user_id` | INTEGER |
| `ts` | INTEGER, seconds |

A **session** is a run of one user's events in which each event follows the previous one by **at most 1800 seconds**. A gap of more than 1800 seconds (strictly) starts a new session. Several events may share a timestamp; order a user's events by `ts`, then `event_id`.

Write **one SQLite query** that returns one row per session:

| column | meaning |
| `user_id` | the user |
| `session_no` | 1 for the user's first session, 2 for the next, ... |
| `start_ts` | `ts` of the session's first event |
| `end_ts` | `ts` of its last event |
| `n_events` | number of events in the session |
| `gap_after` | seconds from this session's `end_ts` to the start of the user's **next** session; NULL for the user's last session |

## Output
- Columns in the order of the table above.
- Sort by `user_id`, then `session_no`.
"""

STARTER = '''-- events(event_id, user_id, ts)
SELECT user_id, 1 AS session_no, MIN(ts) AS start_ts, MAX(ts) AS end_ts
FROM events
GROUP BY user_id;
'''

SOLUTION = '''WITH gaps AS (            -- 1. time since the user's previous event (NULL for the first)
  SELECT user_id, event_id, ts,
         ts - LAG(ts) OVER (PARTITION BY user_id ORDER BY ts, event_id) AS gap
  FROM events
),
numbered AS (             -- 2. a running count of session starts numbers the sessions
  SELECT user_id, ts,
         SUM(CASE WHEN gap IS NULL OR gap > 1800 THEN 1 ELSE 0 END)
           OVER (PARTITION BY user_id ORDER BY ts, event_id
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_no
  FROM gaps
)
SELECT user_id, session_no,
       MIN(ts)  AS start_ts,
       MAX(ts)  AS end_ts,
       COUNT(*) AS n_events,
       -- 3. look ahead one session: next start minus this end
       LEAD(MIN(ts)) OVER (PARTITION BY user_id ORDER BY session_no) - MAX(ts) AS gap_after
FROM numbered
GROUP BY user_id, session_no
ORDER BY user_id, session_no;
'''

HINTS = [
    'Work in three steps: find where sessions start, number the sessions, then aggregate each session. CTEs keep the steps readable.',
    '`ts - LAG(ts) OVER (PARTITION BY user_id ORDER BY ts, event_id)` is the gap to the previous event. A row starts a session when that gap is NULL or above 1800; a running `SUM` of that 0/1 flag is the session number.',
    'Group by `(user_id, session_no)` for `MIN(ts)`, `MAX(ts)` and `COUNT(*)`. Then `LEAD(MIN(ts)) OVER (PARTITION BY user_id ORDER BY session_no) - MAX(ts)` is the gap to the next session; LEAD returns NULL after the last one.',
]

_T = 'CREATE TABLE events (event_id INTEGER PRIMARY KEY, user_id INTEGER, ts INTEGER);\n'
_rng = np.random.default_rng(107)
_rand, _eid = [], 1
for _u in range(1, 6):
    _t = int(_rng.integers(0, 5000))
    for _ in range(int(_rng.integers(1, 25))):
        _t += int(_rng.choice([0, 30, 600, 1800, 1801, 7200]))
        _rand.append(f"({_eid},{_u},{_t})")
        _eid += 1
_rng.shuffle(_rand)
TESTS = [
    _T + "INSERT INTO events VALUES (1,1,0),(2,1,600),(3,1,5000),(4,1,5100),(5,2,100);",
    _T + "INSERT INTO events VALUES (1,7,1000),(2,7,2800),(3,7,4601),(4,7,4601);",            # exactly 1800 vs 1801, equal ts
    _T + "INSERT INTO events VALUES (1,3,50);",
    _T + "INSERT INTO events VALUES (9,4,9000),(3,4,0),(5,4,4000),(1,4,10000),(2,4,20000);",   # inserted out of order
    _T + "INSERT INTO events VALUES (1,5,10),(2,5,10),(3,5,10),(4,6,10);",
    _T + "",
    _T + f"INSERT INTO events VALUES {','.join(_rand)};",
]

EXPLANATION = r"""
## Why it works
1. `LAG(ts)` returns the previous row's value in the window order, so `ts - LAG(ts)` is the gap to the user's previous event; it is NULL on the user's first event.
2. Flag a row 1 if it starts a session (first event, or gap > 1800) and 0 otherwise. A running sum of the flags is 1 throughout the first session, 2 throughout the second, and so on: the classic **gaps-and-islands** pattern.
3. `GROUP BY user_id, session_no` collapses each session. Window functions run after grouping, so `LEAD(MIN(ts))` reads the next session's start, and `LEAD` returns NULL after the last row of the partition.

## Ties and frames
Events with equal `ts` have gap 0 and stay in the same session. The running sum uses an explicit `ROWS` frame and the unique key `(ts, event_id)`. With the default `RANGE` frame and `ORDER BY ts` alone, tied rows would be peers and get the same partial sum even if one of them started a session.

## Complexity
Two window passes and one group-by over each user's events sorted by time: $O(N \log N)$. The equivalent self-join approach ("find the previous event with a correlated subquery") is $O(N^2)$ in the worst case.

## Common bugs
- `gap >= 1800` instead of `> 1800` (test 2 has a gap of exactly 1800).
- Forgetting `gap IS NULL`: the first event then gets flag 0 and the first session is numbered 0.
- Partitioning `LAG` by nothing, so one user's last event is compared with the next user's first.

## Interview follow-ups
- Average session length and events per session per user: wrap this query in a CTE and aggregate again.
- Use a 30-minute **inactivity** timeout but also cap sessions at 4 hours. The cap depends on the session's start, so a single LAG pass is no longer enough; how would you handle it (recursive CTE or application code)?
"""
