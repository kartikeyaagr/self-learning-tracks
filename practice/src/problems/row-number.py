import numpy as np

META = dict(title='Keep the latest version of each record (ROW_NUMBER)', tags=['Data Processing'], diff='Medium',
            runtime='sql', related=['rank-and-dense-rank', 'top-n-per-group'])

STATEMENT = r"""
A change-data-capture feed appends a new row every time a user's profile changes, so the table `profile_versions` holds the full history:

| column | type |
| `version_id` | INTEGER, unique |
| `user_id` | INTEGER |
| `email` | TEXT |
| `updated_at` | INTEGER, Unix seconds |

Clock resolution is coarse, so two versions of the same user can share an `updated_at`. Write **one SQLite query** that returns the **current** profile of every user: the version with the largest `updated_at`, and among those the largest `version_id`. Also report how many versions the user has.

## Output
- Columns, in this order: `user_id`, `email`, `updated_at`, `n_versions`.
- Exactly one row per user.
- Sort by `user_id` ascending.
"""

STARTER = '''-- profile_versions(version_id, user_id, email, updated_at)
SELECT user_id, email, MAX(updated_at) AS updated_at
FROM profile_versions
GROUP BY user_id;
'''

SOLUTION = '''WITH ranked AS (
  SELECT user_id, email, updated_at,
         ROW_NUMBER() OVER (PARTITION BY user_id
                            ORDER BY updated_at DESC, version_id DESC) AS rn,
         COUNT(*) OVER (PARTITION BY user_id) AS n_versions
  FROM profile_versions
)
SELECT user_id, email, updated_at, n_versions
FROM ranked
WHERE rn = 1          -- window results can only be filtered one level up
ORDER BY user_id;
'''

HINTS = [
    'Number the versions of each user from newest to oldest; the current profile is number 1.',
    '`ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC, version_id DESC)` gives each user\'s versions the numbers 1, 2, 3, ... with no ties. `COUNT(*) OVER (PARTITION BY user_id)` gives the version count on every row.',
    'A window function cannot appear in `WHERE` of the same `SELECT`. Compute it in a CTE, then `SELECT ... FROM ranked WHERE rn = 1`.',
]

_T = 'CREATE TABLE profile_versions (version_id INTEGER PRIMARY KEY, user_id INTEGER, email TEXT, updated_at INTEGER);\n'
_rng = np.random.default_rng(71)
_rand = ','.join(f"({v},{int(_rng.integers(1, 16))},'e{v}@mail.test',{1700000000 + int(_rng.integers(0, 6)) * 60})"
                 for v in _rng.permutation(np.arange(1, 61)))
TESTS = [
    _T + "INSERT INTO profile_versions VALUES (1,7,'old@x.io',100),(2,7,'new@x.io',200),(3,8,'b@x.io',150);",
    _T + "INSERT INTO profile_versions VALUES (10,1,'first@x.io',500),(11,1,'second@x.io',500),(9,1,'past@x.io',400);",  # tie on time
    _T + "INSERT INTO profile_versions VALUES (5,2,'late@x.io',900),(6,2,'early@x.io',100);",                   # ids out of time order
    _T + "INSERT INTO profile_versions VALUES (1,3,'a@x.io',1),(2,3,'a@x.io',1),(3,3,'a@x.io',1);",
    _T + "INSERT INTO profile_versions VALUES (1,4,'solo@x.io',42);",
    _T + "",
    _T + f"INSERT INTO profile_versions VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`ROW_NUMBER()` assigns consecutive integers 1, 2, 3, ... within each partition in the window's order, and never repeats a number. With the order `updated_at DESC, version_id DESC`, row 1 is the newest version, and the `version_id` tie-breaker makes "newest" unique even when timestamps collide (test 2). `COUNT(*) OVER (PARTITION BY user_id)` attaches the partition size to every row without collapsing them. The outer query keeps row 1.

## Why GROUP BY with MAX is wrong
`SELECT user_id, email, MAX(updated_at) ... GROUP BY user_id` returns the right timestamp, but `email` is not aggregated. SQLite happens to take `email` from the row holding the max (a documented "bare column" quirk), yet on a tie it picks either one, and other databases reject the query. Joining back on `(user_id, MAX(updated_at))` returns two rows on a tie.

## Complexity
One sort per partition, $O(N \log N)$ overall. An index on `(user_id, updated_at DESC, version_id DESC)` lets the database skip the sort.

## Common bugs
- Filtering `WHERE ROW_NUMBER() OVER (...) = 1` directly: window functions are evaluated after `WHERE`.
- Using `RANK()`, which gives both tied versions rank 1 and returns two rows for the user.
- Ordering ascending and keeping `rn = 1`, which returns the oldest version.

## Interview follow-ups
- Return each user's previous email as well (the second-newest version). Keep `rn <= 2` and pivot, or use `LEAD(email)` in the same window.
- The table has a billion rows and you need this daily. How would you maintain a "current profiles" table incrementally instead of recomputing it?
"""
