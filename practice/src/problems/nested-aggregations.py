import numpy as np

META = dict(title='Per-country spend: aggregate the per-user totals', tags=['Data Processing'], diff='Hard', runtime='sql',
            related=['having-clause', 'common-table-expressions', 'subqueries'])

STATEMENT = r"""
The table `purchases` has one row per purchase. Amounts are in cents; refunds are negative.

| column | type |
| `id` | INTEGER, unique |
| `user_id` | INTEGER |
| `country` | TEXT, never NULL; the same for every purchase of a given user |
| `amount` | INTEGER, cents, may be negative |

A user's **total** is the sum of all their amounts. Write **one SQLite query** that summarises the user totals per country:

| column | meaning |
| `country` | the country |
| `n_users` | number of users in the country |
| `avg_user_total` | the mean of the users' totals (each user counts once, however many purchases they made), a REAL |
| `max_user_total` | the largest user total, an INTEGER |
| `n_above_avg` | number of users whose total is **strictly greater** than `avg_user_total` |

## Output
- Columns in the order of the table above.
- Sort by `country` ascending.
"""

STARTER = '''-- purchases(id, user_id, country, amount)
SELECT country, AVG(amount) AS avg_user_total
FROM purchases
GROUP BY country;
'''

SOLUTION = '''WITH user_totals AS (            -- level 1: one row per user
  SELECT user_id, country, SUM(amount) AS total
  FROM purchases
  GROUP BY user_id, country
),
country_avg AS (                 -- level 2: one row per country
  SELECT country,
         COUNT(*)   AS n_users,
         AVG(total) AS avg_user_total,
         MAX(total) AS max_user_total
  FROM user_totals
  GROUP BY country
)
SELECT c.country, c.n_users, c.avg_user_total, c.max_user_total,
       -- level 3: compare each user back to their own country's average
       SUM(CASE WHEN u.total > c.avg_user_total THEN 1 ELSE 0 END) AS n_above_avg
FROM country_avg AS c
JOIN user_totals AS u ON u.country = c.country
GROUP BY c.country, c.n_users, c.avg_user_total, c.max_user_total
ORDER BY c.country;
'''

HINTS = [
    '`AVG(amount)` per country averages purchases, not users: a user with ten purchases weighs ten times as much. You need two levels of grouping.',
    'Step 1: totals per user (`GROUP BY user_id, country`). Step 2: aggregate those rows per country. Name each step with a CTE: `WITH user_totals AS (...), country_avg AS (...)`.',
    'For `n_above_avg`, join `user_totals` back to the per-country results on `country` and count `CASE WHEN u.total > c.avg_user_total THEN 1 ELSE 0 END`. A window `AVG(total) OVER (PARTITION BY country)` over `user_totals` also works.',
]

_T = 'CREATE TABLE purchases (id INTEGER PRIMARY KEY, user_id INTEGER, country TEXT NOT NULL, amount INTEGER);\n'
_rng = np.random.default_rng(61)
_ucountry = {u: str(_rng.choice(['BR', 'DE', 'IN', 'JP', 'US'])) for u in range(1, 31)}
_rand = []
for _i in range(1, 151):
    _u = int(_rng.integers(1, 31))
    _amt = int(_rng.integers(100, 20000)) * (-1 if _rng.random() < 0.1 else 1)
    _rand.append(f"({_i},{_u},'{_ucountry[_u]}',{_amt})")
TESTS = [
    _T + "INSERT INTO purchases VALUES (1,1,'DE',500),(2,1,'DE',500),(3,1,'DE',500),(4,2,'DE',300),(5,3,'FR',800);",
    _T + "INSERT INTO purchases VALUES (1,10,'US',1000),(2,11,'US',200),(3,11,'US',100),(4,12,'US',600),(5,12,'US',-300);",
    _T + "INSERT INTO purchases VALUES (1,1,'JP',250),(2,2,'JP',100),(3,2,'JP',150),(4,3,'JP',250);",   # all equal: none above
    _T + "INSERT INTO purchases VALUES (1,7,'IN',-500),(2,8,'IN',-100);",                               # refunds only
    _T + "INSERT INTO purchases VALUES (1,1,'BR',4000000000),(2,1,'BR',4000000000),(3,2,'BR',1);",     # large sums
    _T + "",
    _T + f"INSERT INTO purchases VALUES {','.join(_rand)};",
]

EXPLANATION = r"""
## Why it works
The question has three levels, and SQL lets you aggregate only one level per `GROUP BY`:

1. `user_totals` collapses purchases to one row per user.
2. `country_avg` collapses users to one row per country. Now `AVG(total)` weights every user equally.
3. To count users above their country's average, each user total must be compared with a number that was computed from the whole country. Joining the two CTEs puts both on the same row, and a final `GROUP BY` with a conditional sum counts the users above.

Including `country` in the first `GROUP BY` is safe because it is constant per user, and it saves a join later.

## Why not AVG(amount)?
In test 1, user 1 spends 1500 cents in three purchases and user 2 spends 300 in one. The purchase average is $1800 / 4 = 450$, but the user average is $(1500 + 300)/2 = 900$. Averages of averages, or averages over the wrong unit, are one of the most common analytics bugs.

## Exactness
Amounts are INTEGER cents, so the user totals are exact and `AVG` divides an exact sum once. In test 3 every user totals 250, the average is exactly 250.0, and nobody is strictly above it. With REAL amounts, rounding could push a user a hair above an average they really equal.

## Complexity
Two group-bys and one join on the small per-country table: $O(N \log N)$ overall for $N$ purchases.

## Common bugs
- `AVG(amount)` per country (the starter).
- `COUNT(*)` in level 2 computed on `purchases` instead of `user_totals`, which counts purchases.
- Putting `AVG(total)` directly inside `SUM(CASE WHEN total > AVG(total) ...)`: aggregates cannot be nested in one `SELECT`.

## Interview follow-ups
- Return the top-spending user per country, with ties broken by the smallest `user_id`. Add `ROW_NUMBER() OVER (PARTITION BY country ORDER BY total DESC, user_id)` to `user_totals`.
- Report the **median** user total per country instead of the mean. Why is the median a better summary for spend data?
"""
