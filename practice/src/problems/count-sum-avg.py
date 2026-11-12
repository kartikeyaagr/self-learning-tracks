import numpy as np

META = dict(title='Token usage summary: COUNT, SUM and AVG with NULLs', tags=['Data Processing'], diff='Easy', runtime='sql',
            lessons=['sdsd:10.04'], related=['group-by', 'null-handling'])

STATEMENT = r"""
An LLM API logs one row per request. A request that failed has no completion, so its `completion_tokens` is NULL.

| column | type |
| `request_id` | INTEGER, unique |
| `user_id` | TEXT, never NULL |
| `prompt_tokens` | INTEGER, never NULL |
| `completion_tokens` | INTEGER, NULL if the request failed |

Write **one SQLite query** that returns a **single row** summarising the whole table.

## Output
Columns, in this order:

| column | meaning |
| `n_requests` | number of rows |
| `n_completed` | number of rows with a completion |
| `n_users` | number of different users |
| `total_tokens` | all prompt tokens plus all completion tokens; a failed request still counts its prompt tokens |
| `avg_completion` | the average `completion_tokens` over **completed** requests only, a REAL |

On an empty table the counts and `total_tokens` are `0` (INTEGERs) and `avg_completion` is NULL.
"""

STARTER = '''-- usage(request_id, user_id, prompt_tokens, completion_tokens)
SELECT COUNT(*) AS n_requests
FROM usage;
'''

SOLUTION = '''SELECT COUNT(*)                 AS n_requests,
       COUNT(completion_tokens) AS n_completed,     -- COUNT(col) skips NULLs
       COUNT(DISTINCT user_id)  AS n_users,
       -- SUM of an empty set is NULL, so default each half to 0
       COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) AS total_tokens,
       AVG(completion_tokens)   AS avg_completion   -- AVG also skips NULLs
FROM usage;
'''

HINTS = [
    'An aggregate without `GROUP BY` treats the whole table as one group and always returns exactly one row.',
    '`COUNT(*)` counts rows, `COUNT(col)` counts non-NULL values, `COUNT(DISTINCT col)` counts different non-NULL values. `SUM` and `AVG` ignore NULLs too.',
    '`SUM(prompt_tokens + completion_tokens)` loses the prompts of failed requests because `x + NULL` is NULL. Sum the columns separately, and wrap each `SUM` in `COALESCE(..., 0)` for the empty table.',
]

_T = 'CREATE TABLE usage (request_id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER);\n'
_rng = np.random.default_rng(37)
_rand = ','.join(
    f"({i},'u{int(_rng.integers(1, 12))}',{int(_rng.integers(10, 4000))},"
    f"{'NULL' if _rng.random() < 0.2 else int(_rng.integers(1, 1500))})" for i in range(1, 101))
TESTS = [
    _T + "INSERT INTO usage VALUES (1,'ann',100,50),(2,'ann',200,150),(3,'bob',50,100);",
    _T + "INSERT INTO usage VALUES (1,'ann',120,NULL),(2,'bob',80,40),(3,'bob',30,NULL);",     # failed requests
    _T + "INSERT INTO usage VALUES (1,'cat',10,NULL),(2,'cat',20,NULL);",                      # nothing completed
    _T + "",
    _T + "INSERT INTO usage VALUES (1,'big',3000000000,4000000000);",
    _T + "INSERT INTO usage VALUES (1,'a',5,0),(2,'b',5,0),(3,'c',5,1);",
    _T + f"INSERT INTO usage VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
With no `GROUP BY`, the aggregates run over the whole table as one group, and SQL returns exactly one row even when the table is empty. Each aggregate has its own NULL rule:

| aggregate | NULLs | empty input |
| `COUNT(*)` | counted | 0 |
| `COUNT(col)` | skipped | 0 |
| `COUNT(DISTINCT col)` | skipped | 0 |
| `SUM(col)` | skipped | NULL |
| `AVG(col)` | skipped | NULL |

So `AVG(completion_tokens)` is already the average over completed requests, and `SUM` needs a `COALESCE` to turn "no rows" into 0. SQLite also has `TOTAL(col)`, which returns `0.0` on empty input, but as a REAL.

## The row-wise trap
`SUM(prompt_tokens + completion_tokens)` adds per row first. For a failed request `120 + NULL` is NULL, and `SUM` skips it, so its 120 prompt tokens vanish (test 2). Summing each column separately avoids this; so does `SUM(prompt_tokens + COALESCE(completion_tokens, 0))`.

## Complexity
A single pass, $O(N)$. `COUNT(DISTINCT ...)` needs a hash set or a sort of the users, which is the expensive part at scale; approximate counters such as HyperLogLog trade a small error for constant memory.

## Interview follow-ups
- Treat a failed request as 0 completion tokens in the average instead. Use `AVG(COALESCE(completion_tokens, 0))`, and explain why the two averages answer different questions.
- Report the same numbers per user, with the users that have no completed request included. Add `GROUP BY user_id`, then ask what `avg_completion` is for those users.
"""
