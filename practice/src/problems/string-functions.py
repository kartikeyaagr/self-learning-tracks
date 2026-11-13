import numpy as np

META = dict(title='Split and normalise email addresses (string functions)', tags=['Data Processing', 'Feature Engineering'],
            diff='Easy', runtime='sql', related=['distinct-values', 'case-statements'])

STATEMENT = r"""
A sign-up form stored email addresses exactly as typed. The table `signups`:

| column | type |
| `id` | INTEGER, unique |
| `email` | TEXT, may have surrounding spaces and mixed case |

Write **one SQLite query** that cleans each address and splits it into the part before the **first** `@` and the part after it.

- Cleaning means removing leading and trailing spaces, then converting to lower case.
- `username` is everything before the first `@`; `domain` is everything after it (it may contain further `@` characters).
- Rows whose email contains no `@` at all are left out.

## Output
- Columns, in this order: `id`, `username`, `domain`.
- Sort by `domain`, then `id`, ascending.
"""

STARTER = '''-- signups(id, email)
SELECT id, email AS username
FROM signups;
'''

SOLUTION = '''WITH clean AS (
  SELECT id, LOWER(TRIM(email)) AS e
  FROM signups
)
SELECT id,
       SUBSTR(e, 1, INSTR(e, '@') - 1) AS username,   -- characters before the first @
       SUBSTR(e, INSTR(e, '@') + 1)    AS domain      -- everything after it
FROM clean
WHERE INSTR(e, '@') > 0                               -- INSTR is 0 when there is no @
ORDER BY domain, id;
'''

HINTS = [
    "`INSTR(s, '@')` returns the 1-based position of the first `@` in `s`, or 0 if there is none.",
    '`SUBSTR(s, start, length)` takes `length` characters from position `start`; without a length it runs to the end. Clean the text with `LOWER(TRIM(email))` once, in a CTE, so you don\'t repeat it four times.',
    "`username` is `SUBSTR(e, 1, INSTR(e, '@') - 1)`, `domain` is `SUBSTR(e, INSTR(e, '@') + 1)`, and `WHERE INSTR(e, '@') > 0` drops the invalid rows.",
]

_T = 'CREATE TABLE signups (id INTEGER PRIMARY KEY, email TEXT);\n'
_rng = np.random.default_rng(149)
_users = ['ann', 'Bob.Smith', 'c_99', 'DEV', 'eve+ml']
_doms = ['Example.com', 'lab.ORG', 'uni.edu', 'mail.test']
_rand = ','.join(
    f"({i},'{' ' * int(_rng.integers(0, 3))}{_rng.choice(_users)}{'' if _rng.random() < 0.1 else '@'}"
    f"{_rng.choice(_doms)}{' ' * int(_rng.integers(0, 3))}')" for i in range(1, 41))
TESTS = [
    _T + "INSERT INTO signups VALUES (1,'Ann@Example.com'),(2,'  bob@lab.org '),(3,'cat@example.com');",
    _T + "INSERT INTO signups VALUES (1,'no-at-sign.com'),(2,'dee@uni.edu'),(3,'');",
    _T + "INSERT INTO signups VALUES (1,'weird@name@host.io'),(2,'@nouser.io'),(3,'nodomain@');",   # extra @, empty parts
    _T + "INSERT INTO signups VALUES (4,'Z@B.COM'),(3,'y@b.com'),(2,'X@A.com');",
    _T + "INSERT INTO signups VALUES (1,'   ');",
    _T + "",
    _T + f"INSERT INTO signups VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
SQLite strings are 1-indexed. `INSTR(e, '@')` finds the first `@` at position $p$; the username is the $p - 1$ characters before it, `SUBSTR(e, 1, p - 1)`, and the domain starts at $p + 1$ and runs to the end, `SUBSTR(e, p + 1)`. When there is no `@`, $p = 0$ and the `WHERE` drops the row. Cleaning once in a CTE keeps the four uses of the cleaned text consistent.

## Edge cases the statement pins down
- Only the **first** `@` splits, so `weird@name@host.io` gives `weird` and `name@host.io`.
- Empty parts are allowed: `@nouser.io` has an empty username and `nodomain@` an empty domain. They are still rows, with `''` (not NULL).
- `TRIM` runs before `LOWER`, and both before searching, so surrounding spaces never end up in a part.

## Complexity
Linear in the total length of the strings, then a sort: $O(N \log N)$.

## Common bugs
- Off-by-one: `SUBSTR(e, 1, INSTR(e, '@'))` keeps the `@` in the username.
- Splitting on the **last** `@`. SQLite has no "last position" function; you would need `REPLACE` tricks or a recursive CTE.
- Forgetting that `TRIM` only removes spaces by default, not tabs or newlines.

## Interview follow-ups
- Normalise Gmail-style addresses: drop everything after a `+` in the username. Combine `INSTR(username, '+')` with a `CASE` for the "no plus" case.
- Count signups per domain and keep the domains with at least 3. Wrap this query in a CTE, `GROUP BY domain HAVING COUNT(*) >= 3`.
"""
