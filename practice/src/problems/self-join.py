import numpy as np

META = dict(title='Seed variance: pair runs with identical hyperparameters (self join)', tags=['Data Processing'],
            diff='Medium', runtime='sql', related=['inner-join', 'cross-join'])

STATEMENT = r"""
You repeated some experiments with different random seeds and want to see how much the result moves when only the seed changes. The table `runs`:

| column | type |
| `id` | INTEGER, unique |
| `lr` | REAL, learning rate, may be NULL (not recorded) |
| `batch_size` | INTEGER, may be NULL |
| `seed` | INTEGER |
| `acc` | REAL, final accuracy |

Two runs are **replicates** if they have the same `lr` and the same `batch_size`, both non-NULL. Write **one SQLite query** that returns every pair of replicate runs, once per pair.

## Output
- Columns, in this order: `id_a`, `id_b`, `lr`, `batch_size`, `acc_gap`.
- `id_a < id_b`, so each unordered pair appears exactly once and a run is never paired with itself.
- `acc_gap` is the absolute difference of the two accuracies, $|acc_a - acc_b|$.
- Sort by `id_a`, then `id_b`.
"""

STARTER = '''-- runs(id, lr, batch_size, seed, acc)
SELECT id AS id_a, id AS id_b, lr, batch_size
FROM runs;
'''

SOLUTION = '''SELECT a.id AS id_a,
       b.id AS id_b,
       a.lr,
       a.batch_size,
       ABS(a.acc - b.acc) AS acc_gap
FROM runs AS a
JOIN runs AS b
  ON b.lr = a.lr
 AND b.batch_size = a.batch_size   -- = is never true for NULL, so unrecorded settings never pair
 AND a.id < b.id                   -- one row per unordered pair, no self-pairs
ORDER BY id_a, id_b;
'''

HINTS = [
    'You need two rows of the same table side by side. Join `runs` to itself under two different aliases.',
    'Match on `lr` and `batch_size`. Then think about duplicates: without another condition, every pair shows up as (a, b), (b, a) and each run also matches itself.',
    'Add `a.id < b.id` to the join condition. `=` never matches NULL, so the NULL rule needs no extra code. `acc_gap` is `ABS(a.acc - b.acc)`.',
]

_T = 'CREATE TABLE runs (id INTEGER PRIMARY KEY, lr REAL, batch_size INTEGER, seed INTEGER, acc REAL);\n'
_rng = np.random.default_rng(17)
_rand = ','.join(
    f"({i},{_rng.choice(['0.001', '0.0003', '0.01', 'NULL'])},{_rng.choice(['32', '64', 'NULL'])},"
    f"{int(_rng.integers(0, 100))},{round(float(_rng.uniform(0.6, 0.9)), 4)})" for i in range(1, 36))
TESTS = [
    _T + "INSERT INTO runs VALUES (1,0.001,32,0,0.80),(2,0.001,32,1,0.78),(3,0.01,32,0,0.70);",
    _T + "INSERT INTO runs VALUES (4,0.001,64,0,0.75),(1,0.001,64,1,0.79),(9,0.001,64,2,0.77);",   # three replicates
    _T + "INSERT INTO runs VALUES (1,NULL,32,0,0.5),(2,NULL,32,1,0.6),(3,0.1,NULL,0,0.5),(4,0.1,NULL,1,0.7);",
    _T + "INSERT INTO runs VALUES (1,0.001,32,0,0.9);",
    _T + "INSERT INTO runs VALUES (1,0.001,32,0,0.9),(2,0.001,32,7,0.9),(3,0.001,16,0,0.9);",   # gap of zero
    _T + "",
    _T + f"INSERT INTO runs VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
A self join treats one table as two: `a` and `b` range independently over all runs, and the `ON` clause keeps the pairs that share both hyperparameters. The extra `a.id < b.id` does two jobs at once. It removes the pair of a run with itself (`id = id`), and it keeps only one of the mirror images (a, b) and (b, a). A group of $k$ replicates therefore yields $\binom{k}{2}$ rows; test 2 has three runs and three pairs.

## NULLs
`NULL = NULL` is NULL, not true, so runs with an unrecorded `lr` or `batch_size` never match anything, which is the rule the statement asks for. If you wanted NULLs to match each other, you would write `a.lr IS b.lr` (SQLite's null-safe equality).

## Complexity
Worst case the output alone is $O(N^2)$, when every run shares the same settings. With an index on `(lr, batch_size)` the database only compares runs inside the same group, so the cost is $\sum_g k_g^2$ over groups.

## Common bugs
- `a.id <> b.id` instead of `<`, which returns every pair twice.
- Comparing floats from different sources with `=`. It is fine here because both values come from the same column, but `0.1 + 0.2 = 0.3` is false.
- Forgetting the aliases. `FROM runs JOIN runs` is an error because the two copies need different names.

## Interview follow-ups
- Instead of pairs, report one row per hyperparameter group with the number of seeds and the spread `MAX(acc) - MIN(acc)`. That is a `GROUP BY lr, batch_size HAVING COUNT(*) > 1`, and it scales linearly.
- The classic self join is an org chart: `employees e JOIN employees m ON e.manager_id = m.id`. How would you find the employees who earn more than their manager?
"""
