import numpy as np

META = dict(title='Models missing a safety eval (subqueries and the NOT IN trap)', tags=['Data Processing'], diff='Medium',
            runtime='sql', lessons=['sdsd:03.05'], related=['left-join', 'common-table-expressions', 'cross-join'])

STATEMENT = r"""
Before release, every model must pass the `'safety'` benchmark. The data:

| column | type |
| `models.id` | INTEGER, unique |
| `models.name` | TEXT |
| `evals.id` | INTEGER, unique |
| `evals.model_id` | INTEGER, refers to `models.id`; **may be NULL** (rows from a broken import) |
| `evals.benchmark` | TEXT |
| `evals.score` | REAL |

Write **one SQLite query** that lists the models that have **no** eval row with `benchmark = 'safety'`, together with the number of eval rows the model has on any benchmark.

- Eval rows with a NULL `model_id` belong to no model; they must not hide or disqualify any model.
- A model with no evals at all is listed with `n_evals = 0`.

## Output
- Columns, in this order: `id`, `name`, `n_evals`.
- Sort by `id` ascending.
"""

STARTER = '''-- models(id, name), evals(id, model_id, benchmark, score)
SELECT id, name
FROM models
WHERE id NOT IN (SELECT model_id FROM evals WHERE benchmark = 'safety');
'''

SOLUTION = '''SELECT m.id,
       m.name,
       (SELECT COUNT(*) FROM evals e WHERE e.model_id = m.id) AS n_evals   -- correlated scalar subquery
FROM models AS m
WHERE NOT EXISTS (                     -- NULL-safe anti-join, unlike NOT IN
  SELECT 1 FROM evals e
  WHERE e.model_id = m.id AND e.benchmark = 'safety'
)
ORDER BY m.id;
'''

HINTS = [
    'Two subqueries: one in `WHERE` to test "has no safety eval", and one in `SELECT` to count evals per model.',
    'The starter\'s `NOT IN` breaks as soon as the subquery returns a NULL: `x NOT IN (1, NULL)` is NULL, never true, so no model passes. Use `NOT EXISTS (SELECT 1 FROM evals e WHERE e.model_id = m.id AND ...)` instead.',
    'For the count, a correlated scalar subquery `(SELECT COUNT(*) FROM evals e WHERE e.model_id = m.id)` returns 0 when there are no rows, which is exactly what you need.',
]

_T = ('CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT);\n'
      'CREATE TABLE evals (id INTEGER PRIMARY KEY, model_id INTEGER, benchmark TEXT, score REAL);\n')
_rng = np.random.default_rng(113)
_mods = ','.join(f"({i},'model-{i}')" for i in range(1, 21))
_ev = ','.join(f"({i},{'NULL' if _rng.random() < 0.05 else int(_rng.integers(1, 23))},"
               f"'{_rng.choice(['safety', 'mmlu', 'gsm8k', 'code'])}',{round(float(_rng.random()), 3)})" for i in range(1, 61))
TESTS = [
    _T + "INSERT INTO models VALUES (1,'base'),(2,'chat'),(3,'code');\n"
         "INSERT INTO evals VALUES (1,1,'safety',0.9),(2,1,'mmlu',0.6),(3,2,'mmlu',0.7);",
    _T + "INSERT INTO models VALUES (1,'base'),(2,'chat');\n"
         "INSERT INTO evals VALUES (1,NULL,'safety',0.5),(2,2,'safety',0.8);",          # the NULL that breaks NOT IN
    _T + "INSERT INTO models VALUES (1,'a'),(2,'b');\nINSERT INTO evals VALUES (1,1,'safety',0.1),(2,2,'safety',0.2);",
    _T + "INSERT INTO models VALUES (5,'fresh');",                                    # no evals at all
    _T + "INSERT INTO models VALUES (1,'a');\nINSERT INTO evals VALUES (1,1,'Safety',0.9),(2,1,'safety-v2',0.9),(3,9,'safety',1.0);",
    _T + "INSERT INTO evals VALUES (1,1,'safety',0.5);",
    _T + f"INSERT INTO models VALUES {_mods};\nINSERT INTO evals VALUES {_ev};",
]

EXPLANATION = r"""
## Why it works
`NOT EXISTS (subquery)` is true when the subquery returns no rows. The subquery is **correlated**: it refers to `m.id` from the outer query, so it is evaluated for each model and asks "is there a safety eval for *this* model?". An eval with a NULL `model_id` never satisfies `e.model_id = m.id`, so it is simply ignored. The scalar subquery in `SELECT` works the same way and `COUNT(*)` over no rows is 0.

## The NOT IN trap
`m.id NOT IN (a, b, NULL)` means `m.id <> a AND m.id <> b AND m.id <> NULL`. The last comparison is NULL, so the whole condition is NULL or false, never true: as soon as one safety row has a NULL `model_id`, the starter returns nothing (test 2). Fixes:

- use `NOT EXISTS` (preferred: NULL-safe and usually planned as an anti-join);
- or add `WHERE model_id IS NOT NULL` inside the `NOT IN` subquery;
- or `LEFT JOIN ... WHERE e.id IS NULL`.

## Complexity
With an index on `evals(model_id, benchmark)` each subquery is an index probe: $O(M \log E)$ for $M$ models. Modern planners rewrite `NOT EXISTS` into an anti-join anyway; the correlated form is not inherently slow.

## Interview follow-ups
- Return only the models whose best safety score is below the best safety score of all models, using a scalar subquery in `WHERE`.
- Rewrite the `n_evals` column with a `LEFT JOIN` and `GROUP BY`. Why do you need `COUNT(e.id)` there instead of `COUNT(*)`?
"""
