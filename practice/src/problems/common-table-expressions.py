import numpy as np

META = dict(title='Fine-tuning lineage with a recursive CTE', tags=['Data Processing'], diff='Medium', runtime='sql',
            lessons=['sdsd:03.05'], related=['self-join', 'subqueries', 'nested-aggregations'])

STATEMENT = r"""
A model registry records which checkpoint each model was fine-tuned from:

| column | type |
| `id` | INTEGER, unique |
| `name` | TEXT |
| `parent_id` | INTEGER, the model it was fine-tuned from, or NULL for a model trained from scratch |

Every non-NULL `parent_id` refers to an existing model, and there are no cycles, so following parents always ends at a model trained from scratch: its **root**.

Write **one SQLite query** that returns, for every model, the name of its root and its **depth**, the number of fine-tuning steps from the root (0 for a root itself).

## Output
- Columns, in this order: `id`, `name`, `root`, `depth`.
- Sort by `id` ascending.
"""

STARTER = '''-- models(id, name, parent_id)
SELECT id, name, name AS root
FROM models
ORDER BY id;
'''

SOLUTION = '''WITH RECURSIVE lineage(id, root, depth) AS (
  -- anchor: models trained from scratch are their own root
  SELECT id, name, 0
  FROM models
  WHERE parent_id IS NULL
  UNION ALL
  -- recursive step: a child inherits its parent's root, one level deeper
  SELECT m.id, l.root, l.depth + 1
  FROM models AS m
  JOIN lineage AS l ON m.parent_id = l.id
)
SELECT m.id, m.name, l.root, l.depth
FROM models AS m
JOIN lineage AS l ON l.id = m.id
ORDER BY m.id;
'''

HINTS = [
    'A self join can follow one parent link; you do not know in advance how many links a chain has. A recursive CTE repeats a join until no new rows appear.',
    '`WITH RECURSIVE lineage(id, root, depth) AS (anchor UNION ALL step)`. The anchor selects the roots with depth 0; the step joins `models` to the rows already in `lineage` on `m.parent_id = l.id`.',
    'In the step, carry `l.root` along unchanged and add 1 to `l.depth`. Finally join `lineage` back to `models` to get each model\'s own name.',
]

_T = 'CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT, parent_id INTEGER);\n'
_rng = np.random.default_rng(127)
_order = _rng.permutation(np.arange(1, 41))          # ids in creation order: parents always exist earlier
_rand = []
for _k, _i in enumerate(_order):
    _p = 'NULL' if _k < 3 or _rng.random() < 0.1 else str(int(_order[int(_rng.integers(0, _k))]))
    _rand.append(f"({int(_i)},'m{int(_i)}',{_p})")
TESTS = [
    _T + "INSERT INTO models VALUES (1,'llama-base',NULL),(2,'llama-chat',1),(3,'llama-chat-safe',2),(4,'llama-code',1);",
    _T + "INSERT INTO models VALUES (1,'bert',NULL),(2,'gpt2',NULL),(3,'bert-ner',1),(4,'gpt2-poems',2);",
    _T + "INSERT INTO models VALUES (5,'leaf',4),(4,'mid',3),(3,'top',NULL);",                       # children have smaller ids
    _T + "INSERT INTO models VALUES (1,'solo',NULL);",
    _T + "INSERT INTO models VALUES " + ','.join(f"({i},'v{i}',{'NULL' if i == 1 else i - 1})" for i in range(1, 31)) + ";",  # deep chain
    _T + "",
    _T + f"INSERT INTO models VALUES {','.join(_rand)};",
]

EXPLANATION = r"""
## Why it works
A recursive CTE is evaluated in rounds. The **anchor** query produces the first rows: every root, with itself as root and depth 0. Each round runs the **recursive** query against only the rows produced in the previous round: it finds the models whose parent is one of them and emits each child with the parent's root and depth + 1. When a round produces nothing (all leaves reached), the recursion stops. Because each model has exactly one parent and there are no cycles, every model is produced exactly once.

## CTEs in general
`WITH name AS (query)` gives a subquery a name, so a long query reads as a pipeline of steps instead of nested parentheses. A non-recursive CTE is mostly a readability tool; the planner usually inlines it like a subquery. `WITH RECURSIVE` adds something a plain query cannot express in SQL at all: iteration to an unknown depth.

## Complexity
Each round is a join of the newest generation with `models`. With an index on `parent_id`, the total work is $O(N \log N)$ over all rounds; the number of rounds equals the deepest chain (30 in test 5).

## Common bugs
- Recursing upward from every model without a stop condition, or with cyclic data: the query never terminates. A guard such as `WHERE l.depth < 100` is cheap insurance.
- Using `UNION` instead of `UNION ALL`. Here it gives the same result but forces de-duplication each round.
- Forgetting the final join and outputting the root's name as `name`.

## Interview follow-ups
- List the full path of each model, for example `llama-base > llama-chat > llama-chat-safe`, by carrying a string `l.path || ' > ' || m.name` through the recursion.
- How many descendants does every model have? Recurse downward from each model, or reverse the direction and count how often each model appears as an ancestor.
"""
