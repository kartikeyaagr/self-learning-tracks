import numpy as np

META = dict(title='Datasets with their approved license (LEFT JOIN)', tags=['Data Processing'], diff='Easy', runtime='sql',
            lessons=['sdsd:03.01'], related=['inner-join', 'null-handling'])

STATEMENT = r"""
Every dataset in a catalogue needs a license before it can be used for training. Licenses go through review, so a dataset can have several license rows but **at most one** with `approved = 1`.

| column | type |
| `datasets.id` | INTEGER, unique |
| `datasets.name` | TEXT |
| `licenses.dataset_id` | INTEGER, refers to `datasets.id` |
| `licenses.license` | TEXT |
| `licenses.approved` | INTEGER, 1 if approved, 0 if not (yet) |

Write **one SQLite query** that returns **every** dataset with its approved license. If a dataset has no approved license (no license rows at all, or only unapproved ones), show the text `'unknown'`.

## Output
- Columns, in this order: `id`, `name`, `license`.
- Exactly one row per dataset.
- Sort by `id` ascending.
"""

STARTER = '''-- datasets(id, name), licenses(dataset_id, license, approved)
SELECT d.id, d.name
FROM datasets AS d;
'''

SOLUTION = '''SELECT d.id, d.name, COALESCE(l.license, 'unknown') AS license
FROM datasets AS d
LEFT JOIN licenses AS l
       ON l.dataset_id = d.id
      AND l.approved = 1        -- in ON, not WHERE: keeps datasets without an approved license
ORDER BY d.id;
'''

HINTS = [
    'An inner join drops datasets that have no license. Which join keeps every row of the left table?',
    'A `LEFT JOIN` fills the right side with NULLs when nothing matches. `COALESCE(l.license, \'unknown\')` turns that NULL into the text you need.',
    'Put `l.approved = 1` inside the `ON` clause. In `WHERE` it runs after the join and throws away the NULL-filled rows you just kept.',
]

_T = ('CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT);\n'
      'CREATE TABLE licenses (dataset_id INTEGER, license TEXT, approved INTEGER);\n')
_rng = np.random.default_rng(8)
_ds = ','.join(f"({i},'ds{i}')" for i in range(1, 26))
_lic = []
for i in range(1, 26):
    for _ in range(int(_rng.integers(0, 3))):
        _lic.append(f"({i},'{_rng.choice(['MIT', 'CC-BY', 'GPL', 'Apache-2.0'])}',0)")
    if _rng.random() < 0.5:
        _lic.append(f"({i},'{_rng.choice(['MIT', 'CC-BY', 'Apache-2.0'])}',1)")
_rng.shuffle(_lic)
TESTS = [
    _T + "INSERT INTO datasets VALUES (1,'imagenet'),(2,'wiki'),(3,'scraped');\n"
         "INSERT INTO licenses VALUES (1,'custom',1),(2,'CC-BY-SA',1);",
    _T + "INSERT INTO datasets VALUES (1,'faces'),(2,'books');\n"
         "INSERT INTO licenses VALUES (1,'MIT',0),(1,'CC-BY',1),(2,'GPL',0);",
    _T + "INSERT INTO datasets VALUES (1,'a'),(2,'b');\n"
         "INSERT INTO licenses VALUES (2,'MIT',0),(2,'GPL',0),(2,'BSD',0);",          # only unapproved rows
    _T + "INSERT INTO datasets VALUES (4,'lonely');",
    _T + "INSERT INTO datasets VALUES (1,'a');\nINSERT INTO licenses VALUES (9,'MIT',1),(1,'MIT',0),(1,'Apache-2.0',1);",
    _T + "",
    _T + f"INSERT INTO datasets VALUES {_ds};\n" + (f"INSERT INTO licenses VALUES {','.join(_lic)};" if _lic else ''),
]

EXPLANATION = r"""
## Why it works
`LEFT JOIN` keeps every row of `datasets`. For each one it looks for license rows satisfying the **whole** `ON` condition (same dataset *and* approved). If there is one, its `license` is attached; if there are none, the right-hand columns are NULL, and `COALESCE` replaces that NULL with `'unknown'`. Because at most one approved license exists per dataset, no dataset is duplicated.

## ON versus WHERE
For an inner join the two are interchangeable. For an outer join they are not:

- `ON ... AND l.approved = 1` decides **which right rows match**. Datasets with no approved license still appear, with NULLs.
- `WHERE l.approved = 1` runs **after** the join. The NULL-filled rows have `approved` NULL, fail the test, and the datasets you wanted to keep vanish: the query has become an inner join.

Without the approval filter at all, dataset 1 in test 2 would appear twice, once per license row.

## Complexity
One lookup into `licenses` per dataset: $O(N \log M)$ with an index on `licenses(dataset_id, approved)`.

## Interview follow-ups
- List only the datasets that **cannot** be used yet. Keep the same join and add `WHERE l.dataset_id IS NULL`: the anti-join pattern.
- If the "at most one approved license" rule were not guaranteed, how would you make the output one row per dataset anyway? Aggregate first (for example `MIN(license)` per dataset among approved rows) and join to that.
"""
