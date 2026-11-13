import numpy as np

META = dict(title='Distinct labels per item after cleanup (DISTINCT)', tags=['Data Processing'], diff='Easy', runtime='sql',
            related=['string-functions', 'group-by'])

STATEMENT = r"""
Several annotators labelled the same items, and the label text is messy: the same class shows up as `'Cat'`, `'cat'` and `' cat '`. The table `annotations`:

| column | type |
| `id` | INTEGER, unique |
| `item_id` | INTEGER |
| `annotator` | TEXT |
| `label` | TEXT, may be NULL |

A label is normalised by removing leading and trailing spaces and converting it to lower case. Write **one SQLite query** that returns every distinct `(item_id, normalised label)` pair, so you can see which items received conflicting labels.

- Skip rows whose `label` is NULL or becomes the empty string after trimming.
- Each pair appears once, however many annotators gave it.

## Output
- Columns, in this order: `item_id`, `label` (the normalised text).
- Sort by `item_id`, then `label`, both ascending.
"""

STARTER = '''-- annotations(id, item_id, annotator, label)
SELECT DISTINCT item_id, annotator
FROM annotations;
'''

SOLUTION = '''SELECT DISTINCT item_id, LOWER(TRIM(label)) AS label
FROM annotations
WHERE TRIM(label) <> ''          -- also drops NULL: TRIM(NULL) <> '' is NULL, not true
ORDER BY item_id, label;
'''

HINTS = [
    '`SELECT DISTINCT a, b` removes duplicate rows of the selected columns, here pairs of item and label.',
    'Normalise before de-duplicating: `LOWER(TRIM(label))`. DISTINCT compares the values you select, not the raw column.',
    "Filter with `WHERE TRIM(label) <> ''`. That condition is NULL (so false) for NULL labels as well, which drops both kinds of empty row.",
]

_T = 'CREATE TABLE annotations (id INTEGER PRIMARY KEY, item_id INTEGER, annotator TEXT, label TEXT);\n'
_rng = np.random.default_rng(3)
_labels = ['cat', 'Cat', ' cat', 'dog ', 'DOG', 'bird', '  ', 'NULL']
_rand = ','.join(
    f"({i},{int(_rng.integers(1, 9))},'a{int(_rng.integers(1, 4))}',"
    f"{(lambda s: s if s == 'NULL' else repr(s))(str(_rng.choice(_labels)))})" for i in range(1, 51))
TESTS = [
    _T + "INSERT INTO annotations VALUES (1,10,'ann','cat'),(2,10,'bo','Cat'),(3,10,'cy','dog'),(4,11,'ann','dog');",
    _T + "INSERT INTO annotations VALUES (1,5,'ann',' Bird '),(2,5,'bo',NULL),(3,5,'cy','   '),(4,6,'ann','bird');",
    _T + "INSERT INTO annotations VALUES (1,1,'a','x'),(2,1,'a','x'),(3,1,'b','X '),(4,1,'c',' x');",   # all one pair
    _T + "INSERT INTO annotations VALUES (1,3,'a',NULL),(2,3,'b','');",           # nothing usable
    _T + "INSERT INTO annotations VALUES (1,9,'a','Hot Dog'),(2,2,'b','hot dog'),(3,2,'c','hotdog');",
    _T + "",
    _T + f"INSERT INTO annotations VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
`DISTINCT` applies to the **selected** row, after expressions are computed. Selecting `LOWER(TRIM(label))` first maps `'Cat'`, `'cat'` and `' cat '` to the same value, so they collapse into one pair per item. `WHERE` runs before `SELECT`, so the empty and NULL labels never reach the de-duplication step.

## NULL and the empty string
`TRIM(NULL)` is NULL and `NULL <> ''` is NULL, which `WHERE` treats as false. One condition removes both cases. Note that SQLite's `TRIM` with one argument removes spaces only, not tabs or newlines; `TRIM(label, ' ' || char(9) || char(10))` handles those too.

## Complexity
De-duplication needs a sort or a hash of the selected pairs: $O(N \log N)$ or expected $O(N)$.

## Common bugs
- `SELECT DISTINCT item_id, label` followed by normalising in application code. The duplicates that differ only in case survive.
- `WHERE label IS NOT NULL` alone, which keeps the all-space labels as an empty string row.
- Expecting `DISTINCT` to apply to one column. `SELECT DISTINCT a, b` de-duplicates pairs; it is not `DISTINCT a` plus `b`.

## Interview follow-ups
- Return only the items with **conflicting** labels (two or more distinct normalised labels). This needs `GROUP BY item_id HAVING COUNT(DISTINCT LOWER(TRIM(label))) > 1`.
- How would you pick a majority label per item, and what tie rule would you state?
"""
