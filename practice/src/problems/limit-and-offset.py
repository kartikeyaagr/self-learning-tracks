import numpy as np

META = dict(title='Paginate a feed with LIMIT and OFFSET', tags=['Data Processing'], diff='Easy', runtime='sql',
            lessons=['sdsd:04.12'], related=['order-by', 'row-number'])

STATEMENT = r"""
A blog shows its posts newest first, a fixed number per page. The data:

| column | type |
| `posts.id` | INTEGER, unique |
| `posts.title` | TEXT |
| `posts.published` | INTEGER, publication time (Unix seconds) |
| `page.num` | INTEGER, the page being requested, at least 1 |
| `page.size` | INTEGER, posts per page, at least 1 |

The table `page` always has exactly **one row**. Write **one SQLite query** that returns the posts on page `num`, where page 1 holds the first `size` posts in feed order, page 2 the next `size`, and so on.

Feed order is `published` descending, and for posts published at the same time, `id` descending.

## Output
- Columns, in this order: `id`, `title`, `published`.
- Rows in feed order.
- A page past the end returns no rows; the last page may be shorter than `size`.
- Read `num` and `size` from the `page` table; do not hard-code them.
"""

STARTER = '''-- posts(id, title, published), page(num, size)
SELECT id, title
FROM posts
LIMIT 3;
'''

SOLUTION = '''SELECT id, title, published
FROM posts
ORDER BY published DESC, id DESC
-- SQLite accepts any integer expression here, including scalar subqueries
LIMIT (SELECT size FROM page)
OFFSET (SELECT (num - 1) * size FROM page);
'''

HINTS = [
    '`LIMIT n OFFSET k` skips the first `k` rows of the sorted result and returns the next `n`. Without `ORDER BY` the pages are meaningless.',
    'Page `num` starts after `(num - 1) * size` rows. SQLite lets `LIMIT` and `OFFSET` be expressions, including a scalar subquery such as `(SELECT size FROM page)`.',
    '`ORDER BY published DESC, id DESC LIMIT (SELECT size FROM page) OFFSET (SELECT (num - 1) * size FROM page)`.',
]

_T = ('CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT, published INTEGER);\n'
      'CREATE TABLE page (num INTEGER, size INTEGER);\n')
_rng = np.random.default_rng(31)
_rand = ','.join(f"({i},'post {i}',{1700000000 + int(_rng.integers(0, 20)) * 3600})" for i in range(1, 48))
TESTS = [
    _T + "INSERT INTO posts VALUES (1,'hello',100),(2,'gpus',300),(3,'loss',200),(4,'adam',400),(5,'data',500);\n"
         "INSERT INTO page VALUES (2,2);",
    _T + "INSERT INTO posts VALUES (1,'a',100),(2,'b',100),(3,'c',90);\nINSERT INTO page VALUES (1,2);",
    _T + "INSERT INTO posts VALUES (1,'a',10),(2,'b',20),(3,'c',30),(4,'d',40),(5,'e',50);\n"
         "INSERT INTO page VALUES (3,2);",                                        # short last page
    _T + "INSERT INTO posts VALUES (1,'a',10),(2,'b',20);\nINSERT INTO page VALUES (5,2);",   # past the end
    _T + "INSERT INTO posts VALUES (1,'only',1);\nINSERT INTO page VALUES (1,10);",
    _T + "INSERT INTO page VALUES (1,5);",                                        # no posts at all
    _T + f"INSERT INTO posts VALUES {_rand};\nINSERT INTO page VALUES (4,6);",
]

EXPLANATION = r"""
## Why it works
`ORDER BY` fixes a total order (the `id` tie-breaker makes it unique), then `OFFSET (num-1)*size` skips the earlier pages and `LIMIT size` takes one page. If the offset is past the end, nothing is left and the result is empty.

## Why the tie-breaker matters
Without `id DESC`, posts with the same `published` time can come back in any order, and that order can change between queries. A post can then appear on two pages or on none. Test 2 has two posts with the same timestamp.

## Complexity
The database still has to produce and discard the first `(num-1)*size` rows, so page $p$ costs $O(p \cdot \text{size})$ even with an index on `(published, id)`. Deep pages get slow.

## Common bugs
- `OFFSET num * size`, which skips page 1 entirely.
- Paginating without `ORDER BY`.
- Hard-coding the numbers. Real code passes them as bound parameters (`LIMIT ? OFFSET ?`), never by string formatting.

## Interview follow-ups
- Keyset (cursor) pagination: instead of an offset, the client sends the last `(published, id)` it saw and the query uses `WHERE (published, id) < (?, ?) ORDER BY published DESC, id DESC LIMIT ?`. Why does this stay fast on page 10,000, and why does it not skip or repeat posts when new ones are inserted?
- How would you return the total number of pages alongside the rows? A window function `COUNT(*) OVER ()` before the `LIMIT`, or a second query.
"""
