META = dict(title='Running total per customer (SQL window functions)', tags=['Data Processing'], diff='Medium', runtime='sql',
            lessons=['sdsd:03.02'], related=['top-n-per-group'])

STATEMENT = r"""
The table `orders` holds one row per order:

| column | type |
| `id` | INTEGER, unique |
| `customer` | TEXT |
| `day` | INTEGER (days since launch) |
| `amount` | REAL |

Write **one SQLite query** that returns, for every order, the customer's cumulative spend up to and including that order.

## Output
- Columns, in this order and with these names: `customer`, `day`, `amount`, `running_total`.
- Within a customer, accumulate in order of `day`, then `id` (several orders can share a day).
- Sort the result by `customer`, `day`, `id`.

SQLite supports window functions such as `SUM(...) OVER (PARTITION BY ... ORDER BY ...)`.
"""

STARTER = '''-- orders(id, customer, day, amount)
SELECT customer, day, amount
FROM orders;
'''

SOLUTION = '''SELECT customer, day, amount,
       SUM(amount) OVER (
         PARTITION BY customer
         ORDER BY day, id
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_total
FROM orders
ORDER BY customer, day, id;
'''

HINTS = [
    'A window function computes a value per row from a set of related rows without collapsing them as `GROUP BY` would.',
    '`PARTITION BY customer` restarts the sum for each customer. `ORDER BY day, id` inside `OVER (...)` sets the accumulation order.',
    'With only `ORDER BY` in the window, the default frame is RANGE, which treats rows with equal (day, id) as peers. Write `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` to accumulate strictly row by row.',
]

_T = 'CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, day INTEGER, amount REAL);\n'
TESTS = [
    _T + "INSERT INTO orders VALUES (1,'ann',1,10.0),(2,'ann',2,5.0),(3,'bob',1,7.5);",
    _T + "INSERT INTO orders VALUES (1,'ann',3,1.0),(2,'ann',1,2.0),(3,'ann',2,4.0);",
    _T + "INSERT INTO orders VALUES (5,'cat',1,3.0),(2,'cat',1,4.0),(9,'cat',1,1.0),(1,'ann',4,2.5);",
    _T + "INSERT INTO orders VALUES (1,'zed',10,100.0);",
    _T + "INSERT INTO orders VALUES (1,'b',1,1.0),(2,'a',1,2.0),(3,'b',2,3.0),(4,'a',2,4.0),(5,'b',2,-1.5),(6,'a',9,0.25);",
    _T + "",
]

EXPLANATION = r"""
## Window functions in one sentence
`agg(x) OVER (PARTITION BY p ORDER BY o)` computes the aggregate over this row's partition, up to this row in order `o`, and keeps every row in the output.

## The frame clause trap
With `ORDER BY` and no frame, SQL uses `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Rows that tie on the ORDER BY key are peers and **all** get the total through the last peer. Ordering by `day, id` has no ties because `id` is unique, but writing `ROWS` states the intent and is safe if someone drops `id` later. Test 3 has three orders on the same day.

## Performance
The database sorts each partition once and keeps a running sum: $O(N \log N)$ for the sort, then $O(N)$. The alternative, a correlated subquery that sums earlier orders for every row, is $O(N^2)$.

## Interview follow-ups
- A 7-day rolling sum by **date**, not by row count: use `RANGE BETWEEN 6 PRECEDING AND CURRENT ROW` with `ORDER BY day`.
- Each customer's first order and the gap between consecutive orders: `ROW_NUMBER()` and `LAG(day)`.
"""
