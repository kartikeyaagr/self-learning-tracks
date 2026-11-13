import numpy as np

META = dict(title='Order report across four tables (multiple joins)', tags=['Data Processing'], diff='Medium', runtime='sql',
            lessons=['sdsd:03.01'], related=['inner-join', 'left-join'])

STATEMENT = r"""
An online shop keeps a normalised schema:

| column | type |
| `orders.order_id` | INTEGER, unique |
| `orders.customer_id` | INTEGER, refers to `customers` |
| `orders.product_id` | INTEGER, refers to `products` |
| `orders.qty` | INTEGER |
| `customers.customer_id` | INTEGER, unique |
| `customers.name` | TEXT |
| `customers.country_code` | TEXT, may be NULL |
| `products.product_id` | INTEGER, unique |
| `products.title` | TEXT |
| `products.price` | REAL |
| `countries.code` | TEXT, unique |
| `countries.name` | TEXT |

The references are not enforced: an order can point at a customer or product that no longer exists, and a customer's `country_code` can be NULL or missing from `countries`.

Write **one SQLite query** that produces the order report:

- Include an order only if **both** its customer and its product exist.
- `country` is the country's name, or NULL if the customer has no code or the code is not in `countries`. Such orders are still included.
- `revenue` is `qty * price`.

## Output
- Columns, in this order: `order_id`, `customer`, `country`, `product`, `revenue`.
- `customer` is `customers.name`, `product` is `products.title`.
- Sort by `order_id` ascending.
"""

STARTER = '''-- orders(order_id, customer_id, product_id, qty)
-- customers(customer_id, name, country_code), products(product_id, title, price), countries(code, name)
SELECT order_id, customer_id, product_id
FROM orders;
'''

SOLUTION = '''SELECT o.order_id,
       c.name  AS customer,
       co.name AS country,
       p.title AS product,
       o.qty * p.price AS revenue
FROM orders AS o
JOIN customers AS c       ON c.customer_id = o.customer_id   -- required
JOIN products  AS p       ON p.product_id  = o.product_id    -- required
LEFT JOIN countries AS co ON co.code = c.country_code        -- optional: keep the order, NULL country
ORDER BY o.order_id;
'''

HINTS = [
    'Joins chain: `FROM orders JOIN customers ON ... JOIN products ON ... JOIN countries ON ...`. Decide, per table, whether a missing partner should drop the order or not.',
    'Customer and product are required, so inner joins. Country is optional, so that one must be a `LEFT JOIN`.',
    'Two tables have a `name` column. Alias them (`c.name AS customer`, `co.name AS country`) and compute `o.qty * p.price AS revenue`.',
]

_T = ('CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, qty INTEGER);\n'
      'CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, country_code TEXT);\n'
      'CREATE TABLE products (product_id INTEGER PRIMARY KEY, title TEXT, price REAL);\n'
      'CREATE TABLE countries (code TEXT PRIMARY KEY, name TEXT);\n')
_D = ("INSERT INTO customers VALUES (1,'Asha','IN'),(2,'Ben','GB'),(3,'Chen',NULL),(4,'Dara','XX');\n"
      "INSERT INTO products VALUES (10,'GPU hour',2.5),(11,'Dataset',40.0),(12,'Label pack',7.25);\n"
      "INSERT INTO countries VALUES ('IN','India'),('GB','United Kingdom'),('DE','Germany');\n")
_rng = np.random.default_rng(59)
_rand = ','.join(f"({i},{int(_rng.integers(1, 7))},{int(_rng.integers(9, 14))},{int(_rng.integers(1, 20))})"
                 for i in range(1, 41))
TESTS = [
    _T + _D + "INSERT INTO orders VALUES (1,1,10,4),(2,2,11,1);",
    _T + _D + "INSERT INTO orders VALUES (1,3,12,2),(2,4,10,10),(3,9,10,1),(4,1,99,1);",   # NULL/unknown country, missing refs
    _T + _D + "INSERT INTO orders VALUES (5,2,12,3),(6,2,12,3);",
    _T + _D + "INSERT INTO orders VALUES (1,7,10,1),(2,1,13,1);",                        # every order dropped
    _T + _D + "INSERT INTO orders VALUES (1,1,11,1000000);",
    _T + _D,
    _T + _D + f"INSERT INTO orders VALUES {_rand};",
]

EXPLANATION = r"""
## Why it works
A chain of joins is evaluated left to right, each step joining the result so far with the next table. The first two are inner joins, so an order with a missing customer or product has no partner and is dropped. The last is a left join from the customer to `countries`: when the code is NULL or unknown, the country columns are NULL but the order row stays.

## Order of joins matters for outer joins
If you write `LEFT JOIN countries` and then an inner join to a table whose condition uses `co.` columns, the NULL-filled rows would be thrown away again. A safe habit: put the required (inner) joins first, the optional (left) joins after them, and never reference a left-joined table in a later inner join's condition or in `WHERE` unless you want to filter it.

## Complexity
Each join is an index lookup on a primary key: about $O(N \log M)$ overall for $N$ orders. The database's planner may reorder the inner joins; it will not reorder a left join past them in a way that changes the result.

## Common bugs
- Using `JOIN countries`, which silently drops customers with no country (test 2 drops two orders this way).
- Selecting `name` without a prefix: ambiguous between `customers.name` and `countries.name`.
- Computing revenue as `qty * price` with `qty` stored as TEXT. SQLite would coerce it, other databases would fail. Know your column types.

## Interview follow-ups
- Revenue per country, with orders from unknown countries grouped under `'unknown'`: wrap this query in a CTE and `GROUP BY COALESCE(country, 'unknown')`.
- How would you find orders that reference a missing product, to report data-quality issues? `LEFT JOIN products p ... WHERE p.product_id IS NULL`.
"""
