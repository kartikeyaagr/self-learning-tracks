---
id: "03.01"
course: "ml-toolkit"
module: "03"
slug: "select-filtering-and-sorting"
title: "SELECT, filtering and sorting"
keyIdeas: ["A query is evaluated in a logical order that differs from how it is written: FROM, WHERE, GROUP BY, HAVING, SELECT, DISTINCT, ORDER BY, LIMIT. That order explains which names are visible where: a SELECT alias works in ORDER BY but not (in standard SQL) in WHERE.", "`AND` binds tighter than `OR`. Parenthesise every mixed condition, or `split = 'val' OR split = 'test' AND score > 0.9` will keep every validation row whatever its score.", "SQL uses three-valued logic: any comparison with NULL is UNKNOWN, and WHERE keeps only TRUE rows. `x <> 'a'` silently drops the rows where x is NULL; test missing values with `IS NULL`, never `= NULL`.", "Without ORDER BY, row order is undefined; with ties, the order of the tied rows is still undefined. End every ORDER BY with a unique tie-breaker, and state `NULLS FIRST` or `NULLS LAST` because databases disagree.", "For timestamps use half-open ranges (`ts >= '2025-03-01' AND ts < '2025-04-01'`); `BETWEEN` with a date-only upper bound drops the whole last day.", "pandas does not use three-valued logic (a missing value is `!=` anything), so the same filter can return different rows in SQL and in a DataFrame. Count rows before and after each filter to catch it."]
question: "A table `requests(id, model, status, latency_ms)` has NULL latency for timed-out requests. `SELECT COUNT(*) FROM requests WHERE latency_ms < 500 OR latency_ms >= 500` returns a smaller number than the table's row count. Explain why, and say what this means for a query reporting the share of requests that were 'fast' (under 500 ms). A colleague then ports the same report to pandas with `(df.latency_ms < 500).mean()`; does that give the same answer as your SQL version?"
answer: "For a row with NULL latency both comparisons are UNKNOWN, and UNKNOWN OR UNKNOWN is UNKNOWN, so WHERE drops the row. The two conditions look complementary, but under three-valued logic they cover only rows with a known latency; the missing count is exactly the timed-out requests, which you can confirm with `SELECT COUNT(*) FROM requests WHERE latency_ms IS NULL`. For the 'fast share', the denominator matters. If you compute it over rows with `latency_ms IS NOT NULL` (which is what happens implicitly if you filter first), timeouts disappear and the service looks faster than it is. Timeouts are the slowest requests, so they should normally count as not fast: compute the numerator with `latency_ms < 500` (which is never TRUE for NULL, so timeouts are not counted as fast) and the denominator with `COUNT(*)` over all requests, and report the timeout count separately so the choice is visible. Writing it as `SUM(CASE WHEN latency_ms < 500 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)` makes the rule explicit. In pandas, NULL latencies arrive as NaN, and `NaN < 500` is False, so `(df.latency_ms < 500).mean()` divides by all rows and counts timeouts as not fast: it happens to match the recommended SQL version. But it matches by accident of pandas' two-valued comparisons, not by design; `df.latency_ms.lt(500).sum() / df.latency_ms.count()` or a `dropna()` earlier in the pipeline would silently switch to the optimistic denominator. State the NULL rule explicitly in both implementations and check both against the same row counts."
sources: [{"title": "PostgreSQL docs: SELECT", "url": "https://www.postgresql.org/docs/current/sql-select.html", "note": "The reference for clause order, ORDER BY with NULLS FIRST/LAST, LIMIT and DISTINCT."}, {"title": "SQLite: NULL handling in SQLite versus other engines", "url": "https://www.sqlite.org/nulls.html", "note": "A table of how different databases treat NULL in DISTINCT, UNIQUE and sorting."}, {"title": "Modern SQL: Three-valued logic", "url": "https://modern-sql.com/concept/three-valued-logic", "note": "Truth tables for UNKNOWN and how they surface in WHERE, CHECK and NOT IN."}, {"title": "PostgreSQL docs: Comparison functions and operators", "url": "https://www.postgresql.org/docs/current/functions-comparison.html", "note": "BETWEEN, IS DISTINCT FROM and the NULL predicates."}]
practice: ["where-clauses", "order-by", "basic-select", "distinct-values", "limit-and-offset"]
tags: ["NULL", "three-valued logic"]
tools: ["SQL", "SQLite", "PostgreSQL"]
---

## What you will learn

- Why ML work needs SQL at all, and what a `SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT` query does, clause by clause.
- The order in which a query is *evaluated*, which is not the order in which it is written, and why that explains most "column not found" errors.
- How to write filters that mean what you think: comparison operators, `IN`, `BETWEEN`, `LIKE`, and the AND/OR precedence trap.
- How NULL creates a third truth value, UNKNOWN, and how that silently removes rows from your training or evaluation data.
- How to make sorting deterministic with a tie-breaker, and how to run it all from Python with `sqlite3` and pandas.

*Before you start:* you should be comfortable with Python and with pandas boolean filtering (module 02 of this course, lessons 02.01 to 02.03). No previous SQL is assumed. Every snippet runs on Python's built-in `sqlite3` module, so there is nothing to install.

## Where this module starts

Most training data starts life in a database or warehouse: event logs, human labels, model predictions, evaluation results. Pulling a whole table into pandas and filtering there moves gigabytes you then throw away. **SQL** (Structured Query Language) lets you describe the rows and columns you want and have the database do the work where the data lives. This module covers filtering and sorting (this lesson), joins (03.02), aggregation (03.03), window functions (03.04), subqueries and CTEs (03.05), building leak-free training sets (03.06) and query speed (03.07). The examples run in SQLite (the dialect of the practice problems) and, unless noted, in PostgreSQL.

## The idea in one picture

Here is a tiny table of model predictions. Each row is one example: which data split it belongs to, the true label (NULL when nobody labelled it yet), the model's prediction, and the model's confidence.

| id | split | label | pred | confidence
| --- | --- | --- | --- | --- |
| 1 | train | cat | cat | 0.97
| 2 | val | dog | cat | 0.91
| 3 | val | cat | cat | 0.88
| 4 | test | dog | dog | 0.95
| 5 | val | cat | dog | 0.97
| 6 | val | NULL | dog | 0.99
| 7 | train | dog | cat | 0.62
| 8 | val | dog | cat | 0.55
| 9 | test | cat | dog | 0.93
| 10 | val | bird | cat | 0.91

A common debugging question: *which validation examples does the model get wrong most confidently?* They are often label errors. In SQL:

```sql
SELECT id, label, pred, confidence
FROM predictions
WHERE split = 'val' AND pred <> label
ORDER BY confidence DESC, id
LIMIT 3;
```

~~~dg-fig 03.01-pipeline.svg
Left: WHERE gives every row a verdict; only TRUE rows survive, and row 6 gets UNKNOWN because its label is NULL. Right: the four survivors are sorted by confidence (ties broken by id) and LIMIT keeps the first three.
~~~

Reading the figure:

- **FROM** names the source table. **WHERE** keeps the rows whose condition is TRUE: six rows are in `val`, and four of them have `pred <> label` (`<>` means "not equal").
- Row 6's label is **NULL**, the SQL marker for "value missing". Comparing anything with NULL gives **UNKNOWN**, and WHERE drops UNKNOWN rows, so the most confident prediction in the split never reaches you.
- **ORDER BY** sorts the survivors, highest confidence first (`DESC`); rows 2 and 10 tie at 0.91, so the second key, `id`, decides. **LIMIT 3** keeps ids 5, 2, 10. **SELECT** picks the output columns.

## Step 1: the parts of a query and their logical order

A query is written in a fixed order: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT` (`GROUP BY` and `HAVING` come in lesson 03.03). It is not *evaluated* in that order. The **logical order** defines the result: each step receives the output of the previous one.

~~~dg-fig 03.01-order.svg
The order you write clauses in versus the order they are logically evaluated. SELECT and DISTINCT move from the front to the middle; that move explains where column aliases are visible.
~~~

FROM builds the input rows, WHERE filters them, GROUP BY and HAVING form and filter groups, SELECT computes output columns (including new ones named with `AS`), DISTINCT removes duplicates, ORDER BY sorts and LIMIT cuts.

A name created with `AS` in SELECT is a **column alias**. Two consequences of the order:

- An alias is **not** visible in WHERE in standard SQL, because WHERE runs first. PostgreSQL rejects `WHERE examples_seen > 1e6`; SQLite accepts it as an extension, so the query breaks when it moves to a warehouse. Repeat the expression or use a subquery (03.05).
- An alias **is** visible in ORDER BY, which runs after SELECT.

```sql
SELECT run_id, model, steps * batch_size AS examples_seen
FROM training_runs
WHERE status = 'finished' AND (lr >= 1e-4 OR warmup_steps > 0)
ORDER BY examples_seen DESC, run_id
LIMIT 10;
```

The logical order defines the answer, not the work: the **query planner** may use any physical plan with the same result, such as reading an index first ("Going deeper").

## Step 2: the filter vocabulary

WHERE takes a **predicate**: an expression that is TRUE, FALSE or UNKNOWN for each row. The building blocks:

| Operator | Meaning | Example
| --- | --- | --- |
| `=`, `<>` (or `!=`) | equal, not equal | `split = 'val'`
| `<`, `<=`, `>`, `>=` | ordering comparisons | `confidence >= 0.9`
| `BETWEEN a AND b` | `a <= x AND x <= b`, inclusive at both ends | `epoch BETWEEN 1 AND 5`
| `IN (...)` | equal to any item in the list | `split IN ('val', 'test')`
| `LIKE` | pattern match: `%` any run of characters, `_` exactly one | `model LIKE 'resnet%'`
| `IS NULL`, `IS NOT NULL` | value is missing / present | `label IS NOT NULL`
| `AND`, `OR`, `NOT` | combine predicates | see Step 3

Text values go in **single quotes**: `'val'`. Double quotes are for identifiers; SQLite falls back to treating `"val"` as text when no such column exists, PostgreSQL raises an error. `LIKE` is case-insensitive for ASCII letters in SQLite by default and case-sensitive in PostgreSQL (which has `ILIKE`).

<div class="ex">
<div class="lab">Worked example</div>

**evaluate a WHERE clause by hand**

Condition: `split IN ('val', 'test') AND confidence >= 0.9`, applied to the ten-row table.

First predicate, `split IN ('val', 'test')`: TRUE for ids 2, 3, 4, 5, 6, 8, 9, 10 (eight rows); FALSE for the two train rows 1 and 7.

Second predicate, `confidence >= 0.9`: TRUE for ids 1 (0.97), 2 (0.91), 4 (0.95), 5 (0.97), 6 (0.99), 9 (0.93), 10 (0.91); FALSE for 3 (0.88), 7 (0.62) and 8 (0.55).

AND is TRUE only when both sides are TRUE. Intersecting {2, 3, 4, 5, 6, 8, 9, 10} with {1, 2, 4, 5, 6, 9, 10} gives {2, 4, 5, 6, 9, 10}: six rows.

No NULL is involved, so no UNKNOWN appears; SQLite returns exactly ids 2, 4, 5, 6, 9, 10. Boundary check: 0.91 passes `>= 0.9`, 0.88 does not.

</div>

## Step 3: AND binds tighter than OR

Like multiplication before addition, SQL evaluates `NOT` first, then `AND`, then `OR`. This is called **operator precedence**. It means

```sql
WHERE split = 'val' OR split = 'test' AND confidence > 0.9
```

is read as `split = 'val' OR (split = 'test' AND confidence > 0.9)`. The confidence filter only applies to test rows. Every validation row comes through, whatever its confidence. On the ten-row table that returns 8 rows instead of the intended 6: ids 3 (0.88) and 8 (0.55) sneak in.

~~~dg-fig 03.01-precedence.svg
300 synthetic predictions. The intended filter keeps 25 rows (green). Without parentheses, 60 more low-confidence validation rows (crosses) leak in, more than tripling the result.
~~~

On a realistic table the leak is most of a split: 300 synthetic predictions run through both versions in SQLite give 25 rows intended, 85 returned. Feed that into a "high-confidence pseudo-labels" table and you have added 60 shaky labels.

The fix costs two characters:

```sql
WHERE (split = 'val' OR split = 'test') AND confidence > 0.9
-- or, clearer still:
WHERE split IN ('val', 'test') AND confidence > 0.9
```

Rule of thumb: whenever AND and OR meet in one WHERE, parenthesise every OR group, even when precedence already does what you want.

## Step 4: NULL and three-valued logic

NULL is not a value; it is a marker meaning "unknown or missing". A comparison with an unknown cannot be known either, so `NULL = 1`, `NULL <> 1` and even `NULL = NULL` all evaluate to **UNKNOWN**. SQL therefore has **three-valued logic**: TRUE, FALSE, UNKNOWN.

~~~dg-fig 03.01-3vl.svg
Truth tables computed by SQLite. UNKNOWN only disappears when the other side settles the answer: FALSE AND anything is FALSE, TRUE OR anything is TRUE.
~~~

Intuition: treat UNKNOWN as "could be TRUE or FALSE". `FALSE AND UNKNOWN` is FALSE either way; `TRUE AND UNKNOWN` depends on it, so it stays UNKNOWN.

The rule that matters: **WHERE keeps only TRUE rows.** FALSE and UNKNOWN are both dropped. So:

| expression | value when x is NULL | row kept by WHERE?
| --- | --- | --- |
| `x = 1` | UNKNOWN | no
| `x <> 1` | UNKNOWN | no
| `x = NULL` | UNKNOWN, always | no, even for NULL rows
| `x IS NULL` | TRUE | yes
| `NOT (x = 1)` | UNKNOWN | no
| `x <> 1 OR x IS NULL` | TRUE | yes

Practical rules:

- Test for missing values with `IS NULL` / `IS NOT NULL`, never `= NULL`.
- A filter like `label <> 'spam'` also drops unlabelled rows. If you meant "not known to be spam", write `label IS NOT 'spam'` in SQLite or `label IS DISTINCT FROM 'spam'` in PostgreSQL (SQLite accepts that spelling too since version 3.39).
- Complementary-looking conditions are not: `label = 'cat' OR label <> 'cat'` returns 9 of our 10 rows.
- Arithmetic with NULL gives NULL. `COALESCE(x, default)` substitutes a value, but that is a modelling decision: filling missing latency with 0 makes timeouts look fastest.

`NOT IN` with a NULL inside the list is the worst version of this trap; it gets its own section in lesson 03.05.

## Step 5: ORDER BY, LIMIT, OFFSET and DISTINCT

A table is a *set* of rows with no built-in order. Without ORDER BY the database may return rows in any order, and that order can change between runs, after an index is added, or as the table grows. "The first 1,000 rows" of an unordered query is not a reproducible dataset.

**ORDER BY** sorts by one or more keys, each `ASC` (ascending, the default) or `DESC`. Later keys only matter when earlier keys tie. When the keys still tie, the order among the tied rows is again arbitrary. `ORDER BY score DESC LIMIT 10` can return a different tenth row each time if the tenth and eleventh scores are equal. The fix is a **tie-breaker**: end the key list with a unique column, as in `ORDER BY confidence DESC, id`. A sort whose keys are unique is called a **total order**.

Where do NULLs go? Databases disagree. SQLite and MySQL treat NULL as smaller than every value, so NULLs come **first** in ascending order; PostgreSQL treats it as larger, so NULLs come **last**. Say what you want with `NULLS FIRST` / `NULLS LAST` (PostgreSQL, and SQLite since 3.30).

**LIMIT n** keeps the first n sorted rows; **OFFSET k** skips k first. `ORDER BY confidence DESC, id LIMIT 3 OFFSET 3` returns ids 4, 9, 2. The skipped rows are still produced and discarded, so deep pages are slow; lesson 03.07 shows the keyset alternative.

**SELECT DISTINCT a, b** removes duplicate output rows over all listed columns. Here NULLs count as equal, so `SELECT DISTINCT label` returns four rows: cat, dog, NULL, bird.

## Step 6: dates stored as text

SQLite has no date type; timestamps are usually ISO-8601 text like `'2025-03-31 09:00:00'`, which sorts correctly as text. The trap: `'2025-03-31'` is a *prefix* of every timestamp on the 31st, and a prefix sorts before longer strings. So

```sql
WHERE ts BETWEEN '2025-03-01' AND '2025-03-31'
```

stops at the very start of March 31 and silently drops the whole last day, including an event stamped exactly `00:00:00`.

~~~dg-fig 03.01-dates.svg
Seven events near the end of March. BETWEEN with a date-only upper bound keeps only the March 30 event; the half-open range keeps all five March events and excludes April.
~~~

Prefer a **half-open range**, inclusive at the start and exclusive at the end:

```sql
WHERE ts >= '2025-03-01' AND ts < '2025-04-01'
```

It works at any granularity, never double-counts a boundary when you chain months, and needs no "23:59:59" hacks. Text sorting also bites numbers stored as text: `'10' < '9'` is TRUE.

## SELECT and WHERE in code

Python ships with SQLite, so no server is needed. Pass values with `?` placeholders, never by pasting strings into SQL.

```python
import sqlite3
con = sqlite3.connect(":memory:")                      # a throwaway in-memory database
con.execute('''CREATE TABLE predictions(
    id INTEGER PRIMARY KEY, split TEXT, label TEXT, pred TEXT, confidence REAL)''')
con.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", rows)   # the 10 rows above; None = NULL

sql = '''
SELECT id, label, pred, confidence
FROM predictions
WHERE split = 'val' AND pred <> label
ORDER BY confidence DESC, id
LIMIT 3
'''
for row in con.execute(sql):
    print(row)
# (5, 'cat', 'dog', 0.97)
# (2, 'dog', 'cat', 0.91)
# (10, 'bird', 'cat', 0.91)

q = "SELECT id FROM predictions WHERE split = ? AND confidence >= ? ORDER BY id"
print(con.execute(q, ("val", 0.9)).fetchall())         # [(2,), (5,), (6,), (10,)]
```

Checking the NULL rules on the same table:

```python
one = lambda s: con.execute(s).fetchone()[0]
print(one("SELECT COUNT(*) FROM predictions WHERE label = NULL"))        # 0
print(one("SELECT COUNT(*) FROM predictions WHERE label <> 'dog'"))      # 5
print(one("SELECT COUNT(*) FROM predictions WHERE label IS NOT 'dog'"))  # 6
```

pandas does **not** use three-valued logic: a missing value compared with `!=` gives True, so the same filter keeps row 6:

```python
import pandas as pd
df = pd.read_sql("SELECT * FROM predictions", con)
print(df[(df.split == "val") & (df.pred != df.label)].id.tolist())           # [2, 5, 6, 8, 10]
print(df[(df.split == "val") & (df.pred != df.label) & df.label.notna()].id.tolist())  # [2, 5, 8, 10]
```

The tie-breaker, demonstrated: the same rows inserted in two orders give a different second row while the sort has ties, and the same answer once it is total:

```python
for order in (rows, rows[::-1]):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE p(id INTEGER, confidence REAL)")
    c.executemany("INSERT INTO p VALUES (?, ?)", [(r[0], r[4]) for r in order])
    print(c.execute("SELECT id FROM p ORDER BY confidence DESC LIMIT 2").fetchall(),
          c.execute("SELECT id FROM p ORDER BY confidence DESC, id LIMIT 2").fetchall())
# [(6,), (1,)] [(6,), (1,)]
# [(6,), (5,)] [(6,), (1,)]
```

## Checking the result

A query that runs is not yet a query you can trust. Before using its output for training or a report:

- **Count before and after each filter**, adding predicates one at a time. A filter that removes far more or fewer rows than expected is the cheapest bug detector there is.
- **Count the NULLs in every filtered column**: `SELECT COUNT(*) - COUNT(label) FROM predictions` (`COUNT(col)` skips NULLs). If it is not zero, decide what those rows should do.
- **Check that complementary filters add up**: rows with `cond` plus rows with `NOT cond` should equal the table size; any shortfall is UNKNOWN rows.
- **Eyeball rows at the boundaries** of every range filter: the first and last day, values exactly at the threshold.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| Too many rows from an AND/OR filter | AND binds tighter than OR | parenthesise every OR group, or use `IN`
| Rows with missing values vanish from a `<>` filter | comparison with NULL is UNKNOWN | add `OR x IS NULL`, or use `IS NOT` / `IS DISTINCT FROM`
| `WHERE x = NULL` returns nothing | `= NULL` is always UNKNOWN | `WHERE x IS NULL`
| Last day of a month missing | `BETWEEN` with a date-only upper bound on timestamps | half-open range `>= start AND < next_start`
| Top-k list changes between runs | ties in ORDER BY, or no ORDER BY at all | end ORDER BY with a unique column
| Query works in SQLite, fails in PostgreSQL | alias used in WHERE, or `"double quoted"` text | repeat the expression; single-quote strings
| Pandas and SQL counts disagree | pandas `!=` is True for missing values | make the NULL rule explicit in both

## Going deeper

**Why three-valued logic exists.** If `NULL = NULL` were TRUE, two unknown birthdays would count as equal, which is not known. Three-valued (Kleene) logic propagates "not known" consistently, at a price: `p OR NOT p` is not always TRUE, so splitting a table with `cond` and `NOT cond` loses the UNKNOWN rows.

**Different clauses treat UNKNOWN differently.** WHERE, HAVING and JOIN's ON keep a row only for TRUE. A `CHECK` constraint *rejects* a row only for FALSE, so UNKNOWN passes; `CHECK (age > 0)` accepts a NULL age. GROUP BY and DISTINCT treat all NULLs as one group. Remember "WHERE needs TRUE, CHECK needs not-FALSE".

**Logical order versus the physical plan.** Prefix a query with `EXPLAIN QUERY PLAN` in SQLite (or `EXPLAIN` in PostgreSQL) to see what the engine will actually do. With an index on `(split, confidence)`, SQLite 3.53 reports for our query:

```text
SEARCH predictions USING INDEX ix_split_conf (split=?)
USE TEMP B-TREE FOR LAST TERM OF ORDER BY
```

It jumps to the `val` rows through the index, which already delivers them in confidence order, so only the tie-breaker `id` needs an extra sort. The plan changes the cost, never the answer. Predicates of the form `column op constant` can use an index; `WHERE lower(split) = 'val'` usually cannot (lesson 03.07).

## Key terms

| Term | Meaning
| --- | --- |
| Query / SELECT | A request that returns rows; SELECT also names the output columns
| Logical evaluation order | FROM, WHERE, GROUP BY, HAVING, SELECT, DISTINCT, ORDER BY, LIMIT: the order that defines the result
| Predicate | A condition that is TRUE, FALSE or UNKNOWN for each row
| Operator precedence | NOT before AND before OR, unless parentheses say otherwise
| NULL | A marker for a missing or unknown value, not a value itself
| Three-valued logic | TRUE, FALSE, UNKNOWN; WHERE keeps only TRUE rows
| Column alias | A name given to an output column with `AS`; visible in ORDER BY, not in WHERE
| Tie-breaker / total order | A final unique sort key that makes ORDER BY deterministic
| Half-open range | `>= start AND < end`: includes the start, excludes the end

<div class="take">
<div class="lab">Key takeaway</div>

read a query in its logical order (FROM, WHERE, SELECT, ORDER BY, LIMIT), parenthesise every mixed AND/OR, treat every comparison as possibly UNKNOWN because WHERE keeps only TRUE rows, and make every ORDER BY total with a unique tie-breaker.

</div>
