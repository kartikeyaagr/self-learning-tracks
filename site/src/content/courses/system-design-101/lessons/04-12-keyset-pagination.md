---
id: "04.12"
course: "system-design-101"
module: "04"
slug: "keyset-pagination"
title: "Keyset pagination"
keyIdeas: ["Offset pagination (`LIMIT 20 OFFSET 100000`) makes the database read and discard all skipped rows, so deep pages get slower. It also skips or repeats items when rows are inserted between requests.", "Keyset (cursor) pagination remembers the last item seen: `WHERE (created_at, id) < (:last_ts, :last_id) ORDER BY created_at DESC, id DESC LIMIT 20`.", "With an index on (created_at, id), each page is an index seek, so page 10,000 is as fast as page 1.", "Include a unique tiebreaker column so the order is total.", "Trade-off: you can't jump to an arbitrary page number, only next/previous."]
question: "Why does page 10,000 with OFFSET get slower, and why doesn't keyset?"
answer: "With OFFSET the database has no way to jump to the Nth row, so it must produce every earlier row in order and discard it: page 10,000 of 20 rows reads about 200,000 index entries and rows to return 20, so cost grows linearly with depth (and inserts shift rows, causing duplicates and gaps). Keyset pagination remembers the last row's sort values and uses WHERE (sort_key, id) < (last values) ORDER BY sort_key, id with a matching composite index, so the database descends the B-tree straight to that position (3 or 4 page reads) and reads only the next 20 entries. The cost is the same on page 1 and page 10,000, and results stay stable because the position is defined by values, not by counting rows."
sources: [{"title": "Use The Index, Luke: No Offset", "url": "https://use-the-index-luke.com/no-offset", "note": "Why OFFSET is slow and how keyset works."}, {"title": "Use The Index, Luke: Paging through results", "url": "https://use-the-index-luke.com/sql/partial-results/fetch-next-page", "note": "The SQL."}]
tags: ["Code"]
tools: ["PostgreSQL"]
---

## How OFFSET pagination works

Almost every list in an application is paginated: a feed, search results, an admin table, an API that returns 20 items at a time. The most common way to write it is with `LIMIT` and `OFFSET`:

```
SELECT id, title, created_at FROM posts
ORDER BY created_at DESC
LIMIT 20 OFFSET 200000;
```

This asks for page 10,001 (rows 200,001 to 200,020). It reads naturally: "skip 200,000, take 20". What the database actually does is very different. It has no way to jump to "the 200,001st row", because an offset is just a count and carries no information about where that row is. So it **produces the first 200,020 rows in order, throws away 200,000 of them and returns 20**.

Even with a perfect index on `created_at`, the database must walk 200,020 index entries and usually fetch the corresponding rows (to evaluate filters or columns) before discarding them. So:

- **Each page is slower than the last.** Cost grows linearly with page depth. Page 1 reads 20 rows; page 100 reads 2,000; page 10,000 reads 200,000.
- **Deep pages are expensive for everyone.** A crawler or a bot paging through an API to the end can produce queries that each read millions of rows, and a few of them can saturate a database.

~~~dg-bars Rows the database reads to return one page of 20, as the user pages deeper.
OFFSET, page 1 | 20 | 20
OFFSET, page 100 | 2000 | 2,000
OFFSET, page 10,000 | 200000 | 200,000
OFFSET, page 100,000 | 2000000 | 2 million
*Keyset, any page | 20 | 20
~~~

### The correctness problem

OFFSET is also **wrong** on data that changes. The position "row 41" is recomputed on every request. If a new post is inserted at the top while a user is reading page 2, every later row shifts down by one place. The request for page 3 (`OFFSET 40`) now starts one row earlier than it would have, so the user **sees the last row of page 2 again**. A deletion has the opposite effect: everything shifts up, and one row is **skipped**, never shown at all. On a busy feed where many posts arrive per second, users see duplicates and gaps constantly.

~~~dg-timeline A new post arrives while a user is paging with OFFSET, so the next page repeats a post the user has already seen.
10:00:00 | User loads page 1, posts 1 to 20
10:00:05 | User loads page 2, posts 21 to 40
10:00:07 | *New post inserted at the top
10:00:09 | User loads page 3 with OFFSET 40
10:00:09 | Old post 40 is now row 41: shown again
~~~

## Keyset (seek) pagination

**Keyset pagination**, also called **seek pagination** or cursor pagination, fixes both problems. Instead of saying "skip N rows", the client says "**give me rows after the last one I saw**". It sends back the sort values of the last row on the current page, and the query uses them in a `WHERE` clause.

With one unique sort column, it is simple:

```
SELECT id, title FROM posts
WHERE id < :last_id
ORDER BY id DESC
LIMIT 20;
```

With a sort column that is **not unique**, such as a timestamp or a date, you must add a **unique tie-breaker** column, because paging needs a fully determined order. If 50 posts share the same `created_at` and a page boundary falls among them, "after this timestamp" would skip the rest. Adding the ID makes every position unique:

```
SELECT sale_id, sale_date, amount FROM sales
WHERE (sale_date, sale_id) < (:last_date, :last_id)
ORDER BY sale_date DESC, sale_id DESC
FETCH FIRST 10 ROWS ONLY;
```

served by an index on `(sale_date, sale_id)`.

### Row value comparisons

The `(a, b) < (x, y)` syntax is a **row value** comparison. It compares `a` with `x` first, and only if they are equal does it compare `b` with `y`: the same as sorting by `a` then `b`. Support varies:

- **Postgres** (and Db2) evaluate it correctly and use a composite index to seek directly to `(x, y)`.
- **MySQL** evaluates it correctly, but historically cannot always use the index for it, so check the plan.
- **SQL Server** does not support row value comparisons. There you write the equivalent by hand: `sale_date <= :d AND NOT (sale_date = :d AND sale_id >= :id)`. The first part, `sale_date <= :d`, gives the optimiser a range it can seek on.

## Why it stays fast

With keyset pagination the database uses the index to **jump straight to the position just after the last-seen key** (a B-tree descent, lesson 03.04: 3 or 4 page reads), then reads the next 20 entries in order and stops. The cost is the same on page 1 and on page 10,000. Measurements on real databases show OFFSET response time climbing steadily with the page number, clearly visible from about page 20 onwards, while keyset stays flat.

~~~dg-tree How the index serves a keyset query: descend once to the last-seen key, then read forward along the leaf chain.
Index on (created_at DESC, id DESC)
  Root page
    Branch page | covers 2025-03-01
      Leaf page | ... 10:22:07 id 98251, 10:22:05 id 98240
      *Leaf page | seek lands after 10:22:05 id 98213
        10:22:05 id 98190 | row 1 of next page
        10:22:04 id 98177 | row 2, read on to 20
~~~

It is also **stable**. The position is defined by **values**, not by counting rows from the top. A post inserted at the top since the last request has a newer timestamp, so it is simply not "after" the cursor and does not shift anything. A deleted row is just absent. No duplicates, no gaps.

~~~dg-compare OFFSET pagination versus keyset pagination for a feed.
[OFFSET]
Skip N rows -(read and discard N)-> Return 20
+ Jump to any page number
+ Trivial to implement
- Cost grows with page depth
- Inserts and deletes cause repeats and gaps
[*Keyset]
Seek past last key -(index descent)-> Return 20
+ Same cost on every page
+ Stable while data changes
- No jump to page 500
- Needs an index matching ORDER BY
~~~

## Trade-offs

- **No jumping to an arbitrary page number.** You can only go to the next page (or previous). That suits infinite scroll and "next" links, which is what most feeds and APIs use. A "page 1 2 3 ... 500" control needs OFFSET or a different design. In practice nobody reads page 500; offer search and filters instead.
- **Going backwards** needs the comparison and sort order reversed (`> :first_key ORDER BY ... ASC`), then the results reversed again for display.
- **The client must carry the keyset values.** In an API, wrap them in an **opaque cursor token** (lesson 02.01), such as base64-encoded JSON, so clients cannot depend on its contents and you can change the encoding later. Sign or encrypt it if the values are sensitive.
- **The index must match.** Filters and sort order need an index whose columns match the `WHERE` and `ORDER BY`: equality filters first, then the sort columns in order, including the tie-breaker. For a user's feed, `(user_id, created_at, id)`.
- **Total counts are separate.** "Showing 20 of 1,284,331" needs a `COUNT(*)`, which is expensive on large tables whatever the pagination method. Use an estimate or omit it.

## Combining both

Some interfaces want both: fast infinite scroll for most users, and occasional deep jumps. A common compromise allows OFFSET only up to a small maximum (say 50 pages), and uses keyset beyond that. Elasticsearch takes the same approach: `from` and `size` are limited to 10,000 results by default, and deeper paging uses `search_after`, which is keyset pagination over the sort values.

<div class="ex">
<div class="lab">Worked example</div>

**a feed API**

First request: `GET /feed?limit=20` runs `WHERE user_id = 7 ORDER BY created_at DESC, id DESC LIMIT 20`, served by an index on `(user_id, created_at, id)`. The last row returned has `created_at = 2025-03-01T10:22:05Z` and `id = 98213`.

The response includes `next_cursor = base64("2025-03-01T10:22:05Z|98213")`.

Next request: `GET /feed?limit=20&cursor=...`. The server decodes the cursor and runs `WHERE user_id = 7 AND (created_at, id) < ('2025-03-01T10:22:05Z', 98213) ORDER BY created_at DESC, id DESC LIMIT 20`. The database descends the index to `(7, 10:22:05, 98213)`, reads the next 20 entries and stops: about 4 index pages and 20 rows, the same as page 1.

When fewer than 20 rows come back, the server omits `next_cursor`, and the client knows it has reached the end.

</div>

~~~dg-seq A cursor-based API: the server hands out an opaque cursor with each page, and the client sends it back to get the next one.
Client > API : GET /feed?limit=20
API > DB : WHERE user_id = 7 ORDER BY created_at, id DESC LIMIT 20
DB -> API : 20 rows, last is (10:22:05, 98213)
API -> Client : items + next_cursor
Client > API : GET /feed?cursor=...
API > DB : WHERE (created_at, id) < (10:22:05, 98213) LIMIT 20
DB -> API : next 20 rows, same cost
API -> Client : items + next_cursor
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the export job that took the database down**

A nightly job exports 10 million orders with `OFFSET` pages of 1,000. Page n reads n × 1,000 rows, so the job reads 1,000 + 2,000 + ... + 10,000,000 rows: about 10,000 × 10,000 × 1,000 / 2 = 50 billion rows in total, for an export of 10 million. It runs for 14 hours and slows production queries all night.

Rewritten with keyset (`WHERE id > :last_id ORDER BY id LIMIT 1000`), each page reads 1,000 rows. The whole job reads 10 million rows, 5,000 times less, and finishes in about 6 minutes. The same pattern is the right way to write batched backfills (lesson 03.12).

</div>

## Common mistakes

- **Keyset on a non-unique column without a tie-breaker**, which skips or repeats rows at page boundaries.
- **An index that does not match the sort order**, so each "seek" becomes a scan plus sort.
- **Exposing raw keyset values** in the API instead of an opaque cursor.
- **Using OFFSET for batch jobs and exports.**
- **Building deep page-number navigation** that nobody uses but every crawler hammers.

## In an interview

Whenever you design a list endpoint, say "cursor-based (keyset) pagination": the cursor encodes the last item's sort key plus a unique ID, the query seeks with `(sort_key, id) < (cursor values)` on a matching index, and every page costs the same. Mention that it also avoids duplicates when new items arrive, which matters for feeds.

<div class="take">
<div class="lab">Key takeaway</div>

OFFSET makes the database read and discard every skipped row, so deep pages get slower and slower, and inserts or deletes cause repeated or missing rows. Keyset pagination seeks past the last-seen key through an index: constant cost on every page and stable results while data changes, at the price of no random page jumps.

</div>
