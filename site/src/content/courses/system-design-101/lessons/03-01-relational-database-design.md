---
id: "03.01"
course: "system-design-101"
module: "03"
slug: "relational-database-design"
title: "Relational database design"
keyIdeas: ["Relational databases store data in tables with typed columns and enforce rules with constraints: primary keys, foreign keys, unique, not null, check.", "Normalize to avoid duplicate data and update anomalies: each fact lives in one place. Third normal form is the usual target.", "Model relationships: one-to-many with a foreign key on the \"many\" side; many-to-many with a join table (`likes(user_id, post_id)` with a unique pair).", "Transactions (ACID) let you change several rows atomically.", "Denormalize deliberately for read performance (counters, cached joins), and accept the cost of keeping copies in sync.", "Indexes on foreign keys and common filters keep joins fast."]
question: "Model users, posts and likes. Which columns get foreign keys and unique constraints?"
answer: "`posts.user_id` is a NOT NULL foreign key to `users.id`. `likes` has foreign keys to both `users.id` and `posts.id`, and a composite primary key (or unique constraint) on `(user_id, post_id)` so a user can like a post only once, even when two requests race. `users.email` gets a unique constraint so two accounts cannot share an email. Use surrogate BIGINT ids as primary keys. Add an index on `likes.post_id` (Postgres does not create one for a foreign key) for counting likes and fast post deletion, and an index on `posts(user_id, created_at)` for profile pages. Choose ON DELETE behaviour deliberately, for example CASCADE for likes or RESTRICT plus batched cleanup for very large relationships."
sources: [{"title": "PostgreSQL Tutorial (official)", "url": "https://www.postgresql.org/docs/current/tutorial.html", "note": "Tables, joins, constraints from the source."}, {"title": "CMU 15-445 Database Systems", "url": "https://15445.courses.cs.cmu.edu/", "note": "Free lectures; relational model and normal forms early on."}]
tools: ["PostgreSQL"]
---

## Tables, rows and keys

A relational database stores data as **tables**. Each **row** is one thing (a user, a post, an order), and each **column** is one attribute of that thing (an email, a body, a total). Every table has a **primary key**: a column, or a set of columns, whose value identifies exactly one row. The primary key is unique and never null, so "user 42" always means one specific person.

The power of the relational model is that relationships are not stored as pointers or nested objects. They are stored as values: a post row holds the primary key of the user who wrote it. To answer "who wrote this post?" the database matches the two values, which is called a **join**. Because relationships are just values, you can query them in any direction: posts by a user, the user of a post, users who liked posts by a given user, and so on, without having decided those questions in advance.

For example, take two tables: `cities`, with `name` as its primary key, and `weather`, whose `city` column `REFERENCES cities(name)`. That **foreign key** makes the database reject a weather row for a city that does not exist. The rule lives in the database itself, so every application, every batch job and every engineer typing into a console at midnight obeys it.

~~~dg-tree The three tables of a small social app, showing which columns are keys and what each key points at.
Schema
  users
    *id | primary key
    email | UNIQUE, NOT NULL
    name
    created_at
  posts
    *id | primary key
    user_id | FK to users.id, NOT NULL
    body
    created_at
  likes
    *user_id + post_id | composite primary key
    user_id | FK to users.id
    post_id | FK to posts.id
    created_at
~~~

## Choosing a primary key

There are two families of primary key.

- **Natural keys** use a value that already identifies the thing in the real world: an ISO country code, an email address, a passport number.
- **Surrogate keys** are invented by the database: an auto-incrementing integer (`BIGSERIAL` in Postgres, `AUTO_INCREMENT` in MySQL) or a generated ID such as a UUID.

Natural keys look tidy but often turn out not to be stable. Users change their email. A product code gets reused. If the email is the primary key, every foreign key that copied it must change too, across many tables. For that reason most systems use a surrogate `id` as the primary key and put a **unique constraint** on the natural value (`email UNIQUE`). You get a stable identity plus the business rule.

The key type also matters for performance. In InnoDB (MySQL's main storage engine) the table is physically stored in primary-key order (a **clustered index**), and every secondary index stores a copy of the primary key. A 4-byte or 8-byte integer keeps those secondary indexes small. A 36-character UUID string stored as text can triple their size. Random keys also scatter inserts across the whole index, which lesson 04.09 explains in detail. A `BIGINT` (8 bytes, up to about 9.2 × 10^18) is the safe default; a 4-byte `INT` tops out at about 2.1 billion, a limit real companies have hit on busy tables.

## Constraints are free correctness

Constraints turn business rules into guarantees that the database checks on every single write:

- `PRIMARY KEY`: unique and not null.
- `FOREIGN KEY`: the referenced row must exist. You also choose what happens when the parent row is deleted.
- `UNIQUE`: no two rows share this value or this combination of values. This is how you stop a user liking the same post twice.
- `NOT NULL`: the column must have a value.
- `CHECK`: an arbitrary condition on the row, for example `CHECK (price >= 0)` or `CHECK (status IN ('draft','published'))`.

### Why application checks are not enough

A common first instinct is to check in code: "does this email already exist? If not, insert it." This has a race. Two sign-up requests for the same email arrive 5 ms apart. Both run the `SELECT`, both see no row, and both `INSERT`. Now you have two accounts with one email, and the bug will only show up under load, which is the worst time to discover it.

A unique constraint closes the race because the check and the insert happen inside the database, under the protection of the index. The second insert waits for the first to commit, then fails with a unique-violation error. Your code catches that error and returns "email already taken". The application check is still useful for a friendly error message on the form, but the constraint is what guarantees correctness.

~~~dg-seq Two sign-ups race for the same email. The application check passes for both; only the unique constraint stops the duplicate.
Request A > DB : SELECT user WHERE email = x
DB -> Request A : no rows
Request B > DB : SELECT user WHERE email = x
DB -> Request B : no rows
Request A > DB : INSERT email = x
DB -> Request A : OK, committed
Request B > DB : INSERT email = x
DB -> Request B : ERROR unique violation
~~~

### What happens when a parent is deleted

A foreign key needs a rule for the moment the referenced row disappears. The options:

| Rule | What happens to child rows | Typical use |
| --- | --- | --- |
| `RESTRICT` / `NO ACTION` | The delete is refused while children exist | Orders that reference a customer |
| `CASCADE` | Children are deleted too | Likes when a post is deleted |
| `SET NULL` | The child's foreign key becomes null | A post's optional "edited_by" column |
| `SET DEFAULT` | The child's foreign key takes its default | Reassigning to a placeholder record |

`CASCADE` is convenient but dangerous at scale. Deleting one popular post with 2 million likes turns into a 2-million-row delete inside one transaction, holding locks and generating a huge amount of log. Large systems often use `RESTRICT` and delete children in small batches from a background job instead.

One more detail: in Postgres, declaring a foreign key does **not** create an index on the child column. Without an index on `likes.post_id`, every delete of a post must scan the whole likes table to check for children. Always index foreign key columns that you delete or join through.

## Normalisation in practice

**Normalising** means storing each fact exactly once. A user's name lives in `users`, not copied into every post. An update then happens in one place and the copies cannot disagree. The formal theory has numbered "normal forms"; in practice you need three ideas:

1. **One value per cell.** Do not store a comma-separated list of tags in one column; create a `post_tags` table.
2. **Every non-key column describes the whole key.** In `likes(user_id, post_id, user_name)`, `user_name` depends only on `user_id`, so it belongs in `users`.
3. **No column depends on another non-key column.** If `orders` stores `zip_code` and `city`, the city depends on the zip code, so it can drift out of sync.

The problems normalisation prevents are called **anomalies**: an update anomaly (you change a customer's address in 3 of their 40 order rows), an insert anomaly (you cannot record a customer until they place an order) and a delete anomaly (deleting the last order loses the customer's details).

~~~dg-compare Repeating customer details in every order versus storing them once and referencing them by key.
[Denormalised]
Order row holds name, email, address -(repeated per order)-> Many copies
- Change of address must touch every order
- Copies can disagree
+ No join needed to print an order
[*Normalised]
Order row holds customer_id -(join)-> One customer row
+ Each fact stored once
+ Updates touch one row
- Reads need a join
~~~

The cost of normalising is joins at read time. Joins on indexed keys are cheap for small result sets (a few milliseconds), but a page that joins eight tables for every request can become slow. The working rule: **start normalised, then denormalise specific hot read paths on purpose, when you can measure the need.** A `like_count` column on `posts`, updated alongside each insert into `likes`, is a deliberate denormalisation: it avoids counting millions of rows on every page view, at the cost of keeping two things in sync. Lesson 03.10 and Module 4 return to this.

### Mapping relationships to tables

- **One-to-one** (a user has one profile): put the foreign key on either side with a unique constraint, or just add the columns to `users`.
- **One-to-many** (a user has many posts): the "many" side holds the foreign key, `posts.user_id`.
- **Many-to-many** (users like many posts, posts are liked by many users): a **join table**, `likes(user_id, post_id)`, with a composite primary key or unique constraint on the pair. Extra facts about the relationship itself, such as when the like happened, go in the join table.

## Transactions

A **transaction** groups several statements so that they all take effect or none do. A bank transfer subtracts from one account and adds to another between `BEGIN` and `COMMIT`. If anything fails in between, `ROLLBACK` leaves both accounts untouched. **Savepoints** let you undo part of a transaction without abandoning all of it.

The guarantees are summarised as **ACID**:

- **Atomicity**: all or nothing.
- **Consistency**: every constraint holds at commit time.
- **Isolation**: concurrent transactions do not see each other's half-finished work (to a degree you choose, lesson 03.06).
- **Durability**: once `COMMIT` returns, the change survives a crash (lesson 03.08).

Constraints and transactions work together. In a transfer, a `CHECK (balance >= 0)` makes the debit fail if the account would go negative, which aborts the whole transaction, so the credit never happens either.

## What a database engine is made of

The inside of a database is easiest to understand in layers. The rest of this module walks down them.

~~~dg-layers The layers inside a relational database engine, from the SQL you send down to the disk.
Query processing | parser and planner choose how to run SQL (03.05)
Concurrency control | locks and MVCC keep transactions apart (03.06, 03.07)
Access methods | B-tree and other indexes find rows fast (03.03, 03.04)
Buffer pool | in-memory cache of pages, with eviction
Recovery | write-ahead log makes commits durable (03.08)
*Storage | tables in fixed-size pages on disk, 8 KB in Postgres, 16 KB in InnoDB
~~~

Knowing these layers lets you explain why a design is fast or slow instead of guessing. "This query is slow because it reads 400,000 pages that are not in the buffer pool" is an explanation. "Postgres is slow" is not.

<div class="ex">
<div class="lab">Worked example</div>

**users, posts and likes**

`users(id BIGINT PK, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now())`

`posts(id BIGINT PK, user_id BIGINT NOT NULL REFERENCES users(id), body TEXT NOT NULL CHECK (length(body) <= 5000), like_count INT NOT NULL DEFAULT 0, created_at TIMESTAMPTZ NOT NULL)`, with an index on `(user_id, created_at)` for "posts by this user, newest first".

`likes(user_id BIGINT REFERENCES users(id), post_id BIGINT REFERENCES posts(id) ON DELETE CASCADE, created_at TIMESTAMPTZ, PRIMARY KEY (user_id, post_id))`, plus an index on `(post_id)` to list likes for a post and to make post deletion fast.

The composite primary key on likes makes a double like impossible, even if the button is tapped twice in the same 10 ms. The second insert fails, and the application treats that as "already liked".

The like and the counter update go in one transaction: `INSERT INTO likes ...; UPDATE posts SET like_count = like_count + 1 WHERE id = ?; COMMIT`. If the insert fails as a duplicate, the counter is not touched.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**sizing the likes table**

Suppose 50 million users each like 20 posts a day. That is 1 billion new like rows a day, about 11,600 inserts per second on average.

Each row is 8 + 8 + 8 bytes of data plus about 24 bytes of row header in Postgres, so roughly 50 bytes. The primary key index adds about 30 bytes per row and the `post_id` index another 20. Call it 100 bytes per like.

1 billion × 100 bytes = 100 GB per day, or about 36 TB a year. One machine will not hold that for long, which is why lesson 04.05 asks what to shard likes by. The schema stays the same; only where the rows live changes.

</div>

## Common mistakes

- **Trusting the application to enforce uniqueness.** Races will create duplicates. Use a unique constraint.
- **Forgetting indexes on foreign key columns.** Deletes and joins become full scans.
- **Storing lists in a column.** `tags = 'go,rust,sql'` cannot be indexed, joined or constrained.
- **Using a mutable natural key as the primary key.** Emails and usernames change.
- **Cascading deletes on huge relationships.** One click becomes a multi-million-row transaction.
- **Denormalising on day one.** You pay the sync cost before you know you need the speed.

## In an interview

When you sketch a schema, say out loud which columns are primary keys, which are foreign keys, and which have unique constraints, and say *why* (the race on duplicate likes is a good one to mention). Then name the two or three indexes your main queries need. Interviewers listen for whether you know that correctness rules belong in the database and that every index has a write cost.

<div class="take">
<div class="lab">Key takeaway</div>

put relationships in foreign keys and business rules in constraints, so the database enforces them for every writer, including under concurrency. Normalise first, index your foreign keys, and denormalise only where measurements say you must.

</div>
