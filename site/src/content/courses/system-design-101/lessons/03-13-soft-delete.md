---
id: "03.13"
course: "system-design-101"
module: "03"
slug: "soft-delete"
title: "Soft delete"
keyIdeas: ["Soft delete marks a row as deleted (`deleted_at` timestamp) instead of removing it.", "Benefits: easy undo, audit trail, safer against accidental deletes.", "Costs: every query must filter `WHERE deleted_at IS NULL` (easy to forget, leaks deleted data), unique constraints break (a deleted email blocks re-signup unless you use partial unique indexes), foreign keys still point at \"deleted\" rows, and tables grow forever.", "Privacy laws may require real deletion anyway.", "Alternative: hard delete and copy the row to an archive or audit table first."]
question: "What breaks in unique constraints and queries when you soft delete rows?"
answer: "Unique constraints still see soft-deleted rows, so a deleted user's email or username cannot be reused (re-signup fails with a duplicate key) unless every unique constraint becomes a partial unique index WHERE deleted_at IS NULL, and restoring a row can then fail if someone else took the value. Every query and every join must remember WHERE deleted_at IS NULL, or deleted data leaks into results, counts and reports. Foreign keys no longer protect you: live children can point at a soft-deleted parent and ON DELETE CASCADE never fires. Tables and indexes fill with dead rows unless indexes are made partial, and the eventual hard purge must walk every dependent table in order."
sources: [{"title": "Brandur: Soft deletion probably isn't worth it", "url": "https://brandur.org/soft-deletion", "note": "The costs of deleted_at columns."}]
---

## The appeal

**Soft deletion** means never running `DELETE`. Instead you add a column such as `deleted_at timestamptz` and "delete" a row by setting it: `UPDATE projects SET deleted_at = now() WHERE id = 42`. Rows with a timestamp are treated as gone. Undeleting is setting it back to null. A **hard delete**, by contrast, removes the row with a real `DELETE`.

The appeal is easy to see:

- A hard delete is permanent. One bad query or one mistaken click and the data is gone.
- Support staff and engineers can look at deleted data when investigating a bug or a customer complaint.
- An "undo" or a trash bin looks almost free to build.

Soft deletion is common, and frameworks make it a one-line setting. But the costs are spread thinly across the whole codebase and the whole lifetime of the data, and they usually outweigh the benefits. This lesson goes through what breaks and what to do instead.

~~~dg-state The life of a soft-deleted row. The restore path is rarely used, and the purge step is where the hidden cost appears.
[*] -> Active : INSERT
Active -> Soft deleted : set deleted_at
Soft deleted -> Active : restore, rarely used
Soft deleted -> *Purged : hard DELETE after retention
Purged -> [*] : gone, with its dependents
~~~

## What goes wrong

### Every query must remember the filter

Every read needs `WHERE deleted_at IS NULL`. Every join needs it on **each** joined table. Miss it once, in a report, an admin tool, an analytics export, a background job or a hand-written query during an incident, and deleted data appears where it should not: a deleted user in a mention autocomplete, a cancelled order counted in revenue, a removed message shown in search.

ORM plugins and "default scopes" add the filter automatically, which helps in application code. But they do not help the analyst in a SQL console, the data pipeline that reads replicas directly, or raw SQL inside the application. And the filter is easy to forget on joins: `SELECT ... FROM orders JOIN customers ...` with the filter on `orders` but not `customers` shows live orders of deleted customers.

### Foreign keys stop protecting you

A foreign key checks that the parent **row exists**. A soft-deleted customer still exists as a row, so the database is perfectly happy for that customer's invoices, addresses and sessions to stay live underneath it. `ON DELETE CASCADE` never fires, because nothing was deleted. Nothing in the database enforces that children are soft-deleted when the parent is; every relationship needs application code, and every new table added later needs that code updated.

### Unique constraints break

A user deletes their account and later signs up again with the same email. The unique index on `email` rejects it, because the soft-deleted row still holds that email. The same applies to usernames, slugs, SKUs and any other unique business key.

~~~dg-seq Why re-signup fails with soft delete: the unique index still sees the soft-deleted row.
User > App : delete my account
App > users table : UPDATE SET deleted_at = now()
User > App : sign up again, same email
App > users table : INSERT email = ana@x.com
users table -> App : ERROR duplicate key on email
App -> User : email already taken
~~~

The usual fix is a **partial unique index** that only covers live rows:

```
CREATE UNIQUE INDEX users_email_live ON users (email) WHERE deleted_at IS NULL;
```

Now any number of deleted rows can share an email, but only one live row can hold it. Every unique constraint on every soft-deletable table needs the same treatment, and some databases (MySQL among them) have no partial indexes, forcing workarounds such as a generated column that is null for deleted rows. Restoring a deleted row can now fail too, if someone else has taken its email in the meantime.

### Tables and indexes fill with dead data

Soft-deleted rows stay in the table and in every index. If 30% of rows are deleted, scans read 30% more data, indexes are 30% bigger, and the buffer pool caches rows nobody will ever see. Every index used by normal queries should become partial (`WHERE deleted_at IS NULL`) too, which is easy to forget.

### You still have to hard-delete eventually

Privacy law changes the picture. Under GDPR, a user's right to erasure means "deleted" must eventually mean gone, and keeping personal data forever without a reason is itself a problem. So soft delete does not remove the need for a hard-delete path; it just delays it and makes it harder.

Purging a soft-deleted row means hard-deleting its dependents first, in the right order, across every table that references it. One real-world purge query touched about 30 tables and broke whenever someone added a new table with a reference that the purge did not know about.

~~~dg-tree What purging one soft-deleted customer actually requires: every dependent table, in order, deepest first.
customer 42 | soft deleted 40 days ago
  orders | 120 rows
    order_items | 900 rows
    payments | 118 rows
      refunds | 3 rows
    shipments | 115 rows
  addresses | 4 rows
  sessions | 2,000 rows
  *new_feature_table | added last month, purge script unaware
~~~

### Undelete rarely works anyway

Deleting usually has **side effects outside the database**: files removed from object storage, a subscription cancelled at the payment provider, a search index updated, emails sent, a virtual machine shut down, an access token revoked. Flipping `deleted_at` back to null undoes none of those. A "restored" project whose files are gone is worse than a deleted one. In practice, teams that build soft delete for undo find it is almost never used, and when it is, it needs manual work anyway.

## The alternative: an archive table

Delete rows for real, but **copy them into an archive** first. A single generic table works for every table in the schema:

```
CREATE TABLE deleted_record (
  id             bigserial PRIMARY KEY,
  deleted_at     timestamptz NOT NULL DEFAULT now(),
  original_table text NOT NULL,
  original_id    text NOT NULL,
  data           jsonb NOT NULL
);
```

In Postgres one statement can do both the delete and the copy, using a data-modifying common table expression:

```
WITH gone AS (DELETE FROM projects WHERE id = 42 RETURNING *)
INSERT INTO deleted_record (original_table, original_id, data)
SELECT 'projects', id::text, to_jsonb(gone) FROM gone;
```

A trigger on each table can do the same automatically, so no code path can forget.

~~~dg-compare Soft delete flags rows in place; an archive table deletes for real and keeps a JSON copy elsewhere.
[Soft delete column]
DELETE becomes UPDATE -> Row stays in live table -> Every query filters
- Filters leak when forgotten
- Foreign keys and unique keys break
- Purge spans many tables
+ One-step restore, in theory
[*Archive table]
DELETE copies row -> JSON in deleted_record -> Live table only has live rows
+ Queries need no filter
+ Constraints work as designed
+ Purge is one DELETE on one table
- Restore is manual
~~~

The benefits are broad. Normal queries need no filter. Foreign keys work again: deleting a parent that still has children is an error, as it should be, or cascades as designed. Unique constraints need no partial indexes. Live tables and indexes contain only live data. And purging old archived data is one statement on one table: `DELETE FROM deleted_record WHERE deleted_at < now() - interval '90 days'`.

The cost: restoring means reading JSON and re-inserting it by hand or with a small tool, and the archived JSON does not follow later schema changes. That is acceptable because restores are rare, and the archived copy is still there for investigations.

Other alternatives exist for specific needs: **audit or history tables** that record every change (not just deletes), **event sourcing** where state is rebuilt from an append-only log of events, and simply relying on **backups and point-in-time recovery** (lesson 03.08) for the rare "we deleted the wrong thing" disaster.

~~~dg-matrix Ways to keep deleted data, compared on what matters day to day.
Approach | Queries need filter | Constraints work | Easy purge | Easy restore
Soft delete column | yes, everywhere | no | no | yes, in theory
Archive table | no | yes | yes | partial, manual
History or audit table | no | yes | yes | partial
Backups and PITR only | no | yes | yes | no, slow and coarse
~~~

## When soft delete is still reasonable

- A real product feature such as a **trash bin with restore** (documents, emails, files), where the business wants it, users expect it, and the side effects are reversible or deferred until the trash is emptied.
- Small, simple tables where the filter is applied in exactly one place, such as a view or a single repository class, and nothing reads the table any other way.

Even then: use partial unique indexes, put the filter in a view (`CREATE VIEW live_users AS SELECT * FROM users WHERE deleted_at IS NULL`) that all reads use, make normal indexes partial, and run a scheduled purge that hard-deletes rows after a retention period.

<div class="ex">
<div class="lab">Worked example</div>

**fixing a users table that already uses soft delete**

Problem 1: `UNIQUE (email)` blocks re-signup. Fix: `CREATE UNIQUE INDEX CONCURRENTLY users_email_live ON users (email) WHERE deleted_at IS NULL`, then drop the old constraint.

Problem 2: `SELECT count(*) FROM users` on the growth dashboard includes 3.1 million deleted users out of 12 million, overstating users by 35%. Fix: a `live_users` view, and a rule that dashboards and code read only from it.

Problem 3: GDPR erasure requests. Fix: a nightly job that hard-deletes users soft-deleted more than 30 days ago, after deleting their dependents in 14 tables. The job runs in batches of 1,000 users so it never holds long locks, and a test fails if a new table references `users` without being added to the job.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the cost in storage and speed**

A comments table has 500 million rows, and over 4 years 40% have been soft-deleted. Each row with index entries is about 400 bytes.

Dead weight: 200 million rows × 400 bytes = 80 GB, in a table of 200 GB. The hot indexes are 40% larger than they need to be, so less of them fits in the 64 GB buffer pool and the cache hit rate drops.

Switching to an archive table moves those 80 GB to a cold, rarely read table (or out to object storage after 90 days, lesson 04.14). Live indexes shrink by 40%, and the most common queries go back to being served entirely from RAM.

</div>

## Common mistakes

- **Forgetting the filter on joins**, so live rows of deleted parents leak out.
- **Keeping a plain unique constraint** on a soft-deletable column.
- **Assuming restore undoes everything**, when files, payments and messages are already gone.
- **Never purging**, which collides with privacy law and lets tables grow forever.
- **Adding soft delete by default** on every table because a framework makes it easy.

## In an interview

If your design involves deleting data, say how: hard delete with an archive or audit trail for most tables, a trash feature only where the product needs one, and a retention policy for purging. Mention the unique constraint and filtering problems if soft delete comes up; it shows you have lived with the consequences.

<div class="take">
<div class="lab">Key takeaway</div>

soft delete looks safe but leaks deleted data into any query that forgets the filter, breaks unique constraints and foreign keys, bloats tables and indexes, and makes the eventual hard delete harder. Prefer real deletes plus an archive table, and use soft delete only for a genuine trash feature, with partial indexes and a scheduled purge.

</div>
