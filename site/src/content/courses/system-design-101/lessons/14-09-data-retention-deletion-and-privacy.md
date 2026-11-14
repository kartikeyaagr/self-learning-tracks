---
id: "14.09"
course: "system-design-101"
module: "14"
slug: "data-retention-deletion-and-privacy"
title: "Data retention, deletion, and privacy"
keyIdeas: ["Keep data only as long as needed, and be able to delete it everywhere when required (GDPR right to erasure and similar laws).", "A user's data can live in: primary databases, replicas, caches, search indexes, analytics warehouses, logs, event streams, backups, third-party processors, and ML training sets.", "Build deletion as a workflow: find all locations via a data map, delete or anonymize in each, and track completion.", "Backups are hard to edit, so many systems let deleted data age out of backups, or use **crypto-shredding** (encrypt per user, delete the key).", "Google Cloud commits to deleting customer data within about 180 days, including backups."]
question: "A user asks to be deleted. List every place their data might still exist."
answer: "Primary databases across every service (the user row and everything that references it: posts, messages, orders); denormalised copies such as entries in other users' feeds, message threads and notifications; database replicas; caches and CDN copies; search indexes; object storage, including thumbnails and transcodes; event streams, queues and dead-letter queues; the data warehouse, data lake, ML training data and feature stores; application logs and traces; backups and archives; and third-party processors such as email, payments, analytics and support tools. Some data, such as invoices, may legally have to be kept, and backups are usually allowed to expire on a bounded schedule (or are made unreadable by destroying a per-user encryption key)."
sources: [{"title": "GDPR Article 17: Right to erasure", "url": "https://gdpr-info.eu/art-17-gdpr/", "note": "The legal requirement."}, {"title": "Google Cloud: Data deletion on Google Cloud", "url": "https://docs.cloud.google.com/docs/security/deletion", "note": "How deletion propagates through backups."}]
---

## Why deletion is a design problem

Storing data is easy to design; getting rid of it is not. A modern system copies every piece of user data many times on purpose: replicas for availability, caches for speed, search indexes for queries, events for other services, warehouses for analytics, backups for recovery. Each copy was added by a different team for a good reason, and none of them was designed with "and later, delete exactly this one person" in mind.

Privacy law makes that a hard requirement. Beyond the law, there are good engineering reasons to delete: data you no longer hold cannot leak in a breach, does not cost storage, and does not need protecting. **Data retention** is the policy for how long each kind of data is kept; **deletion** is the machinery that enforces it, both on a schedule and on request.

## The legal requirement

Under the EU's GDPR, Article 17 sets out the "right to erasure" (often called the right to be forgotten). A person can ask a company to erase their personal data, and the company must do so **"without undue delay"** when one of several grounds applies:

- the data is no longer needed for the purpose it was collected for;
- the person withdraws consent and there is no other legal basis;
- the person objects to the processing and there are no overriding legitimate grounds;
- the data was processed unlawfully;
- a law requires erasure;
- it was collected from a child for an online service.

The company must respond to the request within one month (extendable in complex cases). If it **made the data public**, it must also take reasonable steps to tell other companies processing it to erase links and copies. California's CCPA/CPRA and many other laws have similar rights.

There are **exceptions**, where keeping data is necessary for: freedom of expression, a legal obligation (such as tax records the law says to keep for years), public health, archiving and research in the public interest, or establishing and defending legal claims. So erasure is not always total. A bank may have to keep transaction records for seven years while deleting marketing data and app analytics immediately. (This is an engineering summary, not legal advice.)

**Personal data** is broader than names and emails: it includes anything that can identify a person directly or indirectly, such as IP addresses, device IDs, location histories, photos and user IDs that link to any of these.

## Where a user's data actually lives

Deleting the row in `users` is the easy part. By the time a request arrives, the data has usually been copied all over the system.

~~~dg-tree Where one user's data ends up: deletion has to reach every branch, not just the users table.
User 42
  Primary stores
    Users DB row | the easy part
    Posts, messages, orders, comments | in other services' databases
    *Other users' copies | feed entries, message threads, mentions
  Derived stores
    Read replicas
    Caches and CDN | profile photo cached at the edge
    Search index
    Object storage | originals, thumbnails, transcodes
  Pipelines
    Kafka topics and dead-letter queues
    Warehouse, data lake, ML training sets
  Operations
    Logs and traces | emails, IPs in log lines
    Backups and archives | snapshots before the request
  Third parties | email, payments, analytics, support desk
~~~

- **Primary databases**: the user row and everything that references it (posts, messages, orders, comments), spread across every service that owns a piece.
- **Denormalised copies**: entries in other users' precomputed feeds (lesson 11.05), message threads, notifications, @mentions.
- **Replicas and read stores**: database replicas, search indexes, caches (Redis, CDN copies of a profile photo).
- **Object storage**: uploaded photos, videos and documents, plus all their derived versions (thumbnails, transcodes).
- **Event streams and queues**: Kafka topics retaining events for 7 days or forever, dead-letter queues holding failed messages.
- **Analytics**: data warehouse tables, data lake files, ML training sets and feature stores.
- **Logs**: application logs and traces that contain emails, IP addresses or user IDs.
- **Backups and archives**: every snapshot taken before the deletion.
- **Third parties**: email providers, payment processors, analytics and advertising vendors, customer support tools.

Each kind of store needs a different deletion method:

| Store | How deletion works | Typical delay |
| --- | --- | --- |
| Primary DB | delete rows, cascade or tombstone references | seconds |
| Cache, CDN | explicit purge, or TTL expiry | seconds to hours |
| Search index | delete by ID, gone after segment merge | minutes |
| Kafka topic | compacted topics take a tombstone; others expire by retention | days |
| Warehouse, lake | rewrite partitions or files without the user | days, batched |
| Logs | retention expiry, or keep personal data out of them | 30 days |
| Backups | expire on rotation, or crypto-shred | months |
| Third parties | API deletion call, contractual obligation | days to weeks |

## How a large provider does it

A large cloud provider (Google Cloud) deletes data in four stages, finishing within about **180 days** end to end:

1. **Request.** A deletion request for a resource, project or account. The data is marked for deletion immediately, with a target of within 24 hours, and stops being served.
2. **Soft delete and recovery period.** Up to about **30 days** during which an accidental deletion can be undone (deleted service accounts, for instance, are disabled for 30 days). After it, a purge signal goes to every product, and it is resent until every product confirms.
3. **Logical deletion from live systems.** The data is marked as free space and overwritten over time. Storage systems built on log-structured files, such as Bigtable, remove it during garbage collection and **compaction** (lesson 07.12) that run on a schedule, because rewriting files is expensive. This takes about **two months**: the recovery period plus two compaction cycles.
4. **Expiry from backups.** Backups are large snapshots on daily, weekly and monthly cycles. They are not edited to remove one user. Instead, deleted data stops being copied into new backups, and old backups **expire** on their normal schedule. This finishes within about **six months**.

~~~dg-timeline One deletion request at a large cloud provider: each stage is bounded, so the whole pipeline finishes within about 180 days.
Day 0 | *Request received, data marked deleted within 24 h
Day 0 to 30 | Recovery window: soft deleted, can be restored
Day 30 | Purge signal sent to every product, resent until confirmed
Day 30 to 60 | Compaction cycles overwrite it in live storage
Day 60 | Gone from all live systems
Day 60 to 180 | Older backups expire on their normal rotation
Day 180 | Gone everywhere
~~~

The usual argument for a tail like this is that "without undue delay" is judged against what is technically reasonable, and that data in an expiring backup is not being used for anything. The key is that the delay is **bounded and documented**, and that a restored backup must re-apply pending deletions before it is used.

## The deletion pipeline

For your own systems, treat deletion as a distributed workflow, not a statement.

~~~dg-seq A deletion request as a tracked workflow: every store must confirm, and the coordinator retries until all have.
User > Account service : delete my account
Account service > Account service : soft delete, schedule purge day 30
Account service > Event bus : UserDeleted(42) on day 30
Event bus > Posts service : UserDeleted(42)
Event bus > Search indexer : UserDeleted(42)
Event bus > Warehouse job : UserDeleted(42)
Posts service -> Deletion tracker : done
Search indexer -> Deletion tracker : done
Deletion tracker > Warehouse job : no ack after 48 h, resend
Warehouse job -> Deletion tracker : done (nightly rewrite)
Deletion tracker > Deletion tracker : all 14 stores confirmed, close request
~~~

1. **Accept and soft delete.** Hide the account immediately so nothing more is shown or processed, and schedule a purge date (lesson 03.13 covers soft delete).
2. **Publish a deletion event** on a durable bus that every service and pipeline holding personal data consumes.
3. **Track confirmations.** A deletion tracker records which stores must confirm and resends to any that have not. The request closes only when every store has said "done".
4. **Handle late arrivals.** Events still in flight (a queued message that creates a comment for user 42 after deletion) must be rejected. Keep a list of deleted IDs, or tombstones, that writers check.
5. **Notify third parties** through their deletion APIs and record the response.

## Crypto-shredding

Editing every backup to remove one person is impractical: backups are large, compressed, sometimes immutable by design (lesson 14.07), and rewriting them risks corrupting them. **Cryptographic erasure** (crypto-shredding) solves this. Encrypt each user's (or each resource's) data with its own key, and keep the keys in a separate key store that is **not** in the long-lived backups (or is backed up with a much shorter retention). To delete, destroy the key. Every copy of the data, including the ones in backups, replicas and forgotten exports, becomes unreadable at once.

~~~dg-compare Deleting a user from backups: rewriting every copy versus destroying one key.
[Rewrite backups]
find user 42 -> rewrite each snapshot -> verify
- hundreds of TB rewritten per request
- breaks immutable backups
- misses copies you do not know about
[*Crypto-shredding]
user 42 data -(encrypted with K42)-> stores and backups
destroy K42 -> every copy unreadable
+ one small operation
+ reaches unknown copies too
- key store becomes critical
- must encrypt per user from day one
~~~

Google Cloud uses this in Cloud Storage, which lets logical deletion finish before every block is physically overwritten. The costs are real: every read needs a key lookup (cached, so a few microseconds), the key store must be extremely available and itself backed up carefully, and losing a key by accident is permanent data loss. It works best for large blobs (files, message bodies) rather than for every small row.

<div class="ex">
<div class="lab">Worked example</div>

**how much key material?**

50 million users, one 256-bit data key each, wrapped (encrypted) by a master key in a key management service. Each wrapped key plus metadata is about 100 bytes, so all keys fit in 5 GB, easily held in a replicated database and a memory cache.

Reading a user's photo means fetching the wrapped key (cache hit, under 1 ms), unwrapping it, then decrypting the photo with AES-GCM at several GB per second on modern CPUs. The overhead is negligible next to the network transfer.

A deletion destroys one 100-byte row. The 20 GB of that user's photos across live storage, two replicas and 12 backup snapshots are unreadable from that moment.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**tracing one deletion through a social app**

User 42 deletes their account on 1 March. The account is hidden at once and the purge is scheduled for 31 March.

On 31 March the `UserDeleted` event goes out. Posts and comments are deleted; replies by other users stay but show "deleted user". 900 entries in followers' precomputed feeds are removed by the feed service. The search indexer deletes 312 documents. The CDN purges the avatar URL. Photos are crypto-shredded.

Kafka topics with 7-day retention drop the old events by 7 April. The warehouse job rewrites affected daily partitions overnight. Logs (which only carry the pseudonymous ID) expire after 30 days. The payment processor is called to delete the stored card; invoices are kept for 7 years under tax law, with the reason documented.

Backups expire by the end of August. If one is restored before then, the restore procedure replays the deletion list first.

</div>

## Designing for deletion from the start

- **Keep a data inventory**: a map of every store that holds personal data, with an owner, kept up to date. A new store is not allowed into production without an entry.
- **Use one stable internal user ID everywhere**, so every store can find a person's data by a single key.
- **Set retention limits by default**, so data expires even without a request: logs 30 days, Kafka 7 days, raw analytics events 13 months, backups 90 days.
- **Keep personal data out of logs**, or log pseudonymous IDs instead of emails and IPs.
- **Use soft delete with a fixed purge date**, so mistakes can be undone but deletion is guaranteed.
- **Separate what you must keep** (invoices required by tax law) from what you must delete, and document the legal reason.
- **Anonymise rather than delete** where aggregate value remains: replacing a user ID with a random value in analytics is acceptable only if the result truly cannot be linked back.

## Common mistakes

- **Deleting only the primary row** and leaving search, caches, feeds and the warehouse untouched.
- **No confirmation tracking**, so a consumer that was down that day never deletes anything.
- **Restoring an old backup** and silently bringing deleted users back.
- **Personal data in logs and metrics labels**, which then live in systems with no deletion path.
- **Unbounded retention** "just in case", which makes every request harder and every breach bigger.

## In an interview

If the system stores personal data, mention deletion briefly: a deletion event consumed by every store, tracked to completion, retention limits by default, backups that expire on a bounded schedule, and per-user encryption keys for large blobs. It shows you think about the whole lifecycle of data, not just the write path.

<div class="take">
<div class="lab">Key takeaway</div>

erasure is a pipeline, not a `DELETE` statement. Map every copy (primary stores, denormalised copies, replicas, caches, search, object storage, streams, warehouse, logs, backups, vendors), send a deletion event to all of them and track that each confirms, let backups expire on a fixed schedule, and use per-user encryption keys so destroying a key erases the copies you cannot reach.

</div>
