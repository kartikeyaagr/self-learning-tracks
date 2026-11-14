---
id: "04.14"
course: "system-design-101"
module: "04"
slug: "hot-and-cold-storage-archival"
title: "Hot and cold storage, archival"
keyIdeas: ["Data has a temperature. Recent data is read often (hot); old data rarely (cold).", "Store hot data on fast, expensive storage (SSD, memory, primary DB) and move cold data to cheap storage (object storage infrequent-access or archive tiers).", "Object storage classes differ a lot in price and retrieval: standard (instant), infrequent access (instant, retrieval fee), archive (minutes to hours to restore).", "Lifecycle rules move or expire data automatically by age.", "Keep an index or metadata for archived data so you can still find it, and design the app to handle slower access for old records."]
question: "Logs older than 30 days are rarely read. What's your storage plan and the retrieval tradeoff?"
answer: "Keep the last 30 days hot: compressed log files in S3 Standard plus a search index, so incidents can be investigated in seconds. A lifecycle rule then moves files older than 30 days to a cheaper class: Glacier Instant Retrieval (or Standard-IA) for the next months, which still reads in milliseconds but charges per GB retrieved, then Glacier Deep Archive for long-term compliance, and finally expires them when the retention period (for example 7 years) ends. Compact hourly files into large compressed objects before archiving to avoid per-object and transition fees. The trade-off: storage becomes many times cheaper (Deep Archive is about 1/23 of Standard's price), but reading old logs costs retrieval fees, is no longer searchable without a restore and a separate query tool, and from Deep Archive takes up to 12 to 48 hours; objects also incur minimum storage durations of 90 to 180 days if deleted early."
sources: [{"title": "AWS S3: Storage classes", "url": "https://aws.amazon.com/s3/storage-classes/", "note": "Hot, infrequent, archive tiers and their prices."}, {"title": "AWS S3: Lifecycle rules", "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html", "note": "Moving data between tiers automatically."}]
---

## Data cools over time

Most data is read heavily when it is new and rarely afterwards. Application logs are searched during an incident today, almost never a year later. Order history is viewed in the week after purchase, then occasionally for a return or a tax question. Photos are viewed a lot in the first days after upload. Backups exist for the rare day you need them. Data scientists sometimes call this the **temperature** of data: **hot** data is read often and needs to be fast, **warm** data is read sometimes, and **cold** data is read rarely, if ever.

Keeping everything on the fastest, most expensive storage wastes money, and the waste grows every day because old data accumulates while new data keeps arriving. The standard pattern is **tiering**: move data to cheaper storage as it ages, and accept slower or costlier access to the old data. The design questions are always the same: how fast must old data be readable, how often will it be read, and how long must it be kept at all?

~~~dg-layers The storage hierarchy from hot to cold: each step down is cheaper per gigabyte and slower or costlier to read.
*RAM and caches | microseconds, most expensive per GB
Local NVMe SSD | about 0.1 ms
Database and search cluster | ms queries, indexed, costly to keep large
Object storage, standard class | tens of ms, about 23 dollars per TB per month
Object storage, infrequent access | ms reads, retrieval fee per GB
Archive, instant retrieval | ms reads, higher retrieval fee
Deep archive | 12 to 48 hours to restore, about 1 dollar per TB per month
~~~

## S3 storage classes

Amazon S3 is the reference example, and other clouds (Google Cloud Storage, Azure Blob Storage) offer very similar ladders. All S3 classes are designed for **eleven nines** (99.999999999%) of durability, so you are not trading safety for price. You are trading **storage price** on one side against **retrieval speed**, **retrieval fees** and **minimum storage duration** on the other.

| Class | For | Retrieval | Minimum duration |
| --- | --- | --- | --- |
| Standard | Frequently accessed data | Milliseconds, no retrieval fee | None |
| Intelligent-Tiering | Unknown or changing access | Milliseconds (optional archive tiers slower) | None; small monitoring fee per object |
| Standard-IA | Infrequent but needs fast access | Milliseconds, per-GB fee | 30 days |
| One Zone-IA | Infrequent, re-creatable data | Milliseconds, per-GB fee; one availability zone only | 30 days |
| Glacier Instant Retrieval | Accessed a few times a year | Milliseconds, higher per-GB fee | 90 days |
| Glacier Flexible Retrieval | Backups, rarely accessed archives | Minutes to hours (about 1 to 5 minutes expedited, 3 to 5 hours standard, up to 12 hours bulk) | 90 days |
| Glacier Deep Archive | Compliance archives | Within 12 hours standard, within 48 hours bulk | 180 days |

Approximate storage prices in a typical US region, which change over time but keep their ratios:

~~~dg-bars Approximate storage price per terabyte per month for each S3 class. Deep Archive is about 23 times cheaper than Standard.
Standard | 23 | about $23
Standard-IA | 12.5 | about $12.50
One Zone-IA | 10 | about $10
Glacier Instant Retrieval | 4 | about $4
Glacier Flexible Retrieval | 3.6 | about $3.60
*Glacier Deep Archive | 1 | about $1
~~~

**Intelligent-Tiering** deserves a note. It watches each object's access and moves it automatically: to an infrequent tier after 30 days without access, to an archive-instant tier after 90 days, and back to the frequent tier as soon as it is read, with no retrieval fees. You pay a small monitoring fee per object. It is a good default when you cannot predict access patterns, and a poor choice for millions of tiny objects, where the per-object fee dominates.

### Things that catch people out

- **Minimum storage duration.** Deleting or moving an object from Standard-IA after 10 days still bills the full 30 days; from Deep Archive, 180 days. Data that is often deleted early should not go to these classes.
- **Retrieval fees.** Cheap storage with heavy reads can cost more than Standard. Standard-IA charges about \$10 per TB retrieved; if you read the whole dataset twice a month, it costs more than leaving it in Standard.
- **Small objects.** The IA and Glacier Instant classes bill each object as at least 128 KB. Glacier Flexible and Deep Archive add about 40 KB of metadata per object, part of it billed at Standard rates. Transitions between classes are charged **per request** (per object). Millions of tiny log files can cost more to transition than to store, so **batch small objects** into large compressed files before archiving.
- **Single-zone classes** keep data in one availability zone and can lose it if that zone is destroyed. Use them only for data you can re-create.
- **Restores are asynchronous.** Reading from Glacier Flexible or Deep Archive means requesting a restore, waiting hours, then reading a temporary copy. Code and runbooks must expect that.

## Lifecycle rules

You do not move objects by hand. **Lifecycle rules** are attached to a bucket, optionally filtered by key prefix or object tags, and the storage service applies them automatically to existing and future objects:

- **Transition actions** move objects to another class N days after creation.
- **Expiration actions** delete objects N days after creation (and can clean up old versions and incomplete multipart uploads).

A common pattern: Standard for 30 days, Standard-IA or Glacier Instant Retrieval until 90 or 180 days, then an archive class, then deletion when the retention period ends.

~~~dg-timeline One log file's life under a lifecycle policy: it gets cheaper to keep and slower to read as it ages.
Day 0 | *Written to S3 Standard and indexed for search
Day 30 | Dropped from search index; moved to Glacier Instant Retrieval
Day 180 | Moved to Glacier Deep Archive
Year 3 | Restore requested for an audit: ready within 12 hours
Year 7 | *Expired and deleted
~~~

## Beyond S3: tiering everywhere

The same idea appears in almost every data system:

- **Databases.** Partition large tables by time (lesson 04.05). Keep recent partitions in the main database on fast disks. Detach old partitions, export them to object storage as **Parquet** files (a compressed columnar format, lesson 04.03), and query them when needed with engines such as Athena, Trino, DuckDB or ClickHouse. Dropping a whole partition is instant, unlike deleting millions of rows.
- **Kafka tiered storage.** Recent log segments stay on broker disks for fast reads; older segments move to object storage, so topics can keep months of history without huge broker disks.
- **Search clusters.** Elasticsearch and OpenSearch support hot, warm, cold and frozen node tiers: new indexes on fast SSD nodes, older ones on cheaper nodes, the oldest searchable directly from object storage at higher latency.
- **Caches.** RAM for the hottest items, then local SSD, then the origin in object storage: a CDN (lesson 13.02) is a cache hierarchy built on the same principle.

## Designing a plan

1. **Measure access by age.** What fraction of reads touch data older than 7, 30, 90 days? Decide what latency old data actually needs. "Instant" is rarely needed for year-old logs.
2. **Decide retention** from legal, regulatory and business needs, not habit. Keeping everything forever costs money and creates privacy risk (lesson 03.13).
3. **Choose the transitions** with the minimum durations and retrieval fees in mind.
4. **Batch and compress** small objects before archiving.
5. **Design retrieval as an explicit, rare, possibly asynchronous operation** in the product: "your export will be ready in a few hours, we will email you".
6. **Test restores.** An archive you have never restored from is a hope, not a plan.

~~~dg-matrix Storage choices for logs of different ages, against what each age actually needs.
Age | Needs full-text search | Needs ms reads | Read frequency | Good home
0 to 30 days | yes | yes | many times a day | search cluster + S3 Standard
30 to 180 days | no | partial, incidents | a few times a month | Glacier Instant Retrieval
180 days to 7 years | no | no | audits, rare | Glacier Deep Archive
Over 7 years | no | no | never | delete
~~~

<div class="ex">
<div class="lab">Worked example</div>

**application logs at 2 TB a day**

A company produces 2 TB of raw logs per day, which compress about 10 times to 200 GB. Compliance requires keeping them for 7 years.

Hot: the last 30 days are indexed in a search cluster for incident investigation, and the compressed files sit in S3 Standard: 6 TB × \$23 = about \$140 per month for the files (the search cluster is a separate, larger cost).

After 30 days, a lifecycle rule moves files to Glacier Instant Retrieval: still millisecond reads for the occasional "what happened in March?" investigation. After 180 days, they move to Deep Archive. After 7 years, they expire.

At steady state there are about 511 TB of compressed logs. Left in S3 Standard, that would cost about \$11,800 per month. With the tiered plan, about 30 TB sits in Glacier Instant (\$120), 6 TB in Standard (\$140) and about 475 TB in Deep Archive (\$475): roughly \$735 per month, about 16 times cheaper.

Hourly files are first compacted into daily files of several gigabytes, so there are a few thousand objects per year instead of millions, keeping transition fees and per-object overhead negligible.

The trade-off: investigating a 2-year-old incident means requesting a restore, waiting up to 12 hours (48 for bulk), paying retrieval fees, and querying the restored files with a separate tool rather than the search cluster.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**when cold storage costs more**

A team moves 50 TB of product images to Standard-IA to save money: storage drops from about \$1,150 to about \$625 per month.

But a nightly job regenerates thumbnails by reading every image. That is 50 TB × 30 nights = 1,500 TB of retrieval a month at about \$10 per TB: \$15,000 in retrieval fees. The "saving" costs \$14,000 a month more.

The fix is to change the job to process only new images, or to keep this bucket in Standard (or Intelligent-Tiering, which charges no retrieval fees). Tiering only pays when cold data is genuinely cold.

</div>

## Common mistakes

- **Keeping everything hot forever** because nobody owns a retention policy.
- **Archiving data that is still read regularly**, so retrieval fees exceed the savings.
- **Archiving millions of tiny objects** without batching them first.
- **Ignoring minimum durations** for data that is deleted soon after creation.
- **Never testing a restore**, and discovering during an audit that it takes days and nobody knows how.

## In an interview

When data grows without bound (logs, events, messages, media), say what happens to it as it ages: how long it stays hot, which cheaper tier it moves to, when it is deleted, and what reading old data costs in time and money. A sentence like "30 days hot and searchable, then archive with 12-hour restores, delete at 7 years" shows you have thought about cost, not just architecture.

<div class="take">
<div class="lab">Key takeaway</div>

data cools with age, so move it down to cheaper tiers with lifecycle rules and delete it when retention ends. You save a great deal on storage and pay in retrieval time, retrieval fees, minimum storage durations and per-object overhead, so tier only data that is genuinely cold and batch small objects first.

</div>
