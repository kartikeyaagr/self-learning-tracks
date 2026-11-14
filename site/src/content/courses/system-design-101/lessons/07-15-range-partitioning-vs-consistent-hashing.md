---
id: "07.15"
course: "system-design-101"
module: "07"
slug: "range-partitioning-vs-consistent-hashing-for-storage"
title: "Range partitioning vs consistent hashing for storage"
keyIdeas: ["**Range partitioning** assigns contiguous key ranges to partitions. Keys stay sorted, so listing by prefix (`photos/2024/...`) is a scan of one or a few partitions. Ranges split when they grow hot or big.", "**Consistent hashing** spreads keys randomly and evenly, but neighbouring keys end up on different nodes, so prefix listing must ask every node.", "Range partitioning needs a partition map and a manager to split and move ranges; hashing needs only the hash function and membership.", "Object stores that support efficient listing (Azure Storage, Bigtable-style systems) use range partitioning."]
question: "Why does range partitioning support listing objects by prefix and hashing doesn't?"
answer: "With range partitioning, keys are stored in sorted order and each server owns a contiguous key range, so all keys sharing a prefix are adjacent and live in one range or a few neighbouring ones; a listing locates the first range from the partition map and reads keys in order from one or two servers, stopping after a page. Hashing deliberately scatters neighbouring keys to unrelated servers, so the system cannot know where keys with a given prefix live: it must send the listing to every node, have each find its matches, and merge-sort the results to return the first page in order, which costs requests on every node and is as slow as the slowest node."
sources: [{"title": "Windows Azure Storage paper (SOSP 2011)", "url": "https://www.microsoft.com/en-us/research/publication/windows-azure-storage-a-highly-available-cloud-storage-service-with-strong-consistency/", "note": "Range-partitioned storage at scale. Lessons 07.15 to 07.19 follow it closely."}, {"title": "Hello Interview: Consistent Hashing", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/consistent-hashing", "note": "The other option."}]
---

## The question: which server owns this key?

Once data outgrows one machine, you split it into **partitions** (also called shards) and spread them across servers. Every request then starts with the same question: given this key, which server owns it? The answer has to be quick to compute, has to spread load evenly, and has to cope when servers are added, removed or fail.

There are two main families of answers:

- **Range partitioning**: divide the key space into contiguous **sorted ranges**, and give each range to one server. Keys `a` to `f` on server 1, `f` to `m` on server 2, and so on.
- **Hash partitioning**, usually via **consistent hashing**: hash each key to a number and assign numbers to servers. Neighbouring keys land on unrelated servers.

The choice decides what queries are cheap. This lesson's focus is the one that matters most for object storage: **listing objects by prefix**, such as "show me everything under `invoices/2024/`".

~~~dg-compare The same five keys under each scheme. Ranges keep a prefix together; hashing scatters it across servers.
[Range partitioning]
invoices/01, /02, /03, /04, /05 -(sorted, one range)-> Server A
+ One server answers a prefix listing
+ Results already in order
- New keys at the end all hit one range
[Consistent hashing]
invoices/01 -(hash)-> Server C, /02 -> A, /03 -> D
+ Load spread evenly by the hash
+ Few keys move when servers change
- A prefix listing must ask every server
~~~

## Range partitioning

In range partitioning, keys are kept in **sorted order** and the sorted sequence is cut into ranges. A small **partition map** records each range's boundaries and its owner. To route a request, look up which range contains the key: a binary search over a few thousand boundaries, fast enough to do on every request from a cached copy of the map.

~~~dg-tree A range-partitioned object table. Each range is a contiguous slice of the sorted keys; listing a prefix touches only the ranges that overlap it.
Object table, sorted by account, container, object
  Range 1: "" to acme/b | Server 4
  Range 2: acme/b to acme/invoices/2024/06 | Server 9
  *Range 3: acme/invoices/2024/06 to acme/m | Server 2
  Range 4: acme/m to globex/ | Server 7
  Range 5: globex/ to end | Server 4
~~~

What range partitioning makes easy:

- **Prefix listing and range scans.** All keys starting with `invoices/2024/` sit next to each other in sorted order, so they live in one range or a few adjacent ones. A listing reads them in order, from one or two servers, and can stop after the first 1,000.
- **Locality for a tenant.** If the key starts with the account name, one customer's data is stored together in a small set of ranges. That makes it easy to measure, throttle and isolate a noisy customer.
- **Flexible rebalancing.** A range that grows too big or too busy can be **split** at any key into two, and cold neighbours can be **merged**. The partition map changes; the keys themselves don't. Lesson 07.16 looks at a partition manager doing this.

What it makes hard:

- **Sequential keys create hot spots.** If keys increase steadily (timestamps, auto-increment IDs, `log-2024-06-01-00001`), every new write lands in the **last range**, on one server, while the others sit idle. Splitting doesn't help much: the new "last range" is still the only one getting writes.
- **Skew needs active management.** Ranges are uneven by nature, so something must watch load and split or move ranges continuously.

Range partitioning is used by Bigtable and HBase (tablets and regions), Spanner and CockroachDB (splits and ranges), MongoDB's ranged shard keys, and the metadata layers of object stores including S3 and Windows Azure Storage.

## Hash partitioning and consistent hashing

### The naive version: hash mod N

The simplest hash scheme is `server = hash(key) % N`. It spreads keys evenly. Its fatal flaw appears when N changes. Going from 4 servers to 5 changes the result of `% N` for about **80%** of keys (in general, a fraction N/(N+1) of them), so nearly all data must move.

### The ring

**Consistent hashing** fixes that. Imagine the hash space (say 0 to 2^32 − 1) bent into a circle. Each server is hashed to one or more points on the circle. Each key is hashed to a point too, and belongs to the **first server clockwise** from it.

- **Adding a server** only takes over the keys on the arc between it and its predecessor; nothing else moves. Going from 4 to 5 servers moves about 1/5, 20% of keys, all of them to the new server.
- **Removing a server** moves only its keys, to the next server clockwise.

~~~dg-bars Share of keys that must move when a fifth server joins four, by scheme.
hash(key) mod N | 80 | about 80%
*Consistent hashing ring | 20 | about 20%, the minimum
~~~

### Virtual nodes

With one point per server, arcs are uneven, and a failed server's entire load lands on the single next server, doubling its traffic. The fix is **virtual nodes**: each physical server is placed at many points (for example 100 to 256 tokens). Then each server owns many small arcs scattered around the ring. When one fails, its load spreads over all the others; when one joins, it takes small slices from many. It also lets a bigger machine take more virtual nodes.

### Hash slots

A common variant fixes the number of partitions up front. **Redis Cluster** uses 16,384 **hash slots**: `slot = CRC16(key) mod 16384`, and each node owns a set of slots. Adding a node means moving some slots, not rehashing keys. Cassandra, DynamoDB, Riak and many caches and CDNs use hashing in one of these forms.

### The cost: order is gone

A good hash function deliberately destroys any relationship between neighbouring keys. `invoices/2024/0001` and `invoices/2024/0002` are adjacent in sorted order but land on unrelated servers. So a query like "list everything under `invoices/2024/`" has no idea where to look and must **scatter** to every server and **gather** the results, then merge-sort them to return the first page in order.

~~~dg-seq A prefix listing on a hash-partitioned store must ask every node and merge the results, even if most have nothing to return.
Client > Router : LIST invoices/2024/
Router > Node 1 : scan prefix
Router > Node 2 : scan prefix
Router > Node 3 : scan prefix
Node 1 -> Router : 340 keys
Node 2 -> Router : 410 keys
Node 3 -> Router : 395 keys
Router > Router : merge, keep 1,000
Router -> Client : page + token
~~~

With 3 nodes that's tolerable. With 1,000 nodes, every listing costs 1,000 requests, the slowest node sets the latency, and each node must scan its own data for the prefix (which, without a sorted local index, means a full scan).

## Range partitioning in Windows Azure Storage

Windows Azure Storage (WAS) is a good example of choosing ranges deliberately. Blobs, tables and queues are all stored in huge **Object Tables** whose primary key is `(AccountName, PartitionName, ObjectName)`. Each table is split into **RangePartitions**: contiguous ranges of rows from a low key to a high key, each served by exactly one partition server.

The reasons given for ranges over hashing:

- **Performance isolation**: a customer's objects are stored together in a set of RangePartitions, so the system can see and throttle an abusive account.
- **Efficient enumeration**: listing a container is a scan of adjacent rows.

The stated downside is exactly the sequential-key problem. If customers use steadily increasing keys, all traffic for them hits the last range. Customers who need high write rates are expected to spread their keys themselves, for example by prefixing a hash or a bucket number to the PartitionName.

## Getting both: hash then range

Many systems combine the two, and this is often the best answer. Hash a **partition key** to choose the server, and keep data **sorted within** the partition by a second key. DynamoDB's partition key plus sort key and Cassandra's partition key plus clustering columns both work this way. "All orders for customer 42, sorted by date" is one partition, read in order from one server. What you can't do is a range scan **across** partition keys.

~~~dg-matrix How the schemes compare on the properties that usually decide the choice.
Property | Range partitioning | Consistent hashing | Hash, then sort within
Prefix listing across keys | yes | no, ask every node | no
Range scan within an entity | yes | no | yes
Even load for sequential keys | no, last range hot | yes | yes
Data moved when nodes change | partial, split or move ranges | yes, about 1/N | yes, about 1/N
Needs active load management | yes | partial | partial
~~~

<div class="ex">
<div class="lab">Worked example</div>

**listing a folder**

A bucket has 50 million objects and a user lists `invoices/2024/`, which matches 120,000 keys; the API returns 1,000 per page. With range partitioning, the 120,000 matching keys sit in one or two adjacent ranges. The first page is a single sorted scan of 1,000 rows on one server, a few milliseconds; the next page resumes from the last key returned.

With hashing across 200 nodes, the matching keys are spread about 600 per node. To return the first 1,000 in order, the router must ask all 200 nodes for their first 1,000 matches (any node might hold the smallest keys), merge 200 sorted streams, and discard most results. Every page repeats that work. Latency is set by the slowest of 200 nodes, and the listing generates 200 times as many requests.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**a hot range from timestamped keys**

A logging service writes 30,000 objects a second with keys like `logs/2024-06-01T10:15:03.123-...`. Under range partitioning, every new key sorts after all existing ones, so all 30,000 writes a second hit the last range on one server, which can handle about 3,000. Splitting the hot range creates a new last range, which immediately becomes the hot one again.

The fix is to break the ordering on purpose: prefix each key with a short hash, `logs/7f/2024-06-01T...`, giving 256 prefixes. Writes now spread across 256 ranges, about 120 a second each. The price: listing "all logs from 10:15" now needs 256 prefix scans. That is the range-versus-hash trade-off made explicit.

</div>

## In an interview

When you choose a partitioning scheme, say what query it serves. "Object metadata is range-partitioned by bucket and key so LIST by prefix is a sequential scan of one or a few ranges; we split hot ranges automatically, and we advise clients not to use purely sequential key prefixes." Or for a key-value cache: "consistent hashing with virtual nodes: no range queries needed, even load, and only about 1/N of keys move when we add a node." Mentioning the hybrid (hash partition key, sorted within) often impresses.

<div class="take">
<div class="lab">Key takeaway</div>

range partitioning keeps keys sorted, so prefix listings and range scans read one or a few adjacent ranges in order and a tenant's data stays together, but sequential keys pile onto the last range and ranges must be split and moved actively. Hashing spreads load evenly and moves only about 1/N of keys when servers change, but destroys key order, so a prefix listing must ask every node and merge.

</div>
