---
id: "02.09"
course: "ai-engineering-course"
module: "02"
slug: "inside-a-vector-database"
title: "Inside a vector database"
keyIdeas: ["Milvus separates storage and compute, and splits the system into access, coordination, worker and storage layers.", "Proxies (access layer) receive requests; coordinators manage metadata, data placement and index building.", "Workers: query nodes serve searches, data nodes persist inserts, index nodes build indexes.", "Writes go to a log (message queue) first, which is the source of truth; data nodes turn it into segments in object storage.", "Growing segments are searched with brute force; sealed segments get an ANN index built in the background.", "Because compute is stateless, query nodes scale out independently of storage."]
question: "In Milvus, what happens between an insert and the moment the new vector is searchable with an index? Name the components involved."
answer: "The proxy receives the insert and writes it to the log broker (durable at that point). Query nodes consume the log so the data is searchable in a growing segment via brute force. Data nodes batch log entries into segments and flush to object storage; once a segment is sealed, the index coordinator schedules an index node to build the ANN index, after which query nodes load the indexed sealed segment."
sources: [{"title": "Milvus: Architecture overview", "url": "https://milvus.io/docs/architecture_overview.md", "note": "Access, coordinator, worker and storage layers."}, {"title": "Wang et al.: Milvus, a purpose-built vector data management system (SIGMOD 2021)", "url": "https://dl.acm.org/doi/10.1145/3448016.3457550", "note": "The design paper."}, {"title": "Milvus on GitHub", "url": "https://github.com/milvus-io/milvus", "note": "Source and docs for the open-source vector DB."}]
tools: ["Milvus"]
---

## Why look inside

Every dedicated vector database faces the same problems: accept writes continuously, keep indexes fresh, search across many machines, and survive failures. Milvus is open source and its architecture docs are detailed, so it is a good one to study.

## Four layers

~~~dg-layers Milvus 2.x splits responsibilities so each layer scales on its own.
Access layer | stateless proxies: validate requests, route, merge results
Coordinators | metadata, data placement, load balancing, index jobs
*Worker nodes | query nodes (search), data nodes (persist), index nodes (build)
Storage | metadata store (etcd), log broker (Pulsar or Kafka), object storage (S3/MinIO)
~~~

The key idea, from the Milvus architecture overview: **storage and compute are separated**, and the log is the backbone. Workers are stateless and can be added or replaced without moving data around.

## The write path

1. A client inserts vectors through a proxy.
2. The proxy writes them to the **log broker**. Once logged, the write is durable.
3. Query nodes subscribe to the log so new data becomes searchable quickly.
4. Data nodes batch log entries into **segments** and flush them to object storage.
5. When a segment reaches its size limit it is **sealed**, and an index node builds an ANN index (HNSW, IVF, DiskANN...) for it.

## The read path

~~~dg-seq A search in Milvus.
Client > Proxy: search(vector, filter, k)
Proxy > Query node: search assigned segments
Query node > Query node: indexed search on sealed, brute force on growing
Query node -> Proxy: local top k
Proxy -> Client: merged top k
~~~

Newly inserted data sits in **growing segments**, which have no index yet and are searched by brute force. That's fine because they are small. Sealed segments use their indexes. The proxy merges the partial results.

## Consistency choices

Because writes flow through a log, a search might not see a write from a moment ago. Milvus lets you choose a consistency level per request (strong, bounded staleness, session, eventually). Stronger means the query waits until the query nodes have caught up with the log.

## What this teaches about any vector DB

- Fresh writes and ANN indexes don't mix well, so systems keep a small unindexed buffer and index in the background.
- Segment or shard results are always merged at the end.
- Separating stateless compute from durable storage is what makes scaling and recovery simple.

## Why deletes and updates are hard

Graph and cluster indexes are built for reads. Changing them in place is expensive, so most vector databases borrow the log-structured design used by Cassandra, RocksDB and Lucene: write to small new segments, and merge them in the background.

~~~dg-raw
%%widget segments Writes create small segments; background compaction merges them and drops deleted rows.
~~~

Walk through it in the figure:

1. **Insert**: new vectors go to a write-ahead log for durability, then into a small in-memory segment. They are searchable within a second or so.
2. **Seal**: when the segment reaches a size limit, it is frozen and indexed (HNSW or IVF built on just that segment).
3. **Delete**: nothing is removed. The id is added to a deletion bitmap, and searches skip it.
4. **Compact**: in the background, several small segments plus their deletion bitmaps are rewritten into one large, clean segment.

A query searches every live segment and merges the results, which is why a database with too many small segments gets slow, and why compaction schedules matter.

## Consequences for your app

- **Updates are delete plus insert.** Re-embedding a document leaves the old vector in place until compaction. Space usage can temporarily double during a full re-embed.
- **Freshness has a lag.** A document written now may not appear in search for a second or more. If users upload a file and immediately ask about it, either wait for the write to be visible or answer from the file directly.
- **Count and recall can drift.** Heavily deleted segments waste search effort. If recall falls over weeks without a code change, check the deleted-row ratio and force a compaction.

<div class="take">
<div class="lab">Key takeaway</div>

a vector database is a log plus segments plus background index builds, with stateless search workers on top. New data is searchable by brute force first and indexed once its segment is sealed.

</div>
