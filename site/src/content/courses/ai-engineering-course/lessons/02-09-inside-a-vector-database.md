---
id: "02.09"
course: "ai-engineering-course"
module: "02"
slug: "inside-a-vector-database"
title: "Inside a vector database"
keyIdeas: ["Milvus separates storage from compute and splits the system into access, coordination, worker and storage layers, so each scales on its own.", "Writes go to a write-ahead log first, which is the source of truth; in Milvus 2.6 streaming nodes own the log and the newest data, and the WAL can live directly on object storage (Woodpecker).", "New data sits in small growing segments searched by brute force; sealed segments get an ANN index built in the background and are served by query nodes.", "Deletes are tombstones and updates are delete plus insert; background compaction merges small segments and drops deleted rows.", "Qdrant segments and Lucene segments (Elasticsearch, OpenSearch) follow the same log-structured pattern, with the same costs: freshness lag, many-segment slowdowns, space spikes."]
question: "In Milvus 2.6, what happens between an insert and the moment the new vector is searchable through an ANN index? Name the components involved."
answer: "The client sends the insert to a stateless proxy, which routes it to the streaming node that owns that shard's channel. The streaming node appends it to the write-ahead log (Woodpecker on object storage, or Kafka or Pulsar), and from that point the write is durable. The streaming node also keeps the row in a growing segment, so it is already searchable, by brute force, within about a second. When the growing segment reaches its size limit it is sealed and flushed to object storage. The coordinator then schedules a data node to build the ANN index for that sealed segment (HNSW, IVF or DiskANN, for example). Once the index is built, a query node loads the sealed segment and its index, and searches for that vector go through the index instead of the brute-force path. The proxy merges results from streaming and query nodes."
sources: [{"title": "Milvus: Architecture overview", "url": "https://milvus.io/docs/architecture_overview.md", "note": "Access, coordinator, worker and storage layers, streaming nodes and the Woodpecker WAL."}, {"title": "Wang et al.: Milvus, a purpose-built vector data management system (SIGMOD 2021)", "url": "https://dl.acm.org/doi/10.1145/3448016.3457550", "note": "The design paper."}, {"title": "Qdrant: Storage", "url": "https://qdrant.tech/documentation/concepts/storage/", "note": "Segments, appendable and non-appendable, in another vector database."}]
practice: ["embedding-similarity", "ann-recall-at-k"]
tools: ["Milvus", "Qdrant", "Lucene"]
---

## Why look inside

Every dedicated vector database faces the same problems: accept writes continuously, keep indexes fresh, search across many machines, and survive failures. Milvus is open source and its architecture documentation is detailed, so it is a good one to study. What you learn here explains behaviour you will see in any of them: a document you just wrote that doesn't show up, recall that drifts over weeks, storage that doubles during a re-embed.

## Four layers

~~~dg-layers Milvus 2.6 splits responsibilities so each layer scales on its own.
Access layer | stateless proxies: validate requests, route, merge results
Coordinator | one active coordinator: metadata, topology, query routing, background job scheduling
*Worker nodes | streaming nodes (WAL, growing data), query nodes (sealed data), data nodes (compaction, index builds)
Storage | metadata (etcd), object storage (S3, MinIO, Azure Blob), WAL storage (Woodpecker, Kafka or Pulsar)
~~~

The key idea from the Milvus architecture overview: **storage and compute are separated, and the log is the backbone**. Workers are stateless and can be added or replaced without moving data around. Milvus 2.6 reorganised the worker layer. Earlier 2.x releases had four coordinators and separate index nodes; 2.6 has a single coordinator, merges index building into data nodes, and adds **streaming nodes** that own the write-ahead log for each shard and serve the newest data. It also adds **Woodpecker**, a write-ahead log that writes directly to object storage instead of needing a separate Kafka or Pulsar cluster, which removes one stateful system to run. From 2.6.4, tiered storage lets query nodes load segment metadata first and fetch data and indexes on demand, trading cold-query latency for less local memory.

## The write path

1. A client inserts vectors through a proxy, which routes them to the streaming node for that shard.
2. The streaming node appends them to the **write-ahead log**. Once logged, the write is durable.
3. The rows go into a **growing segment**, which has no ANN index and is searched by brute force. That is fine because it is small.
4. When the segment reaches its size limit it is **sealed** and flushed to object storage.
5. A data node builds an ANN index (HNSW, IVF, DiskANN and others) for the sealed segment, and a query node loads segment and index to serve them.

## The read path

~~~dg-seq A search in Milvus 2.6.
Client > Proxy: search(vector, filter, k)
Proxy > Streaming node: search growing segments (brute force)
Proxy > Query node: search sealed segments (indexed)
Streaming node -> Proxy: local top k
Query node -> Proxy: local top k
Proxy -> Client: merged top k
~~~

Because writes flow through a log, a search might not see a write from a moment ago. Milvus lets you choose a consistency level per request: strong, bounded staleness, session, or eventually consistent. Stronger levels make the query wait until the serving nodes have caught up with the log, which adds latency. Bounded staleness is the default; session consistency guarantees you see your own writes.

## Deletes, updates and compaction

Graph and cluster indexes are built for reads; changing them in place is expensive. So vector databases borrow the log-structured design of Cassandra, RocksDB and Lucene: write small new segments, never edit old ones, and merge in the background.

~~~dg-raw
%%widget segments Writes create small segments; background compaction merges them and drops deleted rows.
~~~

1. **Insert**: new vectors go to the log, then into a small in-memory segment, searchable within a second or so.
2. **Seal**: at a size limit, the segment is frozen and indexed on its own.
3. **Delete**: nothing is removed. The id is recorded as deleted (a tombstone or deletion bitmap), and searches skip it.
4. **Compact**: in the background, several small segments plus their deletions are rewritten into one large, clean segment with a fresh index.

A query searches every live segment and merges the results. Many small segments mean many small searches per query, which is why a database with a heavy trickle of writes gets slower until compaction catches up.

## The same design elsewhere

Qdrant divides each collection into segments, each with its own vector storage, payload storage and index. Appendable segments take writes; non-appendable ones are read-and-delete only. An optimizer in the background vacuums segments with many deleted points, merges small ones, and builds the HNSW index once a segment passes an indexing threshold, behind its own write-ahead log. Lucene, and so Elasticsearch and OpenSearch, works the same way: immutable segments, a deleted-documents bitmap per segment, a merge policy, and a separate HNSW graph per segment. That last detail is why kNN in a freshly loaded Elasticsearch index is slower than after a force-merge: a query walks one graph per segment. Learn one of these systems and the others' tuning knobs (segment size, merge thresholds, indexing thresholds) look familiar.

<div class="ex">
<div class="lab">Worked example</div>

**Re-embedding 10 M documents (sizes computed, segment counts illustrative)**

The collection holds 10M vectors at 1,024 float32 dimensions: 10M × 4,096 bytes = 41 GB of vectors, in 20 sealed segments of 500,000 vectors.

You switch embedding models and upsert every document. Each upsert is a delete plus an insert, so until compaction runs, the old 41 GB stays on disk next to 41 GB of new vectors: plan for 82 GB, twice the steady state.

The re-embed streams in over a day. If writes seal small segments of 50,000 vectors each, the 10M new rows create 200 segments, and every query now runs 220 per-segment searches instead of 20. p99 latency climbs until compaction merges them back into about 20.

Run bulk re-embeds into a new collection and switch an alias when it is fully indexed, instead of upserting in place.

</div>

## Where this breaks

- **Freshness lag.** A document written now may not appear in search for a second or more. If users upload a file and immediately ask about it, wait for the write to be visible, use a stronger consistency level for that request, or answer from the file directly.
- **Space spikes.** Updates are delete plus insert, so a full re-embed temporarily doubles storage, as above.
- **Silent recall drift.** Heavily deleted segments waste search effort and degrade graph quality. If recall falls over weeks without a code change, check the deleted-row ratio and force a compaction.
- **Too many small segments.** High write rates with small seal sizes multiply per-query work; watch segment counts alongside latency.

<div class="take">
<div class="lab">Key takeaway</div>

A vector database is a log plus segments plus background index builds, with stateless search workers on top. New data is searchable by brute force first and indexed once its segment is sealed; deletes and updates are cheap to write and paid for later in compaction.

</div>
