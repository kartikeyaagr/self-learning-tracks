---
id: "09.11"
course: "system-design-101"
module: "09"
slug: "search-index-sharding"
title: "Search index sharding"
keyIdeas: ["Large indexes are split into shards, each a full mini-index.", "**Document partitioning** (what Elasticsearch does): each shard holds a subset of documents and all their terms. Queries go to every shard, each returns its top-k, and a coordinator merges.", "**Term partitioning**: each shard holds the postings for a subset of terms. Fewer shards per query, but multi-term queries need cross-shard work and load is uneven.", "Document partitioning is more common: simpler, balanced, and shard failures only lose part of the results.", "Replicas serve reads and provide failover. Elastic suggests shards of roughly 10-50 GB."]
question: "Document-partitioned vs term-partitioned index: which is more common and why?"
answer: "Document partitioning is far more common (Elasticsearch, Solr and web search engines use it). Each shard holds a full index of its own documents, so it can intersect postings and score locally and return only a small top-K list; a new document touches only one shard; load spreads evenly because every shard gets a similar slice of every term; and losing a shard only hides some documents. Term partitioning has to ship long postings lists between machines to answer multi-term queries, concentrates load on machines holding popular terms, and makes each new document update many machines. The price of document partitioning is that every query goes to every shard (so tail latency depends on the slowest one) and idf has to be kept global or approximated from local statistics."
sources: [{"title": "Elastic: Size your shards", "url": "https://www.elastic.co/docs/deploy-manage/production-guidance/optimize-performance/size-shards", "note": "Shard sizing guidance."}, {"title": "IR book ch.20.3: Distributing indexes", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "Document vs term partitioning."}]
---

## Why split an index at all

A single machine can hold a surprisingly large inverted index: hundreds of gigabytes, tens or hundreds of millions of documents. But eventually one of three limits is hit:

- **Size**: the index no longer fits on one machine's disks, or the hot part no longer fits in its memory (page cache), so queries slow down sharply.
- **Query throughput**: one machine can't serve the required queries per second.
- **Indexing throughput and recovery**: rebuilding or copying one giant index takes too long when a machine fails.

The answer is to split the index across machines, called **partitioning** or **sharding**. Each piece is a **shard**. For throughput, each shard is also copied to several machines as **replicas**. There are two fundamentally different ways to split an inverted index, and the choice shapes every query.

## Two ways to split

Recall that an inverted index is a dictionary of terms, each with a postings list of documents (lesson 09.02). You can cut that structure along either axis.

**Term partitioning** (a **global** index): each machine holds the **complete postings lists for a subset of terms**. Machine 1 has every document's postings for terms in some range or hash bucket (say, "a" to "f"), machine 2 the next range, and so on. A term lives on exactly one machine.

**Document partitioning** (a **local** index): each machine holds a **complete index**, with every term, but only **for a subset of the documents**. Shard 1 indexes documents 1 to 10 million, shard 2 the next 10 million, or documents are assigned by hashing their ID.

~~~dg-tree The same index cut two ways. Term partitioning splits the dictionary; document partitioning splits the collection.
Inverted index for 30M docs
  By term (global)
    Node 1 | terms a-f, all 30M docs
    Node 2 | terms g-r, all 30M docs
    Node 3 | terms s-z, all 30M docs
  *By document (local)
    Shard 1 | all terms, docs 1-10M
    Shard 2 | all terms, docs 10-20M
    Shard 3 | all terms, docs 20-30M
~~~

### How a query runs in each

- **Term-partitioned**: the query "cat sat" goes only to the machine owning "cat" and the machine owning "sat". But to intersect or score, the two postings lists must meet in one place: one machine ships its whole "cat" list (possibly millions of entries) to the other, or both ship to a coordinator.
- **Document-partitioned**: the query goes to **every** shard (**scatter**). Each shard intersects and scores locally, using only its own documents, and returns its local top K. A coordinator merges them into a global top K (**gather**). Only small result lists cross the network.

~~~dg-flow The two query paths: by term, postings lists travel to meet; by document, every shard works locally and only top K lists travel.
[By term] Query "cat sat" -(cat)-> Node 1 -(ship full cat list)-> *Node 3 intersects with sat
[By document] Query "cat sat" -(scatter to all)-> Shards 1, 2, 3 -(local top K)-> *Coordinator merges
~~~

## Comparing them

| Property | Term-partitioned | Document-partitioned |
| --- | --- | --- |
| Machines per query | Only those holding the query's terms | All shards |
| Multi-term queries | Long postings lists shipped between machines | Each shard intersects locally |
| Network per query | Large (postings lists) | Small (top K IDs and scores) |
| Load balance | Hot terms make hot machines | Even, if documents spread evenly |
| Adding a document | Touches many machines (one per term) | Touches one shard |
| Losing a machine | Whole terms become unsearchable | Some documents missing from results |
| Scoring statistics | Global df for free | Each shard sees local df only |

## Why document partitioning wins

Document partitioning is much more common: Elasticsearch, OpenSearch, Solr and web search engines all use it. The reasons follow from the table:

- **Each shard answers queries on its own.** All the expensive work (intersection, scoring, top-K selection) happens locally, close to the data. Only tiny result lists travel.
- **Writes touch one place.** A new document with 500 distinct terms goes to one shard. Under term partitioning, it would update 500 postings lists spread across most of the cluster.
- **Load spreads evenly.** With documents assigned by hash, each shard gets a similar slice of every term. Under term partitioning, the machine holding "iphone" during a launch week, or "the" always, becomes a hotspot.
- **Failures degrade gracefully.** Losing a document-partitioned shard (with no replica) loses some documents from results; users may not notice. Losing a term partition makes every query containing those terms fail.
- **Operations are simpler.** Shards are self-contained units that can be moved, replicated, snapshotted and recovered independently.

Term partitioning's apparent advantage, fewer machines per query, is undone by the cost of moving long postings lists across the network and by hot spots. It has niche uses (some systems with very high query rates and short queries, or term-sharded auxiliary structures), but it's rarely the default.

### The costs of document partitioning

- **Every query hits every shard.** With 100 shards, each query costs 100 shard-level searches. Query latency is set by the **slowest** shard (tail latency, lesson 02.12), so one slow node slows every query. Mitigations: replicas and adaptive replica selection (send to the fastest copy), hedged requests, and routing (below).
- **Statistics are local.** Each shard computes idf (lesson 09.04) from its own documents. A term rare in one shard but common overall gets over-weighted there. When documents are spread randomly and shards are large, local statistics are close to global and it doesn't matter. When it does matter (small shards, skewed data), engines can gather global statistics first: Elasticsearch's `dfs_query_then_fetch` adds a round trip to collect term frequencies from all shards before scoring. Large systems also compute collection statistics in background jobs and distribute them to shards periodically.
- **Deep pagination is expensive.** Asking for results 1,000 to 1,010 makes every shard return its top 1,010 (lesson 09.01).

### Routing: not every query needs every shard

If queries are always scoped to one tenant or user (a mailbox search, a SaaS customer's documents), route documents by that key: all of tenant 42's documents go to the shard `hash(42) % shards`. Then a tenant's query hits **one** shard instead of all. Elasticsearch supports this with a `routing` value. The risk is uneven shards when one tenant is huge; give large tenants their own index or spread them over a few shards.

## Sizing shards in Elasticsearch

Practical rules for the document-partitioned design in Elasticsearch:

- Aim for shards between **10 GB and 50 GB**, and fewer than about **200 million documents** per shard. Very large shards are slow to recover and rebalance (moving a 200 GB shard to a new node takes a long time); very small shards waste overhead.
- **Each shard is traditionally searched by one thread per search request** (recent versions can also search a shard's segments in parallel), so spreading data over more shards adds parallelism, but each shard also adds fixed overhead: memory for segment metadata, file handles, and coordination work on every query. Thousands of tiny shards make a cluster slow and unstable.
- Keep the shard count per node bounded. Elasticsearch's default limit is **1,000 shards per non-frozen data node**, and real clusters usually stay well below it.
- Shard count is **fixed at index creation**. To change it, use the split or shrink APIs (which multiply or divide by a factor) or reindex into a new index behind an alias (lesson 09.10). So size with growth in mind.
- **Replicas** add read throughput and fault tolerance, not write capacity: every write goes to the primary and all its replicas.

~~~dg-bars Rough cost and benefit of shard sizes. Mid-sized shards balance per-shard overhead against slow recovery.
1 GB shards | 300 | 300 shards for 300 GB: heavy overhead
*30 GB shards | 10 | 10 shards: balanced
300 GB shard | 1 | 1 shard: hours to recover
~~~

### Time-based data

For logs, metrics and events, data arrives in time order, queries mostly target recent data, and old data is deleted. Don't use one giant index. Use **rollover** with index lifecycle management: write to the newest index (`logs-000042`), and when its primary shard reaches 50 GB or the index is a day old, create `logs-000043` and switch writes there. Old indices can move to cheaper hardware (hot, warm, cold tiers) and are deleted **whole** when they expire. Deleting individual documents is expensive in Lucene, because deleted documents are only cleaned up when segments merge (lesson 09.02); dropping a whole index is instant.

~~~dg-flow Time-based indices with rollover: write to the newest index, and drop whole old indices instead of deleting documents.
Writes -> *logs-000042 -(rollover at 50 GB primary shard)-> logs-000043 -(after 30 days)-> Delete old index
~~~

<div class="ex">
<div class="lab">Worked example</div>

**sizing a product search index**

300 million products at about 2 KB each of indexed data is 600 GB. At about 30 GB per shard, that's 20 primary shards; with one replica each, 40 shards. Spread across 8 data nodes, that's 5 shards per node, about 75 GB of index per node, much of which can sit in the page cache on 128 GB machines.

A query fans out to 20 shards (one copy of each) in parallel. Each takes about 15 ms to return its local top 10. The coordinator merges 200 hits into the final 10 and fetches their documents. Total about 30 ms at the median; the p99 depends on the slowest of 20 shard calls, so replica selection and hedging matter. If query load doubles, add replicas (and nodes), not primaries. If the catalogue grows to 900 million products, shards would grow to about 90 GB, so the team plans a reindex into 60 primaries before then.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why term partitioning hurts here**

The same catalogue term-partitioned over 8 nodes. The query "black running shoes" touches 3 terms on up to 3 nodes. "black" matches 30 million products: its postings list, even compressed at about 1 byte per posting, is about 30 MB. To intersect it with "shoes" (20 million) on another node, one of those lists must cross the network: tens of megabytes per query. At 2,000 queries a second, that's tens of gigabytes a second of internal traffic. Document partitioning ships only 20 small top-10 lists per query.

</div>

## Common mistakes

- Thousands of tiny shards ("one index per customer per day"), which overwhelm the cluster's metadata and memory.
- One huge shard that takes hours to recover after a node failure.
- Ignoring routing when every query is tenant-scoped, so each query pays the cost of every shard.
- Adding primaries to increase query throughput, when replicas are what scale reads.
- Forgetting that local idf can skew scores on small or skewed shards.

## In an interview

Say "document-partitioned, like Elasticsearch: each shard is a full index of a subset of documents, queries scatter to all shards and gather local top K." Justify it with local intersection, single-shard writes, even load and graceful failure. Then mention the costs (every query hits every shard, tail latency, local idf) and the fixes (replicas, routing by tenant, global statistics). Size shards around 10 to 50 GB with arithmetic.

<div class="take">
<div class="lab">Key takeaway</div>

indexes can be split by term or by document. Document partitioning is far more common because each shard intersects and scores locally, writes touch one shard, load is even and failures degrade gracefully; its costs are a scatter-gather to every shard, tail latency from the slowest shard, and the need for global statistics. In Elasticsearch, keep shards around 10 to 50 GB and use replicas for read throughput.

</div>
