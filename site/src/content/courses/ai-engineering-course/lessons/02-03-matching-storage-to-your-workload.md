---
id: "02.03"
course: "ai-engineering-course"
module: "02"
slug: "matching-storage-to-your-workload"
title: "Matching storage to your workload"
keyIdeas: ["Four options: an in-process library (Faiss, hnswlib), a vector extension on your database (pgvector), a search engine with vector fields (Elasticsearch, OpenSearch, Vespa), or a dedicated vector database (Milvus, Qdrant, Weaviate, Pinecone).", "Size first: vectors × dimensions × bytes per number, plus index overhead and headroom. 10M × 768 float32 is about 31 GB before the index.", "pgvector filters after the index scan; since 0.8 iterative index scans keep scanning until enough rows pass the filter, and halfvec, bit and sparsevec cut size or add sparse search.", "Multi-tenancy is a choice between a shared index with a tenant filter, a partition per tenant, and a collection per tenant; isolation and recall improve as you move right, at the cost of more objects to manage.", "Managed services bill storage, reads and writes separately; serverless suits spiky or small workloads, provisioned or self-hosted capacity suits steady high QPS."]
question: "You have 2 million product descriptions in Postgres with price and category filters, and some categories hold under 1% of products. Would you add pgvector or a dedicated vector database? Justify with workload numbers and say how you would handle the selective filters."
answer: "Two million 768-dimension float32 vectors are 2M × 768 × 4 bytes, about 6 GB, plus a few hundred MB of HNSW graph: that fits on one Postgres server. The data and both filters already live in Postgres, so pgvector keeps joins with stock and pricing, transactions and one system to back up and operate. A dedicated vector database is justified at hundreds of millions of vectors, thousands of queries per second, or hard multi-tenant isolation, none of which applies. The selective filters are the real risk: pgvector applies WHERE clauses after the index scan, so with the default ef_search of 40 a filter matching 1% of rows usually returns nothing. Enable iterative index scans (pgvector 0.8 or later, hnsw.iterative_scan) so the scan continues until enough rows pass, and for the few large categories consider a partial index or list partitioning. Measure recall on filtered queries before launch."
rubric: ["Computes the raw size (about 6 GB) and concludes it fits one Postgres node.", "Uses the fact that data and filters already live in Postgres (joins, transactions, one system).", "Names the scale or features that would justify a dedicated vector DB.", "Explains post-filtering returning too few rows for a selective filter.", "Proposes iterative index scans, partial indexes or partitioning, and measuring filtered recall."]
sources: [{"title": "pgvector on GitHub", "url": "https://github.com/pgvector/pgvector", "note": "Index types, filtering, iterative index scans, halfvec, bit and sparsevec."}, {"title": "OpenSearch: Vector search", "url": "https://docs.opensearch.org/latest/vector-search/", "note": "Vector fields and k-NN inside a search engine alongside BM25."}, {"title": "Pinecone: What is a vector database", "url": "https://www.pinecone.io/learn/vector-database/", "note": "What a dedicated vector DB adds over a library or extension."}, {"title": "Faiss on GitHub", "url": "https://github.com/facebookresearch/faiss", "note": "The in-process library many vector databases build on."}]
practice: ["ann-recall-at-k", "rag-context-packing", "int8-symmetric-quantize"]
tools: ["pgvector", "Faiss", "Elasticsearch", "OpenSearch", "Vespa", "Milvus", "Qdrant", "Weaviate", "Pinecone"]
---

## Four ways to store vectors

~~~dg-layers From least to most dedicated infrastructure.
Library | Faiss, hnswlib: in-process, you handle persistence, updates and sharding
*Database extension | pgvector: vectors next to your relational data, in one transaction
Search engine | Elasticsearch, OpenSearch, Vespa: vector fields beside BM25 and filters
Dedicated vector DB | Milvus, Qdrant, Weaviate, Pinecone: built for vectors, filtering, tenancy, scale
~~~

A library such as Faiss is very fast, but it is only an index: you manage persistence, sharding and updates yourself, which suits batch jobs and prototypes. A database adds what Pinecone's explainer lists: create, update and delete on vectors, metadata filtering, horizontal scaling, backups, access control and real-time updates. The middle two options give you most of that inside a system you may already run.

## Questions that decide it

- **How many vectors, how many dimensions?** That sets memory, and memory sets machine count.
- **QPS and latency?** Tens of QPS at 50 ms is easy for anything. Thousands of QPS at p99 under 20 ms needs a tuned, sharded system.
- **Update rate?** Nightly bulk rebuilds suit a library. Continuous inserts and deletes need a database that handles them ([Inside a vector database](lesson:aieng:02.09)).
- **Filters?** "Nearest vectors where category = shoes and price < 50" is where stores differ most.
- **Tenants?** Many customers whose data must never mix changes the design.
- **Where does the rest of the data live?** If it is already in Postgres or an Elasticsearch cluster, keeping vectors there avoids a second system.

## Size it before you choose

Set your vector count, dimensions and format below.

~~~dg-raw
%%widget vecmem RAM needed for raw vectors plus index, and how many machines it takes.
~~~

The arithmetic is worth doing by hand once:

- Raw size = vectors × dimensions × bytes per number: float32 is 4 bytes, float16 2, int8 1, binary 1/8.
- An HNSW graph adds roughly M × 2 neighbour ids of 4 bytes each per vector at the bottom layer. With M = 16 that is about 128 bytes per vector, small next to a 1,536-dimension float32 vector (6,144 bytes).
- Leave headroom. Index builds, updates and the operating system need memory too; plan to fill no more than 60 to 70% of RAM.

<div class="ex">
<div class="lab">Worked example</div>

**2 million vs 50 million product embeddings**

2M × 768 dims × 4 bytes = 6.1 GB raw, plus 2M × 128 bytes = 0.26 GB of HNSW graph. At 70% fill that needs about 9 GB of RAM: one Postgres server with pgvector.

50M × 1,024 dims × 4 bytes = 205 GB raw, plus 6.4 GB of graph, so about 300 GB of RAM with headroom. That is three or four large machines. Stored as halfvec it is 102 GB raw; as int8 in a store that supports it, 51 GB, with float32 copies on disk for rescoring. One large server again.

</div>

## pgvector in practice

pgvector adds vector column types, distance operators and two index types, HNSW and IVFFlat. You join vector results with normal tables and filter with SQL. Four details decide whether it works for you.

**Types and limits.** `vector` stores float32 and can be indexed up to 2,000 dimensions. `halfvec` stores float16, half the size, and can be indexed up to 4,000 dimensions, so it is the usual fix for 3,072-dimension models. `bit` stores binary vectors, indexed up to 64,000 dimensions with Hamming or Jaccard distance. `sparsevec` stores sparse vectors (up to 1,000 non-zero entries in an index), which lets you keep learned sparse embeddings next to dense ones.

**Filtering is applied after the index scan.** An HNSW scan returns about `hnsw.ef_search` candidates (40 by default), then the WHERE clause runs. The pgvector docs give the arithmetic: a condition matching 10% of rows returns only about 4 rows on average. Since version 0.8.0, **iterative index scans** fix this by scanning more of the index until enough rows pass the filter, bounded by `hnsw.max_scan_tuples` (20,000 by default) or `ivfflat.max_probes`:

```sql
SET hnsw.iterative_scan = relaxed_order;  -- or strict_order
SELECT id FROM items
WHERE category_id = 123
ORDER BY embedding <=> $1
LIMIT 10;
```

`strict_order` keeps results exactly sorted by distance; `relaxed_order` allows slight reordering for better recall. For a few hot filter values, a partial index (`... WHERE category_id = 123`) is faster still; for many values, partition the table.

**Binary quantization with rescoring** is built in: index `binary_quantize(embedding)::bit(n)`, fetch the top candidates by Hamming distance, then re-order them by the original vectors in an outer query ([Compression and quantization](lesson:aieng:02.05)).

**Build cost.** HNSW builds on tens of millions of rows take hours and need `maintenance_work_mem` large enough to hold the graph; builds that spill to disk are much slower.

## Search engines with vectors

If you already run Elasticsearch or OpenSearch, both store vectors as a field type with HNSW indexes (built on Lucene, plus Faiss in OpenSearch) and support quantized vectors. The draw is one engine for BM25, filters, aggregations and vectors, so hybrid search is a single query ([Hybrid search and score fusion](lesson:aieng:11.02)). Vespa goes further, with filtering integrated into its HNSW search and ranking expressions that combine lexical, vector and business signals in one ranking phase. The cost is the operational weight of a search cluster, and Lucene's per-segment indexes, which make search slower until segments merge.

## Multi-tenancy patterns

| Pattern | How | Good for | Risk |
| --- | --- | --- | --- |
| Shared index + tenant filter | One index, `tenant_id` filter on every query | Many small tenants | One missing filter leaks data; big tenants hurt recall for small ones |
| Partition per tenant | pgvector list partitions, Milvus partition keys, Qdrant tenant-indexed payloads | Hundreds to thousands of tenants of uneven size | Some tuning per large partition |
| Collection or namespace per tenant | Separate tables, collections or Pinecone namespaces | Few large tenants, strict isolation | Many objects to manage and migrate |

pgvector's own guidance is that a shared approximate index lets one tenant's vectors affect another's recall and speed, and recommends partitions or separate tables for isolation. Whatever the pattern, enforce the tenant scope in the data layer, not in prompt-building code.

## Managed cost model

Managed vector services bill three things separately: **storage** (GB-month of vectors and metadata), **reads** (per query, often scaled by how much data the query touches) and **writes** (per upserted vector or byte). Serverless plans charge only for use and suit small, spiky or many-tenant workloads. At steady high QPS, provisioned capacity or self-hosting usually costs less, because per-query pricing grows linearly while a machine's price does not. Model your month as storage + queries × price per query + writes × price per write, at today's volume and at ten times it, before you commit.

## Where this breaks

- **Selective filters return too few rows**, on any store that post-filters. Measure recall on filtered queries, not only on unfiltered ones.
- **Dimension limits bite after a model upgrade.** Moving from a 1,536- to a 3,072-dimension model exceeds pgvector's 2,000-dimension index limit for `vector`; plan for halfvec or the `dimensions` parameter.
- **Benchmarks on someone else's data mislead.** Pick on measured recall, latency and cost at your size.

A rule of thumb: under about 5 to 10 million vectors and already on Postgres, start with pgvector. Tens to hundreds of millions, high QPS, heavy filtering or many tenants: a dedicated vector database or a search engine you already run. Billions of mostly cold vectors: disk-based indexes such as DiskANN, or object storage with a small hot index in memory.

<div class="take">
<div class="lab">Key takeaway</div>

Size the workload first: vectors × dimensions × bytes, QPS, updates, filters and tenants. Use the database you already have until those numbers say otherwise, and if it is Postgres, turn on iterative index scans before filtered queries reach users.

</div>
