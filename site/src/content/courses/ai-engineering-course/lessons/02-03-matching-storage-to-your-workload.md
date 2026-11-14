---
id: "02.03"
course: "ai-engineering-course"
module: "02"
slug: "matching-storage-to-your-workload"
title: "Matching storage to your workload"
keyIdeas: ["Three options: an in-process library (Faiss, hnswlib), a vector extension on your existing DB (pgvector), or a dedicated vector database (Milvus, Qdrant, Weaviate, Pinecone).", "Decide with workload facts: number of vectors, dimensions, QPS, update rate, filter complexity, latency target, and whether data already lives in a relational DB.", "Memory estimate: vectors x dims x 4 bytes. 10M x 768 float32 is about 30 GB before index overhead.", "pgvector is enough for up to a few million vectors with moderate QPS, and keeps transactions and joins with your other data.", "A dedicated vector DB pays off with hundreds of millions of vectors, high QPS, heavy metadata filtering, multi-tenancy or tiered storage.", "A library has no persistence or updates out of the box; fine for batch jobs and prototypes."]
question: "You have 2 million product descriptions in Postgres with price and category filters. Would you add pgvector or a dedicated vector DB? Justify with workload numbers."
answer: "2M x 768 floats is about 6 GB, which fits on one Postgres node with an HNSW index. Data and filters (price, category) already live in Postgres, so pgvector keeps joins, transactions and one system to operate. A dedicated vector DB is only justified at hundreds of millions of vectors, thousands of QPS, heavy filtered search or multi-tenant isolation."
sources: [{"title": "pgvector on GitHub", "url": "https://github.com/pgvector/pgvector", "note": "Vector search inside Postgres: HNSW and IVFFlat indexes."}, {"title": "Pinecone: What is a vector database", "url": "https://www.pinecone.io/learn/vector-database/", "note": "What a dedicated vector DB adds over a library or extension."}, {"title": "Faiss on GitHub", "url": "https://github.com/facebookresearch/faiss", "note": "The in-process library many vector databases build on."}]
tools: ["pgvector"]
---

## Three ways to store vectors

~~~dg-layers From least to most infrastructure.
Library | Faiss, hnswlib, Annoy: in-process, you handle persistence and updates
*Extension | pgvector, Elasticsearch / OpenSearch kNN: vectors next to existing data
Dedicated vector DB | Milvus, Qdrant, Weaviate, Pinecone: distributed, filtering, tenancy
~~~

Pinecone's explainer lists what a database adds over a raw index: CRUD on vectors, metadata filtering, horizontal scaling, backups, access control and real-time updates. Faiss, by contrast, is a library: very fast, but you manage persistence, sharding and updates yourself.

## Questions that decide it

- **How many vectors, how many dimensions?** Raw size = N x d x 4 bytes for float32. 1M x 768 is about 3 GB. 100M x 768 is about 300 GB, which will not fit one machine's RAM without compression.
- **QPS and latency?** Tens of QPS at 50 ms is easy for anything. Thousands of QPS at p99 under 20 ms needs a tuned, sharded system.
- **Update rate?** Bulk nightly rebuilds suit a library. Continuous inserts and deletes need a DB that handles them.
- **Filters?** "Nearest vectors where category = shoes and price < 50" is hard; how well the store combines filters with the ANN index matters a lot.
- **Where does the rest of the data live?** If it's already in Postgres, keeping vectors there avoids a second system and keeps everything in one transaction.

## pgvector in practice

pgvector adds a vector column type, distance operators and two index types: IVFFlat and HNSW. You can join vector results with normal tables and filter with SQL. Limits show up at larger scale: index builds on tens of millions of rows take a long time and a lot of memory, and filtered search can return too few rows when the filter is very selective, because the index finds neighbours first and filters after.

<div class="ex">
<div class="lab">Worked example</div>

**2 million product descriptions**

2M x 768 dims x 4 bytes is about 6 GB raw, plus HNSW overhead: fits comfortably on one Postgres server.

Filters (price, category) are ordinary columns; queries join with stock and pricing tables.

Choose pgvector. Revisit a dedicated DB only if the catalogue grows past tens of millions or QPS reaches thousands.

</div>

## Size it before you choose

Most storage decisions come down to one number: how many bytes your vectors take and whether that fits in memory. Set your vector count, dimensions and format below.

~~~dg-raw
%%widget vecmem RAM needed for raw vectors plus index, and how many machines it takes.
~~~

The arithmetic is simple and worth doing by hand once:

- Raw size = vectors × dimensions × bytes per number. float32 is 4 bytes, float16 is 2, int8 is 1, binary is 1/8.
- An HNSW index adds the graph: roughly M × 2 neighbor ids of 4 bytes each per vector at the bottom layer. With M = 16 that is about 128 bytes per vector, small next to a 1,536-dim float32 vector (6,144 bytes).
- Leave headroom. Index builds, updates and the operating system need memory too; plan to fill no more than 60 to 70% of RAM.

<div class="ex">
<div class="lab">Worked example</div>

**50 million product embeddings**

50 M × 1,024 dims × 4 bytes = 205 GB of raw float32 vectors. With HNSW overhead and headroom you need about 300 GB of RAM.

That is three or four large machines, or one if you store int8 vectors (51 GB raw) and keep float32 copies on disk for rescoring.

The same data at 10 M vectors fits in Postgres with pgvector on one server, which is often the right call.

</div>

## A decision rule

- Under about 5 to 10 million vectors, and you already run Postgres: start with pgvector. One database, transactions, joins with your other tables, one backup story.
- Tens to hundreds of millions, high query rates, or heavy filtering: a dedicated vector database (Qdrant, Milvus, Weaviate, Pinecone and others) with sharding and quantization built in.
- Billions, mostly cold data: disk-based indexes such as DiskANN, or object storage with a small hot index in memory.

Pick on measured recall, latency and cost at your size, not on benchmarks run on someone else's data.

<div class="take">
<div class="lab">Key takeaway</div>

size the workload first (vectors x dimensions x 4 bytes, QPS, updates, filters). Use the database you already have until those numbers say otherwise.

</div>
