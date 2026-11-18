---
id: "11.06"
course: "ai-engineering-course"
module: "11"
slug: "keeping-a-rag-index-correct"
title: "Keeping a RAG index correct"
keyIdeas: ["A RAG index is a derived copy of source systems; stale answers, leaked documents and resurrected deletes are all the copy drifting from the source.", "Drive ingestion from change events, keep a manifest of what is indexed (version, content hash, chunk ids, ACL, embedding model) and reconcile against the source on a schedule to catch missed events, especially deletes.", "Store access principals on every chunk, filter inside the search, and treat a permission change as an event that rewrites metadata without re-embedding.", "A delete must reach every copy: vector and keyword indexes, caches, and eventually compacted segments and backups.", "A new embedding model means a new index: backfill from stored chunk text, dual-write changes, evaluate, then flip an alias with the query encoder in the same release."]
question: "Your company assistant indexes Confluence, Google Drive and Zendesk. A user reports it quoted a document that was deleted last week, and security finds it answered from a Drive file the asker had lost access to. Explain how each failure happened and design the ingestion changes that prevent them."
answer: "Deleted document: the connector probably only processed create and update events, or the delete event was dropped, so the chunks stayed in the index (and possibly in an answer cache). Fix: a manifest of indexed documents with their chunk ids, delete handling that removes every chunk id from the vector and keyword indexes and invalidates caches, and a periodic reconciliation that lists the source and deletes anything indexed but no longer present. Permission leak: ACLs were copied at ingestion and never updated, or filtering ran after retrieval. Fix: store groups (not individual users) on each chunk, treat permission changes as events that update chunk metadata, resolve the user's groups at query time with a short cache, filter inside the search, and optionally re-check the final few chunks against the source. Monitor both: delete-lag and ACL-lag metrics, and a canary document whose deletion and permission change are tested daily."
rubric: ["Identifies missed delete events and caches as the cause of the resurrected document", "Uses a manifest of doc to chunk ids so deletes remove every chunk", "Adds reconciliation against the source to catch missed events", "Stores ACL principals per chunk and updates them on permission-change events", "Filters inside retrieval, not after generation, with groups resolved at query time", "Monitors delete and permission lag with canaries or tests"]
sources: [{"title": "Debezium: Architecture", "url": "https://debezium.io/documentation/reference/stable/architecture.html", "note": "Change data capture from database logs into event streams."}, {"title": "Microsoft: Security filters for trimming results in Azure AI Search", "url": "https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search", "note": "Group ids stored per document and filtered at query time."}, {"title": "pgvector README: filtering and iterative index scans", "url": "https://github.com/pgvector/pgvector", "note": "Why filtered approximate search can return too few rows, and the 0.8 fix."}, {"title": "Qdrant: Collections and collection aliases", "url": "https://qdrant.tech/documentation/concepts/collections/", "note": "Build a new collection in the background and switch an alias atomically."}]
practice: ["rag-context-packing", "text-deduplication-minhash", "perform-stable-exact-deduplication", "rag-pipeline-capstone"]
tools: ["Debezium", "Kafka", "Qdrant", "pgvector", "Azure AI Search", "Elasticsearch"]
tags: ["RAG", "ingestion", "access control", "index maintenance"]
---

## The index is a copy

Everything a RAG system retrieves is a copy of something that lives elsewhere: a wiki page, a ticket, a file in a shared drive. The source keeps changing; the copy changes only when your pipeline makes it. Most production RAG incidents are the copy drifting from the source:

- **Stale**: the policy changed on Monday, the index still holds Friday's text.
- **Resurrected**: a document deleted last week is still quoted.
- **Leaked**: a user lost access to a file, but the chunks still carry the old permissions.
- **Duplicated**: the same handbook lives in three places, and all three copies fill the context.
- **Mixed**: half the chunks were embedded with last year's model.

[Inside a vector database](lesson:aieng:02.09) showed the storage mechanics (deletes as tombstones, compaction, freshness lag). [Search index synchronization](lesson:sdsd:09.10) covers the same problem for keyword search engines. This lesson is about the RAG pipeline above the store, the part you own.

## The ingestion pipeline

~~~dg-flow Incremental ingestion. The manifest records what is indexed, so every later change can be applied exactly.
[Detect] Webhooks / CDC / modified-since poll -(event)-> Queue
[Process] Queue -(fetch, parse)-> Document -(hash, diff vs manifest)-> *Changed chunks only -(embed)-> Vectors
[Apply] Vectors + chunk text + ACL -(upsert)-> Vector + BM25 index -(commit)-> Manifest
[Repair] Nightly listing of source -(diff vs manifest)-> Missed updates and deletes
~~~

**Detect changes three ways.** Push where the source offers it: webhooks from SaaS tools, or change data capture from a database's write-ahead log (Debezium reads Postgres or MySQL logs and publishes row changes to Kafka). Pull where it does not: poll with a modified-since cursor. And always **reconcile**: list the source on a schedule and diff it against what you have indexed. Events get dropped, connectors restart, and many APIs never emit an event for a hard delete. Reconciliation turns "eventually consistent if nothing goes wrong" into "eventually consistent, full stop".

**Keep a manifest**, a small table that is the source of truth for what is in the index:

| Field | Why |
| --- | --- |
| doc_id, source, source version or etag | Idempotent processing: an event for a version already indexed is a no-op |
| content hash per chunk | Re-embed only chunks whose text changed; an edit to one paragraph should cost one embedding, not fifty |
| chunk ids | A delete or re-chunk can remove exactly the old chunks |
| ACL principals | Permission changes can be applied without refetching content |
| embedding model and chunker version | Tells you which chunks a migration still has to touch |
| indexed_at | Freshness lag per document |

Make every step idempotent on (doc_id, version) so retries and replays are harmless, and write the manifest last, after the index upsert succeeds, so a crash leaves work to redo rather than a manifest that claims chunks exist when they do not.

## Freshness as an SLA

Define index lag as

$$
\text{lag} = t_{\text{searchable}} - t_{\text{changed at source}}
$$

and set a target per source, because users' expectations differ: support tickets within minutes, the wiki within an hour, archived reports within a day. Measure it from the outside with **canary documents**: write or edit a synthetic document in each source on a schedule and poll search until it appears. Internal stage timings miss the delay you care about, the one before your pipeline even hears of the change.

## Deduplication

Copies are everywhere: the same PDF attached to ten tickets, a handbook mirrored from Confluence to Drive. Hash normalised text for exact duplicates and use MinHash with LSH for near-duplicates (shingles of a few words, a Jaccard threshold around 0.8 to 0.9). Pick one canonical copy, preferring the authoritative source and the newest version, and keep the others as aliases for citation.

Access control complicates this. Two **exact** copies with different ACLs can share one indexed copy whose ACL is the union, since anyone allowed either copy may read identical text. Two **near**-duplicates cannot: the part that differs may be exactly the sensitive part (a salary table present in one version only). Deduplicate near-duplicates only within the same ACL, and dedupe again at query time when packing the context (the `rag-context-packing` problem does both the ACL filter and the dedupe).

## Access control that follows the source

Rules that hold up in security review:

1. **Store principals on every chunk**, as group ids rather than user ids where the source allows. Group membership then changes in the identity provider without touching the index; you resolve the user's groups at query time (cached for a few minutes).
2. **Filter inside the search**, as a pre-filter on the ANN and keyword queries (Azure AI Search's security trimming pattern is a filter over a group-id field). Never retrieve first and ask the model to ignore what the user should not see; once text is in the prompt it can leak.
3. **Treat a permission change as an event.** It updates the ACL field on all of that document's chunks: a metadata write, no re-embedding.
4. **Mind filtered ANN recall.** With post-filtering an approximate index can return too few rows: pgvector's README notes that with HNSW's default search breadth of 40 and a filter matching 10% of rows, only about 4 rows survive on average, and version 0.8 added iterative scans to keep searching. Highly selective ACLs (a user who can see 0.1% of the corpus) are better served by per-tenant partitions or namespaces.
5. **For the strictest data, re-check at the end**: verify the final 5 to 10 chunks against the source system's permission API before generation. It costs a few calls per query and catches anything the index has not caught up with yet.

## Deletes reach every copy

A source delete must remove the document's chunk ids (from the manifest) from the vector index, the keyword index and every cache that may hold derived text: answer caches, semantic caches, precomputed summaries. In the vector store this marks rows deleted; physical removal waits for compaction, and backups and snapshots keep the data until they expire. For legal erasure requests, know your compaction schedule and backup retention, and write the end-to-end deletion time down rather than assuming "deleted" means gone.

## Re-embedding migrations

Vectors from two embedding models live in different spaces, so you cannot mix them in one index or compare a new-model query against old-model chunks. Switching models, dimensions or chunkers is a migration:

~~~dg-seq Swapping embedding models with dual-write and an alias flip.
Pipeline > Index v2: backfill from stored text
Pipeline > Index v1: live changes
Pipeline > Index v2: same changes (dual-write)
Eval > Index v2: labelled set + shadow queries
Eval -> Pipeline: v2 passes
Pipeline > Index v2: alias "kb" + new query encoder
Pipeline > Index v1: keep for rollback, then drop
~~~

This only works if you **stored the chunk text** (and the parsed documents), not just vectors; otherwise the backfill means refetching and reparsing every source. The query encoder and the index must switch together, so version them as one unit: an alias pointing at v2 while queries are still embedded with the old model returns nonsense with no errors.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a migration of 20 million chunks**

Chunks average 400 tokens: $2 \times 10^7 \times 400 = 8 \times 10^9$ tokens to embed.

*Price*, at an illustrative \$0.02 per million tokens: $8{,}000 \times 0.02 =$ \$160. Cheap.

*Time*, at an illustrative throughput or rate limit of 5 million tokens per minute: $8{,}000 / 5 = 1{,}600$ minutes, about 27 hours. Rate limits, not price, set the schedule, so request a quota increase or run the model yourself.

*Storage*: 1,024-dimensional float32 vectors are 4 KB each, so $2 \times 10^7 \times 4{,}096$ bytes $\approx 82$ GB per index before graph overhead, and about twice that while v1 and v2 coexist.

*Dual-write window*: about 27 hours of backfill plus a few days of evaluation and rollback cover, during which every change is embedded twice.

</div>

## Where this breaks

- **Update-only connectors.** Creates and edits flow; deletes never arrive. Reconciliation is the fix.
- **ACLs copied once.** Correct at ingestion, wrong a month later.
- **Unversioned chunks.** Citations point to text that no longer exists in the source; store the version each chunk came from.
- **Full rebuilds as the update strategy.** Nightly rebuilds give up to a day of staleness and make every model or chunker change a big-bang release.
- **Caches outside the pipeline.** The index is clean, but the answer cache still serves the deleted document.

## What interviewers probe

- How does a permission change reach the index, and how fast? What if a user's group membership changes?
- How do you know a delete was applied everywhere, including caches and compacted segments?
- Walk through an embedding-model migration with zero downtime and a rollback.
- How do you measure freshness, and what is your SLA per source?

<div class="take">
<div class="lab">Key takeaway</div>

Treat the index as a derived copy you must keep in sync: drive it from change events, record every chunk in a manifest, reconcile against the source, carry permissions on each chunk and filter inside the search, push deletes through every cache, and migrate embedding models as a dual-write plus alias flip.

</div>
