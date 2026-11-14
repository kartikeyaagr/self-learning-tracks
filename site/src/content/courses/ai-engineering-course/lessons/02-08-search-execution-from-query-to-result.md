---
id: "02.08"
course: "ai-engineering-course"
module: "02"
slug: "search-execution-from-query-to-result"
title: "Search execution from query to result"
keyIdeas: ["1. Parse the request: text, filters (tenant, date, category), k.", "2. Embed the query with the same model used at indexing.", "3. Apply filters: pre-filter (restrict candidates first), post-filter (search then drop) or filtered graph traversal.", "4. ANN search across shards or segments; each returns its local top-k and results are merged.", "5. Optional rescoring with full-precision vectors, then reranking the top 50-100 with a cross-encoder.", "6. Fetch the stored text and metadata for the final top few and return them (in RAG, place them in the prompt)."]
question: "Walk through a query in a RAG system from user text to the chunks placed in the prompt. Where would you add a reranker, and why there?"
answer: "The query text is embedded with the same model as the corpus; filters (tenant, date) are applied via pre-filtering or filtered index traversal; ANN runs on each shard and results are merged into a global top 50-100; candidates are optionally rescored with full vectors, then a cross-encoder reranks them; the top few chunks' text is fetched and inserted into the prompt with citations. The reranker goes after ANN because it is accurate but too slow to run on the whole corpus."
sources: [{"title": "Pinecone: Rerankers and two-stage retrieval", "url": "https://www.pinecone.io/learn/series/rag/rerankers/", "note": "Why a cross-encoder after ANN improves results."}, {"title": "Qdrant: Filtering", "url": "https://qdrant.tech/documentation/concepts/filtering/", "note": "Payload filters combined with vector search."}, {"title": "Anthropic: Contextual Retrieval", "url": "https://www.anthropic.com/news/contextual-retrieval", "note": "Chunk context, hybrid search and reranking measured end to end."}]
---

## The full path

~~~dg-flow One search request through a typical vector store.
User query -(embed)-> Query vector -(filter + ANN)-> *Candidates per shard -(merge)-> Top 100 -(rerank)-> Top 5 -(fetch text)-> Prompt context
~~~

## 1. Embed the query

The query is embedded with the same model and version as the documents. Some models expect a prefix such as "query: " for queries and "passage: " for documents; skipping it lowers quality.

## 2. Filters

Real queries include constraints: this user's documents only, from the last year, in English. There are three ways to apply them:

- **Pre-filtering**: find matching ids first, then search only among them. Exact, but slow if many items match and the index can't use the filter.
- **Post-filtering**: run ANN, then drop non-matching results. Fast, but with a selective filter you can end up with fewer than k results.
- **Filtered traversal**: the index checks the filter while walking the graph or clusters. Qdrant and other vector DBs do this, and it handles both cases well.

## 3. Search shards and merge

Large collections are split across shards or segments. Each one returns its local top k; a coordinator merges them into a global top k. This is why per-shard k is often set a bit higher than the final k.

## 4. Rescore and rerank

ANN on compressed vectors is approximate, and embedding similarity is a rough signal of relevance. Two fixes stack:

- **Rescore** candidates with full-precision vectors.
- **Rerank** the top 50-100 with a cross-encoder, a model that reads the query and the passage together and outputs a relevance score. Pinecone's reranker guide explains why this is more accurate: a bi-encoder squeezes each document into one vector before seeing the query; a cross-encoder sees both.

Cross-encoders are too slow to run over the whole corpus, so they go at the end, on a short list.

## 5. Return and use

Fetch the text and metadata of the final hits. In RAG they go into the prompt, usually with source ids so the model can cite them. Anthropic's Contextual Retrieval work measured how much each stage helps: adding context to chunks, combining embeddings with BM25, and reranking each cut retrieval failures, and together they reduced failed retrievals by 67% in their tests.

## Budget the whole path

Every step on the query path takes time, and they add up. Set the cost of each stage below and see whether the total fits your latency budget.

~~~dg-raw
%%widget funnel Latency of one search: embed, search shards, rerank, fetch.
~~~

Three observations from playing with it:

- **Shards run in parallel, so the slowest one sets the time.** One overloaded shard turns a 10 ms search into a 60 ms one.
- **Reranking is often the largest cost.** A cross-encoder scores each (query, document) pair with a full model pass. Reranking 100 candidates at 3 ms each is 300 ms on one GPU unless you batch them.
- **The embedding call is a network hop.** If it goes to a hosted API, 50 to 150 ms is common. Caching embeddings for repeated queries removes it.

## Filters: before, during or after

Most real queries carry filters: tenant id, language, date range, permissions. Where you apply them changes both speed and correctness.

| Strategy | How | Risk |
| --- | --- | --- |
| Post-filter | Search top 100, then drop the ones that fail the filter | If only 2% of data matches, you may end with zero results |
| Pre-filter | Find matching ids first, then search only those | Slow if the filter matches a lot; breaks HNSW graph navigation |
| Filtered search | The index checks the filter while it walks | Needs engine support; best of both when available |

For multi-tenant apps, the safest pattern is a separate collection or partition per large tenant. That makes leaking one tenant's data into another's results structurally impossible, instead of depending on a filter being set on every query.

<div class="ex">
<div class="lab">Worked example</div>

**why a filtered query returned nothing**

A user searches their own 300 documents in a 30 M document index. The engine retrieves the global top 100 and then filters by owner.

The chance that any of the user's documents is in the global top 100 is tiny, so they get zero results.

Fix: filtered search, or a per-user partition, or raise k only for highly selective filters.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

query, embed, filter, ANN per shard, merge, rescore, rerank, fetch. Put the expensive, accurate step (the reranker) last, on the shortest list.

</div>
