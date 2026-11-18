---
id: "02.08"
course: "ai-engineering-course"
module: "02"
slug: "search-execution-from-query-to-result"
title: "Search execution from query to result"
keyIdeas: ["A query runs: embed with the same model as the corpus, apply filters, ANN search per shard, merge, rescore, rerank, fetch text.", "Filters can be applied before, during or after the ANN search; post-filtering a selective filter returns too few results, and filtered search needs engine support.", "Shards search in parallel, so the slowest shard sets the latency; each returns a little more than k so the merge is accurate.", "Rescoring fixes compression error; reranking with a cross-encoder fixes the embedding's rough notion of relevance, so it runs last on a short list.", "For multi-tenant data, a partition or collection per large tenant makes cross-tenant leaks structurally impossible."]
question: "Walk through a query in a RAG system from user text to the chunks placed in the prompt. Where would you add a reranker, why there, and what would you check if a user filtering to their own documents gets no results?"
answer: "The query text is embedded with the same model, version and query prefix as the corpus. Filters such as tenant, date or permissions are applied by pre-filtering or by filtered index traversal; ANN then runs on each shard, each returns a slightly larger local top k, and a coordinator merges them into a global top 50 to 100. Candidates are optionally rescored with full-precision vectors, then a cross-encoder reranks them, and the text of the top few chunks is fetched and placed in the prompt with source ids for citations. The reranker goes after ANN because it reads query and passage together, which is accurate but far too slow to run over the whole corpus. If a user filtering to their own documents gets nothing, check whether the engine post-filters: their few hundred documents are unlikely to be in the global top 100, so use filtered search, a per-user partition, or exact search over their documents."
rubric: ["Orders the stages: embed, filter, ANN per shard, merge, rescore or rerank, fetch, prompt.", "Places the reranker after ANN on a short candidate list.", "Justifies it: a cross-encoder is more accurate but too slow for the whole corpus.", "Diagnoses post-filtering as the cause of empty filtered results.", "Proposes filtered search, partitioning or exact search over the user's subset."]
sources: [{"title": "Qdrant: Filtering", "url": "https://qdrant.tech/documentation/concepts/filtering/", "note": "Payload filters combined with vector search."}, {"title": "Pinecone: Rerankers and two-stage retrieval", "url": "https://www.pinecone.io/learn/series/rag/rerankers/", "note": "Why a cross-encoder after ANN improves results."}, {"title": "Anthropic: Contextual Retrieval", "url": "https://www.anthropic.com/news/contextual-retrieval", "note": "Chunk context, hybrid search and reranking measured end to end."}]
practice: ["rag-context-packing", "document-ranking", "mmr-diversity-rerank", "rag-pipeline-capstone"]
---

## The full path

~~~dg-flow One search request through a typical vector store.
User query -(embed)-> Query vector -(filter + ANN)-> *Candidates per shard -(merge)-> Top 100 -(rerank)-> Top 5 -(fetch text)-> Prompt context
~~~

Each stage below has a way to fail quietly. In RAG, the last stage feeds the prompt, so a retrieval mistake becomes a confident wrong answer ([Retrieval-augmented generation](lesson:aieng:03.05)).

## 1. Embed the query

The query is embedded with the same model, version and dimension as the documents, and with the model's query prefix or input type if it has one ([Vector embeddings and semantic space](lesson:aieng:02.02)). If the embedding model is a hosted API, this is a network hop of 50 to 150 ms; caching embeddings for repeated queries removes it.

## 2. Apply filters

Real queries carry constraints: this user's documents only, from the last year, in English. Where you apply them changes both speed and correctness.

| Strategy | How | Risk |
| --- | --- | --- |
| Post-filter | Search the top 100, then drop the ones that fail the filter | If only 2% of data matches, you may end with zero results |
| Pre-filter | Find matching ids first, then search only those | Slow if the filter matches a lot; breaks graph navigation if forced through HNSW |
| Filtered search | The index checks the filter while it walks | Needs engine support; best of both when available |

Qdrant and other dedicated stores implement filtered search, and pgvector 0.8 approximates it with iterative index scans. Strict filters still break graph indexes, so good engines switch to exact search over the matching set when it is small ([Indexing techniques](lesson:aieng:02.07)).

For multi-tenant apps, the safest pattern is a separate partition or collection per large tenant. That makes leaking one tenant's data into another's results structurally impossible, instead of depending on a filter being set on every query.

<div class="ex">
<div class="lab">Worked example</div>

**Why a filtered query returned nothing**

A user searches their own 300 documents in a 30M-document index. The engine retrieves the global top 100 and then filters by owner.

The user owns 300 / 30,000,000 = 0.001% of the index. Unless their documents happen to be the closest in the whole collection, none of them is in the global top 100, so they get zero results. Even raising k to 10,000 would admit, if their documents are no closer than average, 10,000 × 0.00001 = 0.1 of them.

Fix: filtered search, a per-user partition, or simply exact search over the user's 300 vectors, which takes well under a millisecond.

</div>

## 3. Search shards and merge

Large collections are split across shards or segments. Each one returns its local top k; a coordinator merges them into a global top k. Per-shard k is usually set a bit higher than the final k so the merge doesn't lose results that were close to a shard's cut-off. Shards run in parallel, so the slowest one sets the time: one overloaded shard turns a 10 ms search into a 60 ms one.

## 4. Rescore and rerank

ANN on compressed vectors is approximate, and embedding similarity is a rough signal of relevance. Two fixes stack:

- **Rescore** candidates with full-precision vectors, which repairs compression error ([Compression and quantization](lesson:aieng:02.05)).
- **Rerank** the top 50 to 100 with a model that reads the query and each passage together and outputs a relevance score. A bi-encoder has squeezed each document into one vector before seeing the query; a cross-encoder sees both, so it is more accurate but needs a full model pass per pair, which is why it only runs on a short list. Choosing a reranker, LLM rerankers and how deep to rerank are covered in [Reranking](lesson:aieng:11.03).

## 5. Return and use

Fetch the text and metadata of the final hits. In RAG they go into the prompt, usually with source ids so the model can cite them. Anthropic's Contextual Retrieval work measured how much the stages add: prepending a short context to each chunk before embedding, combining embeddings with BM25, and reranking each cut retrieval failures, and together they reduced failed retrievals by 67% in their tests. How chunks are built is in [Chunking and document parsing](lesson:aieng:11.04).

## Budget the whole path

Every stage takes time, and they add up. Set the cost of each stage below and see whether the total fits your latency budget.

~~~dg-raw
%%widget funnel Latency of one search: embed, search shards, rerank, fetch.
~~~

Reranking is often the largest cost. A cross-encoder that takes 3 ms per pair spends 300 ms on 100 candidates unless the pairs are batched on a GPU. With a hosted embedding call (100 ms), parallel shard search (15 ms), batched reranking (60 ms) and a text fetch (5 ms), retrieval alone uses 180 ms of a typical 1 to 2 second budget before the model starts generating.

## Where this breaks

- **Mismatched query embedding.** A new model version or a missing query prefix on the query side lowers recall without any error.
- **Selective filters with post-filtering**, as in the worked example.
- **One slow shard** dominates p99 latency; watch per-shard latency, not just the total.
- **Reranking too few candidates.** A reranker can only reorder what ANN returned; if the right chunk is at rank 300, reranking the top 50 cannot find it.
- **Lost metadata.** Chunks fetched without their source ids or permissions cannot be cited or checked, and may be shown to the wrong user.

<div class="take">
<div class="lab">Key takeaway</div>

Query, embed, filter, ANN per shard, merge, rescore, rerank, fetch. Put the expensive, accurate step (the reranker) last on the shortest list, and decide where filters run before they return empty results in production.

</div>
