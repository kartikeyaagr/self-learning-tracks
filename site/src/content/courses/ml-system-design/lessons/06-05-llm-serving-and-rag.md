---
id: "06.05"
course: "ml-system-design"
module: "06"
slug: "llm-serving-and-rag"
title: "LLM serving and RAG"
keyIdeas: ["LLM serving cost and latency are driven by tokens: prompt length sets prefill time (time to first token) and output length sets decode time. Continuous batching and KV-cache management set throughput.", "**Retrieval-augmented generation (RAG)** grounds answers in your documents: chunk and embed documents, retrieve relevant passages per query, and pass them in the prompt.", "RAG quality depends mostly on retrieval: chunking, hybrid search (keyword plus embeddings), re-ranking and filtering by permissions and freshness.", "Production LLM systems need routing between model sizes, caching, guardrails on inputs and outputs, and evaluation and monitoring designed for generated text."]
question: "A company wants an internal assistant that answers questions over 2 million documents (wikis, tickets, PDFs) with access controls, a p95 time-to-first-token under 2 seconds and a budget of $0.01 per question. Sketch the architecture and the main levers for quality, latency and cost."
answer: "Ingestion: parse documents, split into chunks of a few hundred tokens with titles and section headers attached, embed each chunk, and index both embeddings (ANN) and text (BM25), storing access-control lists and timestamps as metadata; update incrementally from change feeds. Query path: rewrite the question if needed, run hybrid retrieval filtered by the user's permissions (about 50 chunks), re-rank with a cross-encoder to the top 5 to 10, and build a prompt with instructions and cited passages. Generation: route simple questions to a smaller, cheaper model and complex ones to a larger model; stream tokens; cache answers for frequent questions and reuse KV cache for the shared system prompt. Levers: chunking and hybrid retrieval drive quality; prompt length (number of passages) drives prefill latency and cost; model choice and routing drive cost per question. Add guardrails (refuse when retrieval finds nothing relevant, cite sources), evaluate retrieval recall and answer faithfulness offline, and monitor cost per question, latency, feedback and unanswered rates."
sources: [{"title": "Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)", "url": "https://arxiv.org/abs/2005.11401", "note": "The original RAG formulation."}, {"title": "Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)", "url": "https://arxiv.org/abs/2309.06180", "note": "KV-cache paging and continuous batching in vLLM."}, {"title": "Anthropic, Introducing Contextual Retrieval (2024)", "url": "https://www.anthropic.com/news/contextual-retrieval", "note": "Improving chunk retrieval with added context and hybrid search."}]
practice: ["ttft-tpot-itl-and-token-throughput", "kv-cache-memory-for-mha-mqa-gqa-and-mla", "prefix-cache-matching-and-reuse", "document-ranking", "mmr-diversity-rerank"]
tags: ["LLM serving", "retrieval-augmented generation"]
---

## What makes LLM serving different

A classifier produces one output per request in a fixed time. A language model produces tokens one at a time, so latency and cost scale with both the prompt and the answer. Two numbers matter to users: **time to first token** (TTFT), dominated by processing the prompt (prefill), and **time per output token**, dominated by decoding. Throughput depends on batching many sequences and managing the KV cache that stores attention state for each one. The Inference Engineering course covers this machinery in depth (ieng 03, 07, 08, 10); here we look at system design.

## Cost model

For hosted or self-served models, cost is roughly proportional to tokens: input tokens × input price + output tokens × output price. A RAG prompt with 8 passages of 300 tokens adds 2,400 input tokens to every question. Halving the passages, using a smaller model for easy questions, or caching repeated prefixes often cuts cost more than any infrastructure change.

## RAG architecture

~~~dg-flow Retrieval-augmented generation. Most quality problems start in the retrieval half.
Documents -(parse, chunk, embed)-> Index: vectors + keywords + metadata
Question -(rewrite, embed)-> Hybrid retrieval -(permission filter)-> Re-ranker -(top passages)-> *Prompt builder -(LLM)-> Answer with citations
~~~

Key design choices:

- **Chunking**: chunks of a few hundred tokens, split on structure (headings, paragraphs), with the document title and section prepended so each chunk makes sense alone.
- **Hybrid retrieval**: keyword search (BM25) catches exact names, codes and rare terms; embeddings catch paraphrases. Combine both.
- **Re-ranking**: a cross-encoder scores query-passage pairs more accurately than embeddings, applied to the top 50 to 100.
- **Filtering**: enforce access controls at retrieval time, never by asking the model to ignore documents; prefer fresh versions of documents.
- **Prompting**: instruct the model to answer only from the passages, cite them, and say when the answer is not there.

## Routing and caching

Not every question needs the largest model. A router (rules or a small classifier) sends simple or frequent questions to a smaller model and hard ones to a larger model. Caches help at several levels: exact-match answer caching for frequent questions, embedding caches, and prefix caching for shared system prompts so their KV cache is reused across requests.

## Guardrails

- **Input**: detect prompt injection, sensitive data, out-of-scope requests.
- **Output**: check for unsupported claims, policy violations and leaked personal data; require citations.
- **Fallbacks**: if retrieval finds nothing relevant, say so instead of guessing.

<div class="ex">
<div class="lab">Worked example</div>

**retrieval was the bottleneck**

An internal help bot gave wrong answers about expense limits about 30% of the time. Evaluation showed retrieval recall of the correct policy section was only 55%: chunks were split mid-table and lost the heading that said which country they applied to. Re-chunking on section boundaries with headings prepended, plus keyword search for policy codes, raised recall to 88% and halved wrong answers without changing the model.

</div>

## Evaluation and monitoring

Evaluate retrieval (recall of supporting passages) and generation (faithfulness, correctness) separately, as in 05.03. In production, track TTFT and total latency at p95, tokens and cost per request, refusal and "no answer" rates, user feedback, and a weekly human review sample.

## When not to use RAG

If the knowledge is small and stable, putting it directly in the prompt may be simpler. If the task needs a consistent style or format rather than facts, fine-tuning may fit better. If answers must be exact (prices, balances), call the system of record through a tool rather than retrieving text about it.

## Freshness and updates

Documents change, so the index must too. Feed it from change events rather than periodic full rebuilds, delete chunks when their document is deleted or its permissions change, and record the document version each chunk came from so citations point to the right text. Monitor index lag: the time between a document changing and the change being searchable. Stale answers about policies or prices are a common and avoidable RAG failure.

<div class="take">
<div class="lab">Key takeaway</div>

LLM serving is governed by tokens, with prefill setting time to first token and decoding setting speed; RAG quality comes mostly from chunking, hybrid retrieval, re-ranking and permission filtering, and cost comes down through routing, shorter prompts and caching.

</div>
