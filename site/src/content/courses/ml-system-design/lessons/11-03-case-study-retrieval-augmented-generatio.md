---
id: "11.03"
course: "ml-system-design"
module: "11"
slug: "case-study-retrieval-augmented-generation"
title: "Case study: retrieval-augmented generation"
keyIdeas: ["A RAG system answers questions by retrieving relevant passages and having a language model answer from them, so answer quality is limited by retrieval quality.", "Chunking, hybrid retrieval (keyword plus dense embeddings) and a cross-encoder re-ranker usually matter more than the choice of generator.", "Answers must be grounded in the retrieved text, with citations and a way to say \"I don't know\"; access control must apply at retrieval time.", "Evaluate retrieval and generation separately, with a labelled question set, automated judges checked against humans, and online feedback."]
question: "Design a question-answering assistant for a company's internal documents: 5 million documents (wikis, tickets, PDFs, policies) with per-document access permissions. Employees ask about 100,000 questions a day. Answers must cite sources and respect permissions. Cover ingestion, retrieval, generation, evaluation and monitoring."
answer: "Ingest documents through parsers per format, split into chunks of a few hundred tokens with headings kept, and store each chunk with metadata (source, date, permissions). Index chunks in both a keyword index (BM25) and a vector index of dense embeddings. At query time, optionally rewrite the question, retrieve about 50 chunks from each index filtered by the user's permissions, merge them, and re-rank with a cross-encoder to the top 5 to 10. The generator receives the question and these chunks with instructions to answer only from them, cite chunk IDs and say when the answer is not found. Evaluate retrieval with recall at k on a labelled question set, and answers with groundedness and correctness, using an LLM judge calibrated against human ratings. Online, track thumbs up and down, citation clicks and unanswered rates. Monitor index freshness and permission errors. Trade-offs: chunk size, retrieval depth versus latency and cost, and answer coverage versus hallucination risk."
rubric: ["Describes ingestion with parsing, chunking and metadata including permissions", "Uses hybrid retrieval (keyword and dense) and explains why", "Adds a re-ranking stage before generation", "Enforces access control at retrieval time, not after generation", "Instructs the generator to ground answers, cite sources and abstain when unsupported", "Evaluates retrieval and generation separately with a labelled set and calibrated judges", "Keeps the index fresh with incremental updates and handles deletions", "Monitors feedback, unanswered questions and cost and latency per query"]
sources: [{"title": "Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)", "url": "https://arxiv.org/abs/2005.11401", "note": "The original RAG formulation."}, {"title": "Karpukhin et al., Dense Passage Retrieval for Open-Domain Question Answering (2020)", "url": "https://arxiv.org/abs/2004.04906", "note": "Dual-encoder retrieval for question answering."}, {"title": "Liu et al., Lost in the Middle: How Language Models Use Long Contexts (2023)", "url": "https://arxiv.org/abs/2307.03172", "note": "Why passage order and context length matter."}, {"title": "Gao et al., Retrieval-Augmented Generation for Large Language Models: A Survey (2023)", "url": "https://arxiv.org/abs/2312.10997", "note": "A survey of RAG components and evaluation."}]
practice: ["sentence-embeddings", "document-ranking", "hit-rate-and-mrr", "ndcg-at-k"]
tags: ["RAG", "hybrid retrieval", "re-ranking", "grounding", "evaluation"]
---

## Requirements

Clarify the corpus, users and risks. Assume 5 million documents of mixed formats, updated continuously, with permissions per document; 100,000 questions a day; answers within 5 seconds; and answers that must cite sources. Leaking a document a user may not see is a serious failure, and so is a confident wrong answer about a policy.

## Metrics

Retrieval: recall at k (did the top k chunks contain the needed information?) and MRR on a labelled set. Generation: correctness, groundedness (every claim supported by the cited chunks) and abstention quality (says "not found" when it is). Online: thumbs up and down, citation clicks, follow-up questions that rephrase the same need, and escalations to humans. Cost and latency per query.

## Ingestion

- Parse each format (HTML, PDF, slides, tickets) into clean text, keeping headings, tables and lists.
- Split into chunks of 200 to 500 tokens, along section boundaries, with a little overlap; prepend the document title and section heading to each chunk.
- Store metadata: source URL, last updated, author, and the access-control list.
- Embed chunks with a text embedding model and index them in a vector store; index the same chunks in a keyword index.
- Update incrementally from change feeds; delete chunks promptly when documents are deleted or permissions change.

## Retrieval

Keyword search finds exact terms (product codes, error messages, names) that embeddings blur; dense retrieval finds paraphrases. Use both, filter by the user's permissions inside the search, and merge (reciprocal rank fusion is simple and robust). A cross-encoder re-ranker scores each question-chunk pair jointly and is much more accurate than either retriever, at higher cost, so apply it to the top 50 to 100 only.

<div class="ex">
<div class="lab">Worked example</div>

**retrieval was the problem**

An internal assistant answered 61% of test questions correctly. Analysis showed that for most wrong answers, the needed passage was not in the top 5 chunks. Adding keyword retrieval for error codes and part numbers, and a cross-encoder re-ranker, raised recall at 5 from 64% to 86%, and answer correctness to 79%, without changing the generator.

</div>

## Generation

Give the generator the question and the top chunks, each with an ID. Instruct it to answer only from these, cite IDs after each claim, and say clearly when the context does not contain the answer. Put the most relevant chunks first and keep the context short; long contexts can bury relevant passages. Post-check that every cited ID exists and was in the context.

## Serving

~~~dg-flow RAG request. Permissions are applied inside retrieval, before any text reaches the generator.
Documents -(parse, chunk, embed)-> Keyword index + vector index
Question -(rewrite, permission filter)-> Keyword index + vector index -(top 50 each, fuse)-> *Cross-encoder re-ranker -(top 5-10 chunks)-> Generator -(answer with citations)-> User
~~~

## Evaluation

Build a labelled set of a few hundred real questions with the documents that answer them and reference answers. Measure retrieval and generation separately so that you know which part to fix. An LLM judge can score groundedness and correctness at scale, but calibrate it against human ratings on a sample and re-check after changes.

## Monitoring

Track feedback rates, unanswered rates by topic (they reveal missing documentation), latency and cost per query, index lag behind source systems, and any permission-filter errors (which should be zero, and tested).

## Likely follow-up questions

- "Why not put all documents into a long context?" Five million documents do not fit, cost scales with context, and accuracy falls when relevant text is buried.
- "How do you handle questions that need several documents?" Decompose into sub-questions, retrieve for each, then answer; or let the model issue further searches.
- "How do you stop prompt injection inside documents?" Treat retrieved text as data, separate it clearly in the prompt, and restrict what actions the assistant can take.

## Trade-offs

- Chunk size: small chunks retrieve precisely but lose context; large chunks do the reverse.
- Retrieval depth and re-ranking versus latency and cost.
- Abstaining more often reduces wrong answers but frustrates users.

<div class="take">
<div class="lab">Key takeaway</div>

in RAG, retrieval sets the ceiling: chunk well, combine keyword and dense retrieval under permission filters, re-rank with a cross-encoder, make the generator cite and abstain, and evaluate retrieval and generation separately.

</div>
