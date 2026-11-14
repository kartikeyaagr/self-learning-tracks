---
id: "03.05"
course: "ai-engineering-course"
module: "03"
slug: "retrieval-augmented-generation"
title: "Retrieval-augmented generation"
keyIdeas: ["RAG = retrieve relevant documents for a query, then generate an answer grounded in them.", "Indexing: load documents, split into chunks, embed, store with metadata.", "Query time: embed the question, retrieve top chunks (often hybrid + rerank), put them in the prompt with instructions to cite and to say when the answer isn't there.", "Chunking matters: too small loses context, too large dilutes the embedding. Adding a short document-level context to each chunk helps.", "Evaluate the two halves separately: retrieval (recall@k, did the right chunk come back?) and generation (faithfulness to the chunks, answer quality).", "Long contexts don't remove the need for RAG: cost, latency and 'lost in the middle' effects all favour sending only what's relevant."]
question: "Your RAG bot gives wrong answers. List the steps you'd check, in order, to find whether retrieval or generation is at fault."
answer: "1) Check the answer exists in the source corpus. 2) Inspect how the relevant document was chunked (is the needed fact in one coherent chunk with enough context?). 3) Check retrieval: is that chunk in the top-k for this query (recall@k, try hybrid/rerank)? 4) Check the chunk actually reached the prompt (no truncation). 5) Only then check generation: did the model stay faithful to the retrieved text, or ignore/contradict it? Steps 1-4 failing means retrieval; only 5 is generation."
sources: [{"title": "Lewis et al.: Retrieval-Augmented Generation (2020)", "url": "https://arxiv.org/abs/2005.11401", "note": "The paper that named RAG."}, {"title": "Anthropic: Contextual Retrieval", "url": "https://www.anthropic.com/news/contextual-retrieval", "note": "Chunk context, hybrid search and reranking, with measured gains."}, {"title": "Liu et al.: Lost in the Middle (2023)", "url": "https://arxiv.org/abs/2307.03172", "note": "Models use information at the start and end of long contexts best."}]
---

## The idea

Lewis et al. (2020) introduced retrieval-augmented generation: before generating, fetch documents relevant to the question and condition the answer on them. It fixes two core LLM limits from lesson 03.01: knowledge cut-off and no access to private data. It also makes answers checkable, because you can show the sources.

## Two pipelines

~~~dg-flow Offline indexing and online answering.
[Indexing] Documents -(split)-> Chunks -(embed)-> *Vector store + BM25 index
[Query] Question -(retrieve, rerank)-> Top chunks -(prompt)-> LLM -(cite)-> Answer
~~~

**Indexing decisions**

- **Chunking**: a few hundred tokens with some overlap is a common start. Split on structure (headings, paragraphs) rather than fixed character counts where you can.
- **Context per chunk**: a chunk saying "revenue grew 3%" means nothing alone. Anthropic's Contextual Retrieval prepends a short LLM-written summary of where the chunk sits in its document before embedding. With BM25 and reranking added, they cut retrieval failures by 67%.
- **Metadata**: source, date, access permissions, for filtering and citations.

**Query-time decisions**

- Retrieve with hybrid search, rerank, and keep the top few chunks.
- Prompt the model to answer only from the provided sources, cite them, and say when the answer isn't present.
- Order matters: Liu et al.'s "Lost in the Middle" found models use information at the start and end of a long context better than in the middle. Put the best chunks first.

## Debugging RAG

Most RAG failures are retrieval failures. Check in order:

1. **Is the answer in the corpus at all?** If not, no prompt will fix it.
2. **Was it chunked sensibly?** Look at the chunk that should answer the question.
3. **Was it retrieved?** Check recall@k for that query. If it's at rank 30 and you pass 5, you lose.
4. **Was it in the prompt?** Truncation and token limits silently drop context.
5. **Did the model use it faithfully?** Only now is it a generation problem: tighten instructions, use a stronger model, or check for conflicting chunks.

## RAG vs long context

Models with 200k-1M token windows can sometimes take the whole corpus. That costs more per query, is slower (prefill grows with context), and still suffers from middle-of-context loss. RAG remains the default for large or changing corpora; long context helps when the relevant set is small but scattered.

## Chunking decides what can be found

Before retrieval can find the answer, the answer has to sit inside one chunk. Change the chunk size and overlap below and see which chunk the question retrieves, and whether the answer survives intact.

~~~dg-raw
%%widget chunker How chunk size and overlap change what a retriever can find.
~~~

Three regimes:

- **Too small** (a sentence or two): precise matches, but the chunk lacks context. "It was raised to 30 days" is useless without knowing what "it" is.
- **Too large** (several pages): the chunk is about many things, so its vector is a blurry average and matches nothing strongly. It also eats the context window.
- **Middle** (200 to 500 tokens, split on headings and paragraphs, 10 to 20% overlap): the common starting point. Tune it on your own questions.

## Techniques that move the needle

- **Hybrid retrieval**: BM25 plus vectors with RRF (02.06). Catches exact terms like product codes that embeddings miss.
- **Reranking**: retrieve 50 candidates cheaply, re-score with a cross-encoder, keep the best 5. Often the biggest single quality gain.
- **Contextual chunks**: prepend a one-line summary of the document and section to each chunk before embedding ("From the 2025 refund policy, section 3: ..."). Anthropic reported this cut retrieval failures by about a third, and by about two thirds when combined with reranking, on their tests.
- **Metadata filters**: product, version, date. Many "wrong answers" are right answers for last year's version.
- **Cite and check**: ask the model to quote the passage it used, and verify the quote appears in the retrieved text.

<div class="ex">
<div class="lab">Worked example</div>

**debugging one bad answer**

Question: "How long do customers have to return opened items?" The bot answers "14 days". The policy says 30.

Step 1, retrieval: the top 5 chunks include the 2023 policy (14 days) and not the 2025 one. Retrieval failed, not generation.

Step 2, cause: both versions are indexed with no date field. Fix: add a version filter, or remove superseded documents.

Always check retrieval first. If the right chunk wasn't in the context, no prompt change can fix the answer.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

RAG is retrieve then generate. Get chunking and retrieval right first, measure them separately from generation, and put the best evidence where the model will use it.

</div>
