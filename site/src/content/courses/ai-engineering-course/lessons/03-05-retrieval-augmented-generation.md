---
id: "03.05"
course: "ai-engineering-course"
module: "03"
slug: "retrieval-augmented-generation"
title: "Retrieval-augmented generation"
keyIdeas: ["RAG = retrieve relevant documents for a query, then generate an answer grounded in them and cite them.", "Two pipelines: offline indexing (parse, chunk, embed, store with metadata) and online answering (retrieve, rerank, prompt, cite).", "Most wrong answers are retrieval failures; debug in order from corpus to chunk to ranking to prompt, and only then blame generation.", "Measure retrieval and generation separately, because they fail for different reasons and have different fixes.", "Long context does not remove the need for RAG: cost, latency and weaker use of the middle of the prompt all favour sending only what is relevant."]
question: "Your RAG bot gives wrong answers. List the steps you'd check, in order, to find whether retrieval or generation is at fault."
answer: "1) Check the answer exists in the indexed corpus, in the current version. 2) Inspect the chunk that should contain it: is the fact in one coherent chunk with enough context, and was the document parsed correctly? 3) Check retrieval: is that chunk in the top k for this query (recall@k)? If not, try hybrid search, reranking, metadata filters or query rewriting. 4) Check the chunk actually reached the prompt, with no truncation or packing loss. 5) Only then check generation: did the model stay faithful to the retrieved text, or ignore or contradict it? Failures in steps 1 to 4 are retrieval-side; only step 5 is a generation problem, fixed with clearer instructions, better ordering or a stronger model."
rubric: ["Checks the corpus contains the answer first", "Inspects parsing and chunking of the relevant document", "Checks retrieval rank or recall@k for the query", "Checks the chunk survived into the final prompt", "Treats generation faithfulness as the last step", "Names at least one fix per stage"]
sources: [{"title": "Lewis et al.: Retrieval-Augmented Generation (2020)", "url": "https://arxiv.org/abs/2005.11401", "note": "The paper that named RAG."}, {"title": "Anthropic: Contextual Retrieval", "url": "https://www.anthropic.com/news/contextual-retrieval", "note": "Chunk context, hybrid search and reranking, with measured gains."}, {"title": "Liu et al.: Lost in the Middle (2023)", "url": "https://arxiv.org/abs/2307.03172", "note": "Models use information at the start and end of long contexts best."}]
practice: ["rag-context-packing", "rag-pipeline-capstone", "chunk-text-with-overlap", "document-ranking", "hit-rate-and-mrr"]
---

## The idea

Lewis et al. (2020) introduced retrieval-augmented generation: before generating, fetch documents relevant to the question and condition the answer on them. It fixes two core LLM limits from [What a large language model does](lesson:aieng:03.01): the knowledge cut-off and no access to private data. It also makes answers checkable, because you can show the sources.

## Two pipelines

~~~dg-flow Offline indexing and online answering.
[Indexing] Documents -(parse, split)-> Chunks -(embed)-> *Vector store + BM25 index
[Query] Question -(retrieve, rerank)-> Top chunks -(prompt)-> LLM -(cite)-> Answer
~~~

**Indexing.** Parse each document (PDFs and tables are where this goes wrong), split it into chunks, embed them, and store each with metadata: source, date or version, and access permissions for filtering and citations. Keeping the index correct as documents change, are deleted or change permissions is its own job ([Keeping a RAG index correct](lesson:aieng:11.06)).

**Answering.** Retrieve candidates, rerank, keep the top few, and prompt the model to answer only from them:

```text
Answer the question using only the sources below. Cite the source id after
each claim, like [S2]. If the sources do not contain the answer, say
"I can't find that in the documentation" and stop.

<sources>
[S1] (refund-policy-2025.md, section 3) Opened items can be returned within 30 days...
[S2] ...
</sources>

Question: {question}
```

Order matters: Liu et al.'s "Lost in the Middle" found models use information at the start and end of a long context better than in the middle, so put the strongest chunks first.

## Chunking decides what can be found

The answer has to sit inside one retrievable chunk. Too small and a chunk loses its context ("It was raised to 30 days"); too large and its vector is a blurry average that matches nothing strongly. A few hundred tokens split on headings and paragraphs, with some overlap, is the usual start; strategies, parsing and how to measure them are in [Chunking and document parsing](lesson:aieng:11.04).

~~~dg-raw
%%widget chunker How chunk size and overlap change what a retriever can find.
~~~

## Retrieval that finds the right chunk

Pure vector search misses exact terms such as product codes and error ids, so most production systems combine it with BM25 ([Hybrid search and score fusion](lesson:aieng:11.02)) and re-score the top 50 or so candidates with a cross-encoder ([Reranking](lesson:aieng:11.03)). Anthropic's Contextual Retrieval adds a short LLM-written note to each chunk saying where it sits in its document before embedding; on their tests that cut retrieval failures by 35%, by 49% with contextual BM25 added, and by 67% with reranking on top. When the user's question is vague, multi-part or needs several hops, rewrite or decompose it before retrieval, or let an agent search iteratively ([Query transformation and agentic RAG](lesson:aieng:11.05)).

## Debugging RAG

Most RAG failures are retrieval failures. Check in order:

~~~dg-flow Work down the pipeline; stop at the first step that fails.
Answer in corpus? -(yes)-> Chunked sensibly? -(yes)-> *Retrieved in top k? -(yes)-> In the prompt? -(yes)-> Used faithfully?
~~~

1. **Is the answer in the corpus at all,** in the current version? If not, no prompt will fix it.
2. **Was it parsed and chunked sensibly?** Look at the chunk that should answer the question.
3. **Was it retrieved?** If the right chunk is at rank 30 and you pass 5, you lose.
4. **Was it in the prompt?** Truncation and token limits silently drop context.
5. **Did the model use it faithfully?** Only now is it a generation problem: tighten instructions, reorder, use a stronger model, or check for conflicting chunks.

<div class="ex">
<div class="lab">Worked example</div>

**debugging one bad answer**

Question: "How long do customers have to return opened items?" The bot answers "14 days". The policy says 30.

Step 1 passes: the 2025 policy is indexed. Step 3 fails: the top 5 chunks include the 2023 policy (14 days) at rank 1 and not the 2025 one, which sits at rank 9. Retrieval failed, not generation; the model faithfully repeated what it was shown.

Cause: both versions are indexed with no date or version field, and the older one uses the exact words of the question. Fix: add a version filter (or remove superseded documents) and re-run the query set to confirm the 2025 chunk now ranks first.

</div>

## Measuring it

Score the two halves separately. Retrieval: did the labelled relevant chunk come back in the top k (recall@k, MRR)? Generation: is every claim supported by the retrieved text (faithfulness), and does it answer the question? A drop in the first means fix the index or ranking; a drop in the second means fix the prompt or model. Building the labelled set and the metrics is in [Evaluating RAG](lesson:aieng:10.04).

## RAG vs long context

Models with 200k to 1M token windows can sometimes take a whole corpus. That costs more per query, is slower (prefill grows with context), and still uses the middle of the prompt less well. RAG remains the default for large or changing corpora; long context helps when the relevant set is small but scattered, and prompt caching makes re-sending a fixed corpus much cheaper ([Prompt, RAG, fine-tune or long context?](lesson:aieng:07.06)).

## Where this breaks

- Superseded and current documents indexed side by side with no version metadata.
- Permissions not carried into the index, so retrieval leaks documents a user should not see.
- No "not found" path, so the model answers from its own memory when retrieval comes back empty.
- Judging the system only on final answers, which hides whether retrieval or generation regressed.

<div class="take">
<div class="lab">Key takeaway</div>

RAG is retrieve then generate. Get parsing, chunking and retrieval right first, debug in pipeline order, measure retrieval separately from generation, and put the best evidence where the model will use it.

</div>
