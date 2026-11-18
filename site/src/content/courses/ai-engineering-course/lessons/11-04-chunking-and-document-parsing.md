---
id: "11.04"
course: "ai-engineering-course"
module: "11"
slug: "chunking-and-document-parsing"
title: "Chunking and document parsing"
keyIdeas: ["Parsing comes before chunking: reading order, headings and tables recovered from a PDF decide what any chunker can do, and most 'bad chunking' is bad parsing.", "Structure-aware recursive splitting at 200-500 tokens is the default; semantic chunking costs an embedding call per sentence and rarely wins consistently.", "Parent-child (small-to-big) and late chunking separate the unit you match on from the unit you read, so chunks can be small without losing context.", "Every chunk should carry its context: a heading breadcrumb for free, or a 50-100 token LLM-written summary (contextual retrieval) when the corpus justifies the one-off cost.", "Compare chunkers on a labelled question set at an equal context-token budget, never by recall@k alone, because bigger chunks win recall@k by construction."]
question: "You are indexing 50,000 PDF contracts and scanned supplier invoices for a RAG assistant. Recall@10 is 62%, and failures cluster on table values and on clauses that refer to terms defined elsewhere in the contract. Propose a parsing and chunking design and say how you would check it worked."
answer: "Parse first: route born-digital PDFs through a layout-aware parser (reading order, headings, tables as Markdown) and scanned invoices through OCR with layout analysis or a VLM transcription, spot-checking digits against the image. Keep each table whole with its caption and header row, repeating the header if a big table must be split. Chunk contracts by their clause structure with a recursive splitter (about 300-500 tokens, no mid-clause cuts), prepend a breadcrumb (contract, party, section) to every chunk, and add contextual retrieval or parent-child retrieval so a clause is returned with the section that defines its terms. Store page and bounding box for citations. Verify on the labelled set, sliced by question type (table, definition, plain clause), comparing recall at a fixed context-token budget and end-to-end answer accuracy against the old index."
rubric: ["Fixes parsing before chunking, with a separate path for scanned documents (OCR or VLM) and a check on transcribed numbers", "Keeps tables intact with headers, or serialises them so each chunk is self-describing", "Chunks on document structure (clauses, sections) rather than fixed windows", "Adds chunk context: breadcrumbs, contextual retrieval or parent-child retrieval for cross-references", "Evaluates on labelled questions sliced by failure type, at an equal context budget, plus end-to-end accuracy"]
sources: [{"title": "Anthropic: Introducing Contextual Retrieval (2024)", "url": "https://www.anthropic.com/news/contextual-retrieval", "note": "LLM-written chunk context before embedding and BM25, with measured drops in retrieval failure."}, {"title": "Qu, Tu and Bao: Is Semantic Chunking Worth the Computational Cost? (2024)", "url": "https://arxiv.org/abs/2410.13070", "note": "Systematic comparison: semantic chunking's gains over fixed-size chunks are not consistent."}, {"title": "Günther et al.: Late Chunking (2024)", "url": "https://arxiv.org/abs/2409.04701", "note": "Embed the whole document, then pool token vectors per chunk."}, {"title": "Faysse et al.: ColPali (2024)", "url": "https://arxiv.org/abs/2407.01449", "note": "Retrieve page images directly with a vision-language model instead of parsing text."}]
practice: ["semantic-chunk-breakpoints", "chunk-text-with-overlap", "sentence-segmentation", "rag-pipeline-capstone"]
tools: ["Docling", "Unstructured", "Tesseract", "ColPali", "LangChain text splitters"]
tags: ["RAG", "chunking", "document parsing"]
---

## The chunk is the unit of everything

[Retrieval-augmented generation](lesson:aieng:03.05) introduced chunking as one step of indexing. It deserves more attention than that, because the chunk is the unit of every later stage: it is what gets embedded, what [BM25](lesson:aieng:11.01) scores, what a [reranker](lesson:aieng:11.03) reads, what is packed into the prompt and what a citation points to. If the fact a question needs is split across two chunks, or sits in a chunk that never says what it is about, no retriever downstream can recover it.

Two jobs come before any choice of chunk size: getting clean, structured text out of the source files, and deciding what context each chunk carries with it.

## Parsing comes first

~~~dg-flow A document parsing pipeline. Structure recovered here is what the chunker splits on.
[Born-digital] PDF, DOCX, HTML -(text layer + layout model)-> Blocks in reading order
[Scanned] Page images -(OCR or VLM)-> Blocks in reading order
[Both] Blocks -(headings, lists, tables)-> *Structured Markdown + page, bbox -(clean)-> Chunker
~~~

A PDF stores glyphs at positions, not paragraphs. A naive text extractor reads two-column pages straight across, glues running headers and footers into the body, breaks words at line-end hyphens and flattens tables into a stream of numbers. Layout-aware parsers (Docling, Unstructured and similar) run a layout model to find blocks, their reading order and their type (title, paragraph, list, table, figure, caption) and emit Markdown or JSON with page numbers and bounding boxes. Keep those coordinates: they let the interface highlight the cited passage, which is the quickest way for a user to trust or reject an answer.

**Scanned pages** have no text layer, so you choose between three routes:

| Route | How it works | Wins when | Watch for |
| --- | --- | --- | --- |
| OCR + layout | Tesseract or a cloud OCR engine, plus a layout model for blocks and tables | High volume, clean scans, cost matters | Tables and handwriting degrade; low-quality scans produce garbage silently |
| VLM transcription | Send each page image to a vision-language model with a "transcribe to Markdown" prompt | Complex layouts, forms, charts, mixed languages | Plausible but invented text, "corrected" digits; costs on the order of a thousand or more input tokens per page (illustrative) |
| Page-image retrieval | Embed page images directly (ColPali-style multi-vector embeddings) and give the generator the page image | Visually rich documents: slides, diagrams, infographics | Needs a multimodal generator; no text for BM25 or exact-match search |

The VLM risk deserves emphasis. OCR fails visibly (garbled characters); a VLM fails fluently, producing a clean table with one wrong number. For invoices, financial statements and anything where digits matter, cross-check VLM output against OCR on a sample, or ask the model to return values with their page coordinates and verify them.

**Tables** need their own handling. Keep each table whole with its caption and header row and serialise it as Markdown or HTML. If a table is too large for one chunk, split by rows and repeat the header row in every piece, so "€4,200" still sits under "Q3 2025, Germany". For very wide data, also index a one-paragraph summary of the table, and for spreadsheets consider whether the question should be answered by a SQL tool rather than retrieval ([routing](lesson:aieng:11.05)).

## Chunking strategies

| Strategy | Typical size | Wins when | Cost and risk |
| --- | --- | --- | --- |
| Fixed token window with overlap | 256-512 tokens, 10-20% overlap | Uniform prose without structure; a quick baseline | Cuts mid-sentence and mid-table; overlap duplicates text in the context |
| Recursive, structure-aware | 200-500 tokens | The default: Markdown, HTML, parsed PDFs with headings | Needs decent parsing; tiny sections need merging |
| Semantic (similarity breakpoints) | Variable, with min/max | Unstructured transcripts where topics shift without headings | One embedding per sentence at ingestion; gains inconsistent in published tests |
| Parent-child (small-to-big) | Child 100-200 tokens, parent a section of 1-2k | Precise matching but answers need surrounding text | Parents can flood the context; dedupe parents |
| Late chunking | Any chunk size, document up to the embedder's context | Chunks full of pronouns and references to earlier text | Needs a long-context embedder with access to token embeddings |
| Whole document | Under about 500 tokens | FAQs, tickets, product records | None; do not split what is already small |

A **recursive splitter** tries the coarsest separator first (section headings, then blank lines, then sentence ends, then spaces) and only goes finer when a piece is still over the limit. Chroma's chunking evaluation found that a recursive splitter at 200-400 tokens with no overlap came close to the best method they tested, and that overlap mainly added redundant tokens to what was retrieved. Domain structure beats generic separators when you have it: split code on functions and classes, transcripts on speaker turns, contracts on clause numbers.

**Semantic chunking** embeds each sentence and cuts where neighbouring sentences stop being similar. It sounds like the principled choice, but Qu, Tu and Bao found its cost was not justified by consistent gains over fixed-size chunks across retrieval and answer-generation tasks. Treat it as something to test on transcripts, not a default.

**Parent-child retrieval** separates the matching unit from the reading unit: index small children for sharp embeddings, and when a child matches, give the model its parent section. **Late chunking** gets a similar effect without a second index: run a long-context embedding model over the whole document, then mean-pool the token embeddings inside each chunk's span. A chunk that says "the fee" was embedded with attention to the earlier sentence that defined which fee.

## Give every chunk its context

A chunk reading "the limit was raised to 30 days" retrieves poorly because nothing in it says which limit, which product or which year. Two fixes, cheapest first:

- **Breadcrumbs**: prepend the document title and heading path ("Refund policy 2025 > Returns > Opened items"). Free, and often most of the gain.
- **Contextual retrieval**: an LLM reads the whole document and writes 50-100 tokens situating each chunk, which is prepended before embedding and before building the BM25 index. Anthropic reported the top-20 retrieval failure rate falling from 5.7% to 3.7% with contextual embeddings, to 2.9% when contextual BM25 was added, and to 1.9% with reranking on top. With prompt caching of the document, they estimated the one-off cost at \$1.02 per million document tokens at the time of the post.

Store metadata on every chunk as well: document id and version, heading path, page and bounding box, dates, language, document type and the access-control list ([kept in sync in 11.06](lesson:aieng:11.06)).

<div class="ex">
<div class="lab">Worked example</div>

**re-chunking a 4,500-token policy PDF**

The policy has 9 sections: seven of 250-500 tokens, two of 800 and 900, and a returns table in section 4.

*Fixed 512-token windows, 64 overlap.* Stride is $512 - 64 = 448$, so the chunk count is

$$
\left\lceil \frac{4500 - 512}{448} \right\rceil + 1 = 9 + 1 = 10
$$

Chunk 4 starts mid-table: its rows have numbers but no header, so "how long for opened electronics?" matches nothing well.

*Recursive on headings, limit 500.* The seven short sections fit whole; the two long ones split at paragraph breaks into two pieces each, giving 11 chunks, none crossing a heading, with the table kept intact in one chunk.

*Add breadcrumbs* (about 15 tokens each): 165 extra tokens to embed, a negligible cost.

*Contextual retrieval across the whole corpus* of 2,000 such documents (9M tokens) at Anthropic's estimate: $9 \times 1.02 \approx 9.2$, so about \$9 one-off. Re-run whenever the chunker changes.

</div>

## Measuring a chunking change

Use the labelled question set from [Evaluating RAG](lesson:aieng:10.04), with the evidence span for each question. Count a retrieval as a hit if a retrieved chunk contains the evidence span (or most of it). Two traps:

1. **Compare at an equal token budget.** A chunker that makes 1,500-token chunks wins recall@5 easily, because five big chunks cover more text. Compare recall at, say, 2,500 context tokens, which is what the generator actually receives.
2. **Slice by question type.** Table questions, definition lookups and multi-section questions respond differently; an average can hide a regression on the slice users care about.

Then confirm with end-to-end answer accuracy, since a chunker can raise recall while making chunks harder for the model to read.

## Where this breaks

- **Overlap as a fix for bad boundaries.** If cuts land mid-sentence, fix the separators; overlap only duplicates text into the context.
- **Silent truncation.** Many embedding models stop reading after their input limit (256 or 512 tokens for many older open models). A 900-token chunk is embedded from its first 512 tokens with no error.
- **Boilerplate in every chunk.** Repeated headers, footers and legal notices inflate BM25 term counts and make unrelated chunks look similar. Strip them in parsing.
- **Parents that flood the context.** Five children from five sections pull in 8,000 tokens. Cap parent size and dedupe.
- **Chunker changes without re-indexing.** A new chunker means new chunk ids, new embeddings and stale citations; plan it as a migration.

## What interviewers probe

- Why not just use a bigger chunk? (Blurrier embeddings, more tokens per hit, and the recall@k comparison is unfair.)
- How do you handle tables and scanned PDFs, and how do you catch a VLM inventing a number?
- When is semantic chunking worth its cost, and how would you prove it on your data?
- How do you keep a clause's cross-references (defined terms, "as above") retrievable?

<div class="take">
<div class="lab">Key takeaway</div>

Fix parsing before tuning chunks, split on document structure at a few hundred tokens, give each chunk its context with breadcrumbs or contextual retrieval, and judge any chunking change on labelled questions at an equal context budget.

</div>
