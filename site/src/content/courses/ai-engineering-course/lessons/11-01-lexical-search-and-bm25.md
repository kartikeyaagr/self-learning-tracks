---
id: "11.01"
course: "ai-engineering-course"
module: "11"
slug: "lexical-search-and-bm25"
title: "Lexical search and BM25"
keyIdeas: ["An inverted index stores, per term, the documents that contain it with their term counts, plus document lengths and corpus statistics. BM25 needs nothing else.", "BM25 comes from the probabilistic relevance framework: the idf is the Robertson-Sparck Jones weight with no relevance data, tf saturates because a document is either about a term or not, and length is normalised because long documents repeat words.", "The rare query term dominates the score. A chunk that never mentions it rarely wins, however often it repeats the common terms.", "The analyzer (tokenizer, lowercasing, stemming, synonyms) decides what counts as a match, and it must be identical at index and query time. Identifiers need their own unanalysed field.", "Lexical beats dense on exact identifiers, codes, error strings, names and vocabulary the embedding model never saw; it is also cheap, explainable and updates instantly."]
question: "Your RAG assistant over engineering docs uses dense retrieval only. Users complain that searching for an error code like \"ERR_TLS_0x1F\" or a config key returns vaguely related pages. Explain why, how BM25 would score the right page, and what you would change in the index."
answer: "Embedding models map rare strings into tokens they saw rarely in training, so the vector for an exact code lands near other codes and generic TLS text; nearness in meaning is not identity. BM25 scores each query term by idf times a saturating tf part. An error code appears in very few chunks, so its idf is high (with N = 1M and df = 3, about 12.6, against about 1 for a word in a third of all chunks), and any chunk containing it jumps to the top. Changes: add a lexical index next to the vectors and fuse the two lists; index the code-like tokens in an unanalysed keyword field (and maybe a field that splits on punctuation) so the standard tokenizer cannot break \"ERR_TLS_0x1F\" apart or lowercase away meaning; keep the same analyzer at query time; and measure recall@k on a slice of identifier queries before and after."
rubric: ["Explains why dense embeddings blur exact identifiers", "Uses idf to show why a rare term dominates the BM25 score, with a rough number", "Mentions tf saturation or length normalisation correctly", "Addresses tokenization of the code (keyword field or custom analyzer) and same analyzer at query time", "Proposes hybrid retrieval rather than replacing dense", "Measures the change on a labelled slice of identifier queries"]
sources: [{"title": "Robertson and Zaragoza: The Probabilistic Relevance Framework: BM25 and Beyond (2009)", "url": "https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf", "note": "The derivation: binary independence, eliteness, saturation and length normalisation."}, {"title": "Thakur et al.: BEIR, a heterogeneous benchmark for zero-shot retrieval (2021)", "url": "https://arxiv.org/abs/2104.08663", "note": "BM25 as a strong zero-shot baseline across 18 datasets; where dense models fall behind."}, {"title": "Apache Lucene: BM25Similarity", "url": "https://lucene.apache.org/core/9_11_0/core/org/apache/lucene/search/similarities/BM25Similarity.html", "note": "The idf form and the k1 = 1.2, b = 0.75 defaults used by Elasticsearch and OpenSearch."}, {"title": "PostgreSQL: Controlling text search (ranking)", "url": "https://www.postgresql.org/docs/current/textsearch-controls.html", "note": "What ts_rank does, and why it is not BM25."}]
practice: ["document-ranking", "tf-idf", "precision-recall-at-k"]
tools: ["Lucene", "Elasticsearch", "OpenSearch", "Tantivy", "ParadeDB", "PostgreSQL"]
---

## Why a keyword ranker still matters in RAG

Dense retrieval ([Vector search](lesson:aieng:02.06)) matches meaning. That is its strength and its weakness: an error code, a SKU, a function name or a person's surname has no "meaning" beyond its exact spelling, and an embedding model places it near other strings that look similar. Lexical search matches the exact terms, ranks them with a formula that has no trainable parameters, and is the other half of nearly every production RAG retriever. This lesson covers what the index stores, where BM25's formula comes from, a full score by hand, analyzers, and when lexical wins.

The basics of the index and of tf-idf are in System Design: [inverted index and posting lists](lesson:sdsd:09.02), [TF-IDF](lesson:sdsd:09.04), and [BM25 in production](lesson:sdsd:09.05), which walks through the $k_1$ and $b$ curves and per-field tuning. Read that for the knobs; here the focus is the derivation, RAG-specific behaviour and failure modes.

## What the index stores

~~~dg-flow Indexing and querying go through the same analyzer.
[Index] Chunk text -(analyzer)-> Terms -(group by term)-> *Postings: term -> (chunk id, tf)
[Stats] Postings -(count)-> N, df per term, chunk lengths, avgdl
[Query] Query text -(same analyzer)-> Terms -(walk postings, score)-> Top k chunks
~~~

For each term, the postings list holds the ids of chunks that contain it and the term frequency in each. The index also keeps each chunk's length $|d|$, the average length $\text{avgdl}$, the number of chunks $N$ and the document frequency $\text{df}(t)$. A query only touches the postings of its own terms, which is why lexical search over tens of millions of chunks runs in milliseconds on a CPU. Engines also skip postings that cannot reach the current top k (block-max WAND), because each term's contribution has a known ceiling.

## Where the formula comes from

BM25 is not a heuristic someone tuned by hand. It falls out of the probabilistic relevance framework (Robertson and Zaragoza's review is the canonical account) in three steps.

**1. Binary independence gives the idf.** The probability ranking principle says: rank documents by the probability they are relevant. If you treat each term as present or absent, independently, ranking by that probability is equivalent to summing a weight for each query term that the document contains:

$$
w_t = \log \frac{p_t\,(1-q_t)}{q_t\,(1-p_t)}
$$

where $p_t$ is the chance the term appears in a relevant document and $q_t$ in a non-relevant one. With no relevance labels, assume $p_t = 0.5$, and estimate $q_t \approx \text{df}(t)/N$ because almost every document is non-relevant. The weight becomes $\log\frac{N-\text{df}}{\text{df}}$, and adding 0.5 to each count for smoothing gives the Robertson-Spärck Jones idf $\log\frac{N-\text{df}+0.5}{\text{df}+0.5}$. That goes negative for terms in more than half the documents, so Lucene uses $\ln\!\big(1 + \frac{N-\text{df}+0.5}{\text{df}+0.5}\big)$, which stays positive.

**2. Eliteness gives saturation.** A document is either "about" a term (elite) or not, and within each state the term count is roughly Poisson. Working that model through gives a weight that is 0 at $\text{tf}=0$, rises steeply, and levels off at the binary weight: more mentions are more evidence of eliteness, but only up to a point. The exact expression has parameters nobody can estimate, so Robertson replaced it with a curve of the same shape, $\frac{\text{tf}\,(k_1+1)}{\text{tf}+k_1}$, which equals 1 at $\text{tf}=1$ and approaches $k_1+1$.

**3. Verbosity gives length normalisation.** Long documents often say the same thing with more words. Divide tf by a length factor $B = 1 - b + b\,|d|/\text{avgdl}$ before saturating, and the tf part becomes $\frac{\text{tf}}{\text{tf} + k_1 B}$ times $(k_1+1)$. Putting the three together:

$$
\text{BM25}(q,d)=\sum_{t\in q}\text{idf}(t)\cdot\frac{\text{tf}(t,d)\,(k_1+1)}{\text{tf}(t,d)+k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)}
$$

Lucene 8 dropped the constant $(k_1+1)$ from the numerator because it scales every score equally and cannot change the order, so raw scores from newer Lucene are lower than textbook values. Never compare BM25 scores across engines, versions or queries; they only order documents within one query.

## A score by hand

<div class="ex">
<div class="lab">Worked example</div>

**query "rotate api key" over 1,000 chunks**

Corpus: $N = 1{,}000$ chunks, $\text{avgdl} = 100$ terms, $k_1 = 1.2$, $b = 0.75$. Document frequencies: rotate 40, api 300, key 120.

idf (Lucene form): rotate $\ln(1 + 960.5/40.5) = 3.207$; api $\ln(1 + 700.5/300.5) = 1.203$; key $\ln(1 + 880.5/120.5) = 2.117$.

| Chunk | Length | tf (rotate, api, key) | $k_1 B$ | tf parts | BM25 |
| --- | --- | --- | --- | --- | --- |
| A | 60 | 1, 1, 2 | 0.84 | 1.196, 1.196, 1.549 | 3.835 + 1.439 + 3.280 = **8.55** |
| B | 200 | 0, 4, 5 | 2.10 | 0, 1.443, 1.549 | 0 + 1.736 + 3.280 = **5.02** |
| C | 100 | 2, 0, 1 | 1.20 | 1.375, 0, 1.000 | 4.410 + 0 + 2.117 = **6.53** |

Order: A, C, B. Three things to see. The rare term carries the query: "rotate" alone gives C most of its score. Saturation and length cancel repetition: B's five mentions of "key" in a 200-term chunk earn exactly the same tf part (1.549) as A's two mentions in a 60-term chunk. And raw tf-idf gets this wrong: with tf $\times \ln(N/\text{df})$, B scores $4 \times 1.204 + 5 \times 2.120 = 15.4$ against A's 8.7 and C's 8.6, ranking first the one chunk that never mentions rotating anything.

</div>

## Analyzers decide what matches

BM25 only counts exact term matches, so the analyzer that turns text into terms is where most lexical quality is won or lost.

| Step | What it does | Typical trap |
| --- | --- | --- |
| Tokenizer | Splits text into tokens | The standard Unicode tokenizer splits on hyphens and punctuation, so `E-1042` becomes `e` and `1042`, and `1042` now matches every page with that number |
| Lowercasing, ASCII folding | `Café` and `cafe` match | Case can carry meaning in code (`Map` vs `map`) |
| Stop words | Drops "the", "of" | Breaks phrase queries such as "to be or not to be"; BM25's idf already discounts common words, so many teams keep them |
| Stemming (Porter, Snowball) | `rotating`, `rotated`, `rotation` share a stem | Over-stemming merges unrelated words (`university`, `universe`) |
| Synonyms | `2FA` matches `two-factor` | Index-time synonyms need a reindex to change; query-time synonyms do not |
| n-grams / edge n-grams | Partial matches for codes and autocomplete | Index grows several times; short grams match everything |

The usual fix for identifiers is to index the same text twice: an analysed field for prose and an unanalysed `keyword` field (or one that splits only on whitespace) for codes, then query both with a boost on the exact field. Whatever you choose, the query must go through the same analyzer as the index. A query lowercased by one pipeline and an index built by another is the classic "it matches in the console but not in the app" bug.

## Where lexical wins, and where it loses

Lexical wins on exact identifiers (order ids, ticket numbers, SKUs), error messages and stack-trace fragments, function and config names, people and product names, and any vocabulary that appeared after the embedding model was trained. BEIR (Thakur et al.) found BM25 a hard baseline to beat zero-shot: across its 18 datasets, several dense retrievers trained on MS MARCO fell below BM25 out of domain. It also has practical advantages: no GPU, no model version to migrate, an index update is visible as soon as the segment refreshes, and you can explain any score term by term.

It loses on vocabulary mismatch ("can't log in" against a page titled "Recovering account access"), paraphrased questions, cross-lingual queries and misspellings. That is why the answer is almost never "lexical or dense" but both, fused ([Hybrid search and score fusion](lesson:aieng:11.02)).

Where to run it: the Lucene family (Elasticsearch, OpenSearch, Solr), Tantivy (Rust; used by ParadeDB's `pg_search` to add BM25 inside Postgres), Vespa, and several vector databases that now compute BM25 or BM25F natively. Postgres's built-in `ts_rank` is not BM25: it counts matching lexemes in the document but uses no corpus-wide document frequencies, so a rare term weighs the same as a common one. For a few hundred thousand chunks in a notebook, a Python BM25 library over an in-memory index is fine.

## Where this breaks

- **Long natural-language queries.** An LLM-rewritten query of 25 words adds many weak terms; scores grow with query length and the rare term's signal is diluted. Extract key terms, or let the dense side handle the paraphrase.
- **Chunking changes the statistics.** idf and avgdl are computed over chunks, not documents. Boilerplate repeated in every chunk (headers, legal footers) becomes a near-zero-idf term, harmless, but a product name repeated in every chunk of its manual also loses weight. Prepending a short context line to each chunk (Anthropic's "contextual BM25") puts the document's key terms back into chunks that lack them; see [Chunking and document parsing](lesson:aieng:11.04).
- **Per-shard statistics.** Elasticsearch scores with shard-local df by default. On small or skewed shards the same term gets different idf on different shards and merged rankings wobble; `dfs_query_then_fetch` computes global statistics at the cost of an extra round trip.
- **Deletes lag.** Lucene counts deleted documents in df until segments merge, so heavy delete churn skews idf for a while ([Keeping a RAG index correct](lesson:aieng:11.06)).
- **Multilingual corpora.** One analyzer cannot stem English and German correctly; use per-language fields or a language-aware analyzer, and remember CJK text needs a segmenting tokenizer.

## What interviewers probe

- "Derive or explain the idf" (binary independence with no relevance data) and "why does tf saturate" (eliteness). Saying "it's tf-idf with two fixes" is the floor, not the answer.
- "Score this by hand": know the formula well enough to compute one term in your head.
- "When would you not use embeddings at all?" Identifier-heavy search, tiny corpora, strict explainability, or no GPU budget.
- "Why are BM25 scores not comparable across queries, and what does that mean for fusion?" That leads straight into the next lesson.

<div class="take">
<div class="lab">Key takeaway</div>

BM25 is the probabilistic relevance model made practical: an idf that rewards rare terms, a tf part that saturates, and length normalisation against the average chunk. It needs no training, wins on exact identifiers and new vocabulary, and lives or dies by its analyzer. Treat its scores as an order within one query, never as a calibrated number.

</div>
