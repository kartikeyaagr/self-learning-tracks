---
id: "09.04"
course: "system-design-101"
module: "09"
slug: "tf-idf-relevance-scoring"
title: "TF-IDF relevance scoring"
keyIdeas: ["TF-IDF scores how important a term is to a document.", "**TF** (term frequency): how often the term appears in the document. More mentions suggests more relevance.", "**IDF** (inverse document frequency): log(N / df), where df is how many documents contain the term. Rare terms get high weight; terms in every document (\"the\") get weight near zero.", "Score of a doc for a query = sum over query terms of TF × IDF, often in a vector space model with cosine similarity to normalize for document length."]
question: "Why does IDF push down words like \"the\"?"
answer: "IDF is log(N / df), where N is the number of documents and df the number containing the term. A word like \"the\" appears in nearly every document, so df is close to N, the ratio is close to 1, and its logarithm is close to 0. Its tf-idf weight in every document is therefore near zero however many times it appears, so it barely changes any score or ranking. Rare terms that actually distinguish relevant documents get large weights, and the log keeps very rare terms from completely swamping the rest."
sources: [{"title": "IR book ch.6: Scoring, term weighting", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "TF-IDF and the vector space model."}]
---

## From matching to ranking

Boolean search (lesson 09.03) says whether a document matches. Ranked search says **how well**. For a query like "the auto insurance", tens of thousands of documents might contain at least one of those words. You need a number for each, a **relevance score**, so you can show the best ten first.

TF-IDF is the classic way to compute that number from text alone. It's built from two observations about language that are easy to state and surprisingly powerful.

1. **A term that appears often in a document is probably central to it.** A page that says "insurance" twelve times is more likely about insurance than one that says it once. This is **term frequency**, written tf(t, d): the number of times term t occurs in document d.
2. **A term that appears in many documents tells you little.** Almost every English document contains "the", so a match on "the" says nothing about whether a document is relevant. A query for "the auto insurance" should be driven by "insurance" and "auto", not "the". This is captured by **document frequency**, df(t): the number of documents in the collection that contain t.

Term frequency is about one document. Document frequency is about the whole collection. TF-IDF multiplies a local signal by a global one.

## Inverse document frequency

Turn document frequency into a weight that is high for rare terms and low for common ones:

`idf(t) = log(N / df(t))`

where N is the total number of documents. Read it as "how surprising is it to see this term?".

- A term in **every** document: N / df = 1, and log(1) = 0. It contributes nothing.
- A term in **half** the documents: log10(2) ≈ 0.3.
- A term in **1 document of a million**: log10(1,000,000) = 6.

~~~dg-bars idf (base 10) in a collection of 1,000,000 documents. The log keeps a term 1,000 times rarer from being 1,000 times heavier.
the: in 1,000,000 docs | 0.001 | 0
of: in 600,000 docs | 0.22 | 0.22
auto: in 20,000 docs | 1.7 | 1.7
insurance: in 1,000 docs | 3 | 3
*subrogation: in 10 docs | 5 | 5
~~~

Why the logarithm? Without it, idf would be N / df directly: "subrogation" (in 10 documents) would weigh 100,000 and "insurance" (in 1,000) would weigh 1,000, so the rare word would swamp everything. The log dampens this so a term 1,000 times rarer is about 3 points heavier, not 1,000 times heavier. The base of the log doesn't matter for ranking; it scales every weight by the same constant.

### Why document frequency and not total count

There's another way to measure how common a word is: **collection frequency** (cf), its total number of occurrences across all documents. Why not use that? Compare two words from a news collection of 806,791 Reuters articles:

| Term | cf (total occurrences) | df (documents containing it) |
| --- | --- | --- |
| try | 10,422 | 8,760 |
| insurance | 10,440 | 3,997 |

Both words occur about 10,400 times in total, so collection frequency treats them as equally informative. But "try" is spread thinly: it appears about once in each of 8,760 articles. "insurance" is concentrated in fewer than 4,000: when an article is about insurance, it says "insurance" several times. Document frequency captures that "insurance" is the better **discriminator**, the word that separates relevant documents from irrelevant ones. That's exactly what a search weight should measure.

## Combining them

The TF-IDF weight of a term in a document is the product:

`tf-idf(t, d) = tf(t, d) × idf(t)`

So the weight is:

- **Highest** when a term occurs many times in a small number of documents (strong local signal, rare globally).
- **Lower** when a term occurs few times in a document, or occurs in many documents.
- **Zero** when a term occurs in every document.

The simplest score for a document against a query adds up the weights of the query terms:

`Score(q, d) = Σ over t in q of tf-idf(t, d)`

Terms in the query that the document doesn't contain contribute 0.

~~~dg-flow tf-idf for one term in one document: the count in the document times the term's rarity across the collection.
tf: 3 in the doc -(times)-> idf: log10(1,000,000 / 1,000) = 3 -> *Weight 9
~~~

<div class="ex">
<div class="lab">Worked example</div>

**scoring two documents for "the insurance"**

N = 1,000,000. "the" is in all 1,000,000 documents, so idf = log10(1) = 0. "insurance" is in 1,000 documents, so idf = log10(1,000) = 3.

Document A says "the" 40 times and "insurance" once: score = 40 × 0 + 1 × 3 = 3.

Document B says "the" 5 times and "insurance" 6 times: score = 5 × 0 + 6 × 3 = 18.

B wins by a wide margin, and the 40 uses of "the" in A count for nothing. Without idf (raw tf only), A would score 41 and B 11, and the wrong document would win.

</div>

## Sublinear tf

Is a document that says "insurance" 20 times really 20 times as relevant as one that says it once? Almost never. The 20th mention adds much less evidence than the first. A common fix is **sublinear** (log-scaled) term frequency:

`wf(t, d) = 1 + log(tf)` if tf > 0, else 0

With base 10: tf 1 gives 1, tf 10 gives 2, tf 100 gives 3. Repetition still helps, but with sharply diminishing returns. This is the first step towards **saturation**, which BM25 (lesson 09.05) makes explicit and tunable.

## The vector space model

TF-IDF becomes more powerful when you treat documents as **vectors**. Imagine a space with one dimension per term in the dictionary, possibly millions of dimensions. Each document is a point in that space: its coordinate on the "insurance" axis is its tf-idf weight for "insurance", and so on. Most coordinates are zero, since a document contains a tiny fraction of all terms (a **sparse vector**). The query is also a vector in the same space.

Relevance is now geometric: how closely does the document vector point in the same direction as the query vector? The standard measure is **cosine similarity**, the cosine of the angle between them:

`cos(q, d) = (q · d) / (|q| × |d|)`

The dot product `q · d` sums, over shared terms, the product of their weights. Dividing by the vector lengths `|q|` and `|d|` **normalises for length**. Without it, a long document wins just because it contains more words and therefore more matches and higher counts. With it, a short, focused document on the topic can beat a 50-page manual that mentions the topic in passing.

~~~dg-compare Scoring with and without length normalisation. Without it, long documents win simply by containing more words.
[Raw tf-idf sum]
Query -> Sum weights -> Long docs favoured
- 50-page manual beats a focused FAQ
- Keyword stuffing raises the score
[*Cosine similarity]
Query -> Dot product -(divide by lengths)-> Direction match
+ Focused short docs can win
+ Score between 0 and 1
- Can over-penalise long relevant docs
~~~

<div class="ex">
<div class="lab">Worked example</div>

**cosine with two terms**

Query "auto insurance". Use idf: auto 1.7, insurance 3. Query vector q = (1.7, 3), length √(1.7² + 3²) = √11.89 ≈ 3.45.

Doc X: a short page with tf auto 1, insurance 1, so weights (1.7, 3). Its other words also add to its vector length; say its full length is 10. q · X = 1.7 × 1.7 + 3 × 3 = 11.89. Cosine = 11.89 / (3.45 × 10) ≈ 0.34.

Doc Y: a long manual with tf auto 3, insurance 5, so weights (5.1, 15). With thousands of other words, its full length is 80. q · Y = 1.7 × 5.1 + 3 × 15 = 53.67. Cosine = 53.67 / (3.45 × 80) ≈ 0.19.

Raw sums would rank Y first (20.1 versus 4.7). Cosine ranks the short, on-topic X first.

</div>

### How engines compute this fast

The engine doesn't compare the query against every document vector. It walks the postings lists of the query terms only (a handful of lists), accumulating each document's partial dot product in a table of **accumulators**, one per candidate document. Document lengths are precomputed at indexing time and stored per document, so the division is cheap. Then a heap keeps the top K. This **term-at-a-time** (or document-at-a-time, walking all lists in parallel) scoring is why ranking stays fast: work is proportional to the length of the query terms' postings, not to the collection.

## Weaknesses

Plain TF-IDF has known problems:

- **Unbounded tf**: raw term frequency grows without limit, so stuffing a keyword 500 times keeps raising the score. Spammers exploited this in early web search.
- **Crude length handling**: cosine normalisation can over-penalise long documents that are genuinely relevant, and there's no knob to tune it.
- **Bag of words**: word order and proximity are ignored, so "dog bites man" and "man bites dog" score the same.
- **Vocabulary mismatch**: "car" doesn't match "automobile" at all; TF-IDF only rewards exact term matches. Synonyms (lesson 09.07) and vector search (lesson 09.13) address this.

BM25 (next lesson) fixes the first two with saturation and tunable length normalisation, and is the default in Lucene and Elasticsearch today.

## Common mistakes

- Forgetting idf is computed per shard in distributed engines, so a small shard with unusual content can give different scores. Elasticsearch's `dfs_query_then_fetch` exists to gather global statistics first when this matters.
- Expecting TF-IDF to handle synonyms or meaning. It matches strings.
- Removing length normalisation "because long documents are better" without checking with evaluation metrics (lesson 09.09).

## In an interview

Explain TF-IDF in two sentences: tf rewards terms frequent in the document, idf = log(N/df) discounts terms common across the collection, and the score sums tf × idf over query terms. Then name the weaknesses (unbounded tf, crude length normalisation, no synonyms) to lead naturally into BM25 and semantic retrieval.

<div class="take">
<div class="lab">Key takeaway</div>

tf-idf rewards a term that appears often in a document (tf) but discounts terms that appear in many documents (idf = log(N/df)), so common words like "the" contribute almost nothing. A document's score sums tf-idf over the query terms, and cosine similarity normalises for document length so long documents don't win by size alone.

</div>
