---
id: "09.06"
course: "system-design-101"
module: "09"
slug: "stop-words-and-champion-lists"
title: "Stop words and champion lists"
keyIdeas: ["**Stop words** are very common words (the, a, of) that carry little meaning. Dropping them shrinks the index and speeds queries, but breaks phrase queries (\"to be or not to be\"). Modern engines usually keep them and rely on IDF.", "**Champion lists**: for each term, precompute the top r documents by weight. At query time only consider docs in the champion lists of the query terms.", "Champion lists trade recall for speed: a relevant document outside every champion list won't be found.", "Other pruning: only consider high-IDF query terms, or docs containing most query terms."]
question: "What do champion lists trade away to make queries faster?"
answer: "They give up exactness. Only documents in at least one query term's precomputed top r list are scored, so a document that is a strong match for the whole query but not in the top r for any single term (for example ranked 600th for each word alone but best for both together) is never seen. The lists are chosen at index time without knowing the query, so they can't adapt to filters or term combinations; if r is smaller than the number of results needed, or many champions fail filters, the page may not fill; and the lists go stale until rebuilt."
sources: [{"title": "IR book ch.2.2 and 7.1: Stop words, champion lists", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "Pruning for speed."}]
---

## Two ways to do less work

A search engine's query cost is dominated by postings: how many entries it reads and how many documents it scores. This lesson covers two classic ways to cut that cost. **Stop words** remove the most expensive terms from the index altogether. **Champion lists** keep every term but score only a small, precomputed shortlist of documents per term. Both trade something away for speed, and knowing exactly what they trade is the point of the lesson.

## Why common words are expensive

Word frequencies in any language follow **Zipf's law**: the most common word appears about twice as often as the second, three times as often as the third, and so on. In English text, "the" alone is around 5 to 7% of all words, and the top 100 words make up roughly half of all text.

For an inverted index, this means a handful of terms have enormous postings lists. In a collection of 100 million documents, "the" appears in nearly all of them: a 100-million-entry list. A query containing "the" would read that list, and if scored, touch nearly every document. That's the cost stop words were invented to avoid.

~~~dg-bars Approximate share of all word occurrences in English text covered by the most frequent words. A tiny vocabulary makes up a huge share of postings.
Top 1 word (the) | 6 | ~6%
Top 10 words | 23 | ~23%
Top 100 words | 50 | ~50%
*Top 1,000 words | 70 | ~70%
~~~

## Stop words

**Stop words** are extremely common words, such as "the", "a", "of", "to", "and", "in", that carry little meaning on their own. A **stop list** is the set you drop. It's usually built like this: sort terms by **collection frequency** (total occurrences across all documents), take the most frequent ones, and check them by hand for meaning in your domain (in a legal corpus "court" might be extremely common but must stay). At indexing time, stop words are not added to the index; at query time, they're removed from the query.

The benefit is real. Because stop words have the longest postings lists, dropping a few dozen can shrink an uncompressed index by a quarter or more, and queries never read those lists.

### What goes wrong

- **Phrase queries break.** "To be or not to be" consists entirely of stop words. With a typical stop list, the query becomes empty. "President of the United States" loses "of the", and "The Who" (the band) becomes just "who".
- **Meaning changes.** "Flights to London" and "flights from London" become the same query, "flights london". "Vitamin A" loses "a". "Not working" loses "not" in some lists.
- **Names and titles vanish.** "Let It Be", "The The", "IT department", "Take That".
- **Position-based features break**: if stop words are removed before positions are assigned, "new york" might match "new in york".

~~~dg-compare Dropping stop words versus keeping them. Modern engines keep them because compression and IDF remove most of the cost.
[Stop list]
"to be or not to be" -(remove stop words)-> Empty query
+ Smaller index
+ Never reads huge lists
- Phrases and titles break
- "to London" equals "from London"
[*Keep stop words]
"to be or not to be" -(idf near 0)-> Phrase match works
+ Exact phrases and names work
+ Common words barely affect score
- Larger postings, needs compression
~~~

### Why modern engines don't need them

Stop lists have shrunk over time: from lists of 200 to 300 terms, to very small lists of 7 to 12 terms, to none at all. Web search engines generally don't use stop lists, and Elasticsearch's standard analyser has no stop list by default. Three techniques removed the need:

- **Compression.** Long postings lists of a common word have tiny gaps between docIDs (lesson 09.02), so they compress extremely well, often to a bit or two per posting. Storage cost drops sharply.
- **IDF weighting.** Ranking already gives common words almost no weight (lesson 09.04): "the" gets idf near zero, so keeping it doesn't distort scores.
- **Impact-ordered postings and dynamic pruning.** Query processing can stop reading a common term's list early, or skip it entirely during scoring, once it's clear the term can't change the top results. Algorithms like WAND and block-max WAND do this automatically.

A middle path is also common: index stop words, but treat them specially. For example, use **common-word bigrams** ("the_who", "to_be") so phrases with stop words stay cheap, or treat stop words as optional in the query (`should` clauses) so they help ranking when present but aren't required to match.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of keeping "the"**

An index of 100 million documents. "the" appears in 98 million. Uncompressed at 4 bytes per docID, its postings would be 392 MB. With gap encoding, the average gap is about 1, which bit-packs to 1 or 2 bits per posting: roughly 15 to 25 MB. Add positions and it's larger, but the index for one word costs tens of megabytes, not hundreds. At query time, block-max WAND notices "the" can contribute almost nothing to any score and reads only the blocks it needs for phrase checks. Keeping "the" costs a few percent of index size and makes "The Who" findable.

</div>

## Champion lists

Even with an inverted index and no stop words, a single query term can match hundreds of thousands of documents. The query "insurance" in a large collection might match 400,000 documents, and fully scoring all of them (BM25 plus quality signals) for every query is slow. **Champion lists** (also called **fancy lists** or **top docs**) cut that work.

The idea:

1. **At indexing time**, for each term t, precompute the r documents with the highest weight for t (for example, highest tf, or highest tf-idf). This short list is t's champion list.
2. **At query time**, take the **union** of the champion lists of the query terms and compute full scores only for those documents.
3. Return the top K of those.

With r = 500 and a two-word query, at most 1,000 documents are scored, regardless of how many contain the terms.

~~~dg-flow Champion lists: precompute each term's best r documents at index time, then score only their union at query time.
[Index time] Term "insurance" -(keep top r by weight)-> *Champion list: r docs
[Query time] Query terms -(union of champion lists)-> Small candidate set -(full score)-> Top K
~~~

### What you trade away

- **Exactness.** A document that is good overall but not in any single term's top r is never considered. Champion lists approximate the true top K; they don't guarantee it.
- **Combination effects.** A document that mentions both terms moderately (say, ranked 600th for each term alone) may be the best match for the pair, but it's in neither list and will be missed.
- **Filling the page.** If r is smaller than K (the number of results you need), or if most champions fail the query's filters (out of stock, wrong language), you may not have enough results.
- **Query-blind choice.** r is chosen at index time, before the query is known. r can vary by term, for example larger for rarer terms or for terms often combined with filters, but it's always a guess.
- **Freshness.** Champion lists must be recomputed as documents change; a new, excellent document may not be a champion until the next rebuild.

~~~dg-matrix What each speed-up costs. All of them trade some exactness or flexibility for less work per query.
Technique | Index size | Query speed | Exact top K | Phrases work
Stop list | yes, smaller | yes, faster | partial | no
Keep stop words + compression | partial, a bit larger | partial | yes | yes
Champion lists | partial, extra lists | yes, much faster | no | yes
Tiered index | partial, extra tiers | yes, faster | no | yes
Block-max WAND | yes, small metadata | yes, faster | yes | yes
~~~

### Static quality scores

Champion lists get much better when combined with a **static quality score** g(d): a query-independent measure of how good a document is, such as PageRank for web pages, review score and sales for products, or reputation for answers. Two uses:

- Build each champion list from a combination of the term's weight and g(d), so champions are both relevant to the term and good in general.
- Order postings by g(d) descending, so the best documents are read first and a query can stop after finding enough high-scoring candidates.

The final score is then something like `score(q, d) = g(d) + cosine(q, d)`. Because g(d) is the same for every query, high-quality documents get a head start, and the engine can stop once no remaining document could beat the current top K.

A further refinement is **high and low lists**: each term's postings are split into a high list (its champions) and a low list (the rest). The query scores high lists first and only consults low lists if it can't fill K results. This is the per-term version of the tiered index from lesson 09.03.

<div class="ex">
<div class="lab">Worked example</div>

**champion lists for "cheap flights"**

r = 500. "cheap" has 2,000,000 documents and "flights" has 300,000. Fully scoring the union of their postings could mean scoring over 2 million documents. With champion lists, you take the 500 best for "cheap" and the 500 best for "flights", a union of at most 1,000 (in practice maybe 800, since some overlap), and score those fully: roughly 2,500 times less scoring work.

What's lost: a travel deals page ranked 600th for "cheap" and 700th for "flights" individually, but excellent for the phrase "cheap flights", is in neither list and is never seen. If many such pages exist, the results are worse. That's the accuracy given up for speed, and why systems that use champion lists check with evaluation metrics (lesson 09.09) how often the true top 10 is missed.

</div>

## Where you see these ideas today

- **Elasticsearch / Lucene** keep stop words by default and rely on compression and block-max WAND, which gives champion-list-like speed without losing exactness.
- **Web-scale engines** use static quality ordering and tiers so most queries are answered from a small, high-quality slice of the index.
- **Two-stage ranking** (lesson 09.05) is the modern descendant: a cheap stage picks a shortlist, an expensive stage scores only that.

## Common mistakes

- Applying a large stop list to product or name search, so "The North Face" or "Vitamin A" can't be found.
- Removing stop words at indexing time but not at query time (or the reverse), breaking matches.
- Using champion lists with r smaller than the page size plus the expected filter loss.
- Never measuring how often approximation drops the true best result.

## In an interview

If asked about stop words, explain the trade (smaller index versus broken phrases and meanings) and say modern engines keep them, relying on compression, IDF and dynamic pruning. For champion lists, state the trade precisely: faster queries by scoring only each term's precomputed top r, at the cost of missing documents that are strong for the combined query but not top for any single term.

<div class="take">
<div class="lab">Key takeaway</div>

stop lists shrink the index but break phrases and meaning, so modern engines keep stop words and rely on compression, IDF and pruning. Champion lists precompute each term's top r documents so queries only score a small candidate set, trading exactness (and sometimes a full page) for speed.

</div>
