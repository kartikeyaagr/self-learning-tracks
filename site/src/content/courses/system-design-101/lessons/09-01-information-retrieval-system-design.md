---
id: "09.01"
course: "system-design-101"
module: "09"
slug: "information-retrieval-system-design"
title: "Information retrieval system design"
keyIdeas: ["Search turns a query into a ranked list of relevant documents quickly.", "Pipeline: crawl or ingest documents → parse and tokenize text → build an inverted index → at query time, parse the query → retrieve candidate docs → score and rank → return results with snippets.", "Indexing is offline and heavy; querying must be fast (tens of ms).", "Quality depends on text processing (tokenization, stemming, synonyms) and ranking signals (text relevance, popularity, freshness, personalization).", "Elasticsearch/OpenSearch (built on Lucene) are the common production engines."]
question: "Name the stages from raw document to ranked result."
answer: "Indexing: documents are acquired by crawling or ingestion (often CDC from the main database), parsed into fields, analysed (tokenised, lower-cased, accents folded, stemmed, maybe stop words removed) and written into an inverted index mapping each term to the documents that contain it, with frequencies and positions. Querying: the query goes through the same analysis plus spelling correction, synonym expansion and intent detection; the postings lists for its terms are looked up and combined into candidates and filtered; the candidates are scored with BM25 or similar; the top few hundred are re-ranked with a learned model using signals like popularity, freshness and clicks; and the final page is returned with snippets and highlights."
sources: [{"title": "Stanford: Introduction to Information Retrieval (free book)", "url": "https://nlp.stanford.edu/IR-book/", "note": "The standard IR textbook. Chapter 1 here."}, {"title": "Hello Interview: Elasticsearch deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/elasticsearch", "note": "A production search engine."}]
---

## What a search system does

A search system takes a short, vague query, usually two or three words, and returns a ranked list of the documents most likely to satisfy it. It does this in tens of milliseconds, over a collection that may hold millions or billions of items, while the collection keeps changing underneath it.

Start with the naive alternative to see why special machinery is needed. Suppose you want the plays by Shakespeare that contain "Brutus" and "Caesar" but not "Calpurnia". You could scan every play line by line, the way `grep` does. Shakespeare's complete works are about 5 MB, and a modern machine scans that in a few milliseconds. So for small data, scanning works.

It stops working for three reasons:

- **Size.** A product catalogue of 50 million items, or a web index of billions of pages, is terabytes. Scanning a terabyte at 2 GB/s takes over 8 minutes per query.
- **Flexible matching.** Users want "running" to match "run", "colour" to match "color", "NYC" to match "New York", misspellings to be forgiven and words near each other to count more. `grep` matches literal strings.
- **Ranking.** A scan answers yes or no. Users want the best ten results first, out of maybe 100,000 that match.

The answer is to do the expensive work once, ahead of time, by building an **index**, a data structure that turns "which documents contain this word?" into a lookup rather than a scan. Every search system is organised around that idea.

## Two halves on two schedules

Every search system has two halves that run at different times and have very different requirements:

- **Indexing** (offline or near-real-time, heavy, can take seconds to hours): collect documents, turn their text into terms, and build data structures that make lookups fast. It's throughput-bound: you care about documents per second and how quickly a change becomes searchable.
- **Querying** (online, must be fast): turn the user's query into terms, look them up, score the matching documents and return the best few. It's latency-bound: you care about p99 milliseconds per query and queries per second.

Separating them is what makes search fast: the query side does as little work as possible because the indexing side has already done the rest.

~~~dg-flow Indexing runs ahead of time; querying reuses the same text processing so query terms match index terms.
[Indexing] Documents -(crawl or ingest)-> Parse and tokenize -(normalise, stem)-> *Inverted index
[Querying] Query -(same analyser)-> Query terms -(look up postings)-> Candidates -(score, re-rank)-> Results
~~~

## The pipeline, stage by stage

Here is the path from raw document to ranked result. Each stage has its own lesson later in this module.

### Indexing side

1. **Acquire documents.** A **crawler** fetches web pages (lesson 09.12), or, far more often in product work, the application pushes its own records (products, posts, tickets, messages) into the search engine, typically through change data capture from the main database (lesson 09.10).
2. **Parse.** Extract the text and fields from each document: strip HTML tags, pull out the title, body, author, price, date. Decide which fields are searchable text and which are exact values for filtering.
3. **Tokenise and normalise** (together called **analysis**). Split text into **tokens** (roughly, words), lower-case them, fold accents ("café" to "cafe"), optionally remove **stop words** such as "the" (lesson 09.06), and reduce words to a **stem** or base form ("running", "runs" to "run"). The output is **terms**, the units the index stores.
4. **Build the inverted index.** For each term, record which documents contain it, and usually how many times and at which positions (lesson 09.02). Also store per-document data such as length and quality signals.

### Query side

5. **Understand the query.** Apply the same analysis to the query, then correct spelling (lesson 09.15), expand synonyms, recognise entities ("nike" is a brand) and detect intent (lesson 09.07).
6. **Retrieve candidates.** Look up each query term's **postings list** (the list of documents containing that term) and combine them with AND or OR logic (lesson 09.03). Apply filters such as `in_stock = true`.
7. **Score and rank.** Compute a text-relevance score such as TF-IDF or BM25 for each candidate (lessons 09.04, 09.05). Then optionally **re-rank** the top few hundred with a learned model that also uses popularity, freshness, personalisation and click signals (lesson 09.08).
8. **Return results** with titles, **snippets** (short extracts showing the matched words, highlighted) and facets (counts by brand, price band).

The crucial design rule is that **steps 3 and 5 must agree**. If the index stores "run" but the query is processed into "running", nothing matches. Search engines enforce this by attaching one named **analyser** to each field and running it on both documents and queries.

~~~dg-layers The ranking funnel: each stage looks at fewer documents with a more expensive scoring method.
All documents | 100 million, never touched one by one
Candidate retrieval | ~100,000 matching docs via postings lists
*First-stage scoring | BM25 on matches, keep top 1,000
Re-ranking | learned model on top 1,000, keep top 50
Final page | 10 results with snippets
~~~

## Why the funnel matters

Notice the shape of that diagram. Each stage handles roughly 10 to 100 times fewer documents than the one above, and can afford to spend 10 to 100 times more per document. Postings lookups cost nanoseconds per document. BM25 costs maybe a microsecond. A learned re-ranker with hundreds of features might cost 10 to 100 microseconds per document. Running the re-ranker on all 100,000 matches would take seconds; running it on the top 1,000 takes a few milliseconds. Every large search system, from web search to e-commerce to recommendation, is a funnel like this.

## What a production engine looks like

The most common real implementation is **Elasticsearch** (or its fork **OpenSearch**), a distributed layer on top of **Apache Lucene**, the Java library that does the actual indexing and scoring. Solr is another distributed layer on Lucene. Their vocabulary:

| Concept | What it is |
| --- | --- |
| Index | A named collection of documents, roughly a table |
| Document | A JSON object, the unit you search and return |
| Mapping | The schema: which fields exist and how each is analysed |
| Shard | A slice of an index; each shard is one complete Lucene index |
| Replica | A copy of a shard, for fault tolerance and read throughput |
| Segment | An immutable piece of a Lucene index; new writes create new segments |

### A search request, step by step

A search request arrives at any node, which acts as the **coordinating node** for that request. It sends the query to one copy (primary or replica) of every shard of the index.

- In the **query phase**, each shard runs the query against its own documents, scores them, and returns only the IDs and scores of its local top results (if the page size is 10, its top 10).
- The coordinator merges these lists into a global top 10.
- In the **fetch phase**, it asks only the shards that own those 10 documents for their full contents (and snippets).

Returning IDs first and fetching bodies second means the network carries 10 full documents rather than 10 times the number of shards.

~~~dg-seq Elasticsearch's two-phase search: shards return only IDs and scores first, then the coordinator fetches the winning documents.
Client > Coordinator : search "red shoes", size 10
Coordinator > Shard 1 : query phase
Coordinator > Shard 2 : query phase
Shard 1 -> Coordinator : top 10 IDs + scores
Shard 2 -> Coordinator : top 10 IDs + scores
Coordinator > Coordinator : merge, keep global top 10
Coordinator > Shard 1 : fetch docs 4, 9, 17
Coordinator > Shard 2 : fetch docs 2, 8 ...
Coordinator -> Client : 10 ranked documents
~~~

This also shows why **deep pagination** is expensive: asking for results 10,000 to 10,010 forces every shard to return its top 10,010, and the coordinator to sort them all. Elasticsearch caps it by default (`max_result_window` of 10,000) and offers cursor-style `search_after` instead.

### Two practical rules

- **A search engine is not a primary database.** Keep the source of truth in Postgres, MySQL or DynamoDB and copy changes into the search index (lesson 09.10). Search engines are optimised for reading, their durability guarantees and transactional features are weaker, and you will want to rebuild the index from scratch when you change analysers.
- **It's often unnecessary for small data.** Below roughly 100,000 documents with modest query rates, the database's own full-text search (Postgres `tsvector` with a GIN index, MySQL `FULLTEXT`) is usually enough, and saves you running and syncing a second system.

<div class="ex">
<div class="lab">Worked example</div>

**search for a shop with 2 million products**

Indexing: the product database streams every change through CDC into an Elasticsearch index with fields `title` (analysed text with an English stemmer), `brand` (exact keyword), `price` (number), `in_stock` (boolean). A price change is searchable within about a second.

Querying: "red runing shoes" arrives. The analyser lower-cases it and splits it into three tokens. The spell corrector changes "runing" to "running", which the stemmer reduces to "run". The engine looks up postings for red, run and shoe, takes documents containing all three (about 4,000), filters to `in_stock = true` (3,100), scores them with BM25 and keeps the top 500. A re-ranker boosts best-sellers and items with good reviews. The page shows 20 results with matched words highlighted and brand facets, in about 50 ms.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why a scan doesn't work at this size**

The same catalogue holds 2 million products of about 2 KB of text each: 4 GB. Even reading RAM at 5 GB/s, one scan takes about 0.8 s on one core, and you still have to rank the matches. At 500 queries a second you'd need 400 cores scanning flat out. With an inverted index, the same query reads three postings lists of a few thousand entries each, a few hundred kilobytes, and finishes in milliseconds on one core.

</div>

## Common mistakes

- Using different analysis at index and query time (stemming one but not the other), so obvious queries return nothing.
- Treating the search engine as the source of truth, then losing data on a bad reindex.
- Running the expensive model on every candidate instead of using a funnel.
- Adding Elasticsearch for a 20,000-row table that the database could search.

## In an interview

Draw the two halves first: an indexing pipeline fed by CDC from the primary database, and a query path through query understanding, retrieval, scoring and re-ranking. Name the funnel sizes. Then dive where the interviewer points: sharding (lesson 09.11), freshness (lesson 09.10), ranking (lessons 09.05 and 09.08) or autocomplete (lesson 09.14).

<div class="take">
<div class="lab">Key takeaway</div>

search is an offline indexing pipeline (acquire, parse, analyse, build the inverted index) plus an online query pipeline (understand the query, retrieve candidates, score, re-rank, return). Both must analyse text the same way, each ranking stage narrows the candidates so expensive models run on few documents, and the search index is a derived copy of data that lives elsewhere.

</div>
