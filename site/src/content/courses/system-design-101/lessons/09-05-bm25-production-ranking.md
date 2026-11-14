---
id: "09.05"
course: "system-design-101"
module: "09"
slug: "bm25-production-ranking"
title: "BM25 production ranking"
keyIdeas: ["BM25 is the default ranking function in Lucene/Elasticsearch, an improved TF-IDF.", "**Term frequency saturation** (k1, ~1.2): the 10th occurrence of a word adds much less than the 1st, so keyword stuffing doesn't win.", "**Length normalization** (b, ~0.75): matches in short documents count more than in long ones, relative to the average document length.", "IDF works like TF-IDF's but with a slightly different formula.", "In practice BM25 is the first-stage scorer; machine-learned rankers often re-rank its top results."]
question: "What problem with TF-IDF does BM25's saturation (k1) fix?"
answer: "In tf-idf the score grows linearly with term frequency, so a document that repeats a word 50 times scores 50 times higher on that term than one that uses it once. That rewards keyword stuffing and over-weights long, repetitive documents, even though the 50th mention adds almost no evidence of relevance. BM25's term-frequency part, tf × (k1 + 1) / (tf + k1 × L), saturates: the first few occurrences raise the score a lot, later ones add less and less, and the contribution can never exceed k1 + 1 (2.2 with the default k1 = 1.2). Lower k1 saturates sooner; k1 = 0 counts only presence."
sources: [{"title": "Elastic: Practical BM25, part 2", "url": "https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables", "note": "k1 and b explained."}]
tools: ["Lucene"]
---

## What BM25 is

**BM25** ("Best Match 25", the 25th in a series of weighting schemes from the Okapi research system of the 1990s) is the default similarity function in Lucene, and therefore in Elasticsearch, OpenSearch and Solr. If you've used any of them, you've used BM25. It keeps tf-idf's two ideas (lesson 09.04): terms frequent in a document matter more, and terms common across the collection matter less. What it changes is **how term frequency and document length are counted**, fixing tf-idf's two main weaknesses.

The formula looks intimidating, so take it in parts:

`score(D, Q) = Σ IDF(qᵢ) × f(qᵢ, D) × (k1 + 1) / ( f(qᵢ, D) + k1 × (1 − b + b × |D| / avgdl) )`

- The sum runs over each term qᵢ in the query Q. Each term contributes independently, and the document's score is the total.
- `f(qᵢ, D)`: how many times query term qᵢ appears in document D. This is the term frequency, tf.
- `|D|`: the length of document D in terms (for the field being scored). `avgdl`: the average length across all documents in the index.
- `k1` and `b`: two tuning knobs, explained below.
- `IDF(qᵢ)`: Lucene uses `ln(1 + (N − n + 0.5) / (n + 0.5))`, where N is the number of documents and n is the number containing the term. Like tf-idf's log(N/df), it's large for rare terms and close to zero for terms in nearly every document. The `+ 0.5` terms smooth the edge cases, and the `1 +` inside the log keeps the value from ever going negative (the original Okapi formula could give negative IDF for terms in more than half the documents).

So each term's contribution is **IDF × a term-frequency part**, and the term-frequency part is where k1 and b live.

## k1: term frequency saturation

In plain tf-idf, a document that mentions "shoes" 100 times scores 100 times higher on that term than one that mentions it once. That's wrong on two counts. Relevance doesn't really grow that way: the tenth mention adds much less evidence than the first. And it rewards **keyword stuffing**: early web spammers repeated popular words hundreds of times in hidden text to climb rankings.

BM25's term-frequency part has the shape `tf × (k1 + 1) / (tf + k1)` (for an average-length document). Look at what it does:

- At tf = 0 it's 0.
- At tf = 1 it's exactly 1, whatever k1 is.
- As tf grows, the denominator grows almost as fast as the numerator, and the value **flattens towards an asymptote of k1 + 1**. It can never exceed that ceiling.

This is called **saturation**: the first few occurrences raise the score a lot, and later ones add less and less. k1 controls how quickly the curve saturates. The default is **k1 = 1.2**, so the ceiling is 2.2.

- A **higher** k1 (say 2 or 3) lets extra occurrences keep counting for longer; behaviour moves closer to raw tf.
- A **lower** k1 saturates sooner. At **k1 = 0**, term frequency is ignored entirely: only presence matters, and the part is 1 for any tf above 0.

~~~dg-bars BM25's term-frequency part for an average-length document with k1 = 1.2. It can never exceed k1 + 1 = 2.2, so stuffing a word stops paying.
tf 1 | 1.0 | 1.00
tf 2 | 1.375 | 1.38
tf 5 | 1.77 | 1.77
tf 10 | 1.96 | 1.96
*tf 100 | 2.17 | 2.17
raw tf-idf at tf 100 | 100 | 100
~~~

Going from 1 occurrence to 2 adds 0.38. Going from 10 to 100 adds 0.21. A spammer who repeats a word 100 times gets barely more than an honest page that uses it 10 times.

~~~dg-widget bm25 BM25 term-frequency saturation: each extra occurrence adds less, approaching k1 + 1. Longer documents need more occurrences to reach the same score.
~~~

## b: length normalisation

Now the second fix. A match in a 50-word product title means more than the same match in a 10,000-word manual that mentions everything. Long documents contain more words, so they match more queries and have higher term counts by sheer size. Some normalisation is needed, but how much? Sometimes long documents are long because they ramble; sometimes because they genuinely cover more.

BM25 compares each document's length with the average using the **length factor**:

`L = 1 − b + b × |D| / avgdl`

and multiplies k1 by it in the denominator. Look at the cases:

- A document of **exactly average** length has |D| / avgdl = 1, so L = 1, and nothing changes.
- A document **twice** the average length, with b = 0.75: L = 0.25 + 0.75 × 2 = 1.75. The effective k1 becomes 1.2 × 1.75 = 2.1, so the document needs more occurrences to earn the same score.
- A document **half** the average: L = 0.25 + 0.375 = 0.625. Effective k1 is 0.75, so it saturates quickly and a couple of mentions count for a lot.

The default is **b = 0.75**. **b = 0** turns length normalisation off entirely (L is always 1). **b = 1** applies it fully, in direct proportion to length.

~~~dg-compare Two documents scored for "shoes" with k1 = 1.2, b = 0.75 and an average length of 100. The short, focused one wins despite fewer mentions.
[*Doc A: 50 words]
2 mentions -(L = 0.625)-> tf part 1.60
+ Short and focused
+ Saturates after a few mentions
[Doc B: 400 words]
6 mentions -(L = 3.25)-> tf part 1.33
- Long, mentions spread thin
- Needs many more mentions to compete
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two documents for "shoes", worked in full**

Settings: avgdl = 100, k1 = 1.2, b = 0.75, so k1 + 1 = 2.2. Both documents get the same IDF for "shoes", so compare only the term-frequency parts.

Doc A: 50 words, "shoes" twice. L = 1 − 0.75 + 0.75 × (50 / 100) = 0.25 + 0.375 = 0.625. Denominator = 2 + 1.2 × 0.625 = 2.75. Numerator = 2 × 2.2 = 4.4. tf part = 4.4 / 2.75 = 1.60.

Doc B: 400 words, "shoes" 6 times. L = 0.25 + 0.75 × 4 = 3.25. Denominator = 6 + 1.2 × 3.25 = 9.9. Numerator = 6 × 2.2 = 13.2. tf part = 13.2 / 9.9 = 1.33.

Doc A wins despite three times fewer mentions, because it is short and focused. With b = 0 (no length normalisation), Doc A would score 4.4 / 3.2 = 1.38 and Doc B 13.2 / 7.2 = 1.83, and B would win.

</div>

## Tuning the knobs

Defaults work well across many collections, which is why they're defaults. But fields differ, and Elasticsearch lets you set k1 and b per field.

| Knob | Default | Raise it when | Lower it when |
| --- | --- | --- | --- |
| k1 | 1.2 | Repetition genuinely signals relevance (long articles, books) | Fields are short and repetition is noise (titles, tags) |
| b | 0.75 | Long documents ramble and should be penalised | Long documents are long because they have more to say (detailed specs) |

Typical adjustments:

- **Titles and names**: short, every word counts once. Lower k1 (0.5 or less) so one mention is nearly as good as several, and lower b because title length varies little.
- **Tags or keyword lists**: often b = 0 and low k1, since a tag either applies or it doesn't.
- **Long body text**: defaults, or a slightly higher k1.

Change knobs only with an evaluation set (lesson 09.09): a list of queries with known good results, measured before and after. Tuning by eyeballing a few queries usually makes others worse.

### Multi-field scoring

Real documents have several fields: title, body, brand, description. BM25 scores each field separately, using that field's own lengths and statistics. You then combine them: `best_fields` takes the best field's score (plus a bit of the others with `tie_breaker`), `most_fields` sums them, and `cross_fields` treats several fields as one big field so a query like "john smith" can match "john" in first name and "smith" in last name. Field **boosts** (title × 3) express that a title match is worth more than a body match. Getting field combination right often matters more than tuning k1 and b.

## Where BM25 sits in a real system

BM25 is cheap, needs no training data, is easy to explain, and is surprisingly hard to beat as a first-stage scorer. Production systems usually use it to pick the top few hundred or thousand candidates from the inverted index, then re-rank those with a learned model that adds popularity, freshness, personalisation and click signals (lesson 09.08). Many now also combine it with **vector search** (lesson 09.13), because BM25 only matches exact terms: it can't tell that "sneakers" and "trainers" mean the same thing.

~~~dg-flow Two-stage ranking: BM25 cheaply narrows millions of matches to a short list; an expensive model orders that list.
Millions of matches -(BM25 via postings)-> *Top 1,000 candidates -(learned re-ranker)-> Top 10 shown
~~~

BM25's per-term contribution has a known maximum (IDF × (k1 + 1)), which lets engines use dynamic pruning such as **block-max WAND**: skip blocks of postings whose best possible score can't beat the current 10th result. That's why Elasticsearch can return the top 10 BM25 results from millions of matches without scoring them all.

<div class="ex">
<div class="lab">Worked example</div>

**keyword stuffing, before and after**

A spam page repeats "cheap flights" 200 times in 1,000 words. A good travel page uses "cheap flights" 4 times in 300 words. avgdl = 500.

With raw tf × idf, the spam page's term weight is 50 times higher (200 versus 4). With BM25 (k1 = 1.2, b = 0.75): spam L = 0.25 + 0.75 × 2 = 1.75, tf part = 200 × 2.2 / (200 + 2.1) = 2.18. Good page L = 0.25 + 0.75 × 0.6 = 0.7, tf part = 4 × 2.2 / (4 + 0.84) = 1.82. The spam page's advantage shrinks from 50 times to about 1.2 times, small enough that quality signals in the re-ranker easily outweigh it.

</div>

## Variants worth knowing

- **BM25F**: combines fields before saturation (weights each field's tf, sums them, then saturates once), which is more principled than summing per-field scores. Elasticsearch's `combined_fields` query is based on it.
- **BM25+**: adds a small constant so that a very long document containing the term always scores more than one that doesn't contain it at all.

## Common mistakes

- Assuming BM25 scores are comparable across queries or indexes. They aren't probabilities; use them only to rank within one query.
- Tuning k1 and b globally when the real problem is field weighting or query understanding.
- Forgetting that scores use per-shard statistics; on small or skewed shards, IDF can differ between shards.
- Expecting BM25 to handle synonyms or misspellings. It matches analysed terms exactly.

## In an interview

Say "BM25 is tf-idf with two fixes: term frequency saturates (k1, default 1.2) so repetition stops paying, and length is normalised against the average document (b, default 0.75)." Sketch the saturating curve. Then place it: "first-stage retrieval with BM25, re-rank the top few hundred with a learned model, optionally fuse with vector search."

<div class="take">
<div class="lab">Key takeaway</div>

BM25 is tf-idf with saturation (k1, default 1.2, so repeated words approach a ceiling of k1 + 1 and stop adding much) and length normalisation relative to the average document (b, default 0.75). It is the default first-stage ranker in Lucene-based engines, usually followed by a learned re-ranker.

</div>
