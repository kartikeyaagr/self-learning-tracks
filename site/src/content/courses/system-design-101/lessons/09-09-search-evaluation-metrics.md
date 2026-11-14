---
id: "09.09"
course: "system-design-101"
module: "09"
slug: "search-evaluation-metrics"
title: "Search evaluation metrics"
keyIdeas: ["**Precision** = relevant results returned ÷ results returned. **Recall** = relevant results returned ÷ all relevant results.", "Ranked metrics: precision@k, MAP (mean average precision), **NDCG** (rewards putting highly relevant results near the top, with graded relevance), MRR (where the first relevant result appears).", "Offline evaluation uses judged query-document pairs; online evaluation uses A/B tests and interleaving on click and conversion metrics.", "Recall matters more for legal discovery, medical literature or security searches, where missing a relevant item is costly."]
question: "When is recall more important than precision?"
answer: "Recall matters more when missing a relevant document is costly and reading extra results is cheap by comparison: legal discovery (paralegals must find every relevant document), intelligence analysis, patent prior-art and medical systematic-review searches, compliance checks, finding one specific email in your own mailbox, and the first candidate-generation stage of a ranking pipeline, where anything not retrieved can never be ranked later. Web and shopping search usually favour precision because users only look at the first few results."
sources: [{"title": "IR book ch.8: Evaluation", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "Precision, recall, MAP, NDCG."}]
---

## Why you need numbers

Every change to a search system (a new analyser, a synonym list, BM25 settings, a re-ranker) makes some queries better and some worse. Looking at a handful of queries by eye is how teams fool themselves: the queries you think of are the ones you just fixed. To know whether a change helps overall, you need **metrics** computed over many queries, and to choose the right metric you need to know what users lose when the system is wrong.

Evaluation starts with a **test collection**: a set of documents, a set of queries (ideally sampled from real logs, covering frequent and rare ones), and **relevance judgments** saying which documents are relevant to each query. Judgments come from trained human raters, or are derived from user behaviour such as clicks and purchases (lesson 09.08).

## Why accuracy is the wrong measure

The obvious measure, **accuracy** (the fraction of documents classified correctly as relevant or not), is useless for search. For a typical query, well over 99.9% of documents are irrelevant. A system that returns **nothing at all** labels every document "not relevant" and is 99.9% accurate. Search needs measures that look only at the relevant documents and at what was retrieved, ignoring the vast pool of correctly ignored documents.

## Precision and recall

For one query, split all documents by two questions: was it retrieved, and is it relevant?

~~~dg-matrix The four outcomes for one query. Precision and recall ignore the huge true-negative cell.
Outcome | Relevant | Not relevant
Retrieved | true positive (tp) | false positive (fp)
Not retrieved | false negative (fn) | true negative (tn)
~~~

- **Precision** = tp / (tp + fp): of what I returned, how much was relevant?
- **Recall** = tp / (tp + fn): of everything relevant, how much did I return?

Precision is about **noise** in the results. Recall is about **gaps**: things that should have been found and weren't.

<div class="ex">
<div class="lab">Worked example</div>

**computing both**

A collection holds 30 documents relevant to "contract termination clause". The system returns 20 results, of which 12 are relevant. Precision = 12 / 20 = 60%: 8 of the 20 results are noise. Recall = 12 / 30 = 40%: 18 relevant documents were missed. Accuracy over a million documents would be (1,000,000 − 8 − 18) / 1,000,000 = 99.997%, which tells you nothing.

</div>

### They pull against each other

Recall **never decreases** as you return more documents: returning the whole collection gives 100% recall and terrible precision. Returning one sure-fire result can give 100% precision and terrible recall. Most tuning moves along that trade-off: looser matching (OR instead of AND, synonyms, spelling correction, stemming, vector search) raises recall and usually lowers precision; stricter matching does the opposite.

~~~dg-bars How one query's precision and recall change as the system returns more results. Recall only rises; precision usually falls.
P at 5 results | 80 | precision 80%
R at 5 results | 13 | recall 13%
P at 20 results | 60 | precision 60%
R at 20 results | 40 | recall 40%
P at 100 results | 25 | precision 25%
*R at 100 results | 83 | recall 83%
~~~

## When recall matters more

Which measure matters depends on the cost of each kind of mistake for this user.

- **Precision matters** for web search and shopping. Users look at the first page, maybe the first few results, and don't care whether the 5,000th relevant page was found. Irrelevant results on page one cost attention and trust.
- **Recall matters** when missing a relevant document is costly and reading extra results is cheap by comparison:
  - **Legal discovery**: paralegals must find every relevant email or contract in a lawsuit. Missing one can lose the case or bring sanctions; reading a few hundred extra documents is just work.
  - **Intelligence analysis**: an analyst must find every message mentioning a threat.
  - **Patent prior-art search**: one missed earlier patent can invalidate a new one.
  - **Medical systematic reviews**: a review must include every relevant clinical trial, or its conclusions are biased.
  - **Compliance and safety**: finding every transaction matching a sanctions rule, every product affected by a recall.
  - **The candidate-retrieval stage** of any two-stage ranker (lesson 09.05): whatever isn't retrieved can never be re-ranked up, so the first stage should favour recall and leave precision to the re-ranker.
  - **Search over your own data**: "find my email from the plumber" has exactly one relevant result, and failing to return it is total failure.

Professional searchers in recall-heavy fields will read many results, write long Boolean queries and iterate. That's why legal and patent search tools still offer Boolean operators (lesson 09.03).

## The F measure

To compare systems with one number, use the **F measure**, a weighted **harmonic mean** of precision and recall. The balanced version is

`F1 = 2PR / (P + R)`

Why harmonic rather than a plain average? Because the harmonic mean punishes imbalance. A system that returns one relevant document out of 50 has P = 1.0 and R = 0.02. The arithmetic mean says 0.51, which looks respectable. F1 says about 0.04, which is honest. You can't game F1 by maximising one side.

The general form `Fβ = (1 + β²)PR / (β²P + R)` lets you weight the trade: **β > 1 favours recall** (F2 counts recall as twice as important), **β < 1 favours precision** (F0.5).

<div class="ex">
<div class="lab">Worked example</div>

**two systems for one query**

There are 10 relevant documents. System A returns 5 results, 4 relevant: P = 0.8, R = 0.4, F1 = 2 × 0.8 × 0.4 / 1.2 = 0.53. System B returns 20 results, 8 relevant: P = 0.4, R = 0.8, F1 = 0.53.

Same F1, different products. A suits a shopper who wants a short, clean list. B suits a paralegal who must not miss anything. With F2 (favouring recall), A scores 0.44 and B scores 0.67; with F0.5, A scores 0.67 and B 0.44. Choosing β is choosing your user.

</div>

## Measures for ranked results

Precision and recall treat results as an unordered **set**. Users see a **ranked list**, and a relevant document at position 1 is worth far more than one at position 40. Measures that respect order:

### Precision at k

**P@k** is precision among the top k only. P@10 matches what a user sees on one page. It's easy to explain but ignores order within the top k and ignores how many relevant documents exist.

### Recall at k

**R@k** is the share of all relevant documents found in the top k. It's the standard metric for first-stage retrieval: "does the right answer appear anywhere in the top 1,000 candidates?"

### Mean reciprocal rank

**MRR** looks at the rank of the **first** relevant result: score 1/rank (1 for position 1, 0.5 for position 2, 0.33 for 3), averaged over queries. Good for navigational queries and question answering, where one right answer is all the user needs.

### Mean average precision

**MAP**: for each query, compute precision at each rank where a relevant document appears, average those values (counting relevant documents never retrieved as zero), then average over all queries. It rewards putting relevant results early and finding all of them.

<div class="ex">
<div class="lab">Worked example</div>

**average precision for one query**

There are 4 relevant documents. The system places relevant ones at ranks 1, 3 and 6, and misses the fourth. Precision at those ranks: 1/1 = 1.0, 2/3 = 0.67, 3/6 = 0.5. Average precision = (1.0 + 0.67 + 0.5 + 0) / 4 = 0.54. Swapping the rank-6 document with the irrelevant one at rank 2 puts relevant documents at ranks 1, 2 and 3: (1.0 + 1.0 + 1.0 + 0) / 4 = 0.75.

</div>

### NDCG

**NDCG** (normalised discounted cumulative gain) is the standard metric for graded relevance and the one most ranking teams report.

1. Give each result a **gain** from its grade, for example perfect 3, good 2, fair 1, bad 0.
2. **Discount** each gain by its position: divide by log2(rank + 1). Rank 1 divides by 1, rank 2 by 1.58, rank 4 by 2.32, rank 10 by 3.46.
3. Sum to get **DCG**.
4. Divide by the DCG of the **ideal** ordering (all results sorted best first) to get NDCG between 0 and 1.

~~~dg-flow DCG discounts each result's gain by its rank, so a great result at rank 1 counts more than at rank 4.
Rank 1: gain 3 / 1 -> Rank 2: gain 2 / 1.58 -> Rank 3: gain 0 -> Rank 4: gain 3 / 2.32 -> *DCG = 5.55
~~~

For that list (grades 3, 2, 0, 3), DCG = 3 + 1.26 + 0 + 1.29 = 5.55. The ideal order is 3, 3, 2, 0: DCG = 3 + 1.89 + 1.0 + 0 = 5.89. NDCG = 5.55 / 5.89 = 0.94. Moving the perfect result from rank 4 up to rank 2 (pushing the others down one place) gives the ideal order and NDCG 1.0.

~~~dg-matrix Which ranked metric suits which situation.
Metric | Uses grades | Respects order | Best for
P@k | no | partial, only the cut-off | first page quality
R@k | no | no | candidate retrieval
MRR | no | yes, first hit only | one right answer
MAP | no | yes | binary judgments, full lists
NDCG | yes | yes | graded relevance, ranking teams
~~~

## Online metrics

Offline metrics need judgments, which are expensive and can go stale. Online, teams also run A/B tests and interleaving (lesson 09.08) and track behaviour:

- **Click-through rate** on the results page, and **time to first click**.
- **Zero-result rate**: share of searches returning nothing. A direct recall failure.
- **Abandonment** and **reformulation rate**: searches with no click, or immediately followed by a rephrased query. Signs the results missed.
- **Conversion** and **revenue per search** for shopping.
Offline and online metrics should move together. When they don't, the judgments are usually the problem.

## Common mistakes

- Reporting accuracy for retrieval.
- Optimising average precision over all queries when a few high-traffic queries dominate business value; weight by traffic, or report head and tail separately.
- Tuning to the same judgment set you evaluate on, until you overfit it. Hold some out.
- Using P@10 to judge the first stage of a two-stage system, where R@1000 is what matters.

## In an interview

When asked "how do you know search is good?", name precision and recall, explain the trade-off, then choose: "P@10 and NDCG@10 for the shopper-facing ranking, recall@1000 for candidate retrieval, and online zero-result rate, reformulation rate and conversion through A/B tests." Then give one recall-first domain to show you know the cost of a miss differs by user.

<div class="take">
<div class="lab">Key takeaway</div>

precision is the share of returned results that are relevant; recall is the share of relevant documents that were returned, and they trade against each other. F1 combines them, and ranked measures (P@k, MRR, MAP, NDCG) reward putting relevant results first. Choose the measure that matches the user's cost of missing something versus the cost of reading noise.

</div>
