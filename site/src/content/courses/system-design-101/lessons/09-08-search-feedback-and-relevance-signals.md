---
id: "09.08"
course: "system-design-101"
module: "09"
slug: "search-feedback-and-relevance-signals"
title: "Search feedback and relevance signals"
keyIdeas: ["User behaviour shows what's relevant: clicks, dwell time, add-to-cart, purchases, and query reformulations.", "These signals feed ranking features (click-through rate per query-doc pair) and training data for learning-to-rank models.", "Clicks are biased: **position bias** (people click top results more regardless of quality), presentation bias (results with images get more clicks), and popularity loops.", "Correct with click models, randomized interleaving experiments, or inverse propensity weighting.", "Learning to rank: a model (often gradient boosted trees, e.g. LambdaMART) re-ranks the top-k from BM25 using many features."]
question: "Why are clicks a biased relevance signal? Name one bias."
answer: "Clicks depend on where and how a result was shown, not only on how relevant it is. The main example is position bias: people click higher results much more often simply because they look at them first and often stop early, so a mediocre top result can collect more clicks than an excellent one at position 5. Other examples are presentation bias (images and ratings attract clicks), selection bias (results never shown get no clicks) and clickbait (a click followed by a quick return). Training on raw clicks would teach the model to keep ranking whatever was already on top, so clicks are reweighted by the probability each position was examined, combined with stronger signals like purchases, and supplemented with a little exploration."
sources: [{"title": "Query Understanding: relevance feedback", "url": "https://queryunderstanding.com/", "note": "Clicks and reformulations as signals."}, {"title": "Elastic: Learning To Rank", "url": "https://www.elastic.co/docs/solutions/search/ranking/learning-to-rank-ltr", "note": "Training rankers from signals."}]
---

## What feedback is

BM25 (lesson 09.05) ranks by text alone. It knows nothing about which results people actually found useful. Every day, though, your users run millions of queries and react to the results: they click, skip, buy, come back, rephrase. That behaviour is the richest relevance signal you have. **Relevance feedback** is the general name for using people's reactions to improve ranking.

It comes in two kinds:

- **Explicit feedback**: the user says which results were good, for example by marking them relevant, rating them, or answering "Was this helpful?". It's accurate but rare, because few users bother. Human raters hired to judge results are also explicit feedback, and they're expensive.
- **Implicit feedback**: the engine infers relevance from behaviour. A click suggests a result looked relevant. A result skipped while a lower one was clicked suggests it looked worse. A purchase, a long dwell time on the page, or a session that ends after the click (the user found what they wanted) are stronger signals. Implicit feedback is plentiful and free, but noisy and biased, as this lesson explains.

## Rocchio and query modification

The oldest technique is **Rocchio's algorithm**, from the SMART retrieval system of the 1960s. It works in the vector space model (lesson 09.04): documents and queries are vectors of term weights. After the user marks some results relevant and others not, Rocchio moves the query vector **towards** the centroid (average) of the relevant documents and **away from** the centroid of the non-relevant ones:

`q' = α·q + β·mean(relevant) − γ·mean(non-relevant)`

Typical weights are α = 1, β = 0.75, γ = 0.15: the original query matters most, positive evidence a lot, negative evidence a little. The new query picks up terms common in the relevant documents, so a search for "jaguar" where the user marked car pages relevant gains "engine", "sedan" and "dealer", and drifts away from "habitat" and "rainforest".

A variant, **pseudo-relevance feedback** (or blind feedback), skips the user entirely: assume the top 10 results are relevant, extract their most distinctive terms, add them to the query and search again. It often improves recall, but if the top results were wrong, it drifts further off-topic (**query drift**).

Lucene's **MoreLikeThis** query is a modern relative: given a document the user liked, it picks that document's most distinctive terms (high tf-idf) and builds a query from them, so you can show "similar items".

Today the main use of feedback is not re-querying in the moment. It's **training data for ranking models**.

## Learning to rank

**Learning to rank** (LTR) means training a machine-learning model to order results, using examples of good and bad orderings. The standard setup, as in Elasticsearch and OpenSearch:

1. A cheap **first-stage** query (BM25, filters, maybe vector search) retrieves candidates.
2. A trained model **re-scores the top N results** (say 100 to 1,000). It runs as a **rescorer**, not over the whole index, because evaluating hundreds of features per document is expensive.
3. The final order shown to the user comes from the model.

~~~dg-flow Learning to rank: logs and judgments train a model offline; the model re-ranks the first stage's top results online.
[Offline] Click logs and raters -> Judgment list -(features per query and doc)-> *Train model
[Online] Query -(BM25)-> Top N candidates -(rescore with model)-> Final order
~~~

### The pieces

- **Judgment list**: a set of queries, each with candidate documents and a relevance **grade**, such as 0 (bad) to 4 (perfect). Grades come from human raters, or are derived from user behaviour such as clicks and purchases (after correcting for bias, below). A useful list has thousands of queries, sampled across frequent ("head") and rare ("tail") queries.
- **Features**, of three kinds:
  - **Document features**, independent of the query: popularity, price, recency, rating, number of reviews, return rate.
  - **Query features**, independent of the document: query length, detected category, whether it contains a brand.
  - **Query-document features**: how well this document matches this query, such as the BM25 score of the title field, BM25 of the description, whether the brand matches, the vector similarity, the historical click-through rate of this document for this query.
- **Model**: typically **gradient-boosted decision trees** trained with the **LambdaMART** objective, which directly optimises a ranking metric such as NDCG (lesson 09.09) rather than predicting each grade independently. Trained with XGBoost or LightGBM outside the search engine, then uploaded to it. Neural re-rankers (cross-encoders that read query and document together) are increasingly common for text-heavy search.

~~~dg-tree The three families of learning-to-rank features, with typical examples of each.
Features for one query-document pair
  Document only
    popularity, sales last 30 days
    rating, review count
    age, freshness
  Query only
    query length
    detected category
  *Query and document together
    BM25 title, BM25 body
    brand match, category match
    past click rate for this pair
~~~

<div class="ex">
<div class="lab">Worked example</div>

**building judgments from a click log**

For the query "wireless earbuds", the logs over 30 days hold 80,000 searches. For each product shown, compute a label from behaviour: purchased after click = grade 4, added to cart = 3, clicked with long dwell = 2, clicked and bounced back in under 10 seconds = 1, shown but never clicked (after position correction) = 0. Repeat for the top 5,000 queries. For each (query, product) pair, compute 60 features. Train LambdaMART on 80% of queries, test on 20%, and compare NDCG@10 with plain BM25. If it improves from 0.61 to 0.68 offline, run an A/B test before rolling out.

</div>

## Why clicks are biased

Clicks are the cheapest implicit signal and the most dangerous. A click depends not only on whether a result is relevant, but also on **whether the user saw it** and **how attractive it looked**. Several biases distort them.

| Bias | What happens | Common correction |
| --- | --- | --- |
| Position bias | Users click higher results far more often regardless of quality, because they read top-down and often stop early | Estimate how likely each position is to be examined and reweight clicks by its inverse; randomise order slightly for a small share of traffic |
| Presentation bias | Results with images, ratings, badges or bold snippets attract clicks | Include presentation as a feature so the model can separate it |
| Selection bias | You only get feedback on results you showed; anything never shown gets no clicks | Occasional exploration: show new items in some slots |
| Trust bias | Users trust the engine and click its top result even when a lower one is better | Combine with post-click signals |
| Click is not satisfaction | Misleading titles get clicks followed by a quick return | Use dwell time, purchases or no return to results as stronger signals |

~~~dg-bars Typical share of users who even look at each result position on a list page. Clicks at lower positions are rare largely because few people get that far.
Position 1 | 95 | ~95% examine
Position 2 | 70 | ~70%
Position 3 | 50 | ~50%
Position 5 | 30 | ~30%
*Position 10 | 12 | ~12%
~~~

### The feedback loop

The danger of ignoring bias is a **self-reinforcing loop**: the model learns that whatever was on top is good, ranks it on top again, it collects more clicks because it's on top, and the model becomes more certain. New or better items never get shown high enough to collect the clicks they'd need to rise. Rankings freeze.

~~~dg-flow The position-bias loop: training on raw clicks teaches the model to keep whatever it already ranked first.
Result shown first -(gets most clicks)-> Clicks used as labels -(model learns)-> *Ranked first again
~~~

### Correcting it

- **Inverse propensity weighting**: estimate the **propensity** (probability of being examined) for each position, then weight each click by 1 / propensity. A click at position 8, where only 15% of users look, counts about six times as much as a click at position 1. Propensities are estimated with small randomisation experiments: swap results between positions for a tiny fraction of traffic and see how click rates change.
- **Click models**: probabilistic models of how users scan a page (the **cascade model** assumes users read top-down and stop at the first satisfying result) that separate "examined" from "attractive" from "satisfying".
- **Pairwise preferences**: treat "skipped result 2, clicked result 3" as evidence that 3 beats 2 for this query, which is less sensitive to absolute position effects.
- **Exploration**: reserve a small share of slots or traffic for items the model is unsure about, so new items get a chance (a bandit approach).

<div class="ex">
<div class="lab">Worked example</div>

**reading a click log**

For "wireless earbuds", result 1 gets 40% of clicks and result 3 gets 12%. Raw clicks say result 1 is more than three times as attractive. But randomisation experiments on your page (where tall product cards push result 3 below the fold, so it is seen less than the typical curve above suggests) show position 1 is examined by 95% of users and position 3 by 30%. Divide: result 1's click rate among users who looked is 40 / 95 ≈ 42%; result 3's is 12 / 30 = 40%. They are almost equally attractive. The raw numbers mostly measured position. Training on raw clicks would have taught the model that result 1 is far better, keeping it on top forever.

</div>

## Online evaluation

Offline metrics on judgment lists (lesson 09.09) guide development, but the final test is live behaviour. Two common methods:

- **A/B tests**: split users into groups, give each a different ranker, and compare business metrics (conversion, revenue per search, zero-result rate, reformulation rate) over a week or two.
- **Interleaving**: merge results from rankers A and B into one list for the same user, and see whose results get clicked more. It needs far less traffic than an A/B test to detect a difference, because each user compares both directly.

## Common mistakes

- Training on raw clicks without correcting for position, creating a feedback loop.
- Using clicks alone when purchases or dwell time are available; clickbait wins.
- Running the learned model over the whole index instead of re-scoring the top N.
- Features that leak the label, such as "was clicked in the same visit", so offline metrics look great and online results don't move.
- Never exploring, so new products can't earn their way up.

## In an interview

When asked how to use clicks to improve search, say: "log impressions, clicks and conversions with positions; turn them into graded labels with position-bias correction; train a LambdaMART re-ranker on document, query and query-document features; re-score BM25's top few hundred; validate with NDCG offline and an A/B test or interleaving online." Then name position bias as the main trap.

<div class="take">
<div class="lab">Key takeaway</div>

feedback can be explicit (rare, accurate) or implicit (clicks, skips, purchases). Today it mostly trains learning-to-rank models that re-score the first stage's top results using document, query and query-document features. Raw clicks are biased, above all by position, and must be corrected before being used as labels, or the ranker locks in its own past choices.

</div>
