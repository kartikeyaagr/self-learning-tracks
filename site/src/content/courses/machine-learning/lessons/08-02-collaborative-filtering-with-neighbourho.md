---
id: "08.02"
course: "machine-learning"
module: "08"
slug: "collaborative-filtering-with-neighbourhoods"
title: "Collaborative filtering with neighbourhoods"
keyIdeas: ["Neighbourhood methods predict from similar users (**user-kNN**) or similar items (**item-kNN**): $\\hat r_{ui} = \\bar r_u + \\frac{\\sum_{v} s_{uv}(r_{vi}-\\bar r_v)}{\\sum_v |s_{uv}|}$ over the $k$ most similar raters.", "Item-item similarity is usually preferred in production: item vectors are denser and change slowly, so the item-item table can be precomputed offline and served by lookup.", "Cosine, **adjusted cosine** (subtract each user's mean) and Pearson differ only in centring. Centring removes 'this user rates everything 5' effects.", "Similarities computed on two or three co-raters are noise; **shrinkage** $s' = \\frac{n}{n+\\beta}\\,s$ pulls them towards zero in proportion to how little evidence they have.", "kNN cannot recommend items with no interactions and scales with the number of co-occurring pairs, which is why it is often used as a retrieval source rather than the final model."]
question: "Your item-item 'similar products' rail for a phone case shows a different random accessory every night after the rebuild, and many shown items have only a handful of co-purchasers. What is causing the instability, and what two changes would you make?"
answer: "The similarities are being estimated from very few co-occurrences, so they are dominated by chance and change whenever a few new orders arrive. Binary cosine on small counts can be near 1 for items bought together once or twice. First, apply significance shrinkage, $s' = s \\cdot n/(n+\\beta)$, so pairs with little evidence are pulled towards zero and well-supported pairs rise to the top. Second, require a minimum co-occurrence count (for example, at least 5 co-purchasers) before a pair enters the table, and check that the similarity is normalised by item popularity so that a few universally bought accessories do not dominate. Optionally smooth across nights by computing over a longer window or blending with the previous table."
sources: [{"title": "Linden, Smith and York, Amazon.com Recommendations: Item-to-Item Collaborative Filtering (2003)", "url": "https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf", "note": "Four pages; the original argument for item-item over user-user at scale."}, {"title": "Google ML: Collaborative filtering basics", "url": "https://developers.google.com/machine-learning/recommendation/collaborative/basics", "note": "Worked toy example of user and item similarity and why CF needs no content features."}, {"title": "Dacrema, Cremonesi and Jannach, Are We Really Making Much Progress? (2019)", "url": "https://arxiv.org/abs/1907.06902", "note": "Shows tuned kNN baselines beating many published neural recommenders; read the results tables."}]
practice: ["item-item-cf-scores", "user-knn-rating-prediction", "cosine-similarity-matrix", "pearson-correlation-matrix"]
tags: ["collaborative filtering", "similarity"]
tools: ["NumPy"]
---

## The idea

Collaborative filtering (CF) uses only the interaction matrix: no item descriptions, no user profiles. The neighbourhood flavour is the most direct version of "people like you liked this". It is also the first thing to try, because it is easy to explain, has few hyperparameters and makes a strong retrieval source.

There are two mirror images:

- **User-based**: find the $k$ users most similar to $u$ who rated item $i$, and average their ratings of $i$, weighted by similarity.
- **Item-based**: find the $k$ items most similar to $i$ that $u$ has rated, and average $u$'s own ratings of them.

~~~dg-flow Item-based CF at serving time: the expensive similarity table is built offline; a request only does lookups and a weighted sum.
User history -(look up each item)-> Item-item table -(top-k neighbours)-> Candidate scores -(sum, drop seen)-> *Top-N list
~~~

## Similarity measures

Write $r_u$ for user $u$'s rating vector and $I_{uv}$ for the items both $u$ and $v$ rated. Three standard choices:

| Measure | Formula (on co-rated items) | What it ignores
| --- | --- | --- |
| Cosine | $\frac{r_u \cdot r_v}{\lVert r_u\rVert \lVert r_v\rVert}$ | nothing: a harsh rater and a generous one look different
| Pearson | cosine after subtracting each user's mean over $I_{uv}$ | each user's average level
| Adjusted cosine (item-item) | cosine between item columns after subtracting each **user's** mean | user rating scale when comparing items
| Jaccard (implicit) | $\frac{\lvert U_i \cap U_j\rvert}{\lvert U_i \cup U_j\rvert}$ | strength of interaction; only who touched what

For implicit data, ratings are replaced by binary or count values, and cosine on binary columns becomes $\frac{|U_i \cap U_j|}{\sqrt{|U_i||U_j|}}$: co-occurrence normalised by popularity. Without the normalisation every item looks most similar to the bestseller.

The prediction for user-based CF with mean-centring:

$$\hat r_{ui} = \bar r_u + \frac{\sum_{v \in N_k(u,i)} s_{uv}\,(r_{vi} - \bar r_v)}{\sum_{v \in N_k(u,i)} |s_{uv}|}$$

Centring matters: if your neighbour rates everything one star lower than average, their 4 stars is enthusiasm, and the formula converts it to "+1 above my mean" before adding it to yours.

## Shrinkage and significance

Two users who share exactly two rated films and agreed on both have a Pearson correlation of 1. That is coincidence, not similarity. The standard fix multiplies by an evidence factor:

$$s'_{ij} = \frac{n_{ij}}{n_{ij} + \beta}\, s_{ij}$$

where $n_{ij}$ is the number of co-raters and $\beta$ is around 10 to 100. A pair with 2 co-raters and $\beta = 20$ keeps 9% of its similarity; a pair with 500 co-raters keeps 96%.

```python
import numpy as np
# R: users x items, 0 = missing; implicit -> binary
B = (R > 0).astype(float)
co = B.T @ B                          # co-occurrence counts n_ij
norms = np.sqrt(np.diag(co))
S = co / (np.outer(norms, norms) + 1e-9)   # binary cosine
S *= co / (co + 20.0)                 # shrinkage, beta = 20
np.fill_diagonal(S, 0.0)
scores = B[u] @ S                     # sum of similarities to u's items
scores[B[u] > 0] = -np.inf            # never recommend what u already has
```

## Why item-item wins in production

Amazon's 2003 description of item-to-item CF set the pattern that most retailers still follow. The reasons are practical:

- **Stability**: an item's audience changes slowly, a user's taste vector changes with every click. The item table can be rebuilt nightly and stay valid; a user-user table would be stale within hours.
- **Density**: a popular item has thousands of raters, so item similarities are better estimated than similarities between two users who each rated 20 things.
- **Cost at request time**: with the table precomputed and truncated to the top 50 neighbours per item, a request touches (items in history) × 50 entries.
- **Explanations**: "because you bought X" falls out for free.

The offline cost is computing co-occurrence for all item pairs. It is dominated by heavy users: a user with $m$ interactions contributes $m^2$ pairs, so it is common to cap or subsample very long histories.

<div class="ex">
<div class="lab">Worked example</div>

**shrinkage changes the answer**

Item A (a niche documentary) shares 3 viewers with item B and has cosine 0.9. Item C shares 400 viewers with B and has cosine 0.35. With $\beta = 50$: A gets $0.9 \cdot 3/53 = 0.051$, C gets $0.35 \cdot 400/450 = 0.311$. Without shrinkage, B's "similar items" rail would lead with a film three people happened to watch.

</div>

## Limits

Neighbourhood methods have no notion of latent taste dimensions, so two users who like different but related items (two sci-fi fans with no film in common) never become neighbours. They cannot score items without interactions (cold start), and they do not use content. Matrix factorisation (next lesson) addresses the first problem by learning a dense low-dimensional space; two-tower models (08.04) address the last two by putting features into the towers.

<div class="take">
<div class="lab">Key takeaway</div>

item-item CF with a popularity-normalised similarity and shrinkage is a cheap, explainable and surprisingly hard-to-beat retrieval source; most of its quality comes from the normalisation and shrinkage, not the choice of k.

</div>
