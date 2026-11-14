---
id: "08.02"
course: "machine-learning"
module: "08"
slug: "collaborative-filtering-with-neighbourhoods"
title: "Collaborative filtering with neighbourhoods"
keyIdeas: ["Collaborative filtering predicts from the interaction matrix alone. Neighbourhood methods use similar users (**user-kNN**) or similar items (**item-kNN**): $\\hat r_{ui} = \\bar r_u + \\frac{\\sum_{v} s_{uv}(r_{vi}-\\bar r_v)}{\\sum_v |s_{uv}|}$ over the $k$ most similar raters of the item.", "Cosine, **adjusted cosine** (subtract each user's mean) and Pearson differ only in centring. Centring removes 'this user rates everything 5' effects; without it, raw star ratings make even opposite tastes look similar.", "Similarities computed on two or three co-raters are noise; **shrinkage** $s' = \\frac{n}{n+\\beta}\\,s$ pulls them towards zero in proportion to how little evidence they have.", "For implicit data, divide co-occurrence by popularity (binary cosine $\\frac{|U_i \\cap U_j|}{\\sqrt{|U_i||U_j|}}$ or Jaccard); raw counts make every item look most similar to the bestseller.", "Item-item similarity is usually preferred in production: item vectors are denser and change slowly, so the table can be precomputed offline and served by lookup, with 'because you bought X' explanations for free.", "kNN cannot recommend items with no interactions and scales with the number of co-occurring pairs, which is why it is often used as a retrieval source rather than the final model."]
question: "Your item-item 'similar products' rail for a phone case shows a different random accessory every night after the rebuild, and many shown items have only a handful of co-purchasers. What is causing the instability, and what two changes would you make?"
answer: "The similarities are being estimated from very few co-occurrences, so they are dominated by chance and change whenever a few new orders arrive. Binary cosine on small counts can be near 1 for items bought together once or twice: two items with 2 buyers each who are the same 2 people score a perfect 1.0, ahead of a genuinely related item backed by hundreds of co-purchases.\n\nFirst, apply significance shrinkage, $s' = s \\cdot n/(n+\\beta)$, with $\\beta$ somewhere around 10 to 100 tuned offline. Pairs with little evidence are pulled towards zero (2 co-buyers with $\\beta = 20$ keep only 9% of their score) and well-supported pairs rise to the top. Second, require a minimum co-occurrence count (for example, at least 5 co-purchasers) before a pair enters the table at all. While doing this, check that the similarity is normalised by item popularity (cosine or Jaccard rather than raw counts), so that a few universally bought accessories do not take over once the noisy pairs are gone.\n\nTo confirm the fix, compare the top-10 lists of consecutive nightly tables (for example, their overlap) and track a held-out ranking metric on a time split. Optionally smooth across nights by computing over a longer window or blending the new table with the previous one."
sources: [{"title": "Linden, Smith and York, Amazon.com Recommendations: Item-to-Item Collaborative Filtering (2003)", "url": "https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf", "note": "Four pages; the original argument for item-item over user-user at scale."}, {"title": "Google ML: Collaborative filtering basics", "url": "https://developers.google.com/machine-learning/recommendation/collaborative/basics", "note": "Worked toy example of user and item similarity and why CF needs no content features."}, {"title": "Dacrema, Cremonesi and Jannach, Are We Really Making Much Progress? (2019)", "url": "https://arxiv.org/abs/1907.06902", "note": "Shows tuned kNN baselines beating many published neural recommenders; read the results tables."}]
practice: ["item-item-cf-scores", "user-knn-rating-prediction", "cosine-similarity-matrix", "pearson-correlation-matrix"]
tags: ["collaborative filtering", "similarity"]
tools: ["NumPy"]
---

## What you will learn

- What collaborative filtering is, and why it needs nothing but the interaction matrix.
- How to measure how alike two users or two items are (cosine, centred cosine, Pearson, Jaccard), and why centring matters.
- How to turn neighbours' ratings into a prediction, by hand on a five-by-five matrix and then in NumPy.
- Why production systems mostly run item-item CF, including on implicit clicks and purchases.
- How shrinkage and popularity normalisation stop flukes and bestsellers from taking over your lists.

*Before you start:* you should know the rating matrix, explicit versus implicit feedback and the retrieval funnel from lesson 08.01. You also need dot products and vector length; cosine similarity is rebuilt here from scratch.

## From the problem to a first method

Lesson 08.01 set up the task: fill in a huge, mostly empty user-by-item matrix well enough to rank a few items at the top of each list. This lesson gives the oldest, most direct way to fill a blank: to guess how Ana will rate a film, find people whose past ratings look like hers and see what they gave it.

This is **collaborative filtering (CF)**: each user's history helps predict the others'. It uses only the interaction matrix, with no genres or user profiles. The **neighbourhood** flavour (also called **kNN** CF) looks up neighbours at prediction time instead of training a compact model. It is easy to explain, has few knobs, and is the first thing to try.

## The idea in one picture

Five friends have rated five films on a 1 to 5 star scale. Ana has not seen Dune. Should a service recommend it to her?

| User | Alien | Blade Runner | Dune | Casablanca | Notting Hill
| --- | --- | --- | --- | --- | --- |
| Ana | 5 | 4 | ? | 1 | 2
| Ben | 4 | 4 | 5 | 1 | 1
| Cai | 2 | 1 | 2 | 5 | 5
| Dee | 3 | 2 | 3 | 1 | 1
| Eli | 3 | ? | 3 | 5 | 5

~~~dg-fig 08.02-matrix.svg
Left: the ratings, with Ana's missing Dune rating outlined. Right: how much each other user's taste resembles Ana's. Ben and Dee rise and fall with her; Cai and Eli do the opposite.
~~~

By eye: Ana loves the science-fiction films and dislikes the romances. Ben and Dee share that pattern; Cai and Eli are the reverse. Ben gave Dune 5, and Dee gave it 3, which is high for someone who rarely gives more than 2. So Ana will probably like Dune; the hand example below turns that into **4.5 stars**.

Two words used throughout:

- A **neighbour** of Ana is another user whose ratings resemble hers. The $k$ most similar ones who have rated the target film form her **neighbourhood**.
- The **co-rated** items of two users are the items both of them rated. Similarity can only be computed on those.

## Step 1: write tastes as vectors

Each user is a row: a vector with one entry per item, many missing. Write $r_{ui}$ for user $u$'s rating of item $i$, and $I_{uv}$ for the items both $u$ and $v$ rated. Ana's row is $(5, 4, ?, 1, 2)$.

Missing is not zero. Treating a missing rating as "0 stars" would say Ana hates Dune. Every formula below sums only over observed entries.

## Step 2: measure similarity, and centre first

The natural way to compare two vectors is the angle between them. **Cosine similarity** is

$$\text{cos}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui}\, r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2}\,\sqrt{\sum_{i \in I_{uv}} r_{vi}^2}},$$

the dot product divided by the two lengths, on co-rated items: 1 when the vectors point the same way, 0 when perpendicular.

On raw star ratings this goes wrong. All ratings are positive, so every pair of vectors looks fairly similar. Ana and Cai have opposite tastes, yet their raw cosine is $(10 + 4 + 5 + 10)/(\sqrt{46}\sqrt{55}) = 29/50.3 = +0.58$.

The fix is **centring**: subtract each user's average rating first. Ana's mean is 3, so her centred ratings are $(+2, +1, -2, -1)$. Positive now means "above my usual", whatever scale the user habitually uses.

~~~dg-fig 08.02-centring.svg
Left: raw ratings for Ana, Dee and Cai. Dee rates everything low and Cai looks loosely similar to Ana. Right: after subtracting each user's mean, Dee's line has the same shape as Ana's, and Cai's goes the other way.
~~~

The **centred cosine** between Ana and Dee (Dee's mean is 2, so her co-rated ratings become $(+1, 0, -1, -1)$) is

$$\frac{2\cdot 1 + 1 \cdot 0 + (-2)(-1) + (-1)(-1)}{\sqrt{10}\,\sqrt{3}} = \frac{5}{5.48} = 0.913,$$

and between Ana and Cai it is $-0.877$: centring turned a misleading $+0.58$ into a clear "opposite taste".

The common measures differ only in what they centre and what they count:

| Measure | Formula (on co-rated items) | What it ignores
| --- | --- | --- |
| Cosine | $\frac{r_u \cdot r_v}{\lVert r_u\rVert \lVert r_v\rVert}$ | nothing: a harsh rater and a generous one look different
| Pearson | cosine after subtracting each user's mean over $I_{uv}$ | each user's average level
| Adjusted cosine (item-item) | cosine between item columns after subtracting each **user's** mean | user rating scale when comparing items
| Jaccard (implicit) | $\frac{\lvert U_i \cap U_j\rvert}{\lvert U_i \cup U_j\rvert}$ | strength of interaction; only who touched what

Our "centred cosine" subtracts each user's mean over *all* their ratings; strict Pearson uses the mean over co-rated items only (here it gives Ben and Dee 0.95 instead of 0.90 and 0.91). Both remove the "this user rates everything 5" effect.

## Step 3: combine the neighbours' opinions

Now predict. Take the $k$ users most similar to $u$ among those who rated item $i$; call that set $N_k(u, i)$. Each neighbour $v$ reports how far above or below their own mean they rated $i$. Average those reports, weighted by similarity, and add the result to $u$'s own mean:

$$\hat r_{ui} = \bar r_u + \frac{\sum_{v \in N_k(u,i)} s_{uv}\,(r_{vi} - \bar r_v)}{\sum_{v \in N_k(u,i)} |s_{uv}|}$$

What each symbol means:

- $\hat r_{ui}$ is the predicted rating; $\bar r_u$ and $\bar r_v$ are the users' mean ratings.
- $s_{uv}$ is the similarity from step 2; it is the weight a neighbour's opinion gets.
- $r_{vi} - \bar r_v$ is the neighbour's enthusiasm for $i$, measured on their own scale.
- Dividing by $\sum |s_{uv}|$ makes the fraction a weighted average, so the prediction stays on the rating scale.

<div class="ex">
<div class="lab">Worked example</div>

**predict Ana's rating of Dune by hand**

Means: Ana 3, Ben $(4+4+5+1+1)/5 = 3$, Cai 3, Dee $(3+2+3+1+1)/5 = 2$, Eli $(3+3+5+5)/4 = 4$.

Similarities to Ana on co-rated items (centred cosine). Ben's centred ratings on Ana's four films are $(1, 1, -2, -2)$, so $s = (2 + 1 + 4 + 2)/(\sqrt{10}\sqrt{10}) = 9/10 = 0.900$. Dee: $0.913$ (computed above). Cai: $-0.877$. Eli, who shares only three films with Ana: $-0.962$.

Take $k = 2$, the two most similar raters of Dune: Dee (0.913) and Ben (0.900).

Their enthusiasm for Dune: Ben $5 - 3 = +2$; Dee $3 - 2 = +1$.

Weighted average: $(0.900 \times 2 + 0.913 \times 1)/(0.900 + 0.913) = (1.800 + 0.913)/1.813 = 2.713/1.813 = 1.50$.

Prediction: $\hat r = 3 + 1.50 = 4.50$ stars.

Compare with skipping the centring and averaging raw ratings: $(0.900 \times 5 + 0.913 \times 3)/1.813 = 7.239/1.813 = 3.99$. Without centring, Dee's 3 counts as lukewarm, when for her it is a strong yes.

With $k = 4$ (all four raters, negative ones included): numerator $1.800 + 0.913 + (-0.877)(-1) + (-0.962)(-1) = 4.552$, denominator $0.900 + 0.913 + 0.877 + 0.962 = 3.652$, so $\hat r = 3 + 1.246 = 4.25$. The anti-neighbours disliked Dune, which is weak evidence that Ana will like it.

</div>

That is **user-based** CF, or **user-kNN**.

## Step 4: flip it around with item-based CF

The mirror image works on columns instead of rows. To predict $r_{ui}$, find the $k$ items most similar to $i$ *that $u$ has already rated*, and average $u$'s own ratings of them:

$$\hat r_{ui} = \frac{\sum_{j \in N_k(i;u)} s_{ij}\, r_{uj}}{\sum_{j \in N_k(i;u)} |s_{ij}|}$$

Here $s_{ij}$ is an item-item similarity and $N_k(i;u)$ is the set of the $k$ items most like $i$ among those $u$ rated. For ratings, the usual similarity is **adjusted cosine**: compare two item columns after subtracting each *user's* mean from every rating, so that a generous rater does not make all the items she rated look alike.

On our matrix, Dune's centred column over the users who rated both is (Ben $+2$, Cai $-1$, Dee $+1$, Eli $-1$). Alien's is $(+1, -1, +1, -1)$, giving $s = 5/(\sqrt 7 \cdot 2) = 0.945$. Blade Runner, rated by only three of those users, gets $0.730$. Casablanca and Notting Hill come out at $-0.956$. With $k = 2$ the neighbours are Alien (Ana gave 5) and Blade Runner (Ana gave 4):

$$\hat r = \frac{0.945 \times 5 + 0.730 \times 4}{0.945 + 0.730} = \frac{7.646}{1.675} = 4.56.$$

The two views agree here, though not always: user-kNN trusts *people*, item-kNN trusts *Ana's own ratings* of related items.

At serving time item-based CF needs only table lookups:

~~~dg-flow Item-based CF at serving time: the expensive similarity table is built offline; a request only does lookups and a weighted sum.
User history -(look up each item)-> Item-item table -(top-k neighbours)-> Candidate scores -(sum, drop seen)-> *Top-N list
~~~

## Step 5: do not trust thin evidence (shrinkage)

Two users who share exactly two rated films have a Pearson correlation of exactly $+1$ or $-1$. That is coincidence, not similarity.

~~~dg-fig 08.02-shrinkage.svg
Each dot is a pair of random users with no real connection. With only a few shared items, their similarity scatters anywhere from -1 to +1. Shrinkage pulls those low-evidence values towards 0 and leaves well-supported values almost unchanged.
~~~

In a sparse matrix most pairs share few items, so "most similar" lists fill up with flukes. The standard fix multiplies by an evidence factor:

$$s'_{ij} = \frac{n_{ij}}{n_{ij} + \beta}\, s_{ij}$$

where $n_{ij}$ is the number of co-raters (or co-rated items, for users) and $\beta$ is a constant, typically around 10 to 100. A pair with 2 co-raters and $\beta = 20$ keeps $2/22 = 9\%$ of its similarity; a pair with 500 co-raters keeps $500/520 = 96\%$. This is **shrinkage**: an estimate made on little data is pulled towards a safe default, here "not similar". Many systems also add a hard floor, for example ignoring pairs with fewer than 5 co-raters.

<div class="ex">
<div class="lab">Worked example</div>

**shrinkage changes the answer**

Item A (a niche documentary) shares 3 viewers with item B and has cosine 0.9. Item C shares 400 viewers with B and has cosine 0.35.

With $\beta = 50$: A gets $0.9 \times 3/53 = 0.051$, and C gets $0.35 \times 400/450 = 0.311$.

Without shrinkage, B's "similar items" rail would lead with a film three people happened to watch. With it, C leads, backed by 400 viewers.

</div>

## Implicit feedback: co-occurrence and the bestseller problem

Most real data is implicit (lesson 08.01): purchases, plays, clicks, stored as 1 or 0. Item $i$ is described by the set $U_i$ of users who touched it, and the raw signal is the **co-occurrence count** $n_{ij} = |U_i \cap U_j|$, "how many people bought both".

Raw counts have a fatal flaw: a bestseller co-occurs with everything because everyone has it. Normalise by popularity. Cosine on binary columns becomes

$$s_{ij} = \frac{|U_i \cap U_j|}{\sqrt{|U_i|\,|U_j|}},$$

co-occurrence divided by the geometric mean of the two popularities. Jaccard (from the table above) divides by the size of the union instead.

~~~dg-fig 08.02-popularity.svg
Top eight neighbours of a niche item (110 buyers) in a simulated shop with six genres. By raw count, the bestsellers lead, whatever their genre. By cosine, all eight come from the item's own genre.
~~~

## Why item-item wins in production

Amazon's 2003 description of item-to-item CF set the pattern most retailers still follow, for practical reasons:

- **Stability**: an item's audience changes slowly, a user's taste with every click. An item table rebuilt nightly stays valid; a user-user table would be stale within hours.
- **Density**: a popular item has thousands of raters, far more evidence than two users who each rated 20 things.
- **Cost at request time**: with the table precomputed and truncated to the top 50 neighbours per item, a request touches (items in history) × 50 entries.
- **Explanations**: "because you bought X" falls out for free.

The offline cost is computing co-occurrence for all item pairs. It is dominated by heavy users: a user with $m$ interactions contributes $m^2$ pairs, so it is common to cap or subsample very long histories.

## Neighbourhood CF in code

Explicit ratings first. This reproduces the hand example, with missing ratings as `nan`:

```python
import numpy as np
nan = np.nan
#             Alien BladeR Dune Casab NottH
R = np.array([[5,   4,   nan,  1,   2],    # Ana
              [4,   4,   5,    1,   1],    # Ben
              [2,   1,   2,    5,   5],    # Cai
              [3,   2,   3,    1,   1],    # Dee
              [3,   nan, 3,    5,   5]])   # Eli
rated = ~np.isnan(R)
mu = np.nanmean(R, axis=1)                  # each user's mean: [3. 3. 3. 2. 4.]
Z = np.where(rated, R - mu[:, None], 0.0)   # centred, missing -> 0

def centred_cosine(a, b, ma, mb):
    both = ma & mb                          # co-rated positions only
    x, y = a[both], b[both]
    return x @ y / (np.linalg.norm(x) * np.linalg.norm(y))

def predict_user_knn(u, i, k=2):
    raters = [v for v in range(len(R)) if v != u and rated[v, i]]
    s = np.array([centred_cosine(Z[u], Z[v], rated[u], rated[v]) for v in raters])
    top = np.argsort(-s)[:k]                # k most similar raters of item i
    s, dev = s[top], Z[[raters[t] for t in top], i]
    return mu[u] + s @ dev / np.abs(s).sum()

def predict_item_knn(u, i, k=2):
    items = [j for j in range(R.shape[1]) if j != i and rated[u, j]]
    s = np.array([centred_cosine(Z[:, i], Z[:, j], rated[:, i], rated[:, j]) for j in items])
    top = np.argsort(-s)[:k]
    s, r = s[top], R[u, [items[t] for t in top]]
    return s @ r / np.abs(s).sum()

print(round(predict_user_knn(0, 2, k=2), 2))   # 4.5
print(round(predict_user_knn(0, 2, k=4), 2))   # 4.25
print(round(predict_item_knn(0, 2, k=2), 2))   # 4.56
```

Implicit data, the way a retrieval source is usually built: one matrix product gives every co-occurrence count at once.

```python
items = ['phone', 'case', 'charger', 'cable', 'tripod']
B = np.array([[1, 1, 1, 0, 0],      # 6 shoppers x 5 items, 1 = bought
              [1, 1, 0, 1, 0],
              [1, 0, 1, 1, 0],
              [0, 1, 0, 0, 1],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 1, 1]], float)
co = B.T @ B                          # co-occurrence counts n_ij; diagonal = popularity
pop = np.diag(co).copy()
S = co / (np.sqrt(np.outer(pop, pop)) + 1e-9)   # binary cosine
S *= co / (co + 2.0)                  # shrinkage (beta = 2 for this tiny example)
np.fill_diagonal(S, 0.0)
print(np.round(S[0], 2))              # phone's neighbours: [0. 0.45 0.52 0.45 0.]

u = 3                                 # shopper who bought a case and a tripod
scores = B[u] @ S                     # sum of similarities to u's items
scores[B[u] > 0] = -np.inf            # never recommend what u already has
print([items[j] for j in np.argsort(-scores)[:2]])   # ['phone', 'cable']
```

On real data `B` is a `scipy.sparse` matrix; compute `B.T @ B` sparsely and keep the top 50 entries per row. The `implicit` (`CosineRecommender`) and `surprise` (`KNNWithMeans`) libraries package these steps.

## Checking the result

Hide some interactions (a time-based split, lesson 08.01) and see how well you recover them.

**Beat the baselines first**: each user's own mean for ratings, "most popular unseen items" for implicit data. A kNN model that loses to these has a bug or bad similarities.

**Tune $k$ and $\beta$ together.** Very small $k$ trusts one or two neighbours and is noisy; very large $k$ drags in weakly similar users and blurs the signal.

~~~dg-fig 08.02-k.svg
Held-out error of user-kNN on simulated ratings (12% of the matrix observed). With raw similarities, k = 1 to 3 is worse than just predicting each user's mean. Shrinkage helps at every k: best error 0.83 stars at k = 10, against 0.87 without it and 1.08 for the baseline.
~~~

Most of the gain comes from shrinkage rather than from $k$, which is typical. The curves flatten because few films have more than 40 positively similar raters.

**Look at the neighbours.** Print the top 10 neighbours of items you understand. Bestsellers everywhere means normalisation is missing; obscure items with 2 buyers means shrinkage is missing. Also check **coverage**: the fraction of the catalogue that ever appears in anyone's top 10.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Every item's neighbours are the same bestsellers | raw co-occurrence counts, no popularity normalisation | cosine or Jaccard; consider a stronger penalty on the popular side
| Top neighbours are obscure items with 2-3 co-raters | similarity estimated on thin evidence | shrinkage $n/(n+\beta)$ and a minimum co-rater count
| Users with opposite tastes look similar | cosine on raw positive ratings | centre by user mean (Pearson or adjusted cosine)
| New items never get recommended | an item with no interactions has no neighbours (cold start) | content-based or two-tower retrieval (08.04) for fresh items
| Offline build takes hours | a few users with huge histories create $m^2$ pairs each | cap or subsample long histories; drop bots

## Going deeper

**Baseline predictors.** The user mean is the simplest **baseline**. A stronger one is $b_{ui} = \mu + b_u + b_i$ (global mean, user generosity, item likeability), fitted by regularised least squares; kNN is then applied to the residuals $r_{ui} - b_{ui}$. This was a core ingredient of the Netflix Prize models.

**Negative neighbours.** The $k = 4$ example is legitimate because the denominator uses $|s|$, but many implementations keep only positive similarities: "people who disagree with you disliked this" is weaker evidence, and anti-correlation on few items is especially noisy.

**Learned weights.** The prediction is a similarity-weighted average of nearby labels, like a kNN regressor; $k$ and $\beta$ trade bias for variance. Learning the weights from data instead (Koren's neighbourhood models, or SLIM's sparse item-item regression) sits halfway to matrix factorisation.

**Complexity.** All item-item similarities cost $\sum_u m_u^2$ ($m_u$ = user $u$'s history length); user-user costs $\sum_i n_i^2$ ($n_i$ = item $i$'s raters), which explodes on blockbusters. MinHash (for Jaccard) or approximate nearest-neighbour search avoids computing all pairs.

**Cosine still favours popular items a little.** Dividing by $\sqrt{|U_i||U_j|}$ only partly corrects popularity; asymmetric cosine raises the two popularities to different, tuned powers.

**Limits.** Neighbourhood methods have no notion of latent taste dimensions. Two sci-fi fans who have no film in common never become neighbours, because similarity is only defined on co-rated items. They cannot score items without interactions, and they ignore content. Matrix factorisation (next lesson, 08.03) learns a dense low-dimensional space that handles the first problem; two-tower models (08.04) add features to handle the others. Yet a careful 2019 replication study found that well-tuned kNN baselines matched or beat many published neural recommenders, so never skip them.

## Key terms

| Term | Meaning
| --- | --- |
| Collaborative filtering | Predicting a user's preferences from the interaction matrix alone, using other users' behaviour
| User-kNN / item-kNN | Predicting from the most similar users who rated the item / from the most similar items the user rated
| Co-rated items | Items both users rated; similarity is computed only on these
| Centring | Subtracting each user's mean rating so that harsh and generous raters become comparable
| Adjusted cosine | Item-item cosine computed after subtracting each user's mean from their ratings
| Co-occurrence | Number of users who interacted with both items, $\lvert U_i \cap U_j\rvert$
| Shrinkage | Multiplying a similarity by $n/(n+\beta)$ so that pairs with little evidence move towards zero
| Coverage | Share of the catalogue that ever appears in users' recommendation lists

<div class="take">
<div class="lab">Key takeaway</div>

item-item CF with a popularity-normalised similarity and shrinkage is a cheap, explainable and surprisingly hard-to-beat retrieval source; most of its quality comes from the normalisation and shrinkage, not the choice of k.

</div>
