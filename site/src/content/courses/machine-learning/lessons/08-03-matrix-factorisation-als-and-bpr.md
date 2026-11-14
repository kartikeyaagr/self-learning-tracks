---
id: "08.03"
course: "machine-learning"
module: "08"
slug: "matrix-factorisation-als-and-bpr"
title: "Matrix factorisation: ALS and BPR"
keyIdeas: ["Matrix factorisation predicts $\\hat r_{ui} = x_u^\\top y_i$ with short $f$-dimensional user and item vectors ($f$ around 32 to 256). Taste is compressed into a dense **latent space**, so users with no items in common can still be close.", "The explicit-rating objective is squared error over **observed cells only** plus a ridge penalty $\\lambda$; plain SVD is the wrong tool because it would treat every missing cell as a zero rating.", "**ALS** alternates closed-form ridge solves: fix items, and each user's vector is $x_u = (Y_u^\\top Y_u + \\lambda I)^{-1} Y_u^\\top r_u$; then fix users and solve for items. Every solve is independent, the loss never goes up, and there is no learning rate.", "For implicit data (Hu, Koren, Volinsky), every cell is a target $p_{ui} \\in \\{0,1\\}$ with confidence $c_{ui} = 1 + \\alpha r_{ui}$; the trick $Y^\\top C^u Y = Y^\\top Y + Y_u^\\top (C^u_{uu} - I) Y_u$ keeps a user solve at $O(f^2 n_u + f^3)$ instead of touching every item.", "**BPR** optimises ranking directly: for sampled triples (user, observed item $i$, unobserved item $j$) it maximises $\\ln \\sigma(\\hat x_{ui} - \\hat x_{uj})$, a smooth surrogate for per-user AUC. Easy pairs give almost no gradient, so harder negatives speed it up.", "Regularisation $\\lambda$ and, for implicit ALS, the confidence scale $\\alpha$ matter more than $f$; tune them on a time-split ranking metric against a popularity baseline, not on training loss."]
question: "You train implicit ALS on 50M play events over 2M songs with f = 128. Recall@100 is poor and the top recommendations for almost every user are the same 200 hits. Give two causes rooted in the objective and how you would test each."
answer: "First, confidence may be too weak relative to the zeros, or regularisation too strong. With small $\\alpha$ or large $\\lambda$, the sea of confidence-1 zeros dominates and the factors shrink; the only signal that survives is popularity, which a single direction in factor space can encode for everyone. Test it by sweeping $\\alpha$ (for example 1, 10, 40, 100) and $\\lambda$ on a time-split recall@100, and track personalisation alongside it: catalogue coverage of the top-100 lists and the average overlap between two random users' lists. If recall rises while overlap falls as $\\alpha$ grows or $\\lambda$ shrinks, this was the cause.\n\nSecond, raw play counts let a few heavy users and repeat-heavy hits dominate the weighted loss. A track looped 2,000 times gets confidence around 80,000, so the item factors bend towards fitting those cells. Switch to $c = 1 + \\alpha \\log(1 + r/\\epsilon)$, cap counts, or down-weight heavy users, and check whether the hit list shrinks and recall improves for light users in particular (slice the metric by user activity).\n\nA third check is the evaluation itself: confirm the metric excludes songs the user already played in training and that the split is by time, otherwise popular repeats inflate both the score and the apparent popularity bias. Finally, compare against a most-popular baseline; if ALS is barely above it, the model has learned popularity rather than taste."
sources: [{"title": "Hu, Koren and Volinsky, Collaborative Filtering for Implicit Feedback Datasets (2008)", "url": "https://yifanhu.net/PUB/cf.pdf", "note": "The confidence-weighted objective and the $Y^\\top Y$ trick; sections 3 and 4."}, {"title": "Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback (2009)", "url": "https://arxiv.org/abs/1205.2618", "note": "Derivation of the pairwise criterion and its link to AUC; the LearnBPR algorithm box."}, {"title": "Google ML: Matrix factorization", "url": "https://developers.google.com/machine-learning/recommendation/collaborative/matrix", "note": "Weighted MF and why unobserved entries need a separate, smaller weight."}, {"title": "implicit (Python library) documentation", "url": "https://benfred.github.io/implicit/", "note": "Production-grade ALS and BPR implementations with CPU and GPU backends."}]
practice: ["implicit-als-user-factors", "als-explicit-sweep", "bpr-sgd-step", "uniform-negative-sampling"]
tags: ["matrix factorisation", "ALS", "BPR"]
tools: ["NumPy"]
---

## What you will learn

- What **matrix factorisation** (MF) is: a short vector per user and per item whose dot product predicts interest.
- Why training uses only observed cells, and why plain SVD is the wrong tool.
- How **alternating least squares (ALS)** reduces training to small ridge regressions, worked by hand.
- How implicit feedback (plays, clicks) is handled with preferences and confidences, and the trick that keeps it fast.
- How **BPR** trains for ranking with sampled triples, with one SGD step by hand.
- How to check a factor model and the classic ways it fails.

*Before you start:* the interaction matrix and implicit vs explicit feedback (08.01), neighbourhood CF (08.02), ridge regression (02.02) and the sigmoid (02.03). Truncated SVD (ML Maths) helps but is not required.

Lesson 08.02 compared raw rows or columns of the interaction matrix. That breaks down when the matrix is 99.9% empty and two similar users happened to touch different things. Matrix factorisation compresses taste into a handful of learned numbers per user and per item. It powered the Netflix Prize winners, ships as ALS in Spark, and is the simplest two-tower retriever (08.04).

## The idea in one picture

Five people rated films from 1 to 5 stars. Each of them skipped some films, so eight cells are unknown:

| User | Alien | Dune | Matrix | Notebook | Titanic
| --- | --- | --- | --- | --- | --- |
| Ana | 5 | 4 | ? | 1 | ?
| Ben | 4 | ? | 5 | ? | 2
| Cat | ? | 5 | 4 | 2 | 1
| Dan | 1 | ? | 2 | 5 | 4
| Eve | 2 | 1 | ? | 4 | ?

MF gives each user a vector $x_u$ and each film a vector $y_i$, here just $f = 2$ numbers each, and predicts a rating as the global average plus their dot product. Training fits the 17 known cells; every unknown cell then gets a prediction for free.

~~~dg-fig 08.03-complete.svg
Left: the 17 known ratings. Right: the matrix rebuilt from 2-number vectors; bold cells fill the gaps, e.g. Ana is predicted to like Matrix (4.3) but not Titanic (2.2).
~~~

Nobody told the model which films are sci-fi, yet the learned vectors show it:

~~~dg-fig 08.03-latent.svg
The learned 2-D space. Factor 1 separates sci-fi from romance, and each user sits near the films they rate highly.
~~~

- A **latent factor** is one learned coordinate, such as "factor 1". It often lines up with something human, but nothing forces it to.
- The **embedding** of a user or item is its list of $f$ factors; $f$ is the **rank** (32 to 256 in real systems).
- **Low-rank approximation** writes a big $m \times n$ matrix as a product of two thin ones, $(m + n)f$ numbers instead of $mn$.

The payoff over 08.02: users who never rated the same film can still be close, because their films are.

## Step 1: the model is a dot product

Stack the user vectors as rows of $X$ (shape $m \times f$) and the item vectors as rows of $Y$ (shape $n \times f$). The basic prediction is

$$\hat r_{ui} = x_u^\top y_i = \sum_{k=1}^{f} x_{uk}\, y_{ik},$$

where $\hat r_{ui}$ is the predicted rating of user $u$ for item $i$, and all predictions at once are $\hat R = X Y^\top$. Factor $k$ contributes "how much the user likes trait $k$" times "how much the item has it".

Some people rate everything high and some films are loved by everybody. A **biased** model adds these effects explicitly:

$$\hat r_{ui} = \mu + b_u + b_i + x_u^\top y_i.$$

Here $\mu$ is the global average rating, $b_u$ is how far user $u$ tends to rate above or below it, and $b_i$ does the same for item $i$. The factors then explain only the interaction. The toy example keeps just $\mu = 3.06$.

## Step 2: fit only the cells you trust

Match the known ratings while keeping vectors small. With $\Omega$ the set of observed (user, item) pairs, the explicit-rating objective is

$$\min_{X,Y,b} \sum_{(u,i)\in\Omega} \big(r_{ui} - \mu - b_u - b_i - x_u^\top y_i\big)^2 + \lambda\big(\lVert x_u\rVert^2 + \lVert y_i\rVert^2 + b_u^2 + b_i^2\big).$$

The squared error runs **only over observed cells**; $\lambda$ is the **regularisation** strength, as in ridge regression.

Why not plain SVD, the best low-rank approximation? SVD needs every cell filled. Write 0 for "not rated" and it works hard to predict those zeros, so unseen films drift towards 0 stars. MF ignores unknown cells during training and only predicts them afterwards.

## Step 3: alternating least squares

The objective multiplies two unknowns, $x_u$ and $y_i$, so it is not a single bowl. But freeze $Y$ and each user's loss is exactly a ridge regression whose "features" are the vectors of the items that user rated, which has a closed-form answer. The same holds for items with $X$ frozen. **ALS** alternates:

~~~dg-flow ALS alternates two sets of independent ridge regressions until the ranking metric stops improving.
Init X, Y small random -(fix Y)-> Solve every user (ridge) -(fix X)-> Solve every item (ridge) -(repeat 10-20x)-> *Factors
~~~

With $Y_u$ the $n_u \times f$ rows of $Y$ for the items user $u$ rated and $r_u$ their ratings (minus mean and biases):

$$x_u = (Y_u^\top Y_u + \lambda I_f)^{-1}\, Y_u^\top r_u,$$

where $I_f$ is the $f \times f$ identity: the ridge formula from 02.02 with $Y_u$ as the data matrix. The item update is the mirror image.

<div class="ex">
<div class="lab">Worked example</div>

**one ALS user update by hand**

Take $f = 2$, ignore the mean, and use $\lambda = 1$. Frozen items: $y_1 = (1, 0)$, $y_2 = (1, 1)$, and an unrated $y_3 = (0, 1)$. The user rated items 1 and 2 as $r_u = (4, 5)$.

Stack the rated items: $Y_u = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$, so $Y_u^\top Y_u = \begin{pmatrix} 1+1 & 0+1 \\ 0+1 & 0+1 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$ and $A = Y_u^\top Y_u + \lambda I = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$.

Right-hand side: $Y_u^\top r_u = (1\cdot4 + 1\cdot5,\; 0\cdot4 + 1\cdot5) = (9, 5)$.

Invert the $2 \times 2$ matrix: $\det A = 3\cdot2 - 1\cdot1 = 5$, so $A^{-1} = \tfrac15\begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix}$. Then $x_u = \tfrac15(2\cdot9 - 1\cdot5,\; -9 + 3\cdot5) = \tfrac15(13, 6) = (2.6, 1.2)$.

Fitted ratings: $y_1^\top x_u = 2.6$ and $y_2^\top x_u = 2.6 + 1.2 = 3.8$, below the true 4 and 5. That shrinkage is $\lambda$ distrusting just two ratings; with $\lambda = 0$ the solve gives $x_u = (4, 1)$ and fits 4 and 5 exactly.

Unrated item: $y_3^\top x_u = 0\cdot2.6 + 1\cdot1.2 = 1.2$, low, because this user's taste lives in factor 1, which item 3 lacks.

</div>

Why ALS is popular:

- **No learning rate.** Each solve jumps straight to the best $x_u$ for the current $Y$.
- **Monotone progress.** Each solve is an exact minimum, so a sweep never increases the objective; 10 to 20 sweeps usually suffice.
- **Trivially parallel.** User solves are independent $f \times f$ systems.

The alternative, **SGD**, nudges $x_u$ and $y_i$ along the error gradient one rating at a time; it needs a learning rate and many epochs.

~~~dg-fig 08.03-als.svg
Left: on a synthetic 400 x 250 matrix, ALS converges in about five sweeps; SGD depends on its learning rate. Right: held-out RMSE against λ is U-shaped, so tune λ on held-out data.
~~~

## Step 4: implicit feedback, preferences and confidences

Plays, clicks and purchases have no negatives: an unplayed song may be disliked or just unknown. Hu, Koren and Volinsky (2008) use two quantities per cell:

- **Preference** $p_{ui} = 1$ if the user interacted at all ($r_{ui} > 0$), else 0. Now *every* cell is a target, observed or not.
- **Confidence** $c_{ui} = 1 + \alpha r_{ui}$, where $r_{ui}$ is the count and $\alpha$ sets how fast trust grows. A popular variant is $c_{ui} = 1 + \alpha \log(1 + r_{ui}/\epsilon)$. Unobserved cells get confidence 1: a weak "probably not".

$$\min_{X,Y} \sum_{u,i} c_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda\Big(\sum_u \lVert x_u\rVert^2 + \sum_i \lVert y_i\rVert^2\Big)$$

The sum now runs over all $m \times n$ cells, so the zeros count, but gently.

~~~dg-fig 08.03-confidence.svg
How much a cell counts as a function of its play count. With α = 40 one play already outweighs an unplayed song 41 to 1; the log form keeps a 500-play habit from drowning everything else.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what α does**

A user played song A 30 times and song B once. With $\alpha = 40$, the confidences are $1 + 40 \times 30 = 1201$ and $1 + 40 \times 1 = 41$, against 1 for every unplayed song.

With $\alpha = 1$, A is only $1 + 30 = 31$ times as important as an unplayed song. Summed over a million of them, the zeros win, all scores sink, and recall@k typically falls.

</div>

The ALS user update keeps the same shape, now with weights:

$$x_u = (Y^\top C^u Y + \lambda I)^{-1}\, Y^\top C^u p_u,$$

where $C^u$ is the diagonal matrix of user $u$'s confidences over *all* $n$ items and $p_u$ their preferences. Naively that costs $O(f^2 n)$ per user, hopeless for $n = 10^6$. The trick: $C^u - I$ is zero except on the $n_u$ items the user touched, so

$$Y^\top C^u Y = \underbrace{Y^\top Y}_{\text{once per sweep}} + Y_u^\top (C^u_{uu} - I)\, Y_u.$$

$Y^\top Y$ is shared by all users; the correction touches only the user's own items, and $Y^\top C^u p_u$ likewise sums only over them (elsewhere $p_{ui} = 0$). Each user now costs $O(f^2 n_u + f^3)$. The paper used $\alpha = 40$; tune it per dataset, jointly with $\lambda$.

```python
import numpy as np
def implicit_user(Y, YtY, items, counts, alpha=40.0, lam=0.1):
    Yu = Y[items]                         # (n_u, f) vectors of items this user touched
    c = 1.0 + alpha * counts              # confidences of observed items
    A = YtY + (Yu.T * (c - 1.0)) @ Yu + lam * np.eye(Y.shape[1])
    b = (Yu.T * c).sum(axis=1)            # Y^T C^u p_u, with p = 1 on observed items
    return np.linalg.solve(A, b)
```

## Step 5: BPR, optimise the ranking itself

Weighted MF still fits squared error (1 for played, 0 otherwise), but recommenders are judged on order. Rendle et al. (2009) proposed **Bayesian Personalised Ranking (BPR)**: each user's observed items should simply score *above* unobserved ones.

Training data are **triples** $(u, i, j)$: user $u$, an item $i$ they interacted with, and a sampled item $j$ they did not (a **negative**). Write $\hat x_{ui} = x_u^\top y_i$ for the score and $\hat x_{uij} = \hat x_{ui} - \hat x_{uj}$ for the gap. BPR maximises

$$\sum_{(u,i,j)} \ln \sigma\big(\hat x_{ui} - \hat x_{uj}\big) - \lambda \lVert\Theta\rVert^2,$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid from 02.03 and $\Theta$ is all parameters. $\sigma(\hat x_{uij})$ is a smooth version of the indicator "i ranks above j", and averaging that indicator over all (i, j) pairs gives exactly the user's **AUC**, so BPR is a differentiable stand-in for per-user AUC.

The SGD gradient with respect to $x_u$ is $(1 - \sigma(\hat x_{uij}))(y_i - y_j)$; for $y_i$ it is $(1 - \sigma)\,x_u$ and for $y_j$ its negative. The factor $1 - \sigma$ is near 1 for a wrongly ordered pair and near 0 for a well ordered one.

<div class="ex">
<div class="lab">Worked example</div>

**one BPR step by hand**

Take $x_u = (0.5, 0.2)$, liked item $y_i = (0.4, 0.1)$, sampled negative $y_j = (0.3, 0.5)$, learning rate 0.1, and no regularisation.

Scores: $\hat x_{ui} = 0.5\cdot0.4 + 0.2\cdot0.1 = 0.22$ and $\hat x_{uj} = 0.5\cdot0.3 + 0.2\cdot0.5 = 0.25$. The gap is $-0.03$: the wrong order.

Weight: $\sigma(-0.03) = 0.4925$, so $1 - \sigma = 0.5075$, about half strength.

User update: $y_i - y_j = (0.1, -0.4)$, so $x_u \leftarrow (0.5, 0.2) + 0.1 \times 0.5075 \times (0.1, -0.4) = (0.5051, 0.1797)$.

Item updates: $y_i \leftarrow y_i + 0.1 \times 0.5075 \times (0.5, 0.2) = (0.4254, 0.1101)$ and $y_j \leftarrow y_j - 0.1 \times 0.5075 \times (0.5, 0.2) = (0.2746, 0.4899)$.

New gap: $0.5051 \times (0.4254 - 0.2746) + 0.1797 \times (0.1101 - 0.4899) = 0.0762 - 0.0683 = +0.008$. One step flipped the pair into the right order.

</div>

That weight is also BPR's weakness: once the model is decent, most uniform negatives are obscure items it already scores low, their weight is near zero, and learning stalls. **Hard-negative sampling** draws a few candidates and uses the highest-scoring unobserved one; popularity-weighted sampling is a cheaper version.

~~~dg-fig 08.03-bpr.svg
Left: the BPR loss and its gradient weight against the score gap; pairs already in the right order give almost no gradient. Right: on synthetic data, taking the hardest of 10 sampled negatives reaches a given AUC sooner than uniform sampling; both clearly beat recommending the most popular items.
~~~

| Method | Data | Loss | Optimiser | Strength
| --- | --- | --- | --- | --- |
| Biased MF | explicit ratings | squared error on observed cells | SGD or ALS | rating prediction
| Implicit ALS | counts | confidence-weighted squared error on all cells | ALS | fast, stable, few knobs
| BPR | binary implicit | pairwise logistic on sampled triples | SGD | targets ranking directly

## Matrix factorisation in code

Explicit ALS on the toy table, reproducing the first figure's right panel:

```python
import numpy as np

# 5 users x 5 films (Alien, Dune, Matrix, Notebook, Titanic); 0 = not rated
R = np.array([[5, 4, 0, 1, 0],
              [4, 0, 5, 0, 2],
              [0, 5, 4, 2, 1],
              [1, 0, 2, 5, 4],
              [2, 1, 0, 4, 0]], dtype=float)
M = R > 0                                  # mask of observed cells
mu = R[M].mean()                           # global mean, 3.06
f, lam = 2, 0.3
rng = np.random.default_rng(0)
X = rng.normal(0, 0.5, (5, f))             # user factors
Y = rng.normal(0, 0.5, (5, f))             # item factors
Rc = R - mu                                # fit the deviation from the mean

def ridge(F, r, lam):                      # solve (F^T F + lam I) x = F^T r
    return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ r)

for sweep in range(30):
    for u in range(5):                     # fix Y, solve every user
        X[u] = ridge(Y[M[u]], Rc[u, M[u]], lam)
    for i in range(5):                     # fix X, solve every item
        Y[i] = ridge(X[M[:, i]], Rc[M[:, i], i], lam)

P = mu + X @ Y.T
print(np.round(P, 1))           # row Ana: [4.9 4.0 4.3 1.2 2.2]
print("Ana -> Matrix", P[0, 2].round(1), " Ana -> Titanic", P[0, 4].round(1))  # 4.3  2.2
```

For real data use a library: `implicit` (CPU/GPU ALS and BPR on a sparse matrix), Spark's `ALS` with `implicitPrefs=True`, or a PyTorch embedding layer when you will add features. To recommend, compute $Y x_u$, drop already-seen items, and take the top $k$.

## Checking the result

A falling training loss only shows the optimiser works.

**1. Evaluate on the future.** Train before a cut-off date and measure recall@k or NDCG@k (08.05) afterwards; a random split leaks the future.

**2. Beat the baselines.** Report most-popular and item-based CF (08.02) next to MF.

**3. Check diversity.** Count distinct items across all top-10 lists (**catalogue coverage**) and the overlap between two random users' lists.

**4. Sanity-check the space.** List nearest neighbours of well-known items by cosine of $y_i$; sequels near originals is good, random neighbours is not.

**5. Exclude already-consumed items** from recommendations and the metric, unless re-consumption is the product.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Everyone gets the same popular items | $\lambda$ too large or $\alpha$ too small, so only a popularity direction survives | sweep $\lambda$ and $\alpha$ on time-split recall; check coverage
| Great training fit, poor held-out metric | $f$ too large or $\lambda$ too small: memorising cells | raise $\lambda$, lower $f$, stop on validation metric
| A few heavy users or looped tracks dominate | raw counts give confidences in the thousands | log confidence, cap counts, down-weight heavy users
| BPR metric plateaus early | uniform negatives are too easy; gradient weights near 0 | hard or popularity-weighted negatives
| New items never get recommended | no interactions means $y_i$ is just its random start | content features (08.04) or fold-in after the first few events

## Going deeper

**Non-convex but biconvex.** For any invertible $f \times f$ matrix $Q$, replacing $X$ by $XQ$ and $Y$ by $YQ^{-\top}$ leaves $XY^\top$ unchanged; with regularisation the freedom shrinks to rotations, but a family of equally good solutions remains, so the loss is no single bowl. It is convex in $X$ for fixed $Y$ and vice versa, and ALS is exact block coordinate descent on it: non-increasing, converging to a stationary point that in practice is close to the best. The rotation freedom is why "factor 1 = sci-fi" is a convenient rotation, not a law, and why raw vectors from two training runs are not comparable.

**Relation to SVD.** If every cell were observed and $\lambda = 0$, the best rank-$f$ fit is the truncated SVD (the Eckart–Young theorem). With $\lambda > 0$ and full observation, the solution is still built from the top singular vectors, with each singular value shrunk by $\lambda$. Missing cells break this closed form, which is exactly why we need an iterative method.

**Cost per sweep.** Explicit ALS costs $O(|\Omega| f^2 + (m + n) f^3)$; implicit ALS with the $Y^\top Y$ trick has the same order despite fitting all $mn$ cells. Libraries swap the exact solve for a few conjugate-gradient steps, dropping the $f^3$ term. BPR costs $O(f)$ per triple.

**Scaling λ.** **Weighted-λ regularisation** (ALS-WR) multiplies the penalty by $n_u$ or $n_i$; libraries differ, so a tuned $\lambda$ does not transfer between them.

**Fold-in.** A user's vector is one ridge solve given $Y$, so a brand-new user gets a vector from their first few interactions without retraining, cheaply enough to do at request time.

**BPR is a MAP estimate.** $\ln\sigma$ is the log-likelihood of independent pairwise preferences and $\lambda\lVert\Theta\rVert^2$ a Gaussian log-prior. The criterion can wrap any scorer, including the two-tower networks of 08.04.

## Key terms

| Term | Meaning
| --- | --- |
| Matrix factorisation | Approximating the interaction matrix as $XY^\top$
| Latent factor / rank $f$ | One learned coordinate / number of coordinates per vector
| Biases $b_u$, $b_i$ | Offsets for "rates high" and "liked by all"
| ALS | Alternating exact ridge solves for users, then items
| Preference / confidence | Implicit target $p_{ui}$ and its weight $c_{ui} = 1 + \alpha r_{ui}$
| BPR | Pairwise loss $\ln\sigma(\hat x_{ui} - \hat x_{uj})$; a smooth per-user AUC
| Hard negative | An unobserved item the model currently scores high
| Fold-in | One ridge solve for a new user with item factors frozen

<div class="take">
<div class="lab">Key takeaway</div>

matrix factorisation learns a short taste vector for every user and item so that a dot product predicts interest; fit it with ALS (exact ridge solves, confidence-weighted for implicit data) or with BPR when you want a ranking loss, and tune $\lambda$, $\alpha$ and the negative sampling on a time-split ranking metric, not on training loss.

</div>
