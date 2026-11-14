---
id: "08.03"
course: "machine-learning"
module: "08"
slug: "matrix-factorisation-als-and-bpr"
title: "Matrix factorisation: ALS and BPR"
keyIdeas: ["Matrix factorisation models $\\hat r_{ui} = x_u^\\top y_i$ with $f$-dimensional user and item vectors ($f$ around 32 to 256), compressing taste into a dense latent space where users with no items in common can still be close.", "**ALS** alternates closed-form ridge solves: fix items, each user's vector is $x_u = (Y_u^\\top Y_u + \\lambda I)^{-1} Y_u^\\top r_u$; then fix users and solve for items. Every solve is independent, so it parallelises trivially.", "For implicit data (Hu, Koren, Volinsky), every cell is a target $p_{ui} \\in \\{0,1\\}$ with confidence $c_{ui} = 1 + \\alpha r_{ui}$; the trick $Y^\\top C^u Y = Y^\\top Y + Y^\\top (C^u - I) Y$ keeps a user solve at $O(f^2 n_u + f^3)$ instead of touching every item.", "**BPR** optimises ranking directly: for sampled triples (user, observed item $i$, unobserved item $j$) it maximises $\\ln \\sigma(\\hat x_{ui} - \\hat x_{uj})$, a smooth surrogate for per-user AUC.", "Regularisation $\\lambda$ and, for implicit ALS, the confidence scale $\\alpha$ matter more than $f$; tune them on a time-split ranking metric, not on training loss."]
question: "You train implicit ALS on 50M play events over 2M songs with f = 128. Recall@100 is poor and the top recommendations for almost every user are the same 200 hits. Give two causes rooted in the objective and how you would test each."
answer: "First, confidence may be too weak relative to the zeros, or regularisation too strong. With small $\\alpha$ or large $\\lambda$, factors shrink and the only signal that survives is popularity, which a few dimensions can encode for everyone. Sweep $\\alpha$ and $\\lambda$ on a time-split recall@100 and watch whether personalisation (overlap of top lists across users) changes. Second, raw play counts let a few heavy users and repeat-heavy hits dominate the weighted loss. Switch to $c = 1 + \\alpha \\log(1 + r/\\epsilon)$, cap counts, or down-weight heavy users, and check whether the hit list shrinks. A third check is evaluation itself: confirm the metric excludes already-played songs, otherwise popular repeats inflate both the score and the apparent popularity bias."
sources: [{"title": "Hu, Koren and Volinsky, Collaborative Filtering for Implicit Feedback Datasets (2008)", "url": "https://yifanhu.net/PUB/cf.pdf", "note": "The confidence-weighted objective and the $Y^\\top Y$ trick; sections 3 and 4."}, {"title": "Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback (2009)", "url": "https://arxiv.org/abs/1205.2618", "note": "Derivation of the pairwise criterion and its link to AUC; the LearnBPR algorithm box."}, {"title": "Google ML: Matrix factorization", "url": "https://developers.google.com/machine-learning/recommendation/collaborative/matrix", "note": "Weighted MF and why unobserved entries need a separate, smaller weight."}, {"title": "implicit (Python library) documentation", "url": "https://benfred.github.io/implicit/", "note": "Production-grade ALS and BPR implementations with CPU and GPU backends."}]
practice: ["implicit-als-user-factors", "als-explicit-sweep", "bpr-sgd-step", "uniform-negative-sampling"]
tags: ["matrix factorisation", "ALS", "BPR"]
tools: ["NumPy"]
---

## From neighbours to factors

Neighbourhood CF (08.02) compares raw rows. Matrix factorisation (MF) instead learns a low-dimensional vector for every user and item so that their dot product reproduces the observed signal:

$$\hat r_{ui} = x_u^\top y_i, \qquad X \in \mathbb{R}^{m \times f},\; Y \in \mathbb{R}^{n \times f}$$

This is low-rank approximation of $R$, the same idea as truncated SVD (covered in ML Maths, "Singular value decomposition"), with one crucial difference: SVD needs a fully observed matrix, while MF fits only the cells you trust and regularises the rest. The factors become latent taste dimensions. Two users who watched disjoint sets of sci-fi films end up close because their films sit in the same region of item space.

For explicit ratings the objective is regularised squared error over observed cells $\Omega$, usually with biases:

$$\min_{X,Y,b} \sum_{(u,i)\in\Omega} \big(r_{ui} - \mu - b_u - b_i - x_u^\top y_i\big)^2 + \lambda\big(\lVert x_u\rVert^2 + \lVert y_i\rVert^2 + b_u^2 + b_i^2\big)$$

The biases soak up "this user rates high" and "this film is universally loved", leaving the factors to model interaction.

## Alternating least squares

The objective is not jointly convex in $X$ and $Y$, but it *is* a ridge regression in $X$ once $Y$ is fixed, and vice versa. ALS exploits that:

~~~dg-flow ALS alternates two embarrassingly parallel sets of ridge regressions until the ranking metric stops improving.
Init X, Y small random -(fix Y)-> Solve every user (ridge) -(fix X)-> Solve every item (ridge) -(repeat 10-20x)-> *Factors
~~~

For user $u$ with observed items $Y_u$ (rows of $Y$) and ratings $r_u$:

$$x_u = (Y_u^\top Y_u + \lambda I_f)^{-1} Y_u^\top r_u$$

Each solve is an $f \times f$ system, independent of every other user, so a sweep is a map over users. Each sweep cannot increase the loss, so ALS converges monotonically, usually in 10 to 20 sweeps. SGD on the same objective is simpler per step but needs a learning rate and many epochs; ALS is preferred when data is large and distributed, which is why Spark MLlib ships it.

## Implicit feedback: weighted MF

With clicks and plays there are no negatives, only counts. Hu, Koren and Volinsky (2008) reframed the problem:

- **Preference** $p_{ui} = 1$ if $r_{ui} > 0$, else 0. *Every* cell, observed or not, is a target.
- **Confidence** $c_{ui} = 1 + \alpha r_{ui}$ (or $1 + \alpha \log(1 + r_{ui}/\epsilon)$). Unobserved cells get confidence 1, a weak "probably not"; heavily played items get much more.

$$\min_{X,Y} \sum_{u,i} c_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda\Big(\sum_u \lVert x_u\rVert^2 + \sum_i \lVert y_i\rVert^2\Big)$$

The user update becomes $x_u = (Y^\top C^u Y + \lambda I)^{-1} Y^\top C^u p_u$, where $C^u$ is diagonal over *all* $n$ items. Computed naively that is $O(f^2 n)$ per user, hopeless for $n = 10^6$. The trick is that $C^u - I$ is non-zero only on the $n_u$ items the user touched:

$$Y^\top C^u Y = \underbrace{Y^\top Y}_{\text{once per sweep}} + Y_u^\top (C^u_{uu} - I) Y_u$$

So each user costs $O(f^2 n_u + f^3)$. The paper used $\alpha = 40$; in practice $\alpha$ is tuned per dataset and interacts with $\lambda$.

```python
import numpy as np
def implicit_user(Y, YtY, items, counts, alpha=40.0, lam=0.1):
    Yu = Y[items]                         # (n_u, f)
    c = 1.0 + alpha * counts              # confidences of observed items
    A = YtY + (Yu.T * (c - 1.0)) @ Yu + lam * np.eye(Y.shape[1])
    b = (Yu.T * c).sum(axis=1)            # Y^T C^u p_u, p = 1 on observed
    return np.linalg.solve(A, b)
```

## BPR: optimise the ranking, not the reconstruction

Weighted MF still fits squared error. Rendle et al. (2009) argued the real goal is order: for each user, observed items should score above unobserved ones. Bayesian Personalised Ranking samples triples $(u, i, j)$ with $i$ observed and $j$ not, and maximises

$$\sum_{(u,i,j)} \ln \sigma\big(\hat x_{ui} - \hat x_{uj}\big) - \lambda \lVert\Theta\rVert^2$$

$\sigma(\hat x_{ui} - \hat x_{uj})$ is a smooth version of the indicator "i ranks above j", and averaging that indicator over all pairs is exactly per-user AUC. Training is SGD over sampled triples; the gradient on $x_u$ is $(1 - \sigma(\hat x_{uij}))(y_i - y_j)$, so confidently ordered triples contribute almost nothing. That is also BPR's weakness: once most uniform negatives are easy, learning stalls. Popularity-weighted or hard-negative sampling (scoring a few candidates and picking the highest-scored unobserved one) speeds it up considerably.

| Method | Data | Loss | Optimiser | Strength
| --- | --- | --- | --- | --- |
| Biased MF | explicit ratings | squared error on observed | SGD or ALS | rating prediction
| Implicit ALS | counts | confidence-weighted squared error on all cells | ALS | fast, stable, few knobs
| BPR | binary implicit | pairwise logistic on sampled triples | SGD | targets ranking directly

<div class="ex">
<div class="lab">Worked example</div>

**what $\alpha$ does**

A user played song A 30 times and song B once. With $\alpha = 40$, confidences are 1201 and 41 against 1 for every unplayed song. The model must fit A almost exactly, B fairly well, and treats the million unplayed songs as gentle pressure towards zero. Setting $\alpha = 1$ makes A only 31 times as important as an unplayed song, and the sea of zeros pulls all scores down, typically hurting recall@k.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

MF learns dense taste vectors; use ALS with confidence weighting for large implicit datasets and BPR when you want a ranking loss, and in both cases tune regularisation and negative handling on a time-split ranking metric.

</div>
