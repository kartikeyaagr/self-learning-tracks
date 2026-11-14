---
id: "05.04"
course: "machine-learning"
module: "05"
slug: "dimensionality-reduction-in-practice"
title: "Dimensionality reduction in practice"
keyIdeas: ["PCA centres the data and finds the directions of greatest spread (eigenvectors of the covariance, computed via the SVD). Each eigenvalue is the variance along its direction, and the average reconstruction error equals the sum of the eigenvalues you drop.", "In an ML pipeline, fit PCA on the training split only and choose the number of components by validation score; variance targets (90-99%) and the scree elbow are only starting points.", "PCA keeps high-variance directions, not predictive ones: a low-variance direction can carry all the signal, so always compare downstream performance with and without PCA.", "Whitening rescales PCA coordinates to unit variance ($\\Lambda^{-1/2} U^\\top x$); ZCA rotates back to the original axes. Both amplify noise in tiny-eigenvalue directions, so add an $\\epsilon$.", "t-SNE and UMAP make 2-D maps that preserve local neighbourhoods. Cluster sizes and between-cluster distances in the map are not meaningful, and results depend on perplexity or `n_neighbors` and on the seed; use them after PCA to about 50 dimensions, for looking only."]
question: "A teammate shows a UMAP plot of customer embeddings with two clusters far apart and concludes that the two segments are 'very different', then runs k-means with k=2 on the 2-D UMAP coordinates to define the segments. What is wrong with each step, and what would you do instead?"
answer: "UMAP preserves local neighbourhoods. The distance between clusters in the map is not a calibrated measure of how different they are in the original space, so 'far apart' cannot be read as 'very different'. The layout also changes with `n_neighbors`, `min_dist` and the random seed, and the visual size of each cluster says nothing about how spread out it really is. Running k-means on the 2-D coordinates defines segments with distortions the embedding introduced, and new customers would get segments that depend on a stochastic, non-linear map that may shift when it is refitted. Instead I would measure the difference in the original or PCA space: compare cluster centroids, between-cluster and within-cluster distances, or train a classifier to separate the groups and look at its cross-validated accuracy. I would also cluster in that space, using k-means, GMM or HDBSCAN on standardised or PCA-reduced features (with PCA fitted on the data used to build the segments and then applied unchanged to new customers), and evaluate with silhouette and stability across resamples (05.06). The UMAP plot is then only for display, coloured by the segments found in the original space, and the conclusion should hold across several UMAP settings and seeds."
sources: [{"title": "Wattenberg, Viégas, Johnson: How to Use t-SNE Effectively (Distill)", "url": "https://distill.pub/2016/misread-tsne/", "note": "Interactive demonstrations of perplexity, cluster sizes, distances and seeing clusters in noise."}, {"title": "van der Maaten and Hinton (2008): Visualizing Data using t-SNE", "url": "https://www.jmlr.org/papers/v9/vandermaaten08a.html", "note": "The original t-SNE paper; the crowding problem and the Student-t kernel."}, {"title": "McInnes, Healy, Melville: UMAP", "url": "https://arxiv.org/abs/1802.03426", "note": "The UMAP paper; for practical parameter guidance see the umap-learn docs."}, {"title": "scikit-learn User Guide: Decomposing signals in components (PCA)", "url": "https://scikit-learn.org/stable/modules/decomposition.html", "note": "PCA, whitening, incremental and randomised PCA, and truncated SVD for sparse data."}]
practice: ["pca", "pca-projection", "whitening-transform", "t-sne-affinity-matrix", "svd-components"]
tags: ["PCA", "whitening", "t-SNE"]
tools: ["scikit-learn", "UMAP"]
---

## What you will learn

- Why "feed a model" and "look at the data" are two different jobs that need different tools.
- How PCA works: centre, find the directions of largest spread, project, measure what was lost. You will do it by hand on five students.
- How to choose the number of components, best of all by validation score in a pipeline fitted on training data only.
- What whitening (PCA and ZCA) does and why it needs an $\epsilon$.
- How t-SNE and UMAP build 2-D maps, and how those maps mislead.

*Before you start:* variance, covariance and vectors. The mathematics of eigenvectors and the SVD is in ML Maths (lessons on SVD and PCA); here we use the results. Clustering (05.01-05.03) and train/validation pipelines (modules 06-07) help.

## From clusters to directions

Lessons 05.01-05.03 looked for *groups* of similar points. This lesson looks for *directions*: a few combinations of features that capture most of what varies. In 05.03 each Gaussian had a covariance matrix describing its spread; PCA uses the covariance of the whole dataset to find the axes of that spread.

Why bother? Real feature vectors are wide (an image has thousands of pixels, a text embedding 768 numbers), yet many columns move together, so the data lives near a much lower-dimensional surface. Finding it lets you train faster, remove noise, help distance-based methods like k-means, and draw a picture you can look at.

There are two different goals, and they call for different tools:

- **Reduce features for a model**: compress, denoise, decorrelate. The transform must also apply to new data. Use PCA.
- **Look at the data**: a 2-D picture of who is near whom. Use t-SNE or UMAP, and treat it as a hint, not a measurement.

## The idea in one picture

Five students take a maths quiz and a physics quiz:

| Student | A | B | C | D | E
| --- | --- | --- | --- | --- | --- |
| Maths | 45 | 55 | 60 | 65 | 75
| Physics | 45 | 65 | 60 | 55 | 75

The two scores rise and fall together: strong students do well in both. So instead of two numbers per student you could almost get away with one, an "overall ability" score along the diagonal. That diagonal is the **first principal component (PC1)**: the direction in which the data is most spread out.

~~~dg-fig 05.04-students.svg
PC1 runs along the diagonal where students differ most; the second component (PC2) runs across it. Keeping only PC1 moves every student to the diamond on the line, so B, C and D become identical.
~~~

Reading the picture:

- Along PC1 the students spread a lot (variance 225). Across it, along **PC2**, they spread little (variance 25). PC1 holds $225/250 = 90\%$ of the total variance.
- **Projecting** onto PC1 means replacing each student by their position along the line, one number instead of two. That is **dimensionality reduction**: 2 features become 1.
- The cost: B (better at physics) and D (better at maths) land on the same spot as C. The dropped 10% was exactly the "maths versus physics" difference, which may or may not matter for what you predict.

## Step 1: centre (and usually scale) the data

PCA looks for directions of spread around the middle of the data, so the first step is to subtract the mean of each feature. Write $\mu$ for the vector of feature means and $x - \mu$ for a **centred** point.

For the students, $\mu = (60, 60)$, so the centred points are A $(-15,-15)$, B $(-5, 5)$, C $(0, 0)$, D $(5,-5)$, E $(15, 15)$.

Scaling is a separate decision. A column in millimetres has a million times the variance of the same column in metres and becomes PC1 for no real reason. So **standardise** (divide each centred feature by its standard deviation) when units differ; skip it when features share a unit, like pixels or our two quizzes.

## Step 2: find the directions of greatest spread

The **covariance matrix** $C$ summarises the spread. Its diagonal holds each feature's variance; the off-diagonal entries hold how pairs of features move together:

$$C = \frac{1}{n-1} X_c^\top X_c,$$

where $X_c$ is the centred data matrix ($n$ rows, one per example; $d$ columns, one per feature).

The principal components are the **eigenvectors** of $C$: directions $u$ with $Cu = \lambda u$. Each eigenvalue $\lambda$ is the variance along its direction; PC1 has the largest. The eigenvectors are perpendicular, so the components form a rotated set of axes.

Libraries actually take the **singular value decomposition** of the centred data, $X_c = U S V^\top$, where the columns of $V$ are the same directions and the **singular values** $s_j$ (the diagonal of $S$) give the variances as $\lambda_j = s_j^2/(n-1)$. The **explained variance ratio** of component $j$ is $\lambda_j / \sum_k \lambda_k$.

<div class="ex">
<div class="lab">Worked example</div>

**PCA of the five students by hand**

Centred scores: A $(-15,-15)$, B $(-5,5)$, C $(0,0)$, D $(5,-5)$, E $(15,15)$; $n = 5$, so divide by $n - 1 = 4$.

Variance of maths: $(225 + 25 + 0 + 25 + 225)/4 = 500/4 = 125$. Physics is the same: 125.

Covariance: $(225 - 25 + 0 - 25 + 225)/4 = 400/4 = 100$. So $C = \begin{pmatrix}125 & 100\\ 100 & 125\end{pmatrix}$.

Eigenvalues solve $(125-\lambda)^2 - 100^2 = 0$, so $125 - \lambda = \pm 100$, giving $\lambda_1 = 225$ and $\lambda_2 = 25$. The total, 250, equals the sum of the two variances, as it must.

Eigenvectors: for $\lambda_1 = 225$, $u_1 = (1, 1)/\sqrt 2$ (the diagonal); for $\lambda_2 = 25$, $u_2 = (1, -1)/\sqrt 2$ (across it).

PC1 scores $z_1 = (x_c + y_c)/\sqrt 2$: A $-30/\sqrt2 = -21.21$, B $0$, C $0$, D $0$, E $+21.21$. Check: $(450 + 450)/4 = 225 = \lambda_1$.

PC2 scores $z_2 = (x_c - y_c)/\sqrt 2$: B $-7.07$, D $+7.07$, the rest 0. Check: $(50 + 50)/4 = 25 = \lambda_2$.

Explained variance ratio: PC1 $225/250 = 90\%$, PC2 $10\%$. Singular values: $s_1 = \sqrt{4 \times 225} = 30$, $s_2 = \sqrt{4 \times 25} = 10$.

</div>

## Step 3: project, and measure what you lost

To keep $m$ components, stack the top $m$ eigenvectors as columns of a $d \times m$ matrix $V_m$. Each point's new coordinates (its **scores**) are

$$z = V_m^\top (x - \mu),$$

an $m$-vector. To go back to the original space, **reconstruct** with $\hat x = \mu + V_m z$. The difference $x - \hat x$ is what the compression destroyed.

With the students and $m = 1$: B has score 0, so $\hat x_B = (60, 60)$, an error of $(-5, 5)$ with squared length 50. D loses the same 50. A, C and E sit on the line and lose nothing. Total squared error $= 100$, and $100/(n-1) = 25 = \lambda_2$. That is no coincidence: **the average reconstruction error equals the sum of the eigenvalues you dropped.** PCA picks the directions that make this error as small as possible, which is the second, equivalent definition of PCA (maximum variance kept = minimum reconstruction error).

A free diagnostic follows: badly reconstructed points (via `inverse_transform`) are often anomalies, one of the methods in lesson 05.05.

## Step 4: choose how many components to keep

Real data has many components, and you need to pick $m$. Take a 5-feature dataset whose covariance has eigenvalues 4.0, 2.5, 1.0, 0.3 and 0.2 (total variance 8.0). (Had the features been standardised, the total would equal the number of features, 5.)

~~~dg-fig 05.04-scree.svg
Each bar is one component's share of the variance; the line adds them up. A 90% target stops at 3 components, a 95% target at 4, and the bars flatten (the elbow) after component 3.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**choosing m from eigenvalues**

Shares: $4.0/8 = 50\%$, $2.5/8 = 31.25\%$, $1.0/8 = 12.5\%$, $0.3/8 = 3.75\%$, $0.2/8 = 2.5\%$.

Cumulative: $50\%$, $81.25\%$, $93.75\%$, $97.5\%$, $100\%$.

A 95% target keeps 4 components (93.75% is not enough); a 90% target keeps 3. The scree plot's elbow also suggests 3.

But if the label depends on the direction with eigenvalue 0.2, both choices throw the signal away. PCA never looks at the label.

</div>

Three ways to choose, from weakest to strongest:

1. **A variance target** (90%, 95%, 99%): easy, label-free, good for compression.
2. **The scree plot elbow**: stop where the bars stop dropping sharply. Quick but subjective.
3. **Validation score**: if PCA feeds a model, tune $m$ like any hyperparameter. Only this measures what you care about.

The digits dataset (8x8 handwritten digit images, 64 pixel features) shows why the third rule matters.

~~~dg-fig 05.04-digits.svg
Left: 28 of 64 components hold 95% of the pixel variance. Right: a logistic regression on the PCA scores gains a lot up to about 10 components, then creeps up slowly towards the all-pixel score of about 96%.
~~~

At $m = 10$ the classifier reaches 92% cross-validated accuracy, at 30 about 95.5%, and at 40 it matches the all-pixel model (96%). Validation tells you the exact price of every component you drop.

## The pipeline: fit on training data only

PCA learns $\mu$ and $V_m$ from data, so like a scaler it is fitted on the training split only and applied unchanged to validation and test data. Fitting on everything is a small but real leak (module 07).

~~~dg-flow A typical pipeline. Every step is fitted on the training split only, then applied unchanged to validation and test.
Raw features -(fit on train)-> Standardise -(fit on train)-> *PCA to m dims -> Model
~~~

Two more practical rules:

- **Signs are arbitrary.** A component and its negative are the same axis, and refits can flip them. Never hard-code "PC1 high means good student".
- **Sparse data:** centring makes it dense. `TruncatedSVD` skips centring (on text this is latent semantic analysis).

## Dimensionality reduction in code

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)                     # 1797 images, 64 pixels each
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

pca = PCA().fit(Xtr)                                     # fit on TRAIN only
cum = np.cumsum(pca.explained_variance_ratio_)
print(np.searchsorted(cum, 0.95) + 1)                    # 28 components for 95%
print(PCA(n_components=0.95).fit(Xtr).n_components_)     # 28: a float picks m for you

# m as a hyperparameter: the pipeline refits PCA inside every CV fold
pipe = make_pipeline(PCA(), LogisticRegression(max_iter=3000))
grid = GridSearchCV(pipe, {"pca__n_components": [5, 10, 20, 30, 40]}, cv=5).fit(Xtr, ytr)
print(grid.best_params_, round(grid.best_score_, 3))     # {'pca__n_components': 40} 0.961
print(round(grid.score(Xte, yte), 3))                    # 0.967 on the untouched test set

# for LOOKING: PCA first to cut noise, then t-SNE to 2-D
Z = PCA(n_components=30, random_state=0).fit_transform(Xtr)
emb = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(Z)
print(emb.shape)                                         # (1347, 2)
```

Digits have only 64 features, so we reduce to 30 before t-SNE; with wider data use about 50. UMAP (package `umap-learn`) works the same way, `umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(Z)`, and unlike `TSNE` it can `transform` new points. `PCA(whiten=True)` gives the whitening described next.

## Step 5: whitening

PCA scores are uncorrelated, but they still have different variances $\lambda_j$ (225 and 25 for the students). **Whitening** divides each score by its standard deviation $\sqrt{\lambda_j}$, so every direction has variance 1 and the transformed data has identity covariance:

$$x_{\text{PCA-white}} = \Lambda^{-1/2} U^\top (x - \mu) \qquad x_{\text{ZCA}} = U \Lambda^{-1/2} U^\top (x - \mu)$$

where $U$ holds the eigenvectors of the covariance as columns and $\Lambda$ is the diagonal matrix of its eigenvalues. For the students, PCA-whitening turns A's PC1 score $-21.21$ into $-21.21/15 = -1.41$ and B's PC2 score $-7.07$ into $-7.07/5 = -1.41$: the small "maths versus physics" direction is now as loud as overall ability.

**ZCA whitening** rotates back to the original axes with $U$, keeping the output as close as possible to the input: whitened images still look like images, with edges enhanced.

~~~dg-fig 05.04-whiten.svg
Whitening turns a tilted ellipse into a round cloud. PCA-whitening also rotates it onto the component axes; ZCA rotates back, so the highlighted points stay roughly where they started.
~~~

Whitening helps methods that assume equally scaled directions: k-means (05.01), ICA, some older image pipelines. The danger is division by tiny eigenvalues: pure-noise directions get blown up to unit variance, as loud as the signal. Use $(\Lambda + \epsilon I)^{-1/2}$ with a small $\epsilon$, or drop the smallest components first.

## Step 6: t-SNE and UMAP for looking at data

PCA can only rotate and flatten, so curved structure often cannot be shown by any flat projection. **Neighbour embeddings** give up on preserving distances and only try to keep each point's neighbours near it.

**t-SNE** (van der Maaten and Hinton, 2008) works in three moves:

1. Turn original distances into **neighbour probabilities** $p_{j|i}$ with a Gaussian around each point, its width tuned so the effective number of neighbours, the **perplexity** (typically 5-50), is the same everywhere.
2. In 2-D, compute similarities $q_{ij}$ with a heavy-tailed Student-t kernel, which lets moderately distant points sit far apart instead of crowding into the middle.
3. Move the 2-D points to minimise $\mathrm{KL}(P \parallel Q) = \sum p_{ij} \log (p_{ij}/q_{ij})$. This is costly when true neighbours (big $p$) end up far apart (small $q$) but cheap when distant points end up close, so local structure is protected and global structure is not.

**UMAP** (McInnes et al., 2018) builds a weighted $k$-nearest-neighbour graph (`n_neighbors`, typically 15) and lays out a 2-D graph to match it; `min_dist` sets how tightly points pack. It is faster than t-SNE, scales to millions of points and keeps somewhat more global arrangement.

Both maps are easy to misread. The next figure uses three clusters in 50 dimensions: A is tight, B is six times wider, and C is far away from both.

~~~dg-fig 05.04-tsne.svg
PCA keeps the true scale: tight A, wide B, distant C. t-SNE makes the clusters look similar in size and shrinks the big gap to C, and the layout changes with perplexity. Pure noise at a tiny perplexity breaks into clumps that do not exist.
~~~

How to read these maps without fooling yourself (the Distill article on t-SNE demonstrates each trap):

- **Cluster sizes mean nothing.** Every point gets the same number of effective neighbours, so dense clusters expand and sparse ones shrink.
- **Distances between clusters mean little**, in either direction.
- **Settings change the story.** Try several perplexities or `n_neighbors` and seeds; trust only structure that survives.
- **Random noise can look clustered** at low perplexity.
- **Do not cluster on the 2-D output.** Cluster in the original or PCA space and use the map to show the result.

A standard recipe: PCA to about 50 dimensions (less noise, faster neighbour search), then t-SNE or UMAP to 2.

| Method | Preserves | New points | Scale | Use for
| --- | --- | --- | --- | --- |
| PCA | Global variance, linear structure | Yes (linear map) | Millions, fast | Features, compression, denoising
| t-SNE | Local neighbourhoods | Not in the standard form | About $10^5$ with Barnes-Hut | Visual exploration
| UMAP | Local plus some global layout | Yes, approximately | Millions | Visual exploration, sometimes features

## Checking the result

For PCA used as features:

- **Cumulative variance curve:** if it rises in a near-straight line, there is no low-dimensional structure to find.
- **Validation score with and without PCA:** if PCA improves neither speed nor score, leave it out.
- **Loadings** (`pca.components_`): which original features weigh heavily in each component. Interpret loosely.

For t-SNE and UMAP maps:

- **Colour by known labels** or a feature; structure that matches something you know is more trustworthy.
- **Rerun with 2-3 settings and seeds** and report only what stays the same.
- **Confirm in the original space** with a classifier or distance measure.

## What goes wrong

The biggest failure: PCA ranks directions by variance, not by usefulness for your label, so a small consistent signal in a low-variance direction is the first thing it discards.

~~~dg-fig 05.04-lowvar.svg
The two classes are separated across the narrow side of a long cloud. PC1 runs along the cloud, so keeping only PC1 mixes the classes completely.
~~~

Here a logistic regression on PC1 alone scores 50% (coin-flipping); on PC2, the "unimportant" 7% direction, it scores 99.5%. When prediction is the goal, compare validation scores with and without PCA, or use a supervised reduction such as LDA (module 03) or regularisation.

| Mistake | Symptom | Fix
| --- | --- | --- |
| Not scaling mixed units | PC1 is just the column with the biggest units | Standardise first (unless units are shared and meaningful)
| Fitting PCA on all data | Validation score slightly too optimistic | Put PCA inside a pipeline, fit on training folds only
| Choosing m by a 95% rule only | Model worse than without PCA | Tune m by validation score
| Signal in a low-variance direction | PCA features predict badly | Compare with no PCA; use LDA or regularisation
| Whitening without $\epsilon$ | Noisy, unstable features; huge values | Add $\epsilon$, or drop tiny components first
| Reading t-SNE cluster sizes or gaps | "Segment B is twice as diverse" | Measure spread and distances in the original space
| Clustering the 2-D map | Clusters change with seed or settings | Cluster in original or PCA space; map is for display

## Going deeper

**Why PCA is optimal.** The Eckart-Young theorem says the best rank-$m$ approximation of $X_c$ in squared (Frobenius) error is the truncated SVD $U_m S_m V_m^\top$, and its error is $\sum_{j>m} s_j^2$. Dividing by $n-1$ gives the "sum of dropped eigenvalues" you checked by hand. By Pythagoras, kept plus lost squared length equals the total, so maximising one minimises the other.

**Computation.** A full SVD costs about $O(n d \min(n, d))$. For large data scikit-learn uses **randomised SVD** (roughly $O(n d m)$ for the top $m$ directions) or `IncrementalPCA` for batches. The SVD of $X_c$ is also more accurate than forming $C$, for the condition-number reason that favours `lstsq` in 02.01.

**Probabilistic PCA.** PCA is the zero-noise limit of the latent model $x = W z + \mu + \varepsilon$, $z \sim \mathcal N(0, I)$; fitting it with EM (05.03) handles missing values and gives a likelihood for scoring $m$.

**Beyond linear.** **Kernel PCA** runs PCA in a feature space defined by a kernel (02.06) and can unroll curved structure; **autoencoders** learn non-linear encoders and decoders with a neural network, and a linear autoencoder with squared loss recovers exactly the PCA subspace.

**t-SNE details.** Each width $\sigma_i$ is found by binary search so that $2^{H(P_i)}$ (entropy in bits) equals the perplexity; then $p_{ij} = (p_{j|i} + p_{i|j})/2n$. Barnes-Hut cuts each step from $O(n^2)$ to $O(n \log n)$, and "early exaggeration" (multiplying $P$ by about 12 at first) lets clusters form early. The loss is non-convex, so seeds matter; PCA initialisation (scikit-learn's default) steadies the global layout.

**UMAP details.** UMAP minimises a cross-entropy between its fuzzy neighbour graph and the 2-D graph using negative sampling, which is why it scales. Its extra global structure comes largely from a spectral initialisation and is still not a faithful distance map.

## Key terms

| Term | Meaning
| --- | --- |
| Principal component | A direction of maximal remaining variance; an eigenvector of the covariance matrix
| Eigenvalue $\lambda_j$ | The variance of the data along component $j$
| Explained variance ratio | One component's share of the total variance, $\lambda_j / \sum_k \lambda_k$
| Scores | A point's coordinates on the kept components, $V_m^\top(x - \mu)$
| Reconstruction error | What projection loses; on average, the sum of the dropped eigenvalues
| Scree plot | Bar chart of variance per component, used to spot an elbow
| Whitening (PCA / ZCA) | Rescaling components to unit variance; ZCA then rotates back to the original axes
| Perplexity | t-SNE's effective number of neighbours per point
| `n_neighbors` / `min_dist` | UMAP's neighbourhood size and how tightly it packs points

<div class="take">
<div class="lab">Key takeaway</div>

use PCA, fitted on training data and tuned by validation, to reduce features, remembering that it keeps variance, not predictive signal; whiten with an $\epsilon$; use t-SNE/UMAP only to look at neighbourhoods, never to read cluster sizes or distances between clusters.

</div>
