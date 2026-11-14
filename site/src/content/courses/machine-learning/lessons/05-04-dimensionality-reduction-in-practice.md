---
id: "05.04"
course: "machine-learning"
module: "05"
slug: "dimensionality-reduction-in-practice"
title: "Dimensionality reduction in practice"
keyIdeas: ["PCA projects centred data onto the top eigenvectors of its covariance (via SVD). In an ML pipeline it is fitted on the training split only and chosen by explained variance or by validation score.", "PCA keeps high-variance directions, not predictive ones: a low-variance direction can carry all the signal, so check downstream performance rather than trusting a 95% variance rule.", "Whitening rescales PCA coordinates to unit variance ($\\Lambda^{-1/2} U^\\top x$); ZCA rotates back to the original axes. Both amplify noise in tiny-eigenvalue directions, so add an $\\epsilon$.", "t-SNE and UMAP make 2-D maps that preserve local neighbourhoods. Cluster sizes and between-cluster distances in the map are not meaningful, and results depend on perplexity or `n_neighbors` and on the seed.", "Use PCA for features, compression and denoising; use t-SNE/UMAP for looking at data, typically after PCA to about 50 dimensions."]
question: "A teammate shows a UMAP plot of customer embeddings with two clusters far apart and concludes that the two segments are 'very different', then runs k-means with k=2 on the 2-D UMAP coordinates to define the segments. What is wrong with each step, and what would you do instead?"
answer: "UMAP preserves local neighbourhoods. The distance between clusters in the map is not a calibrated measure of how different they are in the original space, so 'far apart' cannot be read as 'very different'. The layout also changes with `n_neighbors`, `min_dist` and the random seed. Running k-means on the 2-D coordinates defines segments with distortions the embedding introduced, and new customers would get segments that depend on a stochastic, non-linear map. Instead I would measure the difference in the original or PCA space: compare cluster centroids, between-cluster and within-cluster distances, or train a classifier to separate the groups and look at its accuracy. I would also cluster in that space, using k-means, GMM or HDBSCAN on standardised or PCA-reduced features, and evaluate with silhouette and stability. The UMAP plot is then only for display, and the conclusion should hold across several UMAP settings."
sources: [{"title": "Wattenberg, Viégas, Johnson: How to Use t-SNE Effectively (Distill)", "url": "https://distill.pub/2016/misread-tsne/", "note": "Interactive demonstrations of perplexity, cluster sizes, distances and seeing clusters in noise."}, {"title": "van der Maaten and Hinton (2008): Visualizing Data using t-SNE", "url": "https://www.jmlr.org/papers/v9/vandermaaten08a.html", "note": "The original t-SNE paper; the crowding problem and the Student-t kernel."}, {"title": "McInnes, Healy, Melville: UMAP", "url": "https://arxiv.org/abs/1802.03426", "note": "The UMAP paper; for practical parameter guidance see the umap-learn docs."}, {"title": "scikit-learn User Guide: Decomposing signals in components (PCA)", "url": "https://scikit-learn.org/stable/modules/decomposition.html", "note": "PCA, whitening, incremental and randomised PCA, and truncated SVD for sparse data."}]
practice: ["pca", "pca-projection", "whitening-transform", "t-sne-affinity-matrix", "svd-components"]
tags: ["PCA", "whitening", "t-SNE"]
tools: ["scikit-learn", "UMAP"]
---

## Where this fits

The mathematics of PCA (covariance, eigenvectors, SVD) is covered in ML Maths, in the lessons on SVD and principal component analysis. This lesson is about using dimensionality reduction inside ML work: what to fit on, how many components to keep, when whitening helps, and how to read 2-D embeddings without fooling yourself.

There are two different goals, and they call for different tools:

- **Reduce features for a model**: compress, denoise, decorrelate, speed things up. The transform must apply to new data. Use PCA (or truncated SVD for sparse data).
- **Look at the data**: a 2-D picture showing which points are neighbours. Use t-SNE or UMAP, and treat the picture as a hint, not a measurement.

## PCA in a pipeline

PCA centres the data, takes the SVD $X_c = U S V^\top$, and keeps the top $m$ right singular vectors as directions. The variance along component $j$ is $\lambda_j = s_j^2 / (n-1)$, and the explained variance ratio is $\lambda_j / \sum_k \lambda_k$.

~~~dg-flow A typical pipeline. Every step is fitted on the training split only, then applied unchanged to validation and test.
Raw features -(fit on train)-> Standardise -(fit on train)-> *PCA to m dims -> Model
~~~

Practical rules:

- **Scale first** when features have different units; otherwise the largest-unit feature becomes PC1. Skip scaling when features share a unit and their variance is meaningful (pixel intensities, for example).
- **Fit on train only.** Fitting PCA on all data leaks information about the test distribution, a small but real leak (module 07 covers pipelines).
- **Choose $m$** with a variance target such as 90-99%, a scree plot, or, best for a downstream model, as a hyperparameter tuned by validation score.
- **Sign and order are arbitrary**: components can flip sign between fits, so do not hard-code their interpretation.

<div class="ex">
<div class="lab">Worked example</div>

**choosing m from eigenvalues**

A standardised 5-feature dataset has eigenvalues 4.0, 2.5, 1.0, 0.3, 0.2 (total 8.0).

Cumulative explained variance: 50%, 81.3%, 93.8%, 97.5%, 100%. A 95% target keeps 4 components; a 90% target keeps 3.

But if the label depends on the direction with eigenvalue 0.2, both choices throw the signal away. PCA never looks at the label.

</div>

That last point is the main failure mode. PCA is unsupervised: it ranks directions by variance, not by usefulness. A small, consistent signal hidden in a low-variance direction (a calibration offset, a rare-class indicator) is the first thing PCA discards. When prediction is the goal, compare validation scores with and without PCA, or use a supervised reduction such as LDA (module 03) or plain regularisation.

## Whitening

PCA leaves components with different variances $\lambda_j$. Whitening rescales them to unit variance, so the transformed data has identity covariance:

$$x_{\text{PCA-white}} = \Lambda^{-1/2} U^\top (x - \mu) \qquad x_{\text{ZCA}} = U \Lambda^{-1/2} U^\top (x - \mu)$$

where $U$ holds the eigenvectors of the covariance and $\Lambda$ its eigenvalues. ZCA whitening rotates back to the original axes, so the output looks as similar as possible to the input (whitened images still look like images, with edges enhanced). Whitening helps methods that assume isotropic inputs, such as k-means, ICA, and some older image pipelines. The danger is division by tiny eigenvalues: directions of pure noise get scaled up to unit variance, as loud as the real signal. Always use $(\Lambda + \epsilon I)^{-1/2}$ or drop the smallest components first.

## t-SNE and UMAP for visualisation

t-SNE (van der Maaten and Hinton, 2008) converts distances into neighbour probabilities. Each point gets a Gaussian whose width is set so that its effective number of neighbours, the perplexity (typically 5-50), is fixed. It then places points in 2-D so that low-dimensional similarities, computed with a heavy-tailed Student-t kernel, match the high-dimensional ones by minimising $\mathrm{KL}(P \parallel Q)$. The heavy tails let moderately distant points sit far apart in the map, which avoids crowding everything into the centre.

UMAP (McInnes et al., 2018) builds a weighted $k$-nearest-neighbour graph (`n_neighbors`, typically 15) and optimises a 2-D layout whose graph matches it, with `min_dist` controlling how tightly points pack. It is usually faster than t-SNE, scales to millions of points, keeps somewhat more global arrangement, and has a `transform` for new points.

How to read these maps without fooling yourself (the Distill article on t-SNE shows each trap):

- **Cluster sizes mean nothing.** t-SNE expands dense clusters and contracts sparse ones.
- **Distances between clusters mean little.** Two clusters far apart in the map may be close in the original space.
- **Hyperparameters change the story.** Try several perplexities or `n_neighbors` and several seeds; trust only structure that survives.
- **Random noise can look clustered** at low perplexity.
- **Do not cluster on the 2-D output** and report it as the structure of the data; cluster in the original or PCA space and use the map to show the result.

| Method | Preserves | New points | Scale | Use for |
| --- | --- | --- | --- | --- |
| PCA | Global variance, linear structure | Yes (linear map) | Millions, fast | Features, compression, denoising |
| t-SNE | Local neighbourhoods | Not in the standard form | About $10^5$ with Barnes-Hut | Visual exploration |
| UMAP | Local plus some global layout | Yes, approximately | Millions | Visual exploration, sometimes features |

A standard recipe is PCA to about 50 dimensions (removing noise and speeding up neighbour search), then t-SNE or UMAP to 2.

<div class="take">
<div class="lab">Key takeaway</div>

use PCA, fitted on training data and tuned by validation, to reduce features; whiten with an $\epsilon$; use t-SNE/UMAP only to look at neighbourhoods, never to read cluster sizes or distances between clusters.

</div>
