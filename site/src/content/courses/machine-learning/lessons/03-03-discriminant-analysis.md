---
id: "03.03"
course: "machine-learning"
module: "03"
slug: "discriminant-analysis"
title: "Discriminant analysis"
keyIdeas: ["Discriminant analysis is a generative classifier with Gaussian class-conditionals $p(x \\mid y = k) = \\mathcal{N}(\\mu_k, \\Sigma_k)$, fitted by class means, covariances and frequencies: no iterative optimisation.", "**QDA** gives each class its own $\\Sigma_k$, so boundaries are quadratic; it needs $K\\,d(d+1)/2$ covariance parameters and breaks when a class has fewer than $d$ samples.", "**LDA** shares one $\\Sigma$ across classes; the quadratic terms cancel and the boundary is linear, $w = \\Sigma^{-1}(\\mu_1 - \\mu_0)$ for two classes.", "LDA and logistic regression have the same functional form but are fitted differently: LDA is more data-efficient if the Gaussian assumption holds, logistic regression more robust when it doesn't.", "Shrinking covariance estimates (toward a pooled or diagonal matrix) interpolates between QDA, LDA and Gaussian naive Bayes."]
question: "You have 200 features, 3 classes and 150 training samples per class. QDA crashes or performs terribly, LDA does reasonably, and LDA with shrinkage does best. Explain each outcome in terms of what is being estimated."
answer: "QDA estimates a separate $200 \\times 200$ covariance per class, 20,100 free parameters each, from 150 points. A sample covariance from 150 points has rank at most 149, so it is singular and cannot be inverted. Even with a pseudo-inverse or a tiny ridge, the near-zero eigenvalues make the Mahalanobis terms explode along directions the data never explored. LDA pools all 450 points into one covariance, rank up to 447, so it is invertible. But with 20,100 parameters from 450 points the small eigenvalues are still underestimated and the inverse is noisy, which costs accuracy. Shrinkage (Ledoit-Wolf, or blending toward a scaled identity or diagonal) trades a little bias for a big drop in variance: it lifts the small eigenvalues and stabilises $\\Sigma^{-1}$. That is the bias-variance trade from module 01, applied to the covariance estimate instead of the weights."
sources: [{"title": "scikit-learn: Linear and Quadratic Discriminant Analysis", "url": "https://scikit-learn.org/stable/modules/lda_qda.html", "note": "The maths of both, LDA as dimensionality reduction, and Ledoit-Wolf shrinkage with the lsqr solver."}, {"title": "The Elements of Statistical Learning, ch. 4.3", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "LDA, QDA, regularised discriminant analysis and Fisher's reduced-rank view."}, {"title": "An Introduction to Statistical Learning", "url": "https://www.statlearning.com/", "note": "Chapter 4 compares LDA, QDA, naive Bayes and logistic regression on the same problems. Free PDF."}]
practice: ["linear-discriminant-analysis", "gaussian-naive-bayes", "mahalanobis-distance", "maximum-likelihood-estimation", "gradient-of-multivariate-gaussian-log-pdf"]
tags: ["generative models", "covariance shrinkage"]
tools: ["NumPy", "scikit-learn"]
---

## Gaussian class-conditionals

Naive Bayes (03.01) made the class-conditional density easy by assuming independent features. Discriminant analysis keeps the correlations but assumes a shape: each class is a multivariate Gaussian,

$$p(x \mid y = k) = \mathcal{N}(x;\, \mu_k, \Sigma_k).$$

Fitting is plug-in maximum likelihood: class frequencies $\hat\pi_k$, class means $\hat\mu_k$ and covariance estimates. Classification picks the largest **discriminant score**, the log of prior times density with shared constants dropped:

$$\delta_k(x) = -\tfrac12\log|\Sigma_k| - \tfrac12(x - \mu_k)^\top\Sigma_k^{-1}(x - \mu_k) + \log\pi_k.$$

The middle term is minus half the squared Mahalanobis distance from $x$ to the class mean, so the rule is "nearest mean, measured in each class's own covariance, adjusted for prior and spread".

~~~dg-flow Discriminant analysis fits in one pass: statistics per class, then a score per class at prediction time.
[Fit] Labelled data -(per class)-> Means and priors -(per class or pooled)-> *Covariance estimates
[Predict] New x -(Mahalanobis distance to each mean)-> Discriminant scores -(argmax)-> Class
~~~

## QDA: one covariance per class

With a separate $\Sigma_k$ per class, $\delta_k$ is quadratic in $x$ and so are the boundaries $\delta_k = \delta_j$: ellipses, parabolas, hyperbolas. That flexibility costs parameters, $K\big(d + d(d+1)/2\big)$ in all; for $d = 50$ and $K = 3$, that is 3,975. Each $\Sigma_k$ is estimated from that class's $n_k$ points only, and if $n_k \le d$ the estimate is singular and $\Sigma_k^{-1}$ does not exist. Even with $n_k$ a few times $d$, the smallest eigenvalues are badly underestimated and their inverses blow up the scores.

## LDA: one shared covariance

Assume all classes share $\Sigma$, estimated by pooling within-class scatter from every class. Then $-\tfrac12 x^\top\Sigma^{-1}x$ is the same for every class and cancels, leaving a linear score:

$$\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \tfrac12\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k.$$

For two classes the boundary is the hyperplane with normal $w = \Sigma^{-1}(\mu_1 - \mu_0)$. If $\Sigma$ were the identity, this would point straight from one mean to the other; $\Sigma^{-1}$ tilts it away from directions where the classes are naturally spread out.

**LDA versus logistic regression.** Both give $\log\frac{p(y=1 \mid x)}{p(y=0 \mid x)} = w^\top x + b$. LDA estimates $w$ from means and a covariance, using the full joint distribution of $x$. Logistic regression maximises the conditional likelihood and ignores how $x$ is distributed. When the classes really are Gaussian with shared covariance, LDA extracts more from each example. When they aren't (heavy tails, outliers, binary or skewed features), logistic regression is more robust, because a far-away outlier moves LDA's mean and covariance but barely moves logistic regression's boundary.

**Fisher's view.** Independently of any Gaussian story, LDA's directions maximise between-class variance relative to within-class variance. There are at most $K - 1$ such directions, which makes LDA a supervised dimensionality reduction: project 50 features onto 2 directions to plot a 3-class problem. In scikit-learn, `LinearDiscriminantAnalysis.transform` does exactly this.

## The family, and shrinkage

| Model | Covariance assumption | Boundary | Covariance parameters | Good when
| --- | --- | --- | --- | --- |
| Gaussian naive Bayes | diagonal, per class | quadratic, axis-aligned | K d | many features, little data
| LDA | full, shared | linear | d(d+1)/2 | moderate data, similar class shapes
| QDA | full, per class | quadratic | K d(d+1)/2 | plenty of data per class, different shapes
| Logistic regression | none (discriminative) | linear | none | non-Gaussian features, enough data

These are points on a continuum. **Regularised discriminant analysis** blends $\Sigma_k(\alpha) = \alpha\hat\Sigma_k + (1 - \alpha)\hat\Sigma_{\text{pooled}}$ to move between QDA and LDA, and shrinks further toward a scaled identity to stabilise inverses. **Ledoit-Wolf shrinkage** chooses the shrinkage amount analytically; in scikit-learn, `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` uses it and is the right default when $d$ is not small relative to $n$.

```python
import numpy as np
def lda_fit(X, y):
    classes = np.unique(y)
    mu = np.stack([X[y == c].mean(0) for c in classes])
    Xc = X - mu[np.searchsorted(classes, y)]          # within-class centring
    S = Xc.T @ Xc / (len(X) - len(classes))           # pooled covariance
    W = np.linalg.solve(S, mu.T).T                    # rows: Sigma^-1 mu_k
    b = -0.5 * (W * mu).sum(1) + np.log([np.mean(y == c) for c in classes])
    return classes, W, b                              # score = X @ W.T + b
```

<div class="ex">
<div class="lab">Worked example</div>

**priors and unequal variances in one dimension**

Class 0 is $\mathcal{N}(0, 1)$ and class 1 is $\mathcal{N}(2, 1)$. With equal priors the LDA boundary is the midpoint, $x = 1$.

With $\pi_1 = 0.2$: the boundary moves to $x = 1 + \sigma^2\log(\pi_0/\pi_1)/(\mu_1 - \mu_0) = 1 + \ln 4/2 = 1.69$, away from the rarer class.

Now let class 1 have variance 4 (QDA, equal priors). Setting $-\tfrac12 x^2 = -\tfrac12\ln 4 - (x-2)^2/8$ gives $3x^2 + 4x - 9.55 = 0$, so $x = 1.24$ or $x = -2.57$. Class 0 wins only between them. The wider class claims both tails, a boundary no linear model can produce.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

discriminant analysis is Gaussian generative classification. Shared covariance gives LDA's linear boundary, per-class covariance gives QDA's quadratic one, and shrinkage between them is how you make either work when features are many and samples few.

</div>
