---
id: "03.03"
course: "machine-learning"
module: "03"
slug: "discriminant-analysis"
title: "Discriminant analysis"
keyIdeas: ["Discriminant analysis is a **generative classifier**: each class is modelled as a Gaussian cloud $\\mathcal{N}(\\mu_k, \\Sigma_k)$, and a new point goes to the class with the highest prior times density. Fitting is just class counts, means and covariances, with no iterative optimisation.", "The decision rule is \"nearest class mean\" measured with the **Mahalanobis distance**, which accounts for how each class is spread and tilted, plus a $\\log\\pi_k$ bonus for common classes.", "**LDA** shares one covariance across classes, so the quadratic terms cancel and the boundary is linear, with $w = \\Sigma^{-1}(\\mu_1 - \\mu_0)$ for two classes. **QDA** keeps one covariance per class, giving curved boundaries but needing $K\\,d(d+1)/2$ covariance parameters, and it breaks when a class has no more samples than features.", "LDA and logistic regression have the same linear log-odds form but are fitted differently: LDA is more data-efficient when the Gaussian assumption holds, logistic regression more robust when it doesn't.", "Shrinking the covariance estimate (Ledoit-Wolf, `shrinkage=\"auto\"`) stabilises its inverse when features are many and samples few, and blending per-class, pooled and diagonal covariances moves between QDA, LDA and Gaussian naive Bayes.", "Fisher's view gives the same LDA direction without Gaussians: maximise between-class over within-class spread. It yields at most $K - 1$ directions, so LDA doubles as a supervised dimensionality reduction."]
question: "You are classifying tissue samples into 3 types from 200 gene-expression features, with 150 training samples per type. QDA crashes or performs terribly, plain LDA does reasonably, and LDA with shrinkage does best. Explain each outcome in terms of what is being estimated, and say what you would try next if shrunk LDA still underperforms logistic regression."
answer: "QDA estimates a separate $200 \\times 200$ covariance per class, $200 \\cdot 201/2 = 20{,}100$ free parameters each, from only 150 points. A sample covariance from 150 points has rank at most 149, so it is singular and cannot be inverted; scikit-learn warns that variables are collinear. Even with a pseudo-inverse or a tiny ridge, the near-zero eigenvalues make the Mahalanobis terms explode along directions the data never explored, so predictions are dominated by noise. LDA pools all 450 points into one covariance with rank up to $450 - 3 = 447$, so it is invertible. But 20,100 parameters from 450 points is still a noisy estimate: the small eigenvalues are underestimated, their inverses are too large, and accuracy suffers. Shrinkage (Ledoit-Wolf, or blending toward a scaled identity or diagonal) trades a little bias for a big drop in variance: it lifts the small eigenvalues and stabilises $\\Sigma^{-1}$. That is the bias-variance trade from module 01, applied to the covariance estimate instead of to the weights. If shrunk LDA still loses to (regularised) logistic regression, the Gaussian shared-covariance assumption is probably wrong: expression data is often skewed and heavy-tailed. Next steps: log-transform the features, check the discriminant-axis plot for multimodal or non-elliptical classes, try a regularised QDA (`reg_param`) in case class shapes differ, and compare calibrated probabilities as well as accuracy with cross-validation."
sources: [{"title": "scikit-learn: Linear and Quadratic Discriminant Analysis", "url": "https://scikit-learn.org/stable/modules/lda_qda.html", "note": "The maths of both, LDA as dimensionality reduction, and Ledoit-Wolf shrinkage with the lsqr solver."}, {"title": "The Elements of Statistical Learning, ch. 4.3", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "LDA, QDA, regularised discriminant analysis and Fisher's reduced-rank view."}, {"title": "An Introduction to Statistical Learning", "url": "https://www.statlearning.com/", "note": "Chapter 4 compares LDA, QDA, naive Bayes and logistic regression on the same problems. Free PDF."}]
practice: ["linear-discriminant-analysis", "gaussian-naive-bayes", "mahalanobis-distance", "maximum-likelihood-estimation", "gradient-of-multivariate-gaussian-log-pdf"]
tags: ["generative models", "covariance shrinkage"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- What discriminant analysis assumes (each class is a Gaussian cloud) and how to fit it in one pass, with no optimiser.
- Why sharing one covariance across classes (**LDA**) gives a straight boundary, and separate covariances (**QDA**) give curved ones.
- How to classify a point by hand with the Mahalanobis distance, and how priors move the boundary.
- When to pick LDA, QDA, shrinkage or logistic regression, and how LDA reduces dimensions.

*Before you start:* Bayes' rule and generative classifiers (naive Bayes, 03.01), distances (kNN, 03.02), means and covariance matrices (ML Maths, module 08), and logistic regression (02.03) for comparison.

## From neighbours back to models

The previous lesson (03.02) classified a point by its nearest training examples, with no model but a heavy dependence on how distance is measured. Naive Bayes (03.01) built a probability model of each class, at the cost of pretending the features are independent.

Discriminant analysis sits between the two. Like naive Bayes it is a **generative classifier**: it models what each class's inputs look like, then uses Bayes' rule. Unlike naive Bayes it keeps the correlations between features. And like kNN, the final rule is "go to the nearest class", with "nearest" measured by a distance that respects each class's shape.

## The idea in one picture

A seed bank measures the length and width of seeds from two varieties. Here is a summary of 80 seeds (spread and correlation are pooled over both varieties, which is what LDA will use):

| Variety | Seeds | Mean length (mm) | Mean width (mm) | Std. dev. of length, width | Length-width correlation
| --- | --- | --- | --- | --- | --- |
| A | 40 | 5.24 | 2.98 | 0.50, 0.47 | 0.73
| B | 40 | 6.41 | 3.47 | 0.50, 0.47 | 0.73

Longer seeds tend to be wider (correlation 0.73), so each class is a tilted oval cloud. Discriminant analysis draws a Gaussian bell over each cloud and asks, for a new seed: under which bell is it more likely, given how common each variety is?

~~~dg-fig 03.03-idea.svg
Each variety gets a fitted Gaussian (ellipses at 1 and 2 standard deviations around the class mean, marked X). Because the two ellipses have the same shape, the boundary where the two bells are equally high is a straight line.
~~~

Reading the figure: the boundary is not the plain halfway line between the means. It is tilted, because "long and wide" is typical of both varieties' shape while "long but narrow" is unusual for both, and the model has measured that shape.

New terms: the **class-conditional density** $p(x \mid y = k)$ describes what inputs of class $k$ look like; the **prior** $\pi_k$ is how common class $k$ is (here 0.5 each); the **covariance matrix** $\Sigma$ holds each feature's variance on its diagonal and how pairs move together off it, setting the size and tilt of the ellipses.

## Step 1: assume each class is a Gaussian cloud

The core assumption is that the features of class $k$ follow a multivariate normal distribution:

$$p(x \mid y = k) = \mathcal{N}(x;\, \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}} \exp\!\Big(-\tfrac12 (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k)\Big).$$

Here $x$ has $d$ features, $\mu_k$ is the class centre, $\Sigma_k$ its $d \times d$ covariance and $|\Sigma_k|$ its determinant (the cloud's volume). Only the exponent depends on $x$, and it has a name: the **Mahalanobis distance** from $x$ to class $k$,

$$D_k^2(x) = (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k).$$

It is an ordinary squared distance after stretching space so the class's ellipse becomes a circle; with $\Sigma_k = I$ it is the Euclidean distance. **Number check:** in one dimension with $\sigma^2 = 4$, a point 3 from the mean has $D^2 = 9/4 = 2.25$, so it is $D = 1.5$ standard deviations away.

## Step 2: fit by counting and averaging

There is no loss to minimise by gradient descent. The maximum-likelihood fit (the idea from 01.01: choose parameters that make the training data most probable) has closed-form answers:

$$\hat\pi_k = \frac{n_k}{n}, \qquad \hat\mu_k = \frac{1}{n_k}\sum_{i:\,y_i = k} x_i, \qquad \hat\Sigma_k = \frac{1}{n_k - 1}\sum_{i:\,y_i = k} (x_i - \hat\mu_k)(x_i - \hat\mu_k)^\top.$$

Here $n_k$ counts class $k$'s points and $n$ all points: the prior is the class's share, the mean its average, the covariance the average outer product of deviations from its own mean. (Strict maximum likelihood divides by $n_k$ rather than $n_k - 1$; the boundary barely changes.)

~~~dg-flow Discriminant analysis fits in one pass: statistics per class, then a score per class at prediction time.
[Fit] Labelled data -(per class)-> Means and priors -(per class or pooled)-> *Covariance estimates
[Predict] New x -(Mahalanobis distance to each mean)-> Discriminant scores -(argmax)-> Class
~~~

## Step 3: turn densities into a decision

Bayes' rule says $p(y = k \mid x) \propto \pi_k\, p(x \mid y = k)$; pick the largest. Taking logs and dropping constants shared by every class gives the **discriminant score**:

$$\delta_k(x) = -\tfrac12\log|\Sigma_k| - \tfrac12 D_k^2(x) + \log\pi_k.$$

Read it term by term:

- $-\tfrac12 D_k^2(x)$: closer to the class centre (in that class's own units) is better.
- $-\tfrac12\log|\Sigma_k|$: a wide class spreads its probability thinly, so it pays a penalty at any single point.
- $\log\pi_k$: common classes get a head start.

For probabilities rather than a label, apply the softmax: $p(y = k \mid x) = e^{\delta_k} / \sum_j e^{\delta_j}$.

The figure shows these effects in one dimension; the boundary is wherever the prior-times-density curves cross.

~~~dg-fig 03.03-1d.svg
Left: equal priors and equal spreads put the boundary at the midpoint, 1. Middle: when class 1 is only 20% of the data its curve shrinks and the boundary moves to 1.69, away from the rare class. Right: a wider class 1 crosses class 0 twice, so class 1 wins both tails (shaded) and class 0 keeps only the middle.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**priors and unequal variances in one dimension**

Class 0 is $\mathcal{N}(0, 1)$ and class 1 is $\mathcal{N}(2, 1)$. With equal priors and equal variances the scores differ only through $-\tfrac12 (x - \mu_k)^2$, which are equal at the midpoint $x = 1$.

Now let $\pi_1 = 0.2$, $\pi_0 = 0.8$. The boundary solves $-\tfrac12 x^2 + \ln 0.8 = -\tfrac12 (x-2)^2 + \ln 0.2$. Expanding, $-\tfrac12 x^2$ cancels on both sides: $0 = 2x - 2 + \ln 0.2 - \ln 0.8$, so $x = 1 + \tfrac12\ln 4 = 1 + 0.693 = 1.69$. In general the shift is $\sigma^2\log(\pi_0/\pi_1)/(\mu_1 - \mu_0)$, away from the rarer class.

Next, equal priors but class 1 has variance 4 (so this is QDA). Its score is $-\tfrac12\ln 4 - (x-2)^2/8$, against class 0's $-\tfrac12 x^2$. Setting them equal and multiplying by $-8$: $4x^2 = 4\ln 4 + (x - 2)^2$, i.e. $3x^2 + 4x - 4 - 5.545 = 0$, i.e. $3x^2 + 4x - 9.545 = 0$.

Quadratic formula: $x = \big(-4 \pm \sqrt{16 + 114.54}\big)/6 = (-4 \pm 11.426)/6$, so $x = 1.24$ or $x = -2.57$. Class 0 wins only between them. The wider class claims both tails, a boundary no linear model can produce.

</div>

## Step 4: LDA, one shared covariance

**Linear discriminant analysis (LDA)** assumes all classes have the same shape: one $\Sigma$ for everyone. It is estimated by **pooling**, which means adding up the within-class scatter of every class and dividing by $n - K$ (with $K$ classes):

$$\hat\Sigma = \frac{1}{n - K}\sum_{k}\sum_{i:\,y_i = k}(x_i - \hat\mu_k)(x_i - \hat\mu_k)^\top.$$

With a shared $\Sigma$, two things cancel when you compare classes: the $\log|\Sigma|$ term, and the part of $D_k^2$ that is quadratic in $x$, namely $x^\top\Sigma^{-1}x$. What remains is linear in $x$:

$$\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \tfrac12\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k.$$

For two classes the boundary $\delta_1 = \delta_0$ is the hyperplane $w^\top x + b = 0$ with

$$w = \Sigma^{-1}(\mu_1 - \mu_0), \qquad b = -\tfrac12(\mu_1 + \mu_0)^\top\Sigma^{-1}(\mu_1 - \mu_0) + \log\frac{\pi_1}{\pi_0}.$$

If $\Sigma$ were the identity, $w$ would point straight from one mean to the other (plain nearest-mean classification). Multiplying by $\Sigma^{-1}$ tilts $w$ away from directions where the classes are naturally spread out, because a difference along a high-variance direction is weak evidence.

<div class="ex">
<div class="lab">Worked example</div>

**LDA by hand on six points**

Class 0: $(1, 1), (2, 3), (3, 2)$. Class 1: $(5, 4), (6, 6), (7, 5)$. Priors are equal, $\pi_0 = \pi_1 = 0.5$.

Means: $\mu_0 = \big(\tfrac{1+2+3}{3}, \tfrac{1+3+2}{3}\big) = (2, 2)$ and $\mu_1 = \big(\tfrac{5+6+7}{3}, \tfrac{4+6+5}{3}\big) = (6, 5)$.

Deviations from the class mean are $(-1, -1), (0, 1), (1, 0)$ in both classes. Per class: sum of squares in feature 1 is $1 + 0 + 1 = 2$, in feature 2 is $1 + 1 + 0 = 2$, cross products $1 + 0 + 0 = 1$.

Pooled covariance: add both classes and divide by $n - K = 6 - 2 = 4$: $\Sigma = \tfrac14\begin{pmatrix}4 & 2\\ 2 & 4\end{pmatrix} = \begin{pmatrix}1 & 0.5\\ 0.5 & 1\end{pmatrix}$.

Inverse: the determinant is $1 \cdot 1 - 0.5 \cdot 0.5 = 0.75$, so $\Sigma^{-1} = \tfrac{1}{0.75}\begin{pmatrix}1 & -0.5\\ -0.5 & 1\end{pmatrix} = \begin{pmatrix}4/3 & -2/3\\ -2/3 & 4/3\end{pmatrix}$.

Direction: $\mu_1 - \mu_0 = (4, 3)$, so $w = \big(\tfrac43 \cdot 4 - \tfrac23 \cdot 3,\ -\tfrac23 \cdot 4 + \tfrac43 \cdot 3\big) = (10/3,\ 4/3)$.

Offset: the midpoint of the means is $(4, 3.5)$, and with equal priors $b = -w^\top(4, 3.5) = -(40/3 + 14/3) = -18$. Score: $f(x) = \tfrac{10}{3}x_1 + \tfrac43 x_2 - 18$; positive means class 1. Multiplying by 3, the boundary is $10x_1 + 4x_2 = 54$.

Classify the new point $(5, 2)$: $f = 50/3 + 8/3 - 18 = 58/3 - 54/3 = 4/3 \approx 1.33 > 0$, so **class 1**, with probability $1/(1 + e^{-1.33}) \approx 0.79$.

Check with distances. Euclidean: $\|(5,2) - (2,2)\|^2 = 9$ and $\|(5,2) - (6,5)\|^2 = 1 + 9 = 10$, so plain nearest-mean says class 0. Mahalanobis: to class 0 the difference is $(3, 0)$ and $D_0^2 = 9 \cdot \tfrac43 = 12$; to class 1 it is $(-1, -3)$ and $D_1^2 = \tfrac43 + 12 - 2 \cdot 3 \cdot \tfrac23 = 9.33$. Class 1 is closer, and $\tfrac12(12 - 9.33) = 1.33$ matches the score exactly.

</div>

~~~dg-fig 03.03-mahal.svg
The six-point example. The star at (5, 2) is closer to the class 0 mean with a ruler (dashed nearest-mean boundary), but the shared covariance is tilted along the diagonal, so a point sitting off that diagonal is unusual for class 0. Measured in Mahalanobis units (dotted ellipses) it is closer to class 1, and LDA's boundary (solid) puts it there.
~~~

## Step 5: QDA, one covariance per class

**Quadratic discriminant analysis (QDA)** drops the shared-shape assumption and keeps a separate $\Sigma_k$ for each class. Now $x^\top\Sigma_k^{-1}x$ differs between classes and does not cancel, so $\delta_k$ is quadratic in $x$ and the boundaries $\delta_k = \delta_j$ are conic sections: ellipses, parabolas, hyperbolas.

~~~dg-fig 03.03-lda-qda.svg
A tight class sitting inside a wide class with the same centre. LDA has to draw a straight line and gets 59% on the training data, barely above chance; QDA draws a closed ellipse around the tight class and gets 88%.
~~~

Each covariance has $d(d+1)/2$ free numbers, so QDA estimates $K\big(d + d(d+1)/2\big)$ means and covariances in all; for $d = 50$ and $K = 3$, that is $3 \times (50 + 1275) = 3975$. Each $\Sigma_k$ comes from that class's $n_k$ points alone. If $n_k \le d$ the estimate is singular (it has rank at most $n_k - 1$) and $\Sigma_k^{-1}$ does not exist. Even with $n_k$ a few times $d$, the smallest eigenvalues are underestimated and their inverses blow up the scores.

## Discriminant analysis in code

scikit-learn has both models. The from-scratch LDA below is the whole algorithm in seven lines, and it agrees with the library on every prediction.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA,
                                           QuadraticDiscriminantAnalysis as QDA)

def lda_fit(X, y):
    classes = np.unique(y)
    mu = np.stack([X[y == c].mean(0) for c in classes])
    Xc = X - mu[np.searchsorted(classes, y)]          # within-class centring
    S = Xc.T @ Xc / (len(X) - len(classes))           # pooled covariance
    W = np.linalg.solve(S, mu.T).T                    # rows: Sigma^-1 mu_k
    b = -0.5 * (W * mu).sum(1) + np.log([np.mean(y == c) for c in classes])
    return classes, W, b                              # score = X @ W.T + b

X, y = load_wine(return_X_y=True)                     # 178 wines, 13 features, 3 classes
classes, W, b = lda_fit(X, y)
ours = classes[np.argmax(X @ W.T + b, axis=1)]
print((ours == LDA().fit(X, y).predict(X)).mean())    # 1.0  (identical predictions)

for name, m in [("LDA", LDA()),
                ("LDA + shrinkage", LDA(solver="lsqr", shrinkage="auto")),
                ("QDA", QDA()),
                ("QDA, reg_param=0.1", QDA(reg_param=0.1))]:
    print(name, cross_val_score(m, X, y, cv=5).mean().round(3))
# LDA 0.966   LDA + shrinkage 0.966   QDA 0.955   QDA, reg_param=0.1 0.972

lda = LDA(n_components=2).fit(X, y)
Z = lda.transform(X)                                  # 13 features -> 2 discriminant axes
print(Z.shape, lda.explained_variance_ratio_.round(2))  # (178, 2) [0.69 0.31]
```

Notes on the code:

- `np.linalg.solve` computes $\Sigma^{-1}\mu_k$ without forming the inverse, which is faster and more accurate.
- Rescaling a feature does not change LDA's predictions (the covariance absorbs the units), so unlike kNN you need no `StandardScaler`. Shrinkage toward a scaled identity *is* scale-sensitive, so standardise when you use it.
- scikit-learn's `covariance_` divides by $n$ rather than $n - K$. On the six-point example every score is scaled by $6/4$, so `predict_proba` gives 0.88 rather than 0.79 for the star; the boundary is the same.

## Reading the result

Checks that tell you whether the Gaussian story is good enough:

- **Cross-validate against logistic regression and regularised QDA.** If LDA matches logistic regression, the linear boundary is fine; if regularised QDA wins clearly, class shapes differ.
- **Plot the first two discriminant axes** from `transform`. Roughly elliptical blobs are good; banana shapes or several clumps per class mean one Gaussian per class is the wrong picture.
- **Check calibration, not just accuracy.** The model trusts its Gaussian tails, so its probabilities are often overconfident on heavy-tailed data (lesson 03.04).
- **Inspect the smallest eigenvalues of $\hat\Sigma$.** Near-zero values mean an unreliable inverse: use shrinkage.

## Shrinkage: making the covariance trustworthy

Shrinkage blends the raw covariance estimate with something simpler:

$$\Sigma(\gamma) = (1 - \gamma)\,\hat\Sigma + \gamma\,\tfrac{\operatorname{tr}\hat\Sigma}{d}\, I.$$

Here $\gamma \in [0, 1]$ is the shrinkage amount and the second term is a round covariance with the same average variance. Blending lifts the tiny eigenvalues and stabilises $\Sigma^{-1}$. **Ledoit-Wolf shrinkage** picks $\gamma$ analytically from the data; in scikit-learn, `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` uses it.

~~~dg-fig 03.03-shrinkage.svg
Synthetic two-class data with 40 correlated features, where LDA's assumptions hold exactly. QDA is at chance until it has more samples than features and still lags at 400 per class. Plain LDA collapses with few samples. LDA with Ledoit-Wolf shrinkage is best everywhere and nearly reaches the best achievable accuracy (dotted) with enough data.
~~~

**Regularised discriminant analysis** blends per-class and pooled covariances, $\Sigma_k(\alpha) = \alpha\hat\Sigma_k + (1 - \alpha)\hat\Sigma_{\text{pooled}}$, so a single knob slides from QDA ($\alpha = 1$) to LDA ($\alpha = 0$). scikit-learn's `QDA(reg_param=r)` shrinks each class covariance toward the identity, which is why it beat plain QDA on wine.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| QDA warns "variables are collinear" or predicts one class everywhere | a class has fewer samples than features, so its covariance is singular | use LDA, `QDA(reg_param=...)`, or reduce features
| LDA accuracy drops as you add features | pooled covariance too noisy for the sample size | `solver="lsqr", shrinkage="auto"`
| A few extreme points move the boundary a lot | means and covariances are not robust to outliers | remove or cap outliers, use logistic regression, or a robust covariance estimate
| Good accuracy, overconfident probabilities | data has heavier tails than a Gaussian | calibrate (03.04) or use a discriminative model
| Rare class almost never predicted | the $\log\pi_k$ term pushes the boundary toward it | set `priors=` explicitly or tune the decision threshold
| Binary, count or skewed features perform poorly | Gaussian shape is badly wrong | transform (log, Box-Cox) or switch to naive Bayes or logistic regression

## LDA versus logistic regression

Both give log-odds that are linear in $x$: $\log\frac{p(y=1 \mid x)}{p(y=0 \mid x)} = w^\top x + b$. They differ in how they choose $w$ and $b$. LDA estimates them from means and a covariance, using the full joint distribution of $x$. Logistic regression maximises the conditional likelihood $p(y \mid x)$ directly and ignores how $x$ is distributed. When the classes really are Gaussian with shared covariance, LDA extracts more from each example and needs less data. When they are not (heavy tails, outliers, binary or skewed features), logistic regression is more robust: a far-away outlier moves LDA's mean and covariance but barely moves logistic regression's boundary. On perfectly separable data, unregularised logistic regression's weights grow without bound while LDA's stay finite.

## Going deeper

**Fisher's view.** Ronald Fisher derived LDA in 1936 without any Gaussian assumption. He asked for the direction $w$ that makes projected classes as separated as possible relative to their spread:

$$J(w) = \frac{w^\top S_B\, w}{w^\top S_W\, w},$$

where $S_B$ is the between-class scatter (how far apart the class means are) and $S_W$ is the within-class scatter (the pooled covariance, up to a constant). For two classes the maximiser is $w \propto S_W^{-1}(\mu_1 - \mu_0)$, the same direction as the Gaussian derivation.

~~~dg-fig 03.03-fisher.svg
Two elongated classes. Projecting onto the line joining their means mixes them (separation ratio 0.41). Projecting onto Fisher's direction, which avoids the long within-class axis, separates them three times better (1.20).
~~~

For $K$ classes the maximisers are the top eigenvectors of $S_W^{-1}S_B$. Since $S_B$ is built from $K$ means it has rank at most $K - 1$, so LDA is a **supervised dimensionality reduction** to at most $K - 1$ axes: wine's 13 features become the 2 axes `transform` returned above. Classifying with nearest mean in that reduced space gives the same answer as full LDA (with equal priors).

**Whitening.** If $\Sigma = LL^\top$, then $z = L^{-1}x$ has identity covariance and $D_k^2(x) = \|z - L^{-1}\mu_k\|^2$. LDA is literally "whiten, then nearest mean, corrected for priors", which is how solvers implement it.

**The family on one axis.** The covariance assumption is the only difference between these models:

| Model | Covariance assumption | Boundary | Covariance parameters | Good when
| --- | --- | --- | --- | --- |
| Gaussian naive Bayes | diagonal, per class | quadratic, axis-aligned | $Kd$ | many features, little data
| LDA | full, shared | linear | $d(d+1)/2$ | moderate data, similar class shapes
| QDA | full, per class | quadratic | $Kd(d+1)/2$ | plenty of data per class, different shapes
| Logistic regression | none (discriminative) | linear | none | non-Gaussian features, enough data

Shrinking toward a pooled covariance moves QDA toward LDA; shrinking toward a diagonal moves toward naive Bayes. Shrinkage picks your point on this bias-variance continuum.

## Key terms

| Term | Meaning
| --- | --- |
| Class-conditional density | $p(x \mid y = k)$, here a Gaussian with mean $\mu_k$ and covariance $\Sigma_k$
| Discriminant score | $\delta_k(x)$, the log of prior times density with shared constants dropped; the largest wins
| Mahalanobis distance | Distance measured in standard deviations along the class's own shape
| Pooled covariance | One covariance estimated from every class's within-class scatter
| LDA | Shared covariance, so the boundary is linear: $w = \Sigma^{-1}(\mu_1 - \mu_0)$
| QDA | Per-class covariance, so boundaries are quadratic; needs many samples per class
| Shrinkage | Blending the covariance estimate toward a simpler matrix to stabilise its inverse
| Fisher discriminant | The projection that maximises between-class over within-class spread; at most $K - 1$ of them

<div class="take">
<div class="lab">Key takeaway</div>

discriminant analysis fits a Gaussian to each class and picks the class with the highest prior-weighted density, which amounts to nearest mean in Mahalanobis distance. A shared covariance gives LDA's straight boundary, per-class covariances give QDA's curves, and shrinkage between them is what makes either work when features are many and samples few.

</div>
