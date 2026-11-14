---
id: "02.06"
course: "machine-learning"
module: "02"
slug: "kernels"
title: "Kernels"
keyIdeas: ["A **kernel** $k(x, x') = \\phi(x)^\\top\\phi(x')$ computes an inner product in a feature space without building $\\phi$; any algorithm that touches data only through inner products (the SVM dual, ridge regression) can be kernelised.", "By the representer theorem, the L2-regularised solution is $f(x) = \\sum_i \\alpha_i k(x_i, x)$: one coefficient per training point, not per feature, even when the feature space is infinite.", "The RBF kernel $e^{-\\gamma\\|x - x'\\|^2}$ is a similarity score: 1 for identical points, near 0 for distant ones. Small $\\gamma$ gives wide bumps and a near-linear model; large $\\gamma$ gives narrow bumps that memorise each point. Standardise features and tune $\\gamma$ together with $C$ or $\\lambda$.", "Kernel ridge regression solves $\\alpha = (K + \\lambda I)^{-1}y$: $O(n^3)$ time and $O(n^2)$ memory, so 50,000 points already need a 20 GB Gram matrix.", "Past that scale, approximate the kernel with explicit features (random Fourier features, Nyström) and go back to a linear model; the random-feature error shrinks like $1/\\sqrt{D}$."]
question: "You fit RBF kernel ridge regression and set gamma to an extremely large value. What does the Gram matrix approach, what do the predictions on training points and on new test points look like, and what does the opposite extreme (gamma near zero) do? How would you find a good value in practice?"
answer: "As $\\gamma \\to \\infty$, $k(x_i, x_j) = e^{-\\gamma\\|x_i - x_j\\|^2}$ goes to 0 for every distinct pair and stays 1 on the diagonal, so $K \\to I$. Then $\\alpha = (I + \\lambda I)^{-1}y = y/(1+\\lambda)$, and training predictions are $y_i/(1+\\lambda)$: near-memorisation, shrunk only by the ridge term. A new point is far, on that tiny length scale, from every training point, so all its kernel values are about 0 and it predicts about 0 (the mean, if $y$ was centred). That is a lookup table: training error near zero, test error near the variance of $y$. As $\\gamma \\to 0$, every kernel value approaches 1 and $K$ approaches the all-ones matrix, so the model can represent little beyond a constant. To second order the kernel behaves like a low-degree polynomial in $x$, so the fit becomes very smooth and underfits: training and test error are both high and close together. The useful $\\gamma$ sits between. In practice, standardise the features, then cross-validate $\\gamma$ and $\\lambda$ jointly on a logarithmic grid (for example $\\gamma$ from 0.001 to 100 and $\\lambda$ from 0.001 to 10), because a larger $\\gamma$ can be partly compensated by a larger $\\lambda$. Pick the cell with the best validation score, check it is not on the edge of the grid, and compare against a plain linear ridge baseline."
sources: [{"title": "Cornell CS4780 lecture notes: Kernels", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote13.html", "note": "Kernelising linear algorithms, well-defined kernels and kernel composition rules."}, {"title": "Random Features for Large-Scale Kernel Machines", "url": "https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html", "note": "Rahimi and Recht (NeurIPS 2007): random Fourier features and their approximation guarantee."}, {"title": "scikit-learn: Kernel ridge regression", "url": "https://scikit-learn.org/stable/modules/kernel_ridge.html", "note": "KernelRidge vs SVR, with timing comparisons that show the n-squared and n-cubed costs."}, {"title": "Gaussian Processes for Machine Learning", "url": "https://gaussianprocess.org/gpml/", "note": "Rasmussen and Williams, chapters 2 and 4: the GP view of kernel ridge and a catalogue of kernels. Free PDF."}]
practice: ["rbf-kernel-matrix", "euclidean-distance", "cholesky-decomposition", "svm-with-hinge-loss-sgd"]
tags: ["kernel trick", "RBF kernel"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- Why adding features such as $x^2$ makes a linear model bend, and why doing it explicitly gets expensive very quickly.
- What a **kernel** is: a shortcut that computes the inner product of two expanded feature vectors without building them.
- How the RBF kernel works as a similarity score, and how its width $\gamma$ moves a model from underfitting to memorising.
- How to fit kernel ridge regression by hand and with scikit-learn, and how to tune an RBF SVM.
- Why kernel methods stop scaling past tens of thousands of points, and how random features fix that.

*Before you start:* you need linear regression and its "linear in the weights" idea (02.01), ridge regression (02.02) and the SVM with its dual form (02.05). Comfort with dot products and the distance $\|x - x'\|$ helps; the ML Maths course (module 08) covers both.

## From straight lines to curves

Lesson 02.01 showed that "linear" refers to the weights: add columns such as $x^2$ and the same algorithm fits curves. The SVM of 02.05 added a second clue: in its dual form the data appear only through dot products $x_i^\top x_j$. This last lesson of the module combines the two. The **kernel trick** lets a linear method work in a huge (even infinite) feature space while only touching an $n \times n$ table of similarities.

## The idea in one picture

Here are eleven measurements on one feature $x$, with two classes:

| x | -3 | -2.5 | -2 | -1.2 | -0.6 | 0 | 0.5 | 1.1 | 2 | 2.6 | 3
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class | B | B | B | A | A | A | A | A | B | B | B

Class A sits in the middle, class B at both ends. A linear classifier on one feature is a single threshold "$x > t$", and no threshold can put the middle on one side and both ends on the other. Now give each point a second coordinate, $x^2$. The middle points have small squares (at most $1.44$), the end points large ones (at least $4$). In the new two-dimensional space a flat line, $x^2 = 2.25$, separates the classes perfectly.

~~~dg-fig 02.06-lift.svg
Left: on the number line, class A sits between two groups of class B, so no single cut works. Right: lift each point to height x². The same points are now split by a flat line, which is still a linear classifier, just in the new features.
~~~

Back on the original line, the rule "$x^2 < 2.25$" means "$-1.5 < x < 1.5$": a curved (here, two-sided) boundary produced by a linear model. Three terms will carry the rest of the lesson:

- A **feature map** $\phi(x)$ turns an input into a new, usually longer, vector. Here $\phi(x) = (x, x^2)$.
- The **feature space** is where those vectors live. The model is linear there, curved back in input space.
- A **kernel** $k(x, x')$ is a function that returns the dot product $\phi(x)^\top \phi(x')$ directly, often far more cheaply than building both vectors.

## Step 1: explicit feature maps get expensive

For one input, adding $x^2$ is free. With $d$ inputs, a degree-$p$ polynomial expansion needs every product of up to $p$ of them; the number of such terms (monomials) is

$$\binom{d + p}{p},$$

read "$d+p$ choose $p$". With $d = 10$ and $p = 2$ that is 66 features, which is fine. With $d = 100$ and $p = 3$ it is 176,851 features *per example*: a dataset of 10,000 rows becomes a matrix with 1.8 billion entries. Yet many algorithms never read individual features; they only need dot products between examples. That is the opening.

## Step 2: algorithms that only need dot products

**The SVM.** In the dual problem from 02.05 the data appear as $x_i^\top x_j$, and the prediction is $f(x) = \sum_i \alpha_i y_i\, x_i^\top x + b$. No single feature is ever read on its own.

**Ridge regression.** The usual ridge solution is $w = (X^\top X + \lambda I)^{-1} X^\top y$, which involves the $d \times d$ matrix $X^\top X$. An algebra identity rewrites it as

$$w = X^\top (X X^\top + \lambda I)^{-1} y.$$

Here $X$ is the $n \times d$ data matrix, $y$ the targets, $\lambda$ the ridge strength, and $XX^\top$ is the $n \times n$ table whose entry $(i, j)$ is $x_i^\top x_j$. (Multiply both versions by $(X^\top X + \lambda I)$ on the left to check they agree.) A new prediction is $w^\top x = \sum_i \alpha_i\, x_i^\top x$ with $\alpha = (XX^\top + \lambda I)^{-1} y$. Again: only dot products.

Replace every $x_i^\top x_j$ with $\phi(x_i)^\top \phi(x_j)$ and the algorithm runs in feature space. What remains is computing those dot products cheaply.

## Step 3: the kernel trick

Take two-dimensional inputs and the kernel $k(x, x') = (x^\top x')^2$. Expanding the square:

$$(x^\top x')^2 = x_1^2 x_1'^2 + 2x_1x_2x_1'x_2' + x_2^2x_2'^2 = \phi(x)^\top\phi(x'),\quad \phi(x) = (x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2).$$

One dot product and a square give the same number as building three quadratic features per point.

<div class="ex">
<div class="lab">Worked example</div>

**checking the polynomial kernel by hand**

Take $x = (1, 2)$ and $x' = (3, 1)$.

Kernel route: $x^\top x' = 1 \cdot 3 + 2 \cdot 1 = 5$, and $5^2 = 25$.

Feature route: $\phi(x) = (1^2,\ \sqrt2 \cdot 1 \cdot 2,\ 2^2) = (1,\ 2\sqrt2,\ 4)$ and $\phi(x') = (3^2,\ \sqrt2 \cdot 3 \cdot 1,\ 1^2) = (9,\ 3\sqrt2,\ 1)$.

Dot product: $1 \cdot 9 + 2\sqrt2 \cdot 3\sqrt2 + 4 \cdot 1 = 9 + 12 + 4 = 25$.

Same answer. For degree 3 in 100 dimensions, the kernel route is still one 100-term dot product and a cube, while the feature route builds 176,851 numbers per point.

</div>

Two facts make the trick general.

- **Representer theorem.** If you minimise any loss on the training predictions plus an L2 penalty $\lambda\|w\|^2$, the best $w$ is a combination of the training points' features: $w = \sum_i \alpha_i\, \phi(x_i)$. So the model is $f(x) = \sum_i \alpha_i\, k(x_i, x)$, and you learn $n$ numbers $\alpha_i$, one per training point, however many features $\phi$ has.
- **Mercer's condition.** Which functions are allowed? A symmetric $k$ is a valid kernel exactly when every **Gram matrix** it produces, $K_{ij} = k(x_i, x_j)$ for any set of points, is positive semi-definite (no negative eigenvalues). Sums, positive multiples and products of kernels are again kernels, so you can build new ones from old.

~~~dg-flow Two routes to the same nonlinear model. The explicit route costs grow with the number of features; the kernel route costs grow with the number of training points.
[Explicit] Input x -(expand)-> Features phi(x) -(dot with w)-> Prediction
[Kernel] Input x -(compare with each x_i)-> *k(x_i, x) for all i -(weight by alpha_i, sum)-> Prediction
~~~

## Step 4: the RBF kernel, a similarity score

The most used kernel is the **RBF** (radial basis function, or Gaussian) kernel:

$$k(x, x') = \exp\!\big(-\gamma\,\|x - x'\|^2\big).$$

Here $\|x - x'\|^2$ is the squared distance between the two points and $\gamma > 0$ is a width knob. It is a similarity: 1 for identical points, decaying toward 0 with distance on a length scale of about $1/\sqrt{\gamma}$. With $\gamma = 1$, points 0.5 apart score $e^{-0.25} = 0.78$, points 1 apart score $e^{-1} = 0.37$, points 2 apart score $e^{-4} = 0.02$.

Its feature space is infinite-dimensional (the exponential's power series contains every polynomial degree), so the explicit route is impossible. Through the representer theorem, though, the model is simply

$$f(x) = \sum_{i=1}^{n} \alpha_i \exp\!\big(-\gamma\|x_i - x\|^2\big),$$

a weighted sum of bumps, one centred on each training point.

The Gram matrix shows what the model "sees". Take eight sorted points on a line and compute every pairwise score:

~~~dg-fig 02.06-gram.svg
The Gram matrix for eight sorted points. Each cell is the RBF similarity of two points; the diagonal is always 1. With gamma 1, neighbours share bright blocks (0, 0.3 and 0.5 look alike). With gamma 10, only very close pairs (3.5 and 3.6) stay similar and the matrix approaches the identity.
~~~

Near neighbours form bright blocks along the diagonal. Small $\gamma$ makes the blocks wide, so each point borrows strength from many neighbours; large $\gamma$ shrinks them until every point resembles only itself.

Concentric rings are the two-dimensional version of the opening example: no straight line separates them, but an RBF SVM draws a closed curve between them.

~~~dg-fig 02.06-circles.svg
Two rings of 200 points. The linear SVM can only draw a straight line and scores about 60% on its own training data. The RBF SVM (gamma 1, C 1) wraps a circle between the rings, using 39 of the 200 points as support vectors.
~~~

## Step 5: choose gamma (and C) together

$\gamma$ sets each point's reach.

- **Small $\gamma$**: wide bumps, every point influences a large area, the boundary is smooth and nearly linear. Too small underfits.
- **Large $\gamma$**: narrow bumps, each point influences only its immediate neighbourhood. Too large draws a little island around each training point and memorises noise.

~~~dg-fig 02.06-gamma.svg
The same 120 noisy two-moons points with an RBF SVM (C = 10) at three widths. Gamma 0.01 is almost a straight line (86% test accuracy); gamma 1 follows the moons (92%); gamma 100 hits 100% on training data by drawing islands, and drops to 76% on 1,000 fresh points.
~~~

$C$ (SVMs) or $\lambda$ (kernel ridge) is the second knob: how hard the model may bend toward individual points. The two interact, so search them together on a logarithmic grid such as $\gamma \in \{0.01, 0.1, 1, 10, 100\}$ and $C \in \{0.1, 1, 10, 100\}$, scored by cross-validation. scikit-learn's default `gamma="scale"` sets $\gamma = 1/(d \cdot \mathrm{Var}(X))$, a reasonable start on standardised data.

Because the RBF kernel is built on distances, **standardise features first**, exactly as for kNN (03.02). A feature measured in thousands would otherwise dominate every distance and the kernel would ignore the rest.

## Step 6: kernel ridge regression

Take the dot-product form of ridge from step 2 and swap $XX^\top$ for the Gram matrix $K$:

$$\alpha = (K + \lambda I)^{-1} y, \qquad f(x) = \sum_{i=1}^{n} \alpha_i\, k(x_i, x).$$

$K$ is the $n \times n$ kernel table, $I$ the identity, $\lambda$ the ridge strength, and $\alpha$ holds one coefficient per training point. Training is one linear solve; prediction compares the new point with every training point.

<div class="ex">
<div class="lab">Worked example</div>

**kernel ridge on two points**

Data: $x = (0, 1)$, $y = (1, 3)$. RBF kernel with $\gamma = 1$, ridge strength $\lambda = 0.1$.

Gram matrix: $k(0,0) = k(1,1) = 1$ and $k(0,1) = e^{-1} = 0.3679$. So $K + \lambda I = \begin{pmatrix} 1.1 & 0.3679 \\ 0.3679 & 1.1 \end{pmatrix}$.

Determinant: $1.1^2 - 0.3679^2 = 1.21 - 0.1353 = 1.0747$.

Solve with the 2-by-2 inverse: $\alpha_1 = (1.1 \cdot 1 - 0.3679 \cdot 3)/1.0747 = (1.1 - 1.1036)/1.0747 = -0.0034$ and $\alpha_2 = (-0.3679 \cdot 1 + 1.1 \cdot 3)/1.0747 = 2.9321/1.0747 = 2.7284$.

Predict at $x = 0.5$: both training points are 0.5 away, so both kernel values are $e^{-0.25} = 0.7788$. $f(0.5) = 0.7788 \times (-0.0034 + 2.7284) = 0.7788 \times 2.7250 = 2.12$, a sensible value between 1 and 3.

Predict at $x = 3$: $k(0, 3) = e^{-9} = 0.0001$ and $k(1, 3) = e^{-4} = 0.0183$. $f(3) = 0.0183 \times 2.7284 = 0.05$. Far from all training data, the RBF model falls back to 0. Centre $y$ first if you want it to fall back to the mean instead.

</div>

The same formula is the posterior mean of a **Gaussian process** with covariance $k$ and noise variance $\lambda$. Unlike the SVM, whose $\alpha$ is zero for every point outside the margin, kernel ridge's $\alpha$ is dense: every training point takes part in every prediction.

## Kernels in code

A from-scratch version is a few lines of NumPy:

```python
import numpy as np

def rbf(A, B, gamma):
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T   # squared distances
    return np.exp(-gamma * np.maximum(d2, 0))

def krr_fit(X, y, gamma, lam):
    K = rbf(X, X, gamma)
    return np.linalg.solve(K + lam * np.eye(len(X)), y)                # alpha (Cholesky in practice)

def krr_predict(Xtr, alpha, Xnew, gamma):
    return rbf(Xnew, Xtr, gamma) @ alpha

alpha = krr_fit(np.array([[0.], [1.]]), np.array([1., 3.]), gamma=1.0, lam=0.1)
print(alpha)                                                            # [-0.0034  2.7284]
print(krr_predict(np.array([[0.], [1.]]), alpha, np.array([[0.5], [3.]]), 1.0))   # [2.12 0.05]
```

In practice use scikit-learn. Here is the rings problem, with a scaler in the pipeline and a joint search over $C$ and $\gamma$:

```python
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_circles(n_samples=1000, factor=0.45, noise=0.1, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

lin = make_pipeline(StandardScaler(), SVC(kernel="linear")).fit(Xtr, ytr)
rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=1.0, C=1.0)).fit(Xtr, ytr)
print(lin.score(Xte, yte), rbf.score(Xte, yte))     # 0.483  1.0
print(rbf[-1].n_support_)                           # [34 31]: 65 of 700 points are support vectors

grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10, 100]}, cv=5)
grid.fit(Xtr, ytr)
print(grid.best_params_, grid.score(Xte, yte))      # {'svc__C': 1, 'svc__gamma': 0.1}  1.0
```

For regression, `KernelRidge(kernel="rbf", gamma=1.0, alpha=0.1)` implements step 6 (scikit-learn calls the ridge strength `alpha`, not to be confused with our $\alpha_i$; the fitted coefficients are in `dual_coef_`, one per training row). `SVR` is the SVM version, with sparse coefficients.

## Reading the result

- **Train versus validation score.** Near-perfect training with a clearly lower validation score means $\gamma$ too large or regularisation too weak. Both low and close together means $\gamma$ too small (or the wrong kernel).
- **Number of support vectors** (SVMs). A handful of support vectors is a confident, simple boundary. If nearly every training point is a support vector, as at $\gamma = 100$ where all 120 points were, the model is memorising.
- **The validation grid.** The best $(C, \gamma)$ cell should sit inside the grid, ideally on a plateau. If it is on an edge, extend the grid that way.
- **Compare with a linear baseline.** If it is within a point or two of the kernel model, keep the linear one.

## What goes wrong

| Problem | Symptom | Fix
| --- | --- | --- |
| Features not scaled | one feature dominates; results change when you change units | put `StandardScaler` in the pipeline
| $\gamma$ too large | 100% train accuracy, poor test accuracy, almost all points are support vectors | lower $\gamma$, cross-validate
| $\gamma$ too small | near-linear boundary, train and test both mediocre | raise $\gamma$
| Tuning $C$ and $\gamma$ separately | best value of one depends on the other you fixed | grid search both on a log scale
| Too many training points | fit takes hours; memory error building $K$ | random features or Nyström plus a linear model
| Invalid "kernel" (e.g. sigmoid at some settings) | solver fails or behaves erratically | use a kernel that satisfies Mercer's condition
| Expecting extrapolation | RBF predictions sag to 0 away from the data | centre $y$; add a linear kernel term; get data in that region

The cost row bites most in practice. Kernel ridge training takes $O(n^3)$ time and $O(n^2)$ memory, and every prediction costs $O(nd)$. At $n = 50{,}000$, $K$ has $2.5 \times 10^9$ entries: 20 GB in float64 before any solving starts. Kernel SVMs, with sparse $\alpha$, fare better but also crawl past roughly $10^5$ points.

## Common kernels

| Kernel | Formula | Feature space | Main knob
| --- | --- | --- | --- |
| Linear | $x^\top x'$ | the inputs themselves | none
| Polynomial | $(\gamma\,x^\top x' + c)^p$ | all monomials up to degree p | degree p, offset c
| RBF (Gaussian) | $\exp(-\gamma\|x - x'\|^2)$ | infinite-dimensional | width gamma
| Sigmoid | $\tanh(\gamma\,x^\top x' + c)$ | not always PSD | rarely worth it

Specialised kernels also exist for strings, graphs and sets: you only need a valid similarity, not a vector.

## Going deeper: scaling back to explicit features

The escape from $O(n^3)$ is to approximate the kernel with a modest number $D$ of explicit features, then fit any linear model from this module. That costs about $O(nD^2)$ instead of $O(n^3)$.

- **Random Fourier features** (Rahimi and Recht). Draw a matrix $\Omega \in \mathbb{R}^{D \times d}$ with entries from $\mathcal{N}(0, 2\gamma)$ and offsets $b$ uniform on $[0, 2\pi]$, and set $z(x) = \sqrt{2/D}\,\cos(\Omega x + b)$. Then $z(x)^\top z(x') \approx e^{-\gamma\|x - x'\|^2}$. This works because the RBF kernel is the Fourier transform of a Gaussian (Bochner's theorem), so it is an average of cosine features that we can sample. The error shrinks like $1/\sqrt{D}$.
- **Nyström.** Pick $m$ landmark points, compute the kernel between every point and the landmarks, and use those columns to build a low-rank approximation of $K$. It adapts to where your data actually are, so it often needs fewer components than random features.

~~~dg-fig 02.06-rff.svg
How well random Fourier features approximate an RBF kernel on 300 points in 5 dimensions. The error falls on a straight line of slope -1/2 in log-log scale: ten times more features buy about three times less error.
~~~

In scikit-learn both are transformers. On the rings, `RBFSampler(gamma=1.0, n_components=300)` or `Nystroem(gamma=1.0, n_components=100)` followed by `LogisticRegression` matches the exact RBF SVM's 100% test accuracy. On real data, $D$ or $m$ in the low thousands is usually enough.

A few more facts that experts use:

- The function space a kernel defines is called its **reproducing kernel Hilbert space (RKHS)**. The penalty $\lambda\|w\|^2$ is the RKHS norm of $f$, which for the RBF kernel penalises roughness: smoother functions have smaller norm.
- As $\gamma \to \infty$ the Gram matrix tends to the identity, so the model becomes a lookup table; as $\gamma \to 0$ it tends to the all-ones matrix, and the fit becomes nearly constant or low-degree polynomial.

## Key terms

| Term | Meaning
| --- | --- |
| Feature map $\phi$ | A function that turns an input into a new (often much longer) feature vector
| Kernel $k(x, x')$ | A function equal to $\phi(x)^\top\phi(x')$ for some feature map, computed without building $\phi$
| Kernel trick | Running an algorithm that needs only dot products in feature space by replacing them with kernel values
| Gram matrix $K$ | The $n \times n$ table of kernel values between all pairs of training points
| RBF kernel | $\exp(-\gamma\|x - x'\|^2)$: similarity that decays with distance; $\gamma$ sets the width
| Representer theorem | With an L2 penalty, the best model is $\sum_i \alpha_i k(x_i, x)$: one coefficient per training point
| Mercer's condition | A function is a valid kernel when all its Gram matrices are positive semi-definite
| Kernel ridge regression | Ridge in feature space: $\alpha = (K + \lambda I)^{-1}y$
| Random Fourier features | Random cosine features whose dot products approximate the RBF kernel

<div class="take">
<div class="lab">Key takeaway</div>

a kernel replaces the dot product of expanded features with a cheap similarity, so linear methods like ridge and the SVM can learn curved functions in huge or infinite feature spaces with one coefficient per training point. Standardise, tune gamma and C together, and past tens of thousands of points switch to random Fourier or Nyström features with a linear model.

</div>
