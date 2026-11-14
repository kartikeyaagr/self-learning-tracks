---
id: "02.05"
course: "machine-learning"
module: "02"
slug: "support-vector-machines"
title: "Support vector machines"
keyIdeas: ["Many lines can separate the training data; the SVM picks the one with the widest **margin**. Scaling $w$ so the closest points have $y_i(w^\\top x_i + b) = 1$ makes the margin $1/\\|w\\|$, so maximising it means minimising $\\tfrac12\\|w\\|^2$.", "The soft margin lets points violate the street at a cost $C$ per unit of slack. Small $C$ gives a wide, stable margin with many violations; large $C$ a narrow margin that follows a few points. Tune $C$ by cross-validation on standardised features.", "The soft-margin SVM is the same as minimising $\\tfrac{\\lambda}{2}\\|w\\|^2 + \\tfrac1n\\sum_i \\max(0, 1 - y_i f(x_i))$, the L2-regularised **hinge loss**, with $\\lambda = 1/(nC)$.", "Hinge loss is exactly zero past the margin, so only **support vectors** (points on or inside the margin) determine the solution; logistic loss is never zero, so every point contributes and its outputs can be probabilities, while SVM scores cannot.", "**Pegasos** trains a linear SVM by SGD: step $\\eta_t = 1/(\\lambda t)$, shrink $w$ by $(1 - \\eta_t\\lambda)$ every step, and add $\\eta_t y_i x_i$ only when the margin is below 1. Unlike the perceptron, it converges to the unique widest-margin solution.", "In the dual, data enter only through dot products $x_i^\\top x_j$ and $w = \\sum_i \\alpha_i y_i x_i$, which is what makes kernels (next lesson) possible."]
question: "A fraud team trains `LinearSVC(C=1.0)` on two raw features: transaction amount in dollars (0 to 50,000) and number of items (1 to 20). Training prints a convergence warning, validation accuracy is barely above always predicting 'legitimate' (fraud is 2% of the data), and the boundary is almost a vertical line in amount. They also want to flag a transaction when its 'fraud probability' exceeds 0.9 by thresholding `decision_function`. Explain each problem and what you would change."
answer: "Four separate issues. (1) Scaling: the SVM penalises $\\|w\\|^2$ and measures the margin as a Euclidean distance. In raw units, a tiny weight on amount produces large scores while the items feature needs a huge weight to matter, so the penalty effectively ignores amount's scale and suppresses items; the boundary ends up driven by amount alone, and the badly conditioned problem is why liblinear fails to converge. Fix: put `StandardScaler` and the SVM in one pipeline so both features are on comparable scales and scaling statistics come only from training folds. (2) Imbalance: with 2% fraud, the slack sum is dominated by the legitimate class, so a solution that labels almost everything legitimate has low objective. Use `class_weight=\"balanced\"` (it multiplies $C$ for the rare class) and judge models with precision/recall or PR-AUC rather than accuracy. (3) $C$ was not tuned: search a log grid such as $10^{-3}$ to $10^{3}$ with stratified cross-validation, after scaling, since the right $C$ depends on the units and on the class weights. (4) Probabilities: `decision_function` returns $f(x) = w^\\top x + b$, a signed distance-like score trained with hinge loss, which carries no probability meaning; 0.9 is just a point inside the margin. Wrap the model in `CalibratedClassifierCV` (Platt or isotonic) to get calibrated probabilities, or skip probabilities and choose the score threshold directly on validation data to hit the required precision. If calibrated probabilities are central to the use case, L2-regularised logistic regression is a natural alternative with similar accuracy."
sources: [{"title": "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", "url": "https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf", "note": "Shalev-Shwartz, Singer, Srebro and Cotter: the algorithm, its step size and the n-independent convergence rate."}, {"title": "Cornell CS4780 lecture notes: Support Vector Machine", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote09.html", "note": "Margin derivation, soft margin and the hinge-loss view."}, {"title": "scikit-learn: Support Vector Machines", "url": "https://scikit-learn.org/stable/modules/svm.html", "note": "SVC vs LinearSVC, the C parameter, complexity and the warning about unscaled data."}, {"title": "Stanford CS229 main notes", "url": "https://cs229.stanford.edu/main_notes.pdf", "note": "The SVM chapter: functional vs geometric margin, Lagrange duality and the KKT conditions."}]
practice: ["svm-with-hinge-loss-sgd", "perceptron", "averaged-perceptron", "subgradients-for-non-smooth-functions"]
tags: ["margin", "hinge loss"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- Why "separates the training data" is not enough, and what the **margin** of a classifier is.
- How the support vector machine (SVM) turns "widest margin" into a small optimisation problem, and how to solve a tiny one by hand.
- What the soft margin and the parameter $C$ do, and how to see their effect in a plot.
- Why an SVM is just L2-regularised **hinge loss**, which puts it in the same family as logistic regression.
- How to train a linear SVM with scikit-learn and with a ten-line SGD solver (Pegasos), and how to check the result.

*Before you start:* you should be comfortable with linear scores $w^\top x + b$ and the $\pm 1$ label convention from logistic regression (02.03), the idea of an L2 penalty from ridge regression (02.02), and vector length $\|w\|$ and dot products (ML Maths, module 08). Subgradients appear once, in the solver section.

Logistic regression (02.03, 02.04) draws a straight boundary and turns distance from it into a probability. This lesson keeps the straight boundary but asks: of all the lines that classify the training data correctly, which one is *safest*? The answer, the support vector machine, sets up kernels in the next lesson (02.06).

## The idea in one picture

A workshop logs two scaled sensor readings for twelve machines and notes which ones failed within a month (label $+1$) and which kept running (label $-1$):

| Machine | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reading 1 | 3 | 4 | 3 | 5 | 4.5 | 6 | 1 | 0.3 | 1.5 | 0.5 | 0 | 1
| Reading 2 | 3 | 3 | 4.5 | 4 | 5.5 | 3.5 | 1 | 1.5 | 0.2 | 0.5 | 0.8 | 0
| Label | +1 | +1 | +1 | +1 | +1 | +1 | −1 | −1 | −1 | −1 | −1 | −1

The groups do not overlap, so many straight lines separate them perfectly. The left panel below shows three. All have zero training error, yet two pass within a hair of some machines; a slightly noisier new machine would land on the wrong side.

~~~dg-fig 02.05-margin.svg
Left: three lines with zero training errors, two of them almost touching a point. Right: the SVM line sits in the middle of the widest empty band; only the two circled points decide where it goes.
~~~

The right panel shows the SVM's choice. It finds the widest band (the shaded "street") that contains no training points, and puts the boundary down its middle. Three terms:

- The **margin** is the distance from the boundary to the nearest training point. Here it is $1.41$ on each side.
- The **support vectors** are the points that touch the edges of the band: machine 7 at $(1, 1)$ and machine 1 at $(3, 3)$. Move any other point a little (without entering the band) and the boundary does not change.
- The **decision function** is the score $f(x) = w^\top x + b$. Its sign is the predicted class. For this data the SVM finds $w = (0.5, 0.5)$ and $b = -2$, so the rule reads "predict failure when $0.5 \cdot \text{reading}_1 + 0.5 \cdot \text{reading}_2 > 2$".

Why the widest band? It tolerates the largest measurement error on every training point before any is misclassified, and test-error bounds for margin classifiers depend on the margin, not on the number of features.

## Step 1: score, sign and the margin of one point

Keep the conventions from 02.03: labels are $y \in \{-1, +1\}$ and the classifier is $f(x) = w^\top x + b$, where $w$ is the weight vector (it points across the boundary, towards the $+1$ side) and $b$ is the bias. Predict $+1$ when $f(x) > 0$.

For one training point, multiply the score by the label:

$$m_i = y_i\, f(x_i).$$

This is the **functional margin** of point $i$. It is positive exactly when the point is on the correct side, and larger means more confidently correct. For machine 4 at $(5, 4)$ with label $+1$: $f = 0.5 \cdot 5 + 0.5 \cdot 4 - 2 = 2.5$, so $m = 2.5$.

It is not yet a distance: doubling $w$ and $b$ doubles every score without moving the line. Dividing by the length of $w$ gives the true distance to the boundary:

$$\text{distance}_i = \frac{y_i f(x_i)}{\|w\|}.$$

Number check: $\|w\| = \sqrt{0.5^2 + 0.5^2} = 0.707$, so machine 4 sits $2.5 / 0.707 = 3.54$ units from the line.

## Step 2: the hard-margin SVM

We want the $(w, b)$ whose *smallest* distance is as large as possible. Use the rescaling freedom from step 1: choose the scale so that the closest points have functional margin exactly 1. Then:

- every point satisfies $y_i(w^\top x_i + b) \ge 1$, and
- the distance from the boundary to the nearest point is $1/\|w\|$.

Maximising $1/\|w\|$ is the same as minimising $\|w\|$, or, more conveniently, $\tfrac12\|w\|^2$ (same minimiser, smooth, and with a tidy derivative). The **hard-margin SVM** is

$$\min_{w,\,b}\; \tfrac12\|w\|^2 \quad \text{subject to}\quad y_i(w^\top x_i + b) \ge 1 \;\text{ for every } i.$$

In words: find the shortest weight vector that still gives every point a score of at least 1 on its correct side. The lines $f(x) = \pm 1$ are the street edges, and the street is $2/\|w\|$ wide. This is a convex quadratic program with a single best answer.

<div class="ex">
<div class="lab">Worked example</div>

**a hard-margin SVM by hand**

Take three of the machines: $A = (1, 1)$ with $y = -1$, $B = (3, 3)$ with $y = +1$, and $C = (4, 3)$ with $y = +1$.

Guess the support vectors. $A$ and $B$ are the closest opposite pair, so assume they sit on the street edges. By symmetry the best $w$ points from $A$ to $B$, along $(1, 1)$, so write $w = (a, a)$.

Put both on the edges: for $B$, $3a + 3a + b = +1$, so $6a + b = 1$. For $A$, $a + a + b = -1$, so $2a + b = -1$.

Subtract: $4a = 2$, so $a = 0.5$ and $b = 1 - 6 \times 0.5 = -2$. The SVM is $w = (0.5, 0.5)$, $b = -2$.

Check the third point: $f(C) = 0.5 \times 4 + 0.5 \times 3 - 2 = 1.5 \ge 1$. The constraint holds with room to spare, so $C$ is not a support vector and our guess was right.

Margin: $\|w\| = \sqrt{0.25 + 0.25} = 0.707$, so $1/\|w\| = 1.414$ on each side and the street is $2.83$ wide. That equals the distance between $A$ and $B$, $\sqrt{2^2 + 2^2} = 2.83$: with only two support vectors, the boundary is their perpendicular bisector.

Adding the other nine machines changes nothing, because every one of them has $|f(x)| > 1$. This is the answer `SVC` returns on the full table.

</div>

## Step 3: the soft margin and the parameter C

Real classes overlap, and a single mislabelled point can make a hard margin impossible (no line separates the data) or absurdly narrow. The **soft-margin SVM** lets points violate the street, at a price. Give each point a **slack** $\xi_i \ge 0$ (Greek "xi"), relax its constraint to $y_i f(x_i) \ge 1 - \xi_i$, and charge for the total slack:

$$\min_{w,\,b,\,\xi}\; \tfrac12\|w\|^2 + C\sum_{i=1}^n \xi_i.$$

What the slack values mean:

- $\xi_i = 0$: the point is on or outside its edge of the street, as before.
- $0 < \xi_i < 1$: the point is inside the street but on the correct side.
- $\xi_i > 1$: the point is on the wrong side of the boundary, a training error.

$C > 0$ sets the exchange rate between a wide street (small $\|w\|$) and few violations (small total slack). Number check: if we add a fourth machine $D = (2.5, 2)$ labelled $+1$ to the hand example and keep $w = (0.5, 0.5)$, $b = -2$, its score is $0.5 \times 4.5 - 2 = 0.25$, so it needs slack $1 - 0.25 = 0.75$. With $C = 1$ the objective is $\tfrac12 \times 0.5 + 1 \times 0.75 = 1.0$. A larger $C$ makes that slack expensive enough that the solver would rather tilt and narrow the street to push $D$ out.

~~~dg-fig 02.05-C.svg
The same 61 overlapping points fitted with a small and a large C. Small C buys a wide street with 38 points on or inside it; large C squeezes the street to fit the data more tightly, leaving 8 support vectors.
~~~

With $C = 0.02$ violations are cheap, so many points sit in the street and the boundary is set by the bulk of both clouds: stable, low variance. With $C = 100$ the street shrinks to less than half the width and a handful of points near the overlap set the boundary: high variance. Choose $C$ by cross-validation.

## Step 4: the same thing as a loss function

At the optimum each slack is as small as allowed, $\xi_i = \max(0,\, 1 - y_i f(x_i))$. Substitute, divide by $nC$ (the minimiser does not move) and write $\lambda = 1/(nC)$:

$$\min_{w,\,b}\; \frac{\lambda}{2}\|w\|^2 + \frac1n \sum_{i=1}^n \max\big(0,\; 1 - y_i f(x_i)\big).$$

The function $\max(0, 1 - m)$ of the margin $m = y f(x)$ is the **hinge loss**. So an SVM is a linear model trained with hinge loss and an L2 penalty: the module 01 recipe. Logistic regression swaps in $\log(1 + e^{-m})$; ridge swaps in squared error. Large $C$ means small $\lambda$, a weak penalty.

| Margin m = y f(x) | 0-1 loss | Hinge max(0, 1-m) | Logistic log(1+e^-m)
| --- | --- | --- | --- |
| -1 (wrong side) | 1 | 2 | 1.31
| 0 (on the boundary) | 1 | 1 | 0.69
| 1 (on the street edge) | 0 | 0 | 0.31
| 2 (safely correct) | 0 | 0 | 0.13

~~~dg-fig 02.05-losses.svg
Three losses as functions of the margin. Hinge and logistic both grow roughly linearly for wrong answers, but only hinge drops to exactly zero, at margin 1.
~~~

The **0-1 loss** (1 for a mistake, 0 otherwise) is what we care about, but it is flat almost everywhere, so gradients tell you nothing. Hinge and logistic are convex **surrogates** that upper-bound it (logistic after dividing by $\ln 2$, the dotted curve). Both charge about linearly for confident mistakes. The right tails are the real difference:

- Hinge loss is exactly zero once a point is past its street edge. Such points exert no force on the solution, which is why only support vectors matter.
- Logistic loss is never zero, so every point keeps pulling, and the scores can be read as probabilities. SVM scores are not probabilities; if you need them, calibrate (03.04).

## Step 5: training it with SGD (Pegasos)

For a linear SVM on a large dataset, the simplest effective solver is stochastic subgradient descent on the hinge objective. The hinge has a kink at $m = 1$, so instead of a gradient we use a **subgradient**: $-y_i x_i$ when the margin is below 1, and $0$ otherwise. **Pegasos** (Shalev-Shwartz and co-authors) picks one random example per step and uses step size $\eta_t = 1/(\lambda t)$:

```python
import numpy as np
def pegasos(X, y, lam=1e-3, T=100_000, seed=0):   # y in {-1,+1}
    rng = np.random.default_rng(seed); w = np.zeros(X.shape[1])
    for t in range(1, T + 1):
        i = rng.integers(len(y)); eta = 1.0 / (lam * t)
        if y[i] * (X[i] @ w) < 1:                    # inside the margin: hinge is active
            w = (1 - eta * lam) * w + eta * y[i] * X[i]
        else:                                        # only the regulariser acts
            w = (1 - eta * lam) * w
    return w
```

Two forces: every step shrinks $w$ by $(1 - \eta_t\lambda) = 1 - 1/t$, the L2 penalty pulling towards a wide street; only points with margin below 1 push back by adding $\eta_t y_i x_i$. The bias is handled by appending a constant feature of 1, which penalises $b$ slightly; with standardised features that rarely matters.

<div class="ex">
<div class="lab">Worked example</div>

**two Pegasos steps**

Setting: $\lambda = 0.1$, current step $t = 10$, so $\eta = 1/(0.1 \times 10) = 1$ and the shrink factor is $1 - \eta\lambda = 0.9$. Current $w = (0.5, -0.2)$.

Example $x = (1, 2)$, $y = +1$: margin $y\, w^\top x = 0.5 \times 1 - 0.2 \times 2 = 0.1 < 1$, so the hinge is active. Update: $w \leftarrow 0.9\,(0.5, -0.2) + 1 \times (1, 2) = (0.45, -0.18) + (1, 2) = (1.45, 1.82)$. Its margin is now $1.45 + 3.64 = 5.09$.

Next step $t = 11$: $\eta = 1/(0.1 \times 11) = 0.909$ and the shrink factor is $1 - 0.909 \times 0.1 = 0.909$. Example $x = (2, 1)$, $y = +1$: margin $2 \times 1.45 + 1.82 = 4.72 \ge 1$, so only the shrink applies: $w \leftarrow 0.909 \times (1.45, 1.82) = (1.32, 1.65)$.

</div>

~~~dg-fig 02.05-pegasos.svg
Three Pegasos runs on the 61-point data with lambda = 0.01. Early steps are huge (step size 1/(lambda t) starts at 100), so the objective jumps around, then it settles close to the exact optimum.
~~~

The steps needed for accuracy $\epsilon$ scale like $1/(\lambda\epsilon)$, independent of $n$, so the method suits very large data. Averaging the last iterates smooths the early spikes.

Compare the **perceptron** (built in the practice problems): it adds $y_i x_i$ only on mistakes (margin $\le 0$, not $< 1$) and never shrinks $w$, so it stops at the first separating line it finds, which depends on data order.

~~~dg-fig 02.05-perceptron.svg
Three perceptron runs that differ only in shuffling order end at three different separating lines, one passing 0.16 units from a point; the SVM returns the same widest-margin line every time.
~~~

## SVMs in code

In practice you will use scikit-learn. `SVC(kernel="linear")` solves the problem exactly and exposes the support vectors; `LinearSVC` is much faster on large data.

```python
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([[3, 3], [4, 3], [3, 4.5], [5, 4], [4.5, 5.5], [6, 3.5],       # failed (+1)
              [1, 1], [0.3, 1.5], [1.5, 0.2], [0.5, 0.5], [0, 0.8], [1, 0]])  # kept running (-1)
y = np.array([1]*6 + [-1]*6)

svm = SVC(kernel="linear", C=1e6).fit(X, y)     # huge C = (almost) hard margin
w, b = svm.coef_[0], svm.intercept_[0]
print(w.round(3), round(b, 3))                  # [0.5 0.5] -2.0
print(svm.support_vectors_)                     # [[1. 1.] [3. 3.]]
print(svm.dual_coef_)                           # [[-0.25  0.25]]  (alpha_i * y_i)
print(round(1 / np.linalg.norm(w), 3))          # 1.414  (margin)
print(svm.decision_function([[2.5, 2], [5, 5]]).round(2))   # [0.25 3.  ]

clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0)).fit(X, y)
print(clf.score(X, y))                          # 1.0
```

The printout matches the hand example. `dual_coef_` holds $\alpha_i y_i$ per support vector (see Going deeper). The new machine at $(2.5, 2)$ scores 0.25: predicted to fail, but inside the street, so the model is unsure.

## Checking the result

- **Tune $C$ on a log grid** ($10^{-3}$ to $10^{3}$) by cross-validation, inside a pipeline with the scaler.
- **Count the support vectors** (`len(svm.support_)`). If nearly every point is one, the classes overlap heavily or $C$ is tiny, and a linear boundary may be the wrong model.
- **Look at scores, not just labels.** `decision_function` values between $-1$ and $1$ are inside the street; a histogram of training scores by class shows the overlap.
- **Convergence warnings** from `LinearSVC` usually mean unscaled features or a very large $C$.
- **Compare with logistic regression.** On linear problems their accuracies are usually within a point or two; a big gap suggests a bug or a scaling problem.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| One feature dominates, poor accuracy, convergence warnings | Features in very different units: the margin is a Euclidean distance | Standardise inside a pipeline
| Boundary jumps when a few points change | $C$ too large, the street is set by a few noisy points | Lower $C$, chosen by cross-validation
| Underfits, almost every point is a support vector | $C$ too small, or the classes are not linearly separable | Raise $C$; try features or a kernel (02.06)
| `SVC` takes hours | Dual solver scales between $n^2$ and $n^3$ | `LinearSVC` or `SGDClassifier(loss="hinge")` for linear problems
| Thresholding `decision_function` as if it were a probability | Hinge loss produces scores, not probabilities | Calibrate with `CalibratedClassifierCV` (03.04)
| Rare class almost never predicted | Errors on the large class dominate the slack sum | `class_weight="balanced"`, or tune the threshold on validation data

## Going deeper: the dual and the KKT conditions

One Lagrange multiplier $\alpha_i \ge 0$ per constraint gives the equivalent **dual problem**:

$$\max_{\alpha}\; \sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\, x_i^\top x_j \quad \text{subject to}\quad 0 \le \alpha_i \le C,\; \sum_i \alpha_i y_i = 0,$$

and the weights are recovered as $w = \sum_i \alpha_i y_i x_i$. In the hand example $\alpha_A = \alpha_B = 0.25$: the constraint $\sum \alpha_i y_i = -0.25 + 0.25 = 0$ holds, and $w = 0.25(3, 3) - 0.25(1, 1) = (0.5, 0.5)$.

Two things to notice. First, the training data appear only as dot products $x_i^\top x_j$. Replacing them with a kernel $k(x_i, x_j)$ gives a non-linear SVM without ever building the features, which is lesson 02.06. Second, the optimality (KKT) conditions sort every point into one of three groups:

~~~dg-layers Where each training point sits decides its multiplier, and only points with a nonzero multiplier shape the boundary.
Beyond the margin | alpha = 0; no influence, could be deleted
*On the margin | 0 < alpha < C; free support vectors, they pin the hyperplane
Inside the margin or misclassified | alpha = C; bounded support vectors, slack is positive
~~~

The bias comes from any free support vector: $b = y_i - w^\top x_i$. Since usually only a fraction of points are support vectors, kernel predictions $\sum_i \alpha_i y_i k(x_i, x) + b$ stay cheap.

A few further facts worth knowing:

- **Why the margin helps.** For data inside a ball of radius $R$ and margin $\gamma$, the perceptron makes at most $(R/\gamma)^2$ mistakes, and margin-based generalisation bounds scale with the same ratio. Neither depends on the number of features, which is why margin methods worked well on high-dimensional text data.
- **Solvers.** libsvm (`SVC`) solves the dual by SMO, two multipliers at a time (because $\sum \alpha_i y_i = 0$). liblinear (`LinearSVC`) defaults to the *squared* hinge and also penalises the bias, so its $C$ is not exactly interchangeable with `SVC`'s.
- **Multiclass.** scikit-learn uses one-vs-rest for `LinearSVC` and one-vs-one for `SVC` (02.04).
- **Probabilities.** `SVC(probability=True)` fits Platt scaling with internal cross-validation; it is slow and can disagree slightly with `predict`.

## Key terms

| Term | Meaning
| --- | --- |
| Margin | Distance from the decision boundary to the nearest training point; the SVM maximises it
| Functional margin | $y_i f(x_i)$: positive when point $i$ is correct, larger when more confident
| Support vector | A training point on or inside the margin (nonzero $\alpha_i$); only these determine the boundary
| Hard margin | SVM that allows no violations; possible only for separable data
| Soft margin / slack | Lets points violate the margin at a cost $C\xi_i$ per point
| C | Trade-off between a wide margin (small C) and few violations (large C); $\lambda = 1/(nC)$
| Hinge loss | $\max(0, 1 - y f(x))$: zero once a point is past its margin edge
| Pegasos | Stochastic subgradient solver for the primal SVM with step size $1/(\lambda t)$
| Dual problem | Equivalent problem in the multipliers $\alpha_i$ that uses data only through dot products

<div class="take">
<div class="lab">Key takeaway</div>

an SVM is a linear classifier trained with L2-regularised hinge loss. It chooses the separating line with the widest margin, $C$ trades margin width against violations, only the support vectors shape the answer, and the dual uses only dot products, which opens the door to kernels.

</div>
