---
id: "03.02"
course: "machine-learning"
module: "03"
slug: "k-nearest-neighbours"
title: "k-nearest neighbours"
keyIdeas: ["kNN stores the training set and predicts by majority vote (or mean) of the $k$ closest points: no training cost, but $O(nd)$ work per query by brute force.", "The distance metric *is* the model: unscaled features let the largest-unit feature dominate, and irrelevant features add noise to every distance. Standardise inside a pipeline.", "Small $k$ means low bias and high variance (1-NN has zero training error); large $k$ smooths toward the majority class. Choose $k$ by cross-validation, and consider distance weighting.", "In high dimensions neighbourhoods stop being local: to capture 1% of uniform data in 10 dimensions you need a cube spanning 63% of each axis, and nearest and farthest distances become nearly equal.", "kNN works when data lie near a low-dimensional manifold or in a learned embedding space; at scale it relies on tree indexes (low dimensions) or approximate nearest-neighbour search such as HNSW."]
question: "A kNN classifier gets 90% accuracy on a 5-feature problem. You append 50 features of pure random noise (same scale as the standardised real features). Predict what happens to kNN's accuracy and to an L1-regularised logistic regression's accuracy, explain the difference, and say what you would change if you had to keep using kNN."
answer: "kNN degrades substantially, and the more noise features you add the closer it drifts to chance. Squared Euclidean distance is a sum over features, so 50 noise features contribute about 10 times as much variation to every distance as the 5 real ones. Neighbours are then picked mostly by noise, and the vote mixes labels from points that are not actually similar. Distances also concentrate as the dimension rises, so the 'nearest' points are barely nearer than the rest. kNN has no mechanism to discount a feature: every coordinate counts in proportion to its spread. L1-regularised logistic regression barely changes. Each noise feature has near-zero correlation with the label, so its coefficient falls below the soft-threshold and is set exactly to zero, leaving essentially the original 5-feature model, with a small variance cost from occasional spurious selections. To keep kNN, do the job the weights would have done: select features (for example by mutual information or by the L1 model's non-zero coefficients), reduce dimension with PCA or a learned embedding, or learn a metric that down-weights useless directions; then re-tune $k$ by cross-validation, since the best $k$ changes with the feature set. The general lesson: models that learn per-feature weights can ignore junk, while kNN needs feature selection or a learned metric done for it."
sources: [{"title": "Cornell CS4780 lecture notes: k-Nearest Neighbors", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html", "note": "The algorithm, the Cover-Hart bound and a clear treatment of the curse of dimensionality."}, {"title": "scikit-learn: Nearest Neighbors", "url": "https://scikit-learn.org/stable/modules/neighbors.html", "note": "Brute force vs KD-tree vs ball tree, when each wins, and the available metrics."}, {"title": "The Elements of Statistical Learning, ch. 2.5 and 13", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Local methods in high dimensions (the edge-length calculation) and prototype methods."}]
practice: ["knn-classifier", "distance-metrics", "euclidean-distance", "feature-scaling", "embedding-similarity"]
tags: ["distance metrics", "curse of dimensionality"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- How k-nearest neighbours (kNN) predicts: find the most similar stored examples and let them vote or average.
- Why the distance you choose, and the scale of each feature, decide almost everything about the result.
- How the single knob $k$ moves the model from jagged and over-fitted to smooth and under-fitted, and how to pick it.
- How to run kNN properly in scikit-learn (scaling inside a pipeline, cross-validated $k$) and from scratch in NumPy.
- Why kNN struggles with many features (the curse of dimensionality) and what people do about it in practice.

*Before you start:* you need the bias-variance picture and validation sets from module 01 (lesson 01.02), and the idea of a distance between two points. Lesson 03.01 (Naive Bayes) is useful contrast but not required.

## From a probability model to no model at all

Lesson 03.01 built a classifier by writing down a probability model and estimating its few parameters. kNN takes the opposite route. It writes down no model, estimates no parameters and keeps the whole training set. To predict, it asks a single question: *which stored examples look most like this one?* Methods that work this way are called **instance-based** (or memory-based) methods, and kNN is the simplest of them.

That simplicity makes it a superb baseline and a great teaching tool, because every concept in this lesson (distance, scale, flexibility, dimension) is visible in a picture.

## The idea in one picture

Five labelled points live in a plane with two features. Two are red (class A), three are blue (class B). A new point, the **query**, arrives at (2.5, 1.5) and we want its colour.

| Point | (1, 1) | (2, 1) | (3, 2.2) | (4, 3) | (5, 4)
| --- | --- | --- | --- | --- | --- |
| Label | red | red | blue | blue | blue
| Distance to query | 1.58 | 0.71 | 0.86 | 2.12 | 3.54

~~~dg-fig 03.02-idea.svg
The query (star) and its five candidates. Each circle encloses the k nearest points: with k = 1 or 3 red wins, with k = 5 blue wins because blue simply has more points overall.
~~~

Read the figure from the inside out. The single closest point is red at distance 0.71, so **1-NN** says red. The three closest are red, blue, red: two votes to one, still red. Widen to all five and blue wins 3 to 2, not because blue points are close but because there are more of them.

Three terms are already on the table:

- A **neighbour** is a stored training point; the **k nearest neighbours** are the $k$ stored points with the smallest distance to the query.
- **Majority vote** means the predicted class is the most common label among those $k$.
- The **hyperparameter** $k$ is a setting you choose rather than learn. The example shows it can flip the answer.

## Step 1: the algorithm

"Training" kNN means storing the data, nothing more. That is why it is called a **lazy learner**: all the work is postponed to prediction time. To predict for a query $x$:

1. Compute the distance from $x$ to every training point $x_1, \dots, x_n$.
2. Keep the $k$ points with the smallest distances. Call that set $N_k(x)$.
3. Classification: return the most common label in $N_k(x)$. Regression: return the average of their targets.

In symbols, for regression the prediction is

$$\hat y(x) = \frac{1}{k} \sum_{i \in N_k(x)} y_i,$$

where $y_i$ is the target of training point $i$ and $N_k(x)$ is the index set of the $k$ nearest points. For classification, the share of neighbours with label $c$,

$$\hat p(c \mid x) = \frac{1}{k} \sum_{i \in N_k(x)} \mathbb{1}[y_i = c],$$

is a rough probability estimate ($\mathbb{1}[\cdot]$ is 1 when the condition holds, 0 otherwise). The vote picks the class with the largest share.

~~~dg-flow Every prediction scans the stored data; the metric and k are the only modelling choices.
Query x -(distance to all n points)-> Sorted distances -(take k smallest)-> *k neighbours -(vote or average)-> Prediction
~~~

Number check with the five points: suppose their targets were 10, 12, 20, 25, 31 (say, prices). With $k = 3$ the neighbours are (2, 1), (3, 2.2) and (1, 1), so the regression prediction is $(12 + 20 + 10)/3 = 14$.

There is no fitted weight vector, so kNN can draw boundaries of any shape if you give it enough data. A classic result by Cover and Hart (1967) says that as the number of training points goes to infinity, the error of 1-NN is at most twice the **Bayes error** (the error of the best possible classifier). For a method with no training at all, that is remarkable. The catch is the phrase "goes to infinity": how much data is "enough" grows very fast with the number of features, as Step 5 shows.

## Step 2: distances, and why the metric is the model

A linear model learns a weight per feature and can turn a useless feature down to zero. kNN cannot: every feature counts in the distance exactly as much as its numbers say. So the choice of distance is the modelling decision.

The default is the **Euclidean distance**, the straight-line length:

$$d(x, x') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2},$$

where $d$ is the number of features and $x_j$ is feature $j$ of point $x$. Check: from the query (2.5, 1.5) to (3, 2.2) the differences are 0.5 and 0.7, so $d = \sqrt{0.25 + 0.49} = \sqrt{0.74} = 0.86$, matching the table.

Other metrics suit other data:

| Metric | Formula | Use when
| --- | --- | --- |
| Euclidean | $\sqrt{\sum_j (x_j - x'_j)^2}$ | continuous, standardised features
| Manhattan | $\sum_j \lvert x_j - x'_j \rvert$ | robust to a single large coordinate difference
| Cosine distance | $1 - \frac{x^\top x'}{\|x\|\|x'\|}$ | direction matters, length doesn't: text, embeddings
| Mahalanobis | $\sqrt{(x - x')^\top S^{-1}(x - x')}$ | correlated features with a known covariance $S$
| Hamming | number of differing coordinates | categorical or binary codes

**Scale is the first trap.** Suppose you match customers by age (years) and income (dollars). The query is 30 years old earning 50,000. Customer A is 31 and earns 52,000; customer B is 60 and earns 50,500. Common sense says A is the look-alike. Raw Euclidean distance disagrees: A is $\sqrt{1^2 + 2000^2} \approx 2000$ away, B only $\sqrt{30^2 + 500^2} \approx 501$. Income differences are measured in thousands, age differences in tens, so income decides everything.

The fix is **standardisation**: subtract each feature's training mean and divide by its training standard deviation, so every feature is measured in "standard deviations". With a spread of 12 years for age and 30,000 for income, A is $\sqrt{(1/12)^2 + (2000/30000)^2} = 0.107$ away and B is $\sqrt{(30/12)^2 + (500/30000)^2} = 2.50$ away. A is now 23 times closer, as it should be.

**Irrelevant features are the second trap.** Each feature that carries no information about the label still adds random variation to every distance. With enough of them, the "nearest" points are chosen mostly by noise.

~~~dg-fig 03.02-scale-noise.svg
Left: on the wine dataset (13 chemistry measurements in very different units), 5-NN scores 69% with raw numbers and 95% after standardising. Right: two informative features plus a growing pile of pure-noise features; accuracy slides from 93% to 75% as noise dilutes the distances.
~~~

## Step 3: worked example by hand

<div class="ex">
<div class="lab">Worked example</div>

**k, weighting and units on five points**

Data: red at (1, 1) and (2, 1); blue at (3, 2.2), (4, 3) and (5, 4). Query $q = (2.5, 1.5)$.

Distances. To (2, 1): $\sqrt{0.5^2 + 0.5^2} = \sqrt{0.5} = 0.71$. To (3, 2.2): $\sqrt{0.25 + 0.49} = 0.86$. To (1, 1): $\sqrt{1.5^2 + 0.5^2} = \sqrt{2.5} = 1.58$. To (4, 3): $\sqrt{2.25 + 2.25} = 2.12$. To (5, 4): $\sqrt{6.25 + 6.25} = 3.54$.

Sorted: 0.71 red, 0.86 blue, 1.58 red, 2.12 blue, 3.54 blue.

$k = 1$: one red vote, predict red. $k = 3$: red 2, blue 1, predict red. $k = 5$: red 2, blue 3, predict blue.

Distance weighting with $k = 5$: give each neighbour the weight $1/\text{distance}$. Red total: $1/0.71 + 1/1.58 = 1.414 + 0.632 = 2.047$. Blue total: $1/0.86 + 1/2.12 + 1/3.54 = 1.162 + 0.471 + 0.283 = 1.917$. Red wins again, with probability estimate $2.047 / (2.047 + 1.917) = 0.516$. Weighting stops far-away points from outvoting close ones.

Units: in the customer example above, the same algorithm picked the wrong look-alike (B at 501 versus A at 2000) until both features were standardised (A at 0.107, B at 2.50). Nothing about the problem changed, only the units, and the answer flipped.

</div>

## Step 4: choosing k

$k$ is the complexity knob, and it runs the opposite way to most knobs: *small* $k$ means a *more* flexible model.

- With $k = 1$ every training point is its own nearest neighbour, so training error is exactly zero. The boundary wraps around every noisy label and makes little islands. That is low bias and high variance.
- As $k$ grows, each prediction averages over more points, and the boundary smooths out.
- At $k = n$ every query gets the overall majority class: maximum bias, almost no variance.

~~~dg-fig 03.02-boundaries.svg
Two-moons data with label noise. k = 1 carves islands around single noisy points; k = 15 follows the real curved shape; k = 100 flattens it into nearly a straight line.
~~~

A handy rule of thumb is that kNN behaves as if it had about $n/k$ effective parameters: the data is split into roughly $n/k$ neighbourhoods, each with its own prediction. So $k = 1$ is the most complex model you can build from $n$ points.

Because training error is always zero at $k = 1$, it tells you nothing about the right $k$. Use validation error instead, as lesson 01.02 taught.

~~~dg-fig 03.02-k-curve.svg
Training error rises steadily with k, but cross-validation error is U-shaped: too-small k over-fits the noise, too-large k smooths away the shape. Here the minimum is around k = 15.
~~~

Practical rules:

- Search $k$ by cross-validation over a range such as 1, 3, 5, ..., 51 (odd values avoid tied votes in two-class problems).
- Expect a larger best $k$ when labels are noisy or the dataset is large.
- Try `weights="distance"` when the closest neighbours are much closer than the $k$-th; it reduces sensitivity to the exact $k$.

## kNN in code

In scikit-learn, put the scaler and the classifier in one **pipeline**. Then cross-validation re-fits the scaler on each training fold only, so no statistics leak from the validation fold.

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X, y = load_wine(return_X_y=True)               # 178 wines, 13 features, 3 classes
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

raw = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
print(raw.score(Xte, yte))                       # 0.722  (unscaled: one feature dominates)

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 15, 21, 31],
                           "kneighborsclassifier__weights": ["uniform", "distance"]}, cv=5)
grid.fit(Xtr, ytr)
print(grid.best_params_)                         # n_neighbors=31, weights='distance'
print(grid.best_score_, grid.score(Xte, yte))    # 0.976 0.981
```

The jump from 72% to 98% comes almost entirely from scaling. To see that there is nothing magic inside, here is the brute-force version in NumPy, checked on the five-point example:

```python
import numpy as np

def knn_predict(Xtr, ytr, Xq, k=5):             # ytr: non-negative int labels
    # squared distances, all queries x all training points, without a Python loop
    d2 = (Xq**2).sum(1)[:, None] + (Xtr**2).sum(1)[None, :] - 2 * Xq @ Xtr.T
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]   # k smallest, unordered: O(n) per row
    return np.array([np.bincount(row).argmax() for row in ytr[idx]])

P = np.array([[1, 1], [2, 1], [3, 2.2], [4, 3], [5, 4.]]); lab = np.array([0, 0, 1, 1, 1])
print([knn_predict(P, lab, np.array([[2.5, 1.5]]), k)[0] for k in (1, 3, 5)])   # [0, 0, 1]
```

For regression, swap in `KNeighborsRegressor`; on the five points with targets 10, 12, 20, 25, 31 and `n_neighbors=3` it returns 14.0, as computed by hand.

## Checking the result

- **Compare against a dumb baseline.** Always predicting the majority class (or the mean) is the floor. kNN that barely beats it usually has a scaling or feature problem.
- **Look at the validation curve over $k$.** A curve that is still falling at the largest $k$ you tried means the data is noisy; extend the range. A best $k$ of 1 on noisy data is suspicious; check for duplicate rows shared between training and validation.
- **Inspect a few neighbours.** Call `kneighbors(X_query)` and look at the actual neighbours of a wrong prediction. If they look nothing like the query to a human, the distance is wrong, not $k$.
- **Check feature scales** before fitting: print each column's standard deviation. Spreads that differ by factors of 100 mean you forgot to scale.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Accuracy jumps when you change units | features not scaled | `StandardScaler` (or `MinMaxScaler`) inside a pipeline
| Accuracy falls as you add features | irrelevant or noisy features dilute distances | feature selection, PCA, or a learned metric
| Perfect training score, poor validation | $k$ too small (often $k = 1$) | choose $k$ by cross-validation
| Rare class never predicted | majority class floods every neighbourhood | distance weighting, smaller $k$, rebalance data
| Predictions slow at serving time | brute force is $O(nd)$ per query | tree index (low $d$) or approximate search (high $d$)
| Scaler fitted on all data | test statistics leak into training | fit the scaler on the training fold only (pipeline does this)
| Categorical features coded as 1, 2, 3 | fake order and fake distances | one-hot encode, or use Hamming distance

## Going deeper: the curse of dimensionality

kNN rests on one assumption: *nearby points have similar labels*. In many dimensions there are no nearby points.

**Volume.** Spread $n$ points uniformly in the unit cube $[0,1]^d$. A smaller cube that captures a fraction $r$ of them needs edge length $r^{1/d}$. For $r = 0.01$: $0.01^{1/2} = 0.1$ in 2 dimensions, $0.01^{1/10} = 0.63$ in 10, and $0.01^{1/100} = 0.955$ in 100. A "neighbourhood" holding just 1% of the data spans most of the range of every feature, so it is not local any more.

**Distance concentration.** With many independent features, the distances from a query to its nearest and farthest points become nearly equal. The relative gap $(\text{max} - \text{min})/\text{min}$ shrinks toward 0 as $d$ grows. When everything is about equally far away, "nearest" carries little information.

~~~dg-fig 03.02-curse.svg
Left: the edge length a neighbourhood needs to hold 1% or 10% of uniform data races toward the full range as features are added (0.63 of each axis at d = 10). Right: in a simulation with 2000 uniform points, the farthest point is over 100 times farther than the nearest in 2-D but less than 1.5 times as far in 100-D.
~~~

**Sample size.** To keep neighbourhoods equally small as $d$ grows, $n$ must grow exponentially in $d$. That is the hidden price in the Cover-Hart guarantee.

kNN still works on real high-dimensional data for two reasons. First, real data usually lie near a **low-dimensional manifold**: a 28 by 28 digit image has 784 pixels but far fewer real degrees of freedom (stroke thickness, slant, position). Second, you can move to a space where distance means something: PCA (module 05) or a learned **embedding**, where kNN becomes "retrieval" and powers search and recommendation.

**Making it fast.** Brute force costs $O(nd)$ per query and vectorises well; it is fine up to around $10^5$ points. KD-trees and ball trees split space so that whole regions can be skipped, giving roughly logarithmic query time in low dimensions (about $d \lesssim 20$), but they degrade toward brute force beyond that, the curse again. For millions of embedding vectors, use **approximate nearest-neighbour** (ANN) indexes such as HNSW (a navigable graph) or IVF (cluster then search a few clusters). They give up a little recall for orders-of-magnitude speed; the AI Engineering course (lessons 02.06-02.07) covers them.

**Why the Cover-Hart bound holds, in one paragraph.** As $n \to \infty$ the nearest neighbour of $x$ lands arbitrarily close to $x$, so its label is (nearly) an independent draw from the same class probabilities $p(c \mid x)$. 1-NN therefore errs when two independent draws from $p(\cdot \mid x)$ disagree. For two classes with $p = p(1 \mid x)$, that probability is $2p(1-p)$, while the Bayes error is $\min(p, 1-p)$. Since $2p(1-p) \le 2\min(p, 1-p)$, the asymptotic 1-NN error is at most twice the Bayes error. Letting $k$ grow too (with $k/n \to 0$) makes kNN **consistent**: its error converges to the Bayes error itself.

## Key terms

| Term | Meaning
| --- | --- |
| Instance-based / lazy learning | Store the training data and do all the work at prediction time
| k nearest neighbours $N_k(x)$ | The $k$ stored points closest to the query under the chosen distance
| Majority vote / averaging | Classification and regression rules applied to the neighbours
| Distance weighting | Each neighbour's vote counts $1/\text{distance}$, so close points matter more
| Standardisation | Rescale each feature to mean 0 and standard deviation 1 using training statistics
| Bayes error | The lowest error any classifier could reach on the problem
| Curse of dimensionality | In many dimensions neighbourhoods stop being local and distances concentrate
| Approximate nearest neighbours | Indexes (HNSW, IVF) that find almost-nearest points far faster than brute force

<div class="take">
<div class="lab">Key takeaway</div>

kNN predicts by letting the k most similar stored examples vote, so it is only as good as its distance. Standardise the features, drop the irrelevant ones, choose k by cross-validation, and in high dimensions move to a space (PCA or a learned embedding) where nearness really means similarity.

</div>
