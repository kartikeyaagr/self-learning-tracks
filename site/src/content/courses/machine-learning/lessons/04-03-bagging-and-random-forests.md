---
id: "04.03"
course: "machine-learning"
module: "04"
slug: "bagging-and-random-forests"
title: "Bagging and random forests"
keyIdeas: ["Averaging $B$ predictors with variance $\\sigma^2$ and pairwise correlation $\\rho$ gives variance $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$: more trees shrink only the second term, while the first is a **floor** that only making the trees more different can lower. Averaging leaves bias unchanged, so use deep, low-bias trees.", "Bagging trains each tree on a **bootstrap sample** ($n$ draws with replacement). Each sample contains about 63.2% of the distinct points, because $(1-1/n)^n \\to e^{-1} \\approx 0.368$; it helps unstable learners like deep trees and does little for stable ones.", "Random forests also offer only a random subset of `max_features` features at every split (classically $\\sqrt{d}$ for classification, $d/3$ for regression), which stops every tree from splitting on the same dominant feature. scikit-learn's `RandomForestRegressor` defaults to all features, which is plain bagging.", "Out-of-bag (OOB) error scores each point using only the trees that did not see it: a nearly free estimate of test error, reliable once there are a few dozen trees.", "Adding trees never causes overfitting in the usual sense, but returns diminish quickly; `max_features` and leaf size are the knobs that matter, and forests cannot extrapolate beyond the training targets."]
question: "A colleague's random forest regressor on 40 features barely beats a single tuned tree, and adding trees from 100 to 2000 changes nothing. They used scikit-learn defaults. What is the likely cause, and what would you change first? How would you check whether your change helped without touching the test set?"
answer: "The flat curve as trees are added says the averaging term $(1-\\rho)\\sigma^2/B$ is already negligible: what remains is the correlation floor $\\rho\\sigma^2$ plus bias, and more trees cannot touch either. With scikit-learn's `RandomForestRegressor`, the default `max_features=1.0` considers all 40 features at every split, so the model is plain bagging of trees. If one or two features dominate, almost every bootstrap tree makes the same top splits, the trees are highly correlated, and the ensemble behaves like a slightly smoothed single tree. I would first lower `max_features`, trying 0.33 (the classic $d/3$, about 13 features), `\"sqrt\"` (6 features) and 0.5, keeping a few hundred trees. To compare without the test set, fit with `oob_score=True` and compare the OOB $R^2$ (or OOB error) across settings; OOB is reliable at this number of trees and needs no extra validation split, provided any preprocessing was fitted inside a pipeline. If many of the 40 features are noise, a small `max_features` can hurt, so the comparison matters rather than a rule. If no setting helps, I would try a larger `min_samples_leaf` for a noisy target, and check whether the target needs extrapolation or a smooth trend (for example a time trend), which no tree ensemble provides; in that case model the trend separately or move to gradient boosting on detrended targets."
sources: [{"title": "Breiman (2001): Random Forests", "url": "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf", "note": "The original paper: strength vs correlation bound, feature subsampling, OOB estimates."}, {"title": "Breiman (1996): Bagging Predictors", "url": "https://www.stat.berkeley.edu/~breiman/bagging.pdf", "note": "Why bagging helps unstable procedures and does little for stable ones."}, {"title": "scikit-learn User Guide: Ensembles (forests)", "url": "https://scikit-learn.org/stable/modules/ensemble.html", "note": "Section on random forests and Extra-Trees, including current max_features defaults and parameter advice."}, {"title": "scikit-learn example: OOB errors for random forests", "url": "https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html", "note": "OOB error as a function of the number of trees for different max_features."}]
practice: ["random-forest", "bagging-classifier", "bootstrap-confidence-interval"]
tags: ["bagging", "random forest", "OOB error"]
tools: ["scikit-learn"]
---

## What you will learn

- Why averaging many deep trees beats one deep tree, and exactly how much variance averaging can remove.
- What a bootstrap sample is, and why each one contains about 63.2% of the training points.
- How a random forest adds a second source of randomness (a random subset of features at every split) and why that matters more than adding trees.
- How to get a nearly free validation score from the points each tree never saw: the out-of-bag (OOB) error.
- How to train, check and tune a forest in scikit-learn, and the situations where forests quietly fail.

*Before you start:* you need decision trees and how they split (04.01), why fully grown trees overfit (04.02), and the bias-variance trade-off (module 01, lesson 02). Variance and correlation of random variables are used in Step 1; the ML Maths course covers them if they are rusty.

Lesson 04.02 ended with a tension. A fully grown tree has low bias (it can carve out any pattern) but high variance (move a few training points and the tree changes shape). Pruning trades some of that variance for bias. This lesson takes a different route: keep the trees deep and unbiased, grow many of them on slightly different data, and average. Averaging cancels much of the noise while leaving the bias alone.

## The idea in one picture

Suppose you run a bike-sharing scheme and want to predict daily rentals from the day's temperature. You have 80 days of records. Rentals peak around 24 °C and fall away on cold and very hot days, with plenty of day-to-day noise.

~~~dg-fig 04.03-smooth.svg
Left: one fully grown tree passes through every noisy point, so its curve is a jagged staircase. Right: three individual bagged trees are just as jagged (thin lines), but the average of 200 of them stays much closer to the true trend, halving the error.
~~~

Zoom in on 30 °C. Five trees, each trained on a different resample of the 80 days, predict:

| Tree | 1 | 2 | 3 | 4 | 5 | average of 5 | average of 200 | true trend
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prediction (hundreds) | 7.8 | 16.2 | 16.2 | 7.8 | 7.8 | 11.2 | 11.0 | 13.5

Each deep tree copies one nearby day: a quiet one (7.8) or a busy one (16.2), depending on which days landed in its resample. So a single tree is off by 3 to 6 hundred either way. The average sits between them, much closer to the truth. Across all temperatures, the squared error against the true trend drops from 6.6 for one tree to 3.2 for the average of 200.

Three terms you will use throughout:

- An **ensemble** is a model made of many models whose predictions are combined, here by averaging.
- **Bagging** (short for *bootstrap aggregating*) trains each member on a random resample of the training data and averages them.
- A **random forest** is bagging of trees plus one extra trick: each split may only choose among a random handful of features.

## Step 1: why averaging removes variance (and where it stops)

Think of a tree's prediction at one input $x$ as a random number: it depends on the luck of which training data it saw. Call its variance $\sigma^2$. If you average $B$ such predictions $f_1(x), \dots, f_B(x)$, the variance of the average depends on how similar the members are. Measure similarity by $\rho$ (rho), the **correlation** between any two members' predictions: 0 means they wobble independently, 1 means they always agree.

$$\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B} f_b(x)\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

Read the two terms separately:

- The second term, $(1-\rho)\sigma^2/B$, shrinks toward zero as you add trees. This is the part averaging can remove.
- The first term, $\rho\sigma^2$, does not depend on $B$ at all. It is a **floor**: the shared part of the trees' errors that no amount of averaging cancels.

Averaging does nothing to **bias**: every tree is built the same way, so they share one expected prediction, and so does their average. The design brief follows: members should be low-bias (deep, unpruned trees) and as different from each other as possible (low $\rho$).

~~~dg-fig 04.03-variance.svg
Variance of an average of B trees, each with variance 1. Every curve drops fast for the first 10 to 30 trees and then flattens at a floor equal to the correlation between trees.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why decorrelation beats more trees**

Take $\sigma^2 = 1$ for a single deep tree's prediction at some point.

Plain bagging with trees correlated at $\rho = 0.5$ and $B = 100$: variance $= 0.5 + 0.5/100 = 0.5 + 0.005 = 0.505$.

Ten times more trees, $B = 1000$: variance $= 0.5 + 0.5/1000 = 0.5005$. The extra 900 trees bought 0.0045.

A random forest that cuts the correlation to $\rho = 0.2$, still $B = 100$: variance $= 0.2 + 0.8/100 = 0.2 + 0.008 = 0.208$.

Decorrelating more than halved the variance (0.505 to 0.208); a tenfold increase in trees bought almost nothing.

</div>

## Step 2: make the trees different with bootstrap samples

Breiman's bagging (1996) creates diversity by giving each tree its own dataset. You cannot collect a fresh dataset for every tree, so you fake it. A **bootstrap sample** is $n$ draws *with replacement* from your $n$ training points: the same size as the original, but some points appear twice or three times and others not at all.

How many points get left out? On a single draw, a particular point is missed with probability $1 - 1/n$. The $n$ draws are independent, so the point is missed every time with probability

$$\Big(1 - \frac{1}{n}\Big)^{n} \;\longrightarrow\; e^{-1} \approx 0.368 \quad\text{as } n \text{ grows.}$$

So each bootstrap sample contains about $1 - 0.368 = 63.2\%$ of the distinct points and leaves out roughly a third. The left-out points are called the **out-of-bag** points for that tree.

~~~dg-fig 04.03-bootstrap.svg
Left: the share of distinct points in a bootstrap sample starts at 1 and settles at 0.632 within a few dozen points. Right: in one bootstrap of 1000 points, about 37% are never drawn, 37% once, 18% twice and 6% three times, as a Poisson distribution with mean 1 predicts.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one bootstrap sample by hand**

Training set: five days, numbered 1 to 5. Roll a fair five-sided die five times and get 2, 4, 4, 1, 2.

The bootstrap sample is days (2, 4, 4, 1, 2): day 2 and day 4 appear twice, day 1 once. The tree trains on these five rows exactly as if they were a normal dataset.

Distinct days used: {1, 2, 4}, that is 3 of 5 = 60%. Out-of-bag days: {3, 5}.

Theory for $n = 5$: a given day is missed with probability $(4/5)^5 = 0.32768$, so on average $1 - 0.328 = 67.2\%$ of days are used. Our 60% is one random draw around that. For large $n$ the expected share falls to 63.2%.

</div>

Each tree is then grown deep on its own sample, with no pruning. To predict:

- For **regression**, average the trees' numbers.
- For **classification**, either take a majority vote or, usually better, average the trees' class-probability estimates and pick the largest. scikit-learn does the latter (**soft voting**).

~~~dg-flow Random forest training and prediction. The trees never talk to each other, so they can be trained in parallel on separate cores.
Training set -(bootstrap b)-> Sample b -(random √d features per split)-> Deep tree b -(average over b)-> *Forest prediction
~~~

Bagging only helps **unstable** learners: ones whose output changes a lot when the training data changes, such as deep trees. For **stable** learners like linear regression or k-NN with large $k$, every bootstrap replica gives nearly the same model, so $\rho$ is close to 1 and the floor $\rho\sigma^2$ is nearly the whole variance. You pay $B$ times the compute for nothing.

## Step 3: random forests also sample features

Bootstrap samples alone leave trees surprisingly similar. If one or two features are much stronger than the rest, almost every bootstrap tree puts the same feature at its root, picks a similar threshold, and grows similar structure underneath. The trees agree on their mistakes, so $\rho$ and the floor stay high.

Breiman's random forest (2001) adds a second source of randomness. At *every split* of *every tree*, only a random subset of $m$ features is offered as candidates; the best split among those $m$ is chosen. In scikit-learn this is `max_features`. Sometimes the dominant feature is not on offer, so the tree has to find a different, useful structure. Classic defaults are:

- $m = \sqrt{d}$ for classification ($d$ is the number of features). With 30 features, about 5 are considered per split.
- $m = d/3$ for regression.

Be careful with current scikit-learn defaults: `RandomForestClassifier` uses `max_features="sqrt"`, but `RandomForestRegressor` defaults to `1.0` (all features), which makes it plain bagging of trees unless you change it.

Smaller $m$ is a trade-off. It lowers $\rho$, but each tree gets worse, because some splits are forced onto weak features. The next figure measures both effects on a synthetic regression problem with 20 features, of which only 5 carry signal.

~~~dg-fig 04.03-maxfeat.svg
Left: the correlation between trees (measured by refitting on 8 independent training sets) rises steadily with max_features. Right: individual trees improve a lot as they see more candidates, but the forest improves far less, because rising correlation eats most of the gain.
~~~

First, the forest is always far better than its average member (around 6 versus 14 or more), which is Step 1 in action. Second, the best $m$ depends on the data. Here 15 of 20 features are pure noise, so a tiny $m$ often offers *only* noise features to a split, and the forest suffers; the best values are large. On data where many features carry some signal, small $m$ wins more often. Treat `max_features` as the one hyperparameter worth tuning.

**Extremely randomised trees** (Extra-Trees) push the idea further: for each candidate feature they draw the split threshold at random instead of searching for the best one, and by default they skip bootstrapping. They are faster and often as accurate.

## Step 4: out-of-bag error, a validation set for free

Every tree skipped about 37% of the training points. Flip that around: every training point was skipped by about $0.37B$ trees. Those trees have never seen it, so they can make an honest prediction for it.

The **out-of-bag (OOB) prediction** for point $i$ averages only the trees whose bootstrap sample did not contain $i$. Score all OOB predictions against the true labels and you get the **OOB error**, an estimate of test error that costs no extra data and almost no extra compute.

<div class="ex">
<div class="lab">Worked example</div>

**one OOB prediction**

A forest has 5 trees. Day 3 was left out of the bootstrap samples of trees 1, 3 and 4.

Those trees give probabilities of "busy day" 0.8, 0.6 and 0.3. Trees 2 and 5 trained on day 3, so they are ignored.

OOB probability $= (0.8 + 0.6 + 0.3)/3 = 1.7/3 \approx 0.57$, so the OOB prediction is "busy". Day 3 really was busy: one OOB hit.

Repeat for every day and report the share of misses. With only 5 trees, a point has no OOB tree at all with probability about $0.632^5 \approx 0.10$, so one day in ten cannot be scored. With 50 trees that chance is effectively zero.

</div>

~~~dg-fig 04.03-oob.svg
Out-of-bag and held-out test error of a random forest as trees are added. With very few trees OOB is pessimistic because each point is judged by only a couple of trees; from a few dozen trees on it follows the test curve closely, and both sit well below a single deep tree.
~~~

OOB behaves much like cross-validation and is a convenient way to choose the number of trees or `max_features` without carving out a validation set. Two cautions: with few trees it is noisy and biased upward (the figure's left end), and it is only honest if every preprocessing step was also learned without peeking at the OOB rows.

## Bagging and random forests in code

A real dataset this time: 569 breast tumours with 30 measurements each, labelled benign or malignant.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier

X, y = load_breast_cancer(return_X_y=True)            # 569 tumours, 30 features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=300,
                        oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                            oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
et = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(X_tr, y_tr)

for name, m in [("tree", tree), ("bagging", bag), ("forest", rf), ("extra", et)]:
    print(name, round(m.score(X_te, y_te), 3), round(getattr(m, "oob_score_", float("nan")), 3))
# tree     0.906  nan     one deep tree
# bagging  0.93   0.96    300 bootstrapped trees, all 30 features per split
# forest   0.953  0.967   plus 5 random features per split
# extra    0.942  nan     random thresholds, no bootstrap, so no OOB
print(rf.predict_proba(X_te[:3]).round(3))   # averaged leaf frequencies, e.g. [0.997 0.003]
```

The ordering is typical: one tree is worst, bagging closes most of the gap, feature subsampling adds a bit more. OOB accuracies (0.96, 0.967) land near the test accuracies (0.93, 0.953): two noisy estimates of the same thing on a few hundred points.

`n_jobs=-1` trains trees on every core, `random_state` makes the random draws reproducible, and no feature scaling is needed because trees only compare a feature with a threshold (04.01).

## Reading and checking a forest

How do you know the forest is doing its job?

1. **Compare with one tree and with a simple baseline.** If the forest barely beats a single tuned tree, trees are too correlated (lower `max_features`) or the signal is not tree-shaped (smooth trends, extrapolation).
2. **Plot OOB error against the number of trees.** It should fall and flatten. If it is still falling at your chosen $B$, add trees; if it flattened long ago, you can cut trees to save memory and latency.
3. **Tune `max_features` by OOB.** Try `"sqrt"`, 0.33, 0.5 and 1.0. This usually matters more than anything else.
4. **Then leaf size.** Fully grown trees (`min_samples_leaf=1`) are a strong default. For noisy regression targets, try 3 to 20.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Adding trees from 100 to 2000 changes nothing, forest barely beats one tree | Correlation floor: all trees split on the same dominant features (often `RandomForestRegressor` with default `max_features=1.0`) | Lower `max_features` to 0.33, `"sqrt"` or 0.5; compare by OOB
| Forest much worse than expected with many columns | Few informative features buried among hundreds of noise features; with small $m$ many splits see only noise | Raise `max_features`, or remove obviously useless columns first
| Predictions flat beyond the training range | Trees cannot extrapolate: every leaf predicts an average of training targets | Model the trend separately (e.g. linear) and let the forest fit residuals; use time-aware features
| Model file of several gigabytes, slow predictions | 500 fully grown trees on a million rows, each with up to ~$2n$ nodes | Cap `max_leaf_nodes` or `max_depth`, raise `min_samples_leaf`, use fewer trees
| OOB score much better than the test score | Leakage: preprocessing or duplicate rows let OOB points be seen indirectly | Fit preprocessing in a pipeline; deduplicate; check with a true hold-out
| OOB warning that some points have no OOB prediction | Too few trees | Use at least 30 to 50 trees before trusting OOB
| Probabilities bunched in the middle | Averaging many votes pulls extremes inward | Calibrate (Platt or isotonic), module 03

Adding trees never makes expected test error worse, but it cannot fix bias or the correlation floor. A forest can still overfit through its trees: with very noisy labels, fully grown trees memorise the noise, and larger leaves help.

## Going deeper

**Where the variance formula comes from.** Write $\bar f = \frac{1}{B}\sum_b f_b$. Then $\operatorname{Var}(\bar f) = \frac{1}{B^2}\sum_{b,c}\operatorname{Cov}(f_b, f_c)$. There are $B$ diagonal terms equal to $\sigma^2$ and $B(B-1)$ off-diagonal terms equal to $\rho\sigma^2$, so $\operatorname{Var}(\bar f) = \frac{\sigma^2}{B} + \frac{B-1}{B}\rho\sigma^2 = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$.

**What $\rho$ really measures.** Given one fixed training set, two trees differ only through their own random draws (bootstrap and features), so they are independent. The correlation comes from the training set they *share*: if the data happen to contain a noisy busy day near 30 °C, many trees copy it. That is why the figure measured $\rho$ by refitting on independent training sets.

**Breiman's strength-correlation bound.** For classification, define the **margin** of the forest on a point as the vote share for the true class minus the largest share for any wrong class, and the **strength** $s$ as the expected margin. Breiman (2001) proved that the generalisation error is at most $\bar\rho\,(1 - s^2)/s^2$, where $\bar\rho$ is the mean correlation between trees' raw margins. It is a loose bound, but it names the trade-off exactly: feature subsampling lowers $\bar\rho$ and also lowers $s$, and the best `max_features` balances them.

**Bootstrap counts are Poisson.** The number of times a point is drawn is Binomial$(n, 1/n)$, which tends to Poisson(1): probabilities $e^{-1}/k!$, so 36.8% zero, 36.8% one, 18.4% two, 6.1% three. This is what the right panel of the bootstrap figure shows; streaming ("online") bagging uses it by giving each arriving point a random Poisson(1) weight per tree.

**Cost.** One reasonably balanced tree on $n$ rows with $m$ candidate features per split costs roughly $m\, n \log n$ work, so a forest costs about $B$ times that, spread across cores. Smaller $m$ makes each tree cheaper, so a random forest often trains faster than plain bagging.

**Next.** Forests cut variance by averaging low-bias trees built in parallel. Boosting (04.04, 04.05) cuts bias by adding small trees in sequence, each fixing earlier mistakes; tuned, it usually wins on accuracy, while forests need far less tuning.

## Key terms

| Term | Meaning
| --- | --- |
| Ensemble | A model that combines the predictions of many models
| Bootstrap sample | $n$ draws with replacement from $n$ training points; holds about 63.2% of distinct points
| Bagging | Training each model on its own bootstrap sample and averaging their predictions
| Correlation floor $\rho\sigma^2$ | The part of the variance shared by all trees, which more trees cannot remove
| Random forest | Bagged deep trees that consider only a random subset of features at each split
| `max_features` ($m$) | Number of candidate features per split; lower means more different but weaker trees
| Out-of-bag (OOB) error | Error measured on each point using only the trees that never saw it
| Extra-Trees | Forest variant with random split thresholds and, by default, no bootstrap
| Soft voting | Averaging class probabilities across trees instead of counting votes

<div class="take">
<div class="lab">Key takeaway</div>

averaging deep trees removes variance only down to the floor $\rho\sigma^2$, so random forests add per-split feature sampling to lower $\rho$, which helps far more than adding trees; bootstrapping also gives OOB error as a free validation estimate, and `max_features` is the knob worth tuning.

</div>
