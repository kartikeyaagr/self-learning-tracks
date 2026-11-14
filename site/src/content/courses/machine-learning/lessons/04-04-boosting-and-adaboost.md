---
id: "04.04"
course: "machine-learning"
module: "04"
slug: "boosting-and-adaboost"
title: "Boosting and AdaBoost"
keyIdeas: ["Bagging trains strong learners in parallel to reduce variance; boosting trains **weak learners** (often one-split stumps) in sequence, each focused on what the ensemble so far gets wrong, to reduce bias.", "AdaBoost keeps a weight per sample. After each weak learner with weighted error $\\varepsilon_m$, it gives the learner a vote $\\alpha_m = \\tfrac12 \\ln\\frac{1-\\varepsilon_m}{\\varepsilon_m}$ and multiplies the weights of misclassified points by $e^{\\alpha_m}$ (correct ones by $e^{-\\alpha_m}$).", "After reweighting, mistakes and correct points each hold exactly half the weight, so the last learner has weighted error 0.5 and the next learner is forced to find something new.", "AdaBoost is forward stagewise additive modelling with exponential loss $e^{-yF(x)}$, whose minimiser is half the log-odds. That view generalises to gradient boosting.", "Exponential loss punishes confident mistakes exponentially, so AdaBoost is fragile under label noise: pick the number of rounds on a validation curve, use a learning rate below 1, or switch to a log-loss booster."]
question: "You train AdaBoost with stumps on 20,000 customer records to predict churn. Training error reaches 0 after 50 rounds. You later learn that about 5% of the churn labels were entered wrongly, and the validation curve from `staged_predict` shows error falling until round 200 and rising after that. Explain the mechanism, say whether 'training error is 0' was a useful stopping signal, and give two or three concrete changes you would make."
answer: "Each wrongly labelled customer sits in a region where the true pattern says the opposite, so most stumps misclassify it. Under AdaBoost's update its weight is multiplied by $e^{\\alpha_m}$ almost every round, so after a few dozen rounds the 5% noisy records hold a far larger share of the total weight (in a simulation with 14% flipped labels they held over a third of it). Later stumps are therefore chosen mainly to fix those few records, carving small regions around them that do not exist in new customers, and validation error rises. This is the exponential loss at work: a point with margin $-4$ costs $e^4 \\approx 55$, so persistent mistakes dominate the fit. Training error reaching 0 was not a useful signal either way: on clean data AdaBoost often keeps improving after that point because later rounds enlarge margins, and here the rise came 150 rounds later. Changes: (1) choose the number of rounds from the validation curve (about 200 here) or refit with `n_estimators` set there; (2) use a learning rate below 1, such as 0.1, with more rounds, which slows the weight explosion and flattens the curve; (3) switch to gradient boosting with log-loss (lesson 04.05) or LogitBoost, whose loss grows only linearly in the negative margin, so noisy points gain weight far more slowly. Inspecting the records with the largest final weights is also a cheap way to find and fix bad labels."
sources: [{"title": "Freund and Schapire (1997): A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting", "url": "https://doi.org/10.1006/jcss.1997.1504", "note": "The original AdaBoost paper and its training-error bound."}, {"title": "Friedman, Hastie, Tibshirani (2000): Additive Logistic Regression", "url": "https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf", "note": "Shows AdaBoost is stagewise fitting of exponential loss; introduces LogitBoost."}, {"title": "Schapire: Explaining AdaBoost", "url": "https://rob.schapire.net/papers/explaining-adaboost.pdf", "note": "Short survey of the different explanations, including margins and the noise problem."}, {"title": "Zhu, Zou, Rosset, Hastie: Multi-class AdaBoost (SAMME)", "url": "https://hastie.su.domains/Papers/samme.pdf", "note": "The multiclass variant used by scikit-learn."}]
practice: ["adaboost", "stacking-ensemble", "bagging-classifier"]
tags: ["boosting", "AdaBoost", "exponential loss"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- What boosting is and how it differs from bagging: weak models trained in sequence, each correcting the last.
- The AdaBoost algorithm step by step: sample weights, weighted error, the vote $\alpha$, and the reweighting rule.
- How to run three rounds of AdaBoost by hand on ten points, then in NumPy and scikit-learn.
- How to read boosting curves and pick the number of rounds.
- Why AdaBoost is secretly minimising exponential loss, which makes it fragile under label noise.

*Before you start:* you need decision trees and stumps (04.01), the idea of limiting a tree's complexity (04.02), and bagging (04.03), which this lesson contrasts with. The loss-function view at the end uses the margin and log-odds ideas from logistic regression (02.03).

## From averaging to correcting

Bagging (04.03) averages many deep, independent trees, which cancels their jumpiness and lowers **variance**. Boosting goes the other way: it starts from a model far too simple, such as a single split, and adds simple models one at a time, each trained to fix what the ensemble so far still gets wrong. That mainly lowers **bias**, the systematic error of a model that is not flexible enough.

The building block is a **weak learner**: any model slightly better than random guessing. The classic choice is a **decision stump**, a tree with one split. One stump is a poor classifier; a weighted vote of a few hundred, each covering the previous ones' blind spots, can be excellent. **AdaBoost** (adaptive boosting, Freund and Schapire, 1997) was the first practical boosting method and is still the clearest to learn from.

| Aspect | Bagging / random forest | Boosting
| --- | --- | --- |
| Base learner | Deep, low-bias tree | Shallow, high-bias tree or stump
| Training | Parallel, independent | Sequential, each depends on the last
| Main effect | Reduces variance | Reduces bias (and some variance)
| Too many rounds | Harmless, just slower | Can overfit; needs a stopping rule
| Label noise | Robust | AdaBoost in particular is sensitive

## The idea in one picture

Take ten points on a line, $x = 1, 2, \dots, 10$, with labels $+1$ or $-1$:

| x | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label y | +1 | +1 | +1 | −1 | −1 | −1 | +1 | +1 | +1 | −1

No single stump can classify this correctly, because the labels change sign three times and a stump can only make one cut. AdaBoost gives every point a **sample weight** (how much the next learner should care about it), starting equal at 0.1 each. Then it repeats: fit the best stump to the weighted points, give that stump a vote, and make the points it got wrong heavier.

~~~dg-fig 04.04-rounds.svg
Three AdaBoost rounds on ten points. Marker size is the sample weight and ringed markers are this round's mistakes. Mistakes grow before the next round, so the next stump is pushed to fix them. Bottom: the weighted sum of the three stumps has the right sign everywhere.
~~~

Read it top to bottom:

- **Round 1** cuts at 3.5 (left $+1$, right $-1$) and misses points 7, 8, 9.
- Those three become heavy, so **round 2** must care about them. It cuts at 9.5, gets 7 to 9 right, and misses 4, 5, 6.
- **Round 3** sees heavy points 4 to 9 and cuts at 6.5 (left $-1$, right $+1$).
- Each stump gets a **vote** $\alpha$. The score $F(x)$ is the sum of votes times stump outputs; its sign is the prediction. All ten points end up correct, though each stump alone gets 3 or 4 wrong.

Boosting in one sentence: each learner is weak, but each is trained on data reweighted to stress what the team gets wrong.

## Step 1: weighted error

Labels are coded $y_i \in \{-1, +1\}$, and each stump $h(x)$ outputs $-1$ or $+1$. With sample weights $w_i$ that add up to 1, the **weighted error** of a stump is the total weight of the points it gets wrong:

$$\varepsilon = \sum_{i=1}^{n} w_i \,[h(x_i) \ne y_i],$$

where $[\cdot]$ is 1 if the condition holds, else 0. With equal weights this is the ordinary error rate. In round 1 of the picture, the stump misses three points of weight 0.1, so $\varepsilon_1 = 0.3$.

To fit a stump to weighted data, search features and thresholds as in 04.01 but count each point by its weight: `DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)`. A learner that cannot take weights can be fitted on a resample drawn with probabilities $w$.

## Step 2: give the stump a vote

A stump with low weighted error deserves a big say in the final vote. AdaBoost uses

$$\alpha = \tfrac12 \ln\frac{1-\varepsilon}{\varepsilon}.$$

Here $\ln$ is the natural log and $\frac{1-\varepsilon}{\varepsilon}$ is the stump's odds of being right. For $\varepsilon = 0.3$, $\alpha = \tfrac12 \ln 2.333 = 0.424$; for $\varepsilon = 0.1$, $\alpha = \tfrac12 \ln 9 = 1.10$; a coin flip ($\varepsilon = 0.5$) gets $\alpha = 0$; a perfect stump would get $\alpha = \infty$.

~~~dg-fig 04.04-alpha.svg
Left: the vote falls smoothly as the weighted error rises, reaching zero at 0.5. Right: the factors used to reweight points. A strong stump (small error) moves the weights a lot; a near-coin-flip stump barely moves them.
~~~

A stump with error above 0.5 gets a negative vote, meaning "flip what it says", so a weak learner only has to avoid being an exact coin flip.

## Step 3: reweight the points

Now make the mistakes heavier and the correct points lighter:

$$w_i \leftarrow w_i \exp\big(-\alpha\, y_i\, h(x_i)\big), \quad\text{then divide every } w_i \text{ by their sum}.$$

The product $y_i h(x_i)$ is $+1$ when the stump is right and $-1$ when it is wrong. So a correct point is multiplied by $e^{-\alpha}$ (it shrinks) and a mistake by $e^{\alpha}$ (it grows). Dividing by the sum keeps the weights a probability distribution. That divisor is usually written $Z$.

<div class="ex">
<div class="lab">Worked example</div>

**one round in numbers**

A stump has weighted error $\varepsilon = 0.3$ (round 1 of the picture: points 7, 8, 9 wrong, weight 0.1 each). Its vote is $\alpha = \tfrac12 \ln(0.7/0.3) = 0.424$.

Multipliers: mistakes $\times\, e^{0.424} = 1.528$; correct points $\times\, e^{-0.424} = 0.655$.

Before normalising, each mistake weighs $0.1 \times 1.528 = 0.1528$ and each correct point $0.1 \times 0.655 = 0.0655$. Totals: mistakes $3 \times 0.1528 = 0.458$, correct $7 \times 0.0655 = 0.458$. They are equal, and $Z = 0.917$.

After dividing by $Z$: each mistake weighs $0.1528/0.917 = 0.167 = 1/6$, each correct point $0.0655/0.917 = 0.071 = 1/14$. These are the sizes in round 2 of the figure.

Check: the stump just used now has weighted error $3 \times 1/6 = 0.5$. On the new weights it is a coin flip, so the next round must find something different.

</div>

This holds in every round: mistakes have total weight $\varepsilon e^{\alpha} = \varepsilon\sqrt{(1-\varepsilon)/\varepsilon} = \sqrt{\varepsilon(1-\varepsilon)}$, and correct points have $(1-\varepsilon)e^{-\alpha} = \sqrt{\varepsilon(1-\varepsilon)}$: the same amount. Hence a shortcut: **after each round, the mistakes share half of the weight and the correct points share the other half**, each group keeping its internal proportions.

## Step 4: repeat, then add up the votes

Rounds 2 and 3 of the picture, with the shortcut.

<div class="ex">
<div class="lab">Worked example</div>

**rounds 2 and 3, and the final vote**

Round 2 weights: points 7, 8, 9 weigh $1/6$ each; the other seven weigh $1/14$. Cutting again at 3.5 would now have error $3 \times 1/6 = 0.5$, which is useless. The best stump cuts at 9.5 (left $+1$, right $-1$) and misses points 4, 5, 6: $\varepsilon_2 = 3/14 = 0.214$, so $\alpha_2 = \tfrac12 \ln(11/3) = \tfrac12 \times 1.299 = 0.650$.

Reweight with the shortcut. The mistakes 4, 5, 6 share $1/2$, so each becomes $1/6$. The correct points held $11/14$ and must now hold $1/2$, so each is scaled by $(1/2)/(11/14) = 7/11$: weights of $1/14$ become $1/22 = 0.045$, and weights of $1/6$ become $7/66 = 0.106$.

Round 3: the best stump cuts at 6.5 with left $-1$ and right $+1$. It misses points 1, 2, 3 (truly $+1$) and point 10 (truly $-1$), each of weight $1/22$: $\varepsilon_3 = 4/22 = 0.182$, so $\alpha_3 = \tfrac12 \ln(18/4) = \tfrac12 \times 1.504 = 0.752$.

Final score $F(x) = 0.424\, h_1(x) + 0.650\, h_2(x) + 0.752\, h_3(x)$. For points 1 to 3: $0.424 + 0.650 - 0.752 = +0.32$. Points 4 to 6: $-0.424 + 0.650 - 0.752 = -0.53$. Points 7 to 9: $-0.424 + 0.650 + 0.752 = +0.98$. Point 10: $-0.424 - 0.650 + 0.752 = -0.32$.

Every sign matches the label, so the training error is 0 after three stumps, while the individual stumps had error rates of 30%, 30% and 40% on the plain data.

</div>

The final classifier is

$$F(x) = \sum_{m=1}^{M} \alpha_m h_m(x), \qquad \hat y = \operatorname{sign}\big(F(x)\big).$$

The size $|F(x)|$ is the ensemble's confidence: points 7 to 9 ($0.98$) are far surer than points 1 to 3 ($0.32$).

In two dimensions each stump is one axis-parallel cut, and their weighted vote is a staircase.

~~~dg-fig 04.04-boundary.svg
AdaBoost with stumps on a round class (dotted circle). One stump makes one cut. Ten stumps already outline a box. Two hundred stumps carve a staircase that fits the training points exactly.
~~~

## The whole algorithm

Freund and Schapire's AdaBoost for labels $y_i \in \{-1, +1\}$:

1. Start with equal weights $w_i = 1/n$.
2. For $m = 1, \dots, M$: fit a weak learner $h_m$ to the weighted data and compute its weighted error $\varepsilon_m = \sum_i w_i\, [h_m(x_i) \ne y_i]$.
3. Give it the vote $\alpha_m = \tfrac12 \ln\big((1-\varepsilon_m)/\varepsilon_m\big)$, positive whenever $\varepsilon_m < 0.5$.
4. Update $w_i \leftarrow w_i \exp(-\alpha_m y_i h_m(x_i))$ and renormalise so the weights sum to 1.
5. Predict with $\operatorname{sign}\big(F(x)\big)$ where $F(x) = \sum_m \alpha_m h_m(x)$.

~~~dg-flow One AdaBoost round. The sample weights carry everything the earlier rounds learned.
Weights w -(fit stump)-> h_m -(weighted error ε)-> Vote α_m -(upweight mistakes)-> *New weights
~~~

## AdaBoost in code

First from scratch, using scikit-learn only for the weighted stump. On the ten points it reproduces the hand calculation exactly.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def adaboost(X, y, M=100):                          # y must be -1 / +1
    w = np.full(len(y), 1 / len(y))                 # equal weights
    stumps, alphas = [], []
    for _ in range(M):
        h = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        pred = h.predict(X)
        eps = np.clip(w[pred != y].sum(), 1e-10, 1 - 1e-10)   # weighted error
        a = 0.5 * np.log((1 - eps) / eps)                     # vote
        w = w * np.exp(-a * y * pred); w /= w.sum()           # reweight
        stumps.append(h); alphas.append(a)
    def F(Z):                                       # the score; sign(F) is the class
        return sum(a * h.predict(Z) for a, h in zip(alphas, stumps))
    return F, alphas

x = np.arange(1, 11).reshape(-1, 1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
F, alphas = adaboost(x, y, M=3)
print(np.round(alphas, 3))           # [0.424 0.65  0.752]
print(np.round(F(x), 2))             # [ 0.32  0.32  0.32 -0.53 -0.53 -0.53  0.98  0.98  0.98 -0.32]
print(np.mean(np.sign(F(x)) != y))   # 0.0
```

The `np.clip` stops a perfect stump from getting $\alpha = \infty$.

In practice use `AdaBoostClassifier`. Its default weak learner is already a stump. `staged_predict` replays the ensemble after 1, 2, 3, ... rounds, which gives you a validation curve from a single fit.

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 5))
y = np.where(X[:, 0]**2 + X[:, 1]**2 > 1.4, 1, -1)   # a circle hidden in 2 of 5 features
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.5, random_state=0)

stump = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(stump, n_estimators=500, learning_rate=0.5, random_state=0)
model.fit(Xtr, ytr)

val_err = [np.mean(p != yva) for p in model.staged_predict(Xva)]
best = int(np.argmin(val_err)) + 1
print(f"1 stump: {val_err[0]:.3f}  50: {val_err[49]:.3f}  500: {val_err[-1]:.3f}")
# 1 stump: 0.379  50: 0.064  500: 0.020
print("best round", best, "val error", round(min(val_err), 3))   # best round 486 val error 0.02
print(model.estimator_weights_[:3].round(3), model.estimator_errors_[:3].round(3))
# [0.277 0.196 0.209] [0.365 0.403 0.397]
```

scikit-learn implements **SAMME**, whose vote is $\ln\frac{1-\varepsilon}{\varepsilon} + \ln(K-1)$ for $K$ classes. With two classes that is twice the classic $\alpha$ for every learner, so predictions are unchanged. `learning_rate` scales every vote: $0.5 \times \ln(0.635/0.365) = 0.277$.

## Reading the result

**Plot error against rounds.** The main knob is the number of rounds $M$, and `staged_predict` gives its curve for free.

~~~dg-fig 04.04-curves.svg
Error against boosting rounds (log scale), 400 training points, a circle hidden in 2 of 5 features. Left, clean labels: training error hits 0 at round 175, yet test error keeps creeping down to 4.2% at round 802. Right, 14% flipped labels: test error is best at round 24 (6.4%) and then roughly doubles.
~~~

- **Training error reaching zero is not a stop signal.** On clean data AdaBoost often keeps improving afterwards (left panel; the last section explains why).
- **A rising validation curve is.** Pick $M$ at its minimum, or use a smaller `learning_rate` with more rounds, which flattens the curve.
- **Watch `estimator_errors_`.** Weighted errors climb towards 0.5 as easy structure is used up (0.365, 0.403, 0.397 above). Stuck at 0.5: nothing new to find. Near 0: the learner is too strong for boosting.
- **Compare with baselines:** one stump (37.9% error above) and a random forest (04.03), which often wins on noisy tabular data.

## What goes wrong

**Label noise is AdaBoost's weak spot.** A mislabelled point sits where the true rule says the opposite, so most stumps get it wrong and its weight is multiplied by $e^{\alpha}$ round after round. Later stumps are then chosen mainly to classify those few points, carving regions that do not exist in new data.

~~~dg-fig 04.04-noise.svg
Same data, 57 of 400 labels flipped (14%). Within about 50 rounds those points hold over a third of all sample weight, and they stay there. After 500 rounds, the ten heaviest points in the training set are all flipped labels.
~~~

| Problem | Symptom | Fix
| --- | --- | --- |
| Noisy or wrong labels | test error has an early minimum then rises; a few points carry huge weights | early stopping, `learning_rate` below 1, clean labels, or switch to gradient boosting with log-loss (04.05)
| Weak learner too strong (deep trees) | weighted error near 0, huge votes, fast overfitting | stumps or depth 2-3 trees
| Weak learner too weak | weighted errors stuck at 0.5, curve flat | slightly deeper trees, better features
| Perfect learner ($\varepsilon = 0$) | division by zero, infinite vote | clip $\varepsilon$, as every library does
| Many classes with the binary rule | requiring error below 0.5 is too strict with $K$ classes | SAMME, which only needs accuracy above $1/K$
| Wrong label coding in your own code | nonsense weights | the formulas need $y \in \{-1, +1\}$, not $\{0, 1\}$

## Going deeper: AdaBoost minimises exponential loss

Friedman, Hastie and Tibshirani (2000) showed it is **forward stagewise additive modelling**: build $F$ as a sum of terms, add one term per round, and never revisit earlier terms. Each added term $\alpha h$ is the one that most reduces the **exponential loss**

$$L(F) = \sum_{i=1}^{n} \exp\big(-y_i F(x_i)\big).$$

The quantity $m_i = y_i F(x_i)$ is the **margin** of point $i$: positive when it is on the correct side, and larger when the ensemble is more confident.

Suppose $F_{m-1}$ is fixed and we add $\alpha h$. Write $w_i \propto \exp(-y_i F_{m-1}(x_i))$. Then

$$L = \sum_i w_i\, e^{-\alpha y_i h(x_i)} = (1-\varepsilon)\, e^{-\alpha} + \varepsilon\, e^{\alpha}$$

(after normalising the weights), where $\varepsilon$ is $h$'s weighted error. For any fixed $\alpha > 0$ this is smallest when $\varepsilon$ is smallest, so the best $h$ is the weighted classifier of step 1. Setting the derivative in $\alpha$ to zero gives $e^{2\alpha} = (1-\varepsilon)/\varepsilon$, which is exactly the vote of step 2. And the weights are nothing but each point's current exponential loss, which is why step 3 multiplies them by $e^{-\alpha y h}$. Lesson 04.05 replaces the exponential loss with any differentiable loss and gets gradient boosting.

~~~dg-fig 04.04-losses.svg
Losses as a function of the margin. All three smooth losses upper-bound the 0-1 loss we actually count. Exponential loss explodes on the wrong side; logistic loss grows only linearly, which is why it tolerates mislabelled points far better.
~~~

**What $F$ estimates.** The exponential loss is minimised, point by point, by $F^*(x) = \tfrac12 \ln\frac{P(y=1\mid x)}{P(y=-1\mid x)}$, half the log-odds. So $p = 1/(1+e^{-2F})$ turns scores into probabilities, but in practice these are poorly calibrated.

**A training-error guarantee.** With the best $\alpha$, the round's normaliser is $Z_m = 2\sqrt{\varepsilon_m(1-\varepsilon_m)}$. Since $[y_i F(x_i) \le 0] \le e^{-y_i F(x_i)}$ for every point, the training error obeys

$$\text{train error} \le \frac{1}{n}\sum_i e^{-y_i F(x_i)} = \prod_{m=1}^{M} 2\sqrt{\varepsilon_m(1-\varepsilon_m)} \le \exp\Big(-2\sum_m \gamma_m^2\Big),$$

where $\gamma_m = 0.5 - \varepsilon_m$ is how much learner $m$ beats a coin flip. If every learner has $\gamma_m \ge \gamma > 0$, training error falls exponentially in $M$. In the hand example: $0.917 \times 0.821 \times 0.771 = 0.580$, and the average of $e^{-y_i F(x_i)}$ over the ten points is also 0.580, exactly as the identity says. The true training error is already 0, so the bound is loose but it falls fast.

**Why test error keeps falling after training error hits zero.** Schapire and colleagues' **margin** explanation: once every point has a positive margin, later rounds still push margins up, especially the smallest ones, and generalisation bounds depend on the margin distribution rather than on $M$. This is the SVM's margin idea (02.05) again. On noisy data the flipped points' margins can never be made positive, and the push to raise them is what overfits.

**Variants.** SAMME handles $K$ classes with $\alpha_m = \ln\frac{1-\varepsilon_m}{\varepsilon_m} + \ln(K-1)$, positive whenever accuracy beats $1/K$. LogitBoost swaps in logistic loss. Scaling every vote by a learning rate $\nu < 1$ ("shrinkage") trades more rounds for better generalisation, as in gradient boosting.

**Cost and a third pattern.** With features pre-sorted, a weighted stump costs $O(nd)$, so $M$ rounds cost $O(Mnd)$; rounds are inherently sequential. Bagging averages with equal weights and boosting learns weights greedily; **stacking** instead trains a second-level model on the out-of-fold predictions of several different base models.

## Key terms

| Term | Meaning
| --- | --- |
| Boosting | Building an ensemble sequentially, each new model focused on the current ensemble's mistakes
| Weak learner | A model only slightly better than chance, such as a stump
| Decision stump | A tree with a single split
| Sample weight | How much a training point counts when fitting the next learner
| Weighted error $\varepsilon$ | Total weight of the points a learner misclassifies
| Vote $\alpha$ | A learner's say in the final sum, $\tfrac12 \ln\frac{1-\varepsilon}{\varepsilon}$
| Margin | $y F(x)$: positive when correct, larger when more confident
| Exponential loss | $e^{-yF(x)}$, the loss AdaBoost minimises stagewise
| SAMME | The multiclass AdaBoost used by scikit-learn

<div class="take">
<div class="lab">Key takeaway</div>

AdaBoost fits weak learners one after another, each on data reweighted so that the previous learner looks like a coin flip, and adds them with votes $\alpha = \tfrac12 \ln\frac{1-\varepsilon}{\varepsilon}$. It is greedy minimisation of exponential loss: very effective on clean data, fragile on noisy labels, so choose the number of rounds on a validation curve.

</div>
