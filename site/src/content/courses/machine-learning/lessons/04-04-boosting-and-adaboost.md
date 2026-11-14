---
id: "04.04"
course: "machine-learning"
module: "04"
slug: "boosting-and-adaboost"
title: "Boosting and AdaBoost"
keyIdeas: ["Bagging trains strong learners in parallel to reduce variance; boosting trains weak learners in sequence, each focused on what the ensemble so far gets wrong, to reduce bias.", "AdaBoost keeps a weight per sample. After each weak learner with weighted error $\\varepsilon_m$, it gives the learner a vote $\\alpha_m = \\tfrac12 \\ln\\frac{1-\\varepsilon_m}{\\varepsilon_m}$ and multiplies the weights of misclassified points by $e^{\\alpha_m}$.", "After reweighting, the last learner has weighted error exactly 0.5, so the next learner is forced to find something new.", "AdaBoost is forward stagewise additive modelling with exponential loss $e^{-yF(x)}$, whose minimiser is half the log-odds. That view generalises to gradient boosting.", "Exponential loss punishes confident mistakes exponentially, so AdaBoost is fragile under label noise."]
question: "Your AdaBoost model's training error reaches 0 after 50 rounds. On a dataset you know contains about 5% flipped labels, test error starts rising after round 200. Explain the mechanism, and give two changes that would help."
answer: "Each flipped label sits in a region where the correct rule says the opposite, so most stumps misclassify it. Under exponential loss its weight is multiplied by $e^{\\alpha_m}$ almost every round, so after many rounds a few noisy points hold a large share of the total weight. Later stumps are then chosen mainly to classify those points, carving small regions around them that do not generalise, and test error rises. Two fixes: (1) switch to a loss that grows only linearly with negative margin, such as log-loss gradient boosting or LogitBoost, so a persistent mistake gains weight far more slowly; (2) regularise the existing model by using a learning rate below 1 together with early stopping on a validation set, which stops before the noise-chasing phase. Cleaning the labels or capping sample weights also helps."
sources: [{"title": "Freund and Schapire (1997): A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting", "url": "https://doi.org/10.1006/jcss.1997.1504", "note": "The original AdaBoost paper and its training-error bound."}, {"title": "Friedman, Hastie, Tibshirani (2000): Additive Logistic Regression", "url": "https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf", "note": "Shows AdaBoost is stagewise fitting of exponential loss; introduces LogitBoost."}, {"title": "Schapire: Explaining AdaBoost", "url": "https://rob.schapire.net/papers/explaining-adaboost.pdf", "note": "Short survey of the different explanations, including margins and the noise problem."}, {"title": "Zhu, Zou, Rosset, Hastie: Multi-class AdaBoost (SAMME)", "url": "https://hastie.su.domains/Papers/samme.pdf", "note": "The multiclass variant used by scikit-learn."}]
practice: ["adaboost", "stacking-ensemble", "bagging-classifier"]
tags: ["boosting", "AdaBoost", "exponential loss"]
tools: ["NumPy", "scikit-learn"]
---

## From averaging to correcting

Bagging (lesson 04.03) builds many strong, independent models and averages away their variance. Boosting takes the opposite route: it builds many weak models in sequence, each one trained to fix the mistakes of the ensemble so far. A weak learner is anything slightly better than chance, and the canonical one is a decision stump, a tree with a single split. One stump is a terrible classifier; a weighted vote of a few hundred well-chosen stumps can be excellent. Boosting reduces bias first, and because the learners are dependent, it can also overfit if run too long, unlike bagging.

| | Bagging / random forest | Boosting |
| --- | --- | --- |
| Base learner | Deep, low-bias tree | Shallow, high-bias tree or stump |
| Training | Parallel, independent | Sequential, each depends on the last |
| Main effect | Reduces variance | Reduces bias (and some variance) |
| Too many rounds | Harmless, just slower | Can overfit; needs a stopping rule |
| Label noise | Robust | AdaBoost in particular is sensitive |

## The AdaBoost algorithm

Freund and Schapire's AdaBoost (1997) for labels $y_i \in \{-1, +1\}$:

1. Start with uniform weights $w_i = 1/n$.
2. For $m = 1, \dots, M$: fit a weak learner $h_m$ to the weighted data, and compute its weighted error $\varepsilon_m = \sum_i w_i\, [h_m(x_i) \ne y_i]$.
3. Give it a vote $\alpha_m = \tfrac12 \ln\big((1-\varepsilon_m)/\varepsilon_m\big)$, which is positive whenever $\varepsilon_m < 0.5$ and grows without bound as $\varepsilon_m \to 0$.
4. Update $w_i \leftarrow w_i \exp(-\alpha_m y_i h_m(x_i))$, then renormalise so the weights sum to 1. Misclassified points are multiplied by $e^{\alpha_m}$, correct ones by $e^{-\alpha_m}$.
5. Predict with $\operatorname{sign}\big(F(x)\big)$ where $F(x) = \sum_m \alpha_m h_m(x)$.

~~~dg-flow One AdaBoost round. The sample weights carry all the information from previous rounds.
Weights w -(fit stump)-> h_m -(weighted error ε)-> Vote α_m -(upweight mistakes)-> *New weights
~~~

If the weak learner cannot take weights directly, resample the training set in proportion to $w$ instead.

<div class="ex">
<div class="lab">Worked example</div>

**one round in numbers**

A stump has weighted error ε = 0.3. Its vote is α = ½ ln(0.7 / 0.3) = ½ ln 2.333 = 0.424.

Misclassified weights are multiplied by e^0.424 = 1.528 and correct ones by e^−0.424 = 0.655. Total misclassified weight becomes 0.3 × 1.528 = 0.458, total correct weight 0.7 × 0.655 = 0.458.

After normalisation each group holds exactly half the weight. The stump just fitted now has weighted error 0.5 on the new weights, so it is useless for the next round, which must find a different split.

</div>

That equalisation is the mechanism in one line: AdaBoost always moves the weights so that the newest learner looks like a coin flip.

## Why it works: exponential loss

Friedman, Hastie and Tibshirani (2000) showed that AdaBoost is forward stagewise additive modelling. At each round, it adds the term $\alpha h$ that most reduces the exponential loss

$$L(F) = \sum_i \exp\big(-y_i F(x_i)\big)$$

holding earlier terms fixed. Writing $w_i \propto \exp(-y_i F_{m-1}(x_i))$, minimising over $h$ is exactly fitting the weighted classifier, and minimising over $\alpha$ gives the formula above. The weights are just the current loss of each point.

Two consequences follow. First, the population minimiser of exponential loss is $F^*(x) = \tfrac12 \ln\frac{P(y=1\mid x)}{P(y=-1\mid x)}$, half the log-odds, so $F$ can be turned into probabilities with $p = 1/(1+e^{-2F})$, although those probabilities are often poorly calibrated. Second, the training error is bounded by $\prod_m 2\sqrt{\varepsilon_m(1-\varepsilon_m)}$, which falls exponentially fast as long as every learner beats chance by some margin.

Practitioners noticed that AdaBoost's test error often kept falling after training error hit zero. The margin explanation (Schapire et al.) is that later rounds keep increasing $y_i F(x_i)$, the confidence of correct predictions, and larger margins give better generalisation bounds. It does eventually overfit on noisy data.

## Failure modes and variants

- **Label noise**: a mislabelled point is misclassified round after round, its weight grows exponentially, and later learners contort themselves around it. Losses that grow linearly in the margin (log-loss, used by gradient boosting) are much more forgiving.
- **Weak learner too weak or too strong**: if $\varepsilon_m \ge 0.5$, boosting stalls; if a learner reaches $\varepsilon_m = 0$, $\alpha_m$ is infinite, so implementations clip it. Deep trees overfit the weighted data and make boosting unstable.
- **Multiclass**: requiring error below 0.5 is too strict with $K$ classes. SAMME uses $\alpha_m = \ln\frac{1-\varepsilon_m}{\varepsilon_m} + \ln(K-1)$, which only needs accuracy above $1/K$; this is what scikit-learn's `AdaBoostClassifier` implements.
- **Learning rate**: scaling every $\alpha_m$ by $\nu < 1$ and using more rounds usually generalises better, as with gradient boosting.

```python
import numpy as np
def adaboost(X, y, fit_stump, M=200):          # y in {-1, +1}
    w = np.full(len(y), 1 / len(y)); model = []
    for _ in range(M):
        h = fit_stump(X, y, w); pred = h(X)
        eps = np.clip(w[pred != y].sum(), 1e-10, 1 - 1e-10)
        a = 0.5 * np.log((1 - eps) / eps)
        w *= np.exp(-a * y * pred); w /= w.sum()
        model.append((a, h))
    return lambda Z: np.sign(sum(a * h(Z) for a, h in model))
```

Stacking is a third ensemble pattern worth knowing: instead of fixed weights, a second-level model learns how to combine base models from their out-of-fold predictions. Lesson 04.05 generalises AdaBoost's idea to any differentiable loss.

<div class="take">
<div class="lab">Key takeaway</div>

AdaBoost fits weak learners in sequence, each on data reweighted towards current mistakes; it is greedy minimisation of exponential loss, which makes it powerful on clean data and fragile on noisy labels.

</div>
