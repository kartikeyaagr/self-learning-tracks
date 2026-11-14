---
id: "02.04"
course: "machine-learning"
module: "02"
slug: "multiclass-classification"
title: "Multiclass classification"
keyIdeas: ["Two routes to $K$ classes: **reduce** to binary problems (one-vs-rest with $K$ models, one-vs-one with $K(K-1)/2$), or use a natively multiclass model such as softmax regression.", "One-vs-rest scores come from separately trained models, so they are not on a common scale and do not sum to 1; each subproblem is also imbalanced ($1$ class against $K-1$).", "Softmax regression sets $p(y = k \\mid x) = e^{z_k}/\\sum_j e^{z_j}$ with $z = Wx + b$; the cross-entropy gradient on the logits is just $p - \\text{onehot}(y)$.", "Softmax is over-parameterised (adding a vector to every row of $W$ changes nothing); with $K = 2$ it reduces exactly to logistic regression on $w_1 - w_0$.", "If labels are not mutually exclusive (multi-label), use $K$ independent sigmoids, not a softmax."]
question: "An OvR logistic model on 4 balanced classes outputs per-class sigmoid probabilities (0.30, 0.25, 0.20, 0.10) for a test point. What does this pattern tell you, how would a softmax model trained on the same data present the same point, and which output would you rather feed into a downstream cost-sensitive decision?"
answer: "The four binary models each say this point is more likely not their class, and their probabilities sum to only 0.85. The point sits in a region none of them claims, near the shared boundaries or away from the training data. The argmax still predicts class 1, but by a narrow margin, and the numbers are not a distribution: rescaling them to sum to 1 (0.35, 0.29, 0.24, 0.12) is an ad hoc fix with no likelihood behind it. A softmax model is trained so its outputs form a distribution over the four classes. It would likely give a similarly flat but properly normalised vector, and because it was fitted with multiclass cross-entropy, its probabilities are usually better calibrated. For a cost-sensitive decision, which needs expected costs $\\sum_k p_k c_k$, use the softmax output (after a calibration check as in 03.04), and treat the flat profile as a signal to abstain or route for review."
sources: [{"title": "In Defense of One-Vs-All Classification", "url": "https://www.jmlr.org/papers/v5/rifkin04a.html", "note": "Rifkin and Klautau (JMLR 2004): well-tuned one-vs-rest matches more complex multiclass schemes."}, {"title": "Dive into Deep Learning: Softmax Regression", "url": "https://d2l.ai/chapter_linear-classification/softmax-regression.html", "note": "Softmax, cross-entropy and its gradient, with the information-theory view of the loss."}, {"title": "scikit-learn: Multiclass and multioutput algorithms", "url": "https://scikit-learn.org/stable/modules/multiclass.html", "note": "OneVsRest, OneVsOne and which estimators are inherently multiclass, plus multi-label targets."}]
practice: ["softmax-regression", "softmax-cross-entropy-gradient", "cross-entropy-logits", "softmax-numerical-stability-and-log-sum-exp", "top-k-classification-accuracy"]
tags: ["softmax", "one-vs-rest"]
tools: ["NumPy", "scikit-learn"]
---

## Two routes to K classes

Logistic regression and SVMs are binary. With $K > 2$ classes there are two strategies:

- **Reduction**: train several binary classifiers and combine their outputs. Works with any binary learner, including ones that have no natural multiclass form.
- **Native multiclass**: change the model so it scores all $K$ classes jointly. For linear models that is softmax (multinomial logistic) regression.

scikit-learn does this silently: `LogisticRegression` uses the multinomial form by default with most solvers, `SVC` uses one-vs-one internally, and `LinearSVC` uses one-vs-rest. Knowing which one you got matters when you read the scores.

## One-vs-rest and one-vs-one

**One-vs-rest (OvR)** trains classifier $k$ to separate class $k$ from everything else, then predicts $\arg\max_k s_k(x)$. It needs $K$ models, each trained on all $n$ examples. Three weaknesses:

- The scores come from independently trained models, so a score of 2.0 from model 3 and 1.5 from model 5 are not directly comparable, and OvR probabilities do not sum to 1.
- Each subproblem is imbalanced: with 10 balanced classes, every binary problem is 10% positive.
- Regions where every classifier says "not mine", or several say "mine", are resolved only by the argmax.

Rifkin and Klautau showed that, with each binary model well tuned and regularised, OvR is as accurate as more elaborate schemes. It is a fine default when the binary learner is strong.

**One-vs-one (OvO)** trains a classifier for every pair of classes on just those two classes' data and predicts by majority vote. For $K = 10$ that is 45 models instead of 10, but each sees only about a fifth of the data. For kernel SVMs, whose training cost grows faster than linearly in $n$, many small problems can be cheaper than a few big ones, which is why libsvm uses OvO.

| Property | One-vs-rest | One-vs-one | Softmax
| --- | --- | --- | --- |
| Models trained | K | K(K-1)/2 | 1 (K weight vectors)
| Data per model | all n | two classes only | all n
| Scores sum to 1 | no | no (votes) | yes
| Typical use | linear SVMs, any binary learner | kernel SVMs | logistic-style models, neural nets

## Softmax regression

Give each class a weight vector, stacked as $W \in \mathbb{R}^{K \times d}$, compute logits $z = Wx + b$, and normalise:

$$p(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

The loss is cross-entropy, the negative log-probability of the true class:

$$\ell = -\log p_y = -z_y + \log\sum_j e^{z_j}.$$

Its gradient with respect to the logits is $p - e_y$, where $e_y$ is the one-hot label: push the true class's logit up by $1 - p_y$ and every other logit down by its probability. The weight gradient is that vector times $x^\top$. The loss is convex in $W$, so the fitting story from 02.03 carries over: gradient methods or L-BFGS, with L2 regularisation.

```python
import numpy as np
def softmax_xent(Z, y):                 # Z: (n, K) logits, y: (n,) int labels
    Z = Z - Z.max(axis=1, keepdims=True)              # stability: exp never overflows
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    G = np.exp(logp); G[np.arange(len(y)), y] -= 1    # dL/dZ = p - onehot
    return loss, G / len(y)
```

Subtracting the row maximum leaves the probabilities unchanged (softmax ignores a constant added to every logit) and keeps `exp` from overflowing; computing log-probabilities directly, the log-sum-exp trick, avoids `log(0)`. The AI Research course (lesson 02.02) covers softmax intuition and temperature in more depth.

That invariance has a second consequence: adding the same vector $c$ to every row of $W$ leaves predictions unchanged, so the unregularised model has no unique solution. Any L2 penalty pins it down. With $K = 2$, only the difference matters: $p(y = 1) = \sigma\big((w_1 - w_0)^\top x\big)$, which is exactly logistic regression.

~~~dg-flow Softmax regression: one score per class, normalised jointly, so the outputs compete and sum to 1.
Features x -(W x + b)-> K logits -(softmax)-> *K probabilities -(argmax)-> Predicted class
~~~

**Decision regions.** Class $k$ wins where $z_k \ge z_j$ for all $j$, an intersection of $K - 1$ half-spaces. So the regions are convex polytopes and boundaries are piecewise linear, the same for OvR and softmax, though the exact hyperplanes differ.

## Multi-label is a different problem

If an example can carry several labels at once (a photo tagged both "beach" and "dog"), softmax is wrong: it forces the probabilities to compete and sum to 1. Use $K$ independent sigmoid outputs with binary cross-entropy on each, which is OvR with an honest name, and pick a threshold per label. For very large $K$ (thousands of products or words), the full softmax becomes the computational bottleneck; sampled and hierarchical softmax are covered in the Deep Learning and Recommender Systems material.

<div class="ex">
<div class="lab">Worked example</div>

**softmax and its gradient for three classes**

Logits $z = (2.0, 1.0, 0.1)$. Exponentials $7.389, 2.718, 1.105$, sum $11.212$, so $p = (0.659, 0.242, 0.099)$.

If the true class is the second one, the loss is $-\ln 0.242 = 1.42$ and the gradient on the logits is $p - e_2 = (0.659, -0.758, 0.099)$: lower the first and third logits, raise the second.

Adding 100 to every logit gives the same $p$; computing $e^{102}$ directly would still work in float64 but overflows float32 near $e^{89}$, which is why the max is subtracted first.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

reduce to binary problems when your learner is binary, but prefer softmax regression when you want comparable scores and probabilities that sum to 1. Use independent sigmoids when labels are not exclusive.

</div>
