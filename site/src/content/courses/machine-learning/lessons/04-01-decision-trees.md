---
id: "04.01"
course: "machine-learning"
module: "04"
slug: "decision-trees"
title: "Decision trees"
keyIdeas: ["A decision tree partitions feature space into axis-aligned boxes with a sequence of yes/no questions like $x_j \\le \\theta$, and predicts a constant in each box: the majority class or the mean target.", "Splits are chosen greedily: at each node, try every feature and threshold and keep the one that most reduces **impurity**, weighted by the number of samples sent each way.", "Gini impurity $1-\\sum_k p_k^2$ and entropy $-\\sum_k p_k \\log_2 p_k$ both peak at a 50/50 mix and are zero for a pure node; they pick the same split most of the time.", "Regression trees use squared error: a leaf predicts the mean, and the best split maximises the drop in sum of squared errors. Sorting plus prefix sums makes one feature scan $O(n \\log n)$.", "Trees need no scaling, handle mixed feature types and interactions, but are high-variance, axis-bound and cannot extrapolate beyond the target values they saw."]
question: "You have a two-class problem where the label is 1 exactly when $x_1$ and $x_2$ have the same sign, with both features uniform on $[-1, 1]$. Will a CART tree with `min_impurity_decrease=0.01` learn it? What happens without that rule, and why?"
answer: "No. The label is an XOR pattern of the signs. Any single threshold on $x_1$ or on $x_2$ leaves each child with about half positives, so the expected Gini decrease at the root is close to zero (only sampling noise). A rule requiring a decrease of at least 0.01 will usually stop at the root and predict the majority class, giving about 50% accuracy. Without the rule, the tree splits on whichever threshold happens to look best through noise; once it has split near zero on one feature, the next split on the other feature near zero becomes very informative, so a depth-2 tree recovers the pattern. The lesson is that greedy splitting judges each split only by its immediate gain, so early stopping can block splits whose value appears only one level deeper. This is one argument for growing large and pruning afterwards."
sources: [{"title": "scikit-learn User Guide: Decision Trees", "url": "https://scikit-learn.org/stable/modules/tree.html", "note": "Section 1.10.7 gives the exact impurity formulas and complexity; read the tips on practical use."}, {"title": "The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free PDF. Section 9.2 covers CART regression and classification trees and why misclassification error is a poor split criterion."}, {"title": "An Introduction to Statistical Learning", "url": "https://www.statlearning.com/", "note": "Free PDF. Chapter 8.1 is a gentle walk through regression and classification trees with figures."}]
practice: ["decision-tree-classifier-cart", "decision-tree-regressor"]
tags: ["CART", "Gini impurity", "information gain"]
tools: ["NumPy", "scikit-learn"]
---

## What a tree computes

This module covers the most widely used family of models for tabular data: trees and the ensembles built from them. It starts with a single tree, because every later lesson (forests, boosting, importance) reuses the same splitting machinery.

A decision tree is a sequence of questions. Each internal node asks one thing about one feature, such as "is `income` at most 42,000?", and sends the sample left or right. Each leaf holds a constant prediction. Geometrically, the tree carves the feature space into axis-aligned rectangles and paints each one a single value.

~~~dg-flow One root-to-leaf path. Every sample follows exactly one path; the leaf it lands in decides the prediction.
Sample x -(income ≤ 42k?)-> Node: age -(age > 30?)-> *Leaf: p(default) = 0.08
~~~

This structure explains most of a tree's personality. A single feature is compared with a threshold, so monotone transformations of a feature (log, rank, standardisation) do not change the tree at all, and there is no need to scale inputs. Interactions come for free: a split on `age` below a split on `income` only applies to low-income customers. And because each leaf predicts a constant learned from training targets, a regression tree can never predict above the largest or below the smallest target in its training data.

## Impurity: Gini and entropy

To grow a tree you need a score for how mixed a node is. With class proportions $p_k$ in a node, the two standard choices are:

$$G = 1 - \sum_k p_k^2 \qquad H = -\sum_k p_k \log_2 p_k$$

Gini is the probability that two samples drawn at random from the node (with replacement) have different labels. Entropy is the expected number of bits needed to encode a label. Both are zero for a pure node and maximal for a uniform mix: for two classes, Gini peaks at 0.5 and entropy at 1 bit.

A split sends $n_L$ of the node's $n$ samples left and $n_R$ right. Its quality is the impurity decrease

$$\Delta = I(\text{parent}) - \frac{n_L}{n} I(L) - \frac{n_R}{n} I(R)$$

With entropy this decrease is called information gain. In practice the two criteria agree on the chosen split in the large majority of cases; Gini is slightly cheaper (no logarithm) and is the default in scikit-learn. Misclassification error, $1-\max_k p_k$, is a poor splitting criterion because it is not strictly concave: many useful splits leave it unchanged.

<div class="ex">
<div class="lab">Worked example</div>

**Gini vs entropy on one node**

A node holds 10 samples: 6 positive, 4 negative. Gini = 1 − (0.36 + 0.16) = 0.48; entropy = 0.971 bits.

Split A sends 4 positives left (pure) and 2 positive / 4 negative right. Right Gini = 1 − (1/9 + 4/9) = 0.444, weighted 0.6 × 0.444 = 0.267, so the decrease is 0.213. Right entropy = 0.918, weighted 0.551, information gain 0.420.

Split B sends 4+/1− left and 2+/3− right. Weighted Gini = 0.5 × 0.32 + 0.5 × 0.48 = 0.40, decrease only 0.08.

Both criteria prefer A: it creates one pure child, which is exactly what concave impurity rewards.

</div>

## Greedy splitting

Finding the globally optimal tree is NP-hard, so CART (Classification and Regression Trees) grows the tree greedily, top down:

1. At the current node, for each feature, sort the samples by that feature.
2. Evaluate every threshold between consecutive distinct values (midpoints).
3. Keep the (feature, threshold) pair with the largest impurity decrease.
4. Split, then recurse on each child until a stopping rule fires (pure node, too few samples, maximum depth).

Sorting costs $O(n \log n)$ per feature, and with running class counts each candidate threshold is scored in $O(1)$, so one node costs $O(d\, n \log n)$ for $d$ features. Summed over a balanced tree's levels this gives roughly $O(d\, n \log^2 n)$. Libraries such as LightGBM and scikit-learn's `HistGradientBoosting*` replace sorting with a histogram of at most 255 bins per feature, which turns each scan into $O(\text{bins})$ after one pass over the data.

Greedy means myopic. The classic failure is XOR: with two features where the label is $x_1 \oplus x_2$, no single split reduces impurity, so a greedy tree with a minimum-gain rule may refuse to split at all even though a depth-2 tree is perfect.

## Regression trees

For a continuous target, a leaf predicts the mean of its training targets, and impurity becomes the sum of squared errors (SSE). The best split maximises the drop in SSE, which is the same as maximising variance reduction. The prefix-sum trick makes this fast, because $\text{SSE} = \sum y^2 - (\sum y)^2 / n$ for any contiguous block of sorted samples:

```python
import numpy as np

def best_split(x, y):
    # Best threshold on one feature for squared error. O(n log n).
    o = np.argsort(x); x, y = x[o], y[o]
    n = len(y); cs, cs2 = np.cumsum(y), np.cumsum(y * y)
    nl = np.arange(1, n)                            # left size if we cut after i
    sse_l = cs2[:-1] - cs[:-1] ** 2 / nl
    sse_r = (cs2[-1] - cs2[:-1]) - (cs[-1] - cs[:-1]) ** 2 / (n - nl)
    cost = np.where(x[1:] != x[:-1], sse_l + sse_r, np.inf)   # only between distinct values
    i = int(np.argmin(cost))
    return (x[i] + x[i + 1]) / 2, cost[i]
```

Other losses swap the leaf statistic: absolute error uses the median, Poisson deviance a mean on the log scale.

## Strengths and failure modes

| Property | Consequence |
| --- | --- |
| Threshold splits | No feature scaling; robust to monotone transforms and outliers in inputs |
| Axis-aligned boxes | A diagonal boundary like $x_1 > x_2$ needs a staircase of many splits |
| Greedy, data-dependent | Small changes in data can change the root split and the whole tree (high variance) |
| Constant leaves | Piecewise-constant predictions; no extrapolation outside the training target range |
| Deep trees memorise | A fully grown tree reaches zero training error; lesson 04.02 deals with this |

Those last three rows are the reason single trees are rarely the final model. Bagging (lesson 04.03) attacks the variance, and boosting (04.04 and 04.05) builds accurate models from many shallow trees.

<div class="take">
<div class="lab">Key takeaway</div>

a tree is greedy recursive partitioning: at each node, pick the feature and threshold that most reduce weighted impurity (Gini, entropy or SSE), and predict a constant in each leaf.

</div>
