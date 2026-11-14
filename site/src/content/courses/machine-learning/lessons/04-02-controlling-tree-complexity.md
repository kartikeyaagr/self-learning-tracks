---
id: "04.02"
course: "machine-learning"
module: "04"
slug: "controlling-tree-complexity"
title: "Controlling tree complexity"
keyIdeas: ["A tree grown until every leaf is pure has zero training error and very high variance; its complexity must be controlled either while growing (pre-pruning) or afterwards (post-pruning).", "Pre-pruning knobs: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`. `min_samples_leaf` is usually the most robust single knob.", "Cost-complexity pruning minimises $R_\\alpha(T) = R(T) + \\alpha\\,|T|$; weakest-link pruning produces a nested sequence of subtrees, one per range of $\\alpha$, and cross-validation picks $\\alpha$.", "Post-pruning avoids the horizon effect of early stopping: a split with little immediate gain can enable valuable splits below it.", "How much to control depends on the role of the tree: prune single trees, grow forest trees deep, keep boosting trees shallow."]
question: "Two configurations reach the same validation accuracy on a dataset with noisy labels: (a) `max_depth=6` and (b) `min_samples_leaf=25` with unlimited depth. The dataset has a dense region with 90% of samples and a sparse tail. Which would you expect to generalise better to next month's data, and why?"
answer: "Usually (b). `max_depth=6` caps every path at six questions regardless of how many samples a region holds. In the dense region that may be too coarse, and in the sparse tail it still allows leaves with just one to three samples, which copy label noise. `min_samples_leaf=25` ties capacity to the data: the dense region can be split finely because each leaf still has 25 or more samples, while the sparse tail stays coarse because no leaf may drop below 25 samples. Every leaf prediction is therefore an average over at least 25 labels, which bounds its variance. With noisy labels that is the property that matters for new data. Equal validation accuracy can hide this, so I would also compare the scores' variance across CV folds or bootstrap resamples, where (a) should look less stable."
sources: [{"title": "scikit-learn example: Post pruning decision trees with cost complexity pruning", "url": "https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html", "note": "Shows the pruning path, how leaves and depth shrink with alpha, and train vs test accuracy."}, {"title": "scikit-learn User Guide: Decision Trees", "url": "https://scikit-learn.org/stable/modules/tree.html", "note": "Section on minimal cost-complexity pruning gives the exact definition of the effective alpha."}, {"title": "The Elements of Statistical Learning", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Section 9.2.2 derives weakest-link pruning and cross-validating the penalty."}]
practice: ["decision-tree-classifier-cart", "decision-tree-regressor", "k-fold-cross-validation"]
tags: ["pruning", "overfitting"]
tools: ["scikit-learn"]
---

## Why a full tree overfits

Lesson 04.01 grew a tree until a stopping rule fired. Without any rule, CART keeps splitting until every leaf is pure or holds identical feature values. With $n$ distinct training points that can mean close to $n$ leaves, each predicting the label of a single sample. Training error is zero, and the model is a lookup table with rectangular cells.

In bias-variance terms (module 01, lesson 02), a deep tree has very low bias and very high variance. Leaves with one or two samples copy label noise directly into predictions, and the greedy root split can change if a handful of samples move. The cure is to limit the number of leaves, which is the tree's measure of capacity. There are two ways to do that: stop early, or grow fully and cut back.

## Pre-pruning: stopping rules

Stopping rules are cheap because they prevent the work from being done at all. The common ones in scikit-learn:

| Parameter | Effect | Typical range |
| --- | --- | --- |
| `max_depth` | Caps the number of questions on any path; at most $2^{\text{depth}}$ leaves | 3-12 for single trees |
| `min_samples_split` | A node needs this many samples to be considered for a split | 2-50 |
| `min_samples_leaf` | Every child must keep at least this many samples | 1-100, or a fraction like 0.01 |
| `max_leaf_nodes` | Grows best-first and stops at this many leaves | 8-256 |
| `min_impurity_decrease` | A split must reduce weighted impurity by at least this | 0-0.01 |

`min_samples_leaf` is usually the most useful single knob. It directly limits how much a leaf can rely on a few noisy labels, it smooths regression trees (a leaf mean over 20 samples is far less noisy than over 1), and unlike `max_depth` it adapts to the data: dense regions can still be split finely while sparse regions stay coarse.

The weakness of every stopping rule is the horizon effect. A rule judges a split by its immediate gain, but some splits are only valuable because of what they enable below. The XOR example from lesson 04.01 is the extreme case. Post-pruning sidesteps this by looking at whole subtrees.

## Cost-complexity pruning

Breiman's CART book proposed growing a large tree $T_0$ and then pruning it with a penalty on size. For a subtree $T$ with $|T|$ leaves and training error (or impurity) $R(T)$, define

$$R_\alpha(T) = R(T) + \alpha\,|T|$$

For $\alpha = 0$ the best subtree is $T_0$ itself; as $\alpha$ grows, smaller trees win; for large enough $\alpha$ only the root survives. The key result is that the optimal subtrees for increasing $\alpha$ are nested, and they can be found by weakest-link pruning. For each internal node $t$ with subtree $T_t$, compute the penalty at which collapsing $t$ into a leaf breaks even:

$$g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$$

Here $R(t)$ is the error if $t$ were a leaf, and $|T_t| - 1$ is the number of leaves saved. Collapse the node with the smallest $g(t)$, recompute, and repeat until only the root is left. Each collapse gives the next tree in the sequence and the $\alpha$ at which it becomes optimal.

~~~dg-flow Cost-complexity pruning: grow big, generate a nested sequence of subtrees, and let validation pick one.
Grow full tree T0 -(weakest link)-> Subtrees T1 ⊃ T2 ⊃ … -(CV score per α)-> *Pick α -(refit)-> Final tree
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one weakest-link step**

Errors are weighted by the share of all training samples. Node A has a 3-leaf subtree with R(T_A) = 0.05; as a single leaf it would have R(A) = 0.11. So g(A) = (0.11 − 0.05) / (3 − 1) = 0.03.

Node B has a 2-leaf subtree with R(T_B) = 0.02 and R(B) = 0.04, so g(B) = 0.02 / 1 = 0.02.

B is the weakest link: it buys the least error reduction per extra leaf. It is collapsed first, at α = 0.02. A survives until α reaches 0.03 (after recomputing, since collapsing B can change other values).

</div>

## Doing it in practice

scikit-learn exposes the full pruning path and a `ccp_alpha` parameter. A standard recipe:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas[:-1]                     # last alpha prunes to the root
scores = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=0),
                          X_tr, y_tr, cv=5).mean() for a in alphas]
best = alphas[int(np.argmax(scores))]
```

A common refinement is the one-standard-error rule: pick the largest $\alpha$ (smallest tree) whose CV score is within one standard error of the best. It trades a statistically invisible amount of accuracy for a simpler, more stable tree. Cross-validation strategies themselves are covered in module 06.

Two cautions. First, `ccp_alpha` is on the scale of impurity, not of accuracy, so values like 0.001-0.02 are typical and a grid must be data-driven (the pruning path gives it). Second, pruning a tree that is going to be bagged or boosted is usually wasted effort, as the next section explains.

## Complexity depends on the job

The right amount of control depends on what the tree is for:

- **Single interpretable tree**: prune hard; a tree with 8-20 leaves can be read and checked by a domain expert.
- **Random forest member**: grow deep (default `max_depth=None`), maybe with a small `min_samples_leaf`. Averaging (lesson 04.03) removes variance, so you want each tree to have low bias.
- **Boosting member**: keep shallow (depth 3-8, or 15-63 leaves). Each tree only needs to fit a small correction, and deep trees make boosting overfit quickly (lesson 04.05).

<div class="take">
<div class="lab">Key takeaway</div>

control tree size with `min_samples_leaf`-style stopping rules or, better for a single tree, grow fully and pick a cost-complexity $\alpha$ by cross-validation; ensembles change the target, with deep trees for bagging and shallow ones for boosting.

</div>
