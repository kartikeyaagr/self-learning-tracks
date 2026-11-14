---
id: "04.02"
course: "machine-learning"
module: "04"
slug: "controlling-tree-complexity"
title: "Controlling tree complexity"
keyIdeas: ["A tree grown until every leaf is pure has zero training error and very high variance: its leaves copy label noise and its greedy splits are unstable. Its size (number of leaves) must be controlled, either while growing (**pre-pruning**) or afterwards (**post-pruning**).", "Pre-pruning knobs are `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes` and `min_impurity_decrease`. `min_samples_leaf` is usually the most robust single knob, because every prediction is then an average over enough samples and dense regions can still be split finely.", "Stopping rules suffer from the **horizon effect**: a split with little immediate gain (as in XOR) can enable valuable splits below it. Post-pruning judges whole subtrees and avoids this.", "Cost-complexity pruning minimises $R_\\alpha(T) = R(T) + \\alpha\\,|T|$. Weakest-link pruning repeatedly collapses the node with the smallest $g(t) = \\frac{R(t) - R(T_t)}{|T_t| - 1}$, giving a nested sequence of subtrees; cross-validation (often with the one-standard-error rule) picks $\\alpha$.", "How much to control depends on the tree's role: prune single trees hard, grow random-forest trees deep, keep boosting trees shallow."]
question: "Two configurations reach the same validation accuracy on a dataset with noisy labels: (a) `max_depth=6` and (b) `min_samples_leaf=25` with unlimited depth. The dataset has a dense region with 90% of samples and a sparse tail. Which would you expect to generalise better to next month's data, and why? How would you check, and what would you try if you wanted a single tree that a domain expert could read?"
answer: "Usually (b). `max_depth=6` caps every path at six questions regardless of how many samples a region holds. In the dense region that may be too coarse, and in the sparse tail it still allows leaves with just one to three samples, which copy label noise into their predictions. `min_samples_leaf=25` ties capacity to the data: the dense region can be split finely because each fine leaf still has 25 or more samples, while the sparse tail stays coarse because no leaf may drop below 25. Every leaf prediction is therefore an average over at least 25 labels, which bounds its variance (the noise in a mean of 25 values is about a fifth of the noise in a single value). With noisy labels that is the property that matters for new data.\n\nEqual mean validation accuracy can hide this, so I would check (1) the leaf sizes of each fitted tree (`tree_.n_node_samples` at the leaves), where (a) should show tiny leaves in the tail; (2) the spread of scores across CV folds or bootstrap resamples, where (a) should be less stable; and (3) accuracy on the sparse tail separately, since that is where the two models differ most. If month-to-month drift is a concern, validating on a later time slice is better than a random split.\n\nFor a readable single tree I would grow the tree fully (perhaps with a small `min_samples_leaf`), compute `cost_complexity_pruning_path`, score each `ccp_alpha` with cross-validation, and apply the one-standard-error rule to pick the smallest tree whose score is within one standard error of the best. That typically gives a tree of roughly 5-20 leaves that the expert can check split by split, without giving up measurable accuracy."
sources: [{"title": "scikit-learn example: Post pruning decision trees with cost complexity pruning", "url": "https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html", "note": "Shows the pruning path, how leaves and depth shrink with alpha, and train vs test accuracy."}, {"title": "scikit-learn User Guide: Decision Trees", "url": "https://scikit-learn.org/stable/modules/tree.html", "note": "Section on minimal cost-complexity pruning gives the exact definition of the effective alpha."}, {"title": "The Elements of Statistical Learning", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Section 9.2.2 derives weakest-link pruning and cross-validating the penalty."}]
practice: ["decision-tree-classifier-cart", "decision-tree-regressor", "k-fold-cross-validation"]
tags: ["pruning", "overfitting"]
tools: ["scikit-learn"]
---

## What you will learn

- Why a fully grown tree memorises its training data, and how to see it on a training/test accuracy plot.
- The five stopping rules (pre-pruning) in scikit-learn and why `min_samples_leaf` is usually the safest.
- The horizon effect: why stopping early can miss splits that pay off one level down.
- Cost-complexity pruning: $R(T) + \alpha|T|$, weakest-link pruning by hand, and choosing $\alpha$ by cross-validation.
- How much to restrict a tree for its job: readable single tree, forest member or boosting member.

*Before you start:* you need lesson 04.01 (how CART grows a tree, Gini impurity and impurity decrease) and the bias-variance trade-off from module 01, lesson 02. Cross-validation is used as a tool here and explained properly in module 06.

## From growing to stopping

Lesson 04.01 showed how a tree picks its splits greedily. It said little about when to stop, which is the most important decision for a single tree: the same algorithm can give a sensible model with 6 leaves or a useless one with 600.

## The idea in one picture

Take 300 training points from two interleaved half-moons (two classes, two features) and flip 10% of the labels at random, as real labels contain mistakes. Grow trees of increasing depth and score each on 2,000 fresh points:

| max_depth | 1 | 2 | 6 | 9 | 13 (no limit)
| --- | --- | --- | --- | --- | --- |
| Leaves | 2 | 4 | 25 | 47 | 64
| Training accuracy | 0.72 | 0.81 | 0.89 | 0.96 | 1.00
| Test accuracy | 0.71 | 0.79 | 0.80 | 0.75 | 0.74

~~~dg-fig 04.02-overfit.svg
Training accuracy climbs all the way to 100% as the tree deepens, but accuracy on new data peaks around depth 6 and then falls. The gap between the two curves is overfitting.
~~~

Each extra level carves the plane into smaller boxes. At first that helps both curves. After about depth 6 the new boxes are spent on isolated points whose label was flipped: training accuracy keeps rising, but new points landing in those little boxes are mostly of the other class, so test accuracy falls.

Vocabulary for the rest of the lesson:

- **Capacity** is how flexible a model is. For a tree, it is the number of leaves $|T|$: each leaf is one box with its own constant prediction.
- **Overfitting** is spending capacity on noise: training error falls while error on new data rises.
- **Pre-pruning** (early stopping) refuses some splits while the tree grows. **Post-pruning** grows the tree fully, then cuts branches back.

## Step 1: why a full tree overfits

Without a stopping rule, CART splits until every leaf is pure or holds identical feature values: with $n$ distinct points, close to $n$ leaves. Training error is zero and the model is a lookup table with rectangular cells. In bias-variance terms (module 01, lesson 02) it has very low bias and very high variance, for two reasons:

- **Leaves copy label noise.** A one-sample leaf predicts that sample's label, right or wrong, with no averaging.
- **Greedy splits are unstable.** Move a handful of samples and a different root split can win, changing every split below it.

~~~dg-fig 04.02-regions.svg
Left: the unrestricted tree spends many of its 64 leaves on thin slivers and islands around single mislabelled points. Middle and right: requiring 15 samples per leaf, or pruning with a small penalty, leaves 7 to 14 large boxes that follow the true moon shapes and score better on new data.
~~~

The cure is to limit the number of leaves. The figure shows both families of cure, a stopping rule (middle) and post-pruning (right); both beat the full tree on new data (0.81 and 0.82 against 0.74) with a fraction of the leaves.

## Step 2: pre-pruning with stopping rules

Stopping rules are cheap: each is a test applied at a node before it is split, and if the test fails the node becomes a leaf. The rules in scikit-learn:

| Parameter | What it limits | Typical range
| --- | --- | --- |
| `max_depth` | The number of questions on any root-to-leaf path; at most $2^{\text{depth}}$ leaves | 3-12 for a single tree
| `min_samples_split` | A node needs at least this many samples to be considered for a split | 2-50
| `min_samples_leaf` | Every child created by a split must keep at least this many samples | 1-100, or a fraction such as 0.01
| `max_leaf_nodes` | Grows the tree best-first and stops at this many leaves | 8-256
| `min_impurity_decrease` | A split must reduce weighted impurity by at least this much | 0-0.01

`max_leaf_nodes` changes the order of growth: instead of depth-first, the tree always makes the split with the largest impurity decrease anywhere in the tree next, until the leaf budget is spent. `min_impurity_decrease` compares a threshold with the split's decrease weighted by how much of the data reaches the node:

$$\frac{n_t}{N}\Big(I(t) - \frac{n_L}{n_t} I(L) - \frac{n_R}{n_t} I(R)\Big),$$

where $N$ is the number of training samples, $n_t$, $n_L$, $n_R$ the numbers reaching the node and its left and right child, and $I$ the impurity (Gini by default). Deep splits on few samples must be very effective to pass. Number check: a node holding 10% of the data with Gini 0.48 whose children have weighted Gini 0.40 scores $0.1 \times 0.08 = 0.008$, which passes a threshold of 0.005 but not 0.01.

**Why `min_samples_leaf` is usually the best single knob.** It acts directly on the cause of variance: a leaf's prediction is an average over its samples, so a leaf of 20 is far less noisy than a leaf of 1. It also adapts to the data: a dense region can still be split finely, while a sparse region stays coarse. `max_depth` gives every region the same number of questions, which can be too coarse where data is plentiful and still leave one-sample leaves in a sparse tail.

~~~dg-fig 04.02-leafsize.svg
A regression tree on 120 noisy points. With one sample per leaf (left) the prediction jumps to every point, noise included. With at least 20 per leaf (right) each step is the mean of 20 or more targets, and the steps trace the true curve. Since the noise in a mean of $m$ values shrinks like $\sigma/\sqrt{m}$, going from 1 to 20 samples per leaf cuts it by about $\sqrt{20} \approx 4.5$.
~~~

## Step 3: the horizon effect

Every stopping rule judges a split by its immediate gain. Some splits are worth little alone and only pay off through the splits they enable below; a rule that looks one step ahead cannot see past that horizon. XOR (lesson 04.01) is the extreme case: any single cut leaves both sides half and half, so its gain is zero, yet two levels of cuts separate the classes perfectly.

~~~dg-fig 04.02-xor.svg
Balanced XOR data. A stopping rule that demands a minimum gain refuses the first split and stops at a single leaf (50% accuracy). Growing the tree fully and then pruning keeps the splits, because together they remove almost all the error.
~~~

Post-pruning avoids the problem because it judges whole subtrees after they exist: a branch whose first split gains nothing but whose lower splits remove all the error is kept. (The pruned tree has 7 leaves, not the ideal 4: with zero gain everywhere the greedy root split lands at an arbitrary threshold, which pruning cannot repair.)

## Step 4: the cost-complexity criterion

The CART book (Breiman et al.) grows a large tree $T_0$, then finds the subtree that best balances fit against size. For any subtree $T$ obtained by collapsing internal nodes of $T_0$, define

$$R_\alpha(T) = R(T) + \alpha\,|T|.$$

- $R(T)$ is training error: misclassification rate in the hand example below; in scikit-learn, the leaves' impurities weighted by their share of samples.
- $|T|$ is the number of leaves, and $\alpha \ge 0$ is the price charged per leaf.

At $\alpha = 0$ the best subtree is $T_0$ itself. As $\alpha$ rises, a leaf must reduce the error by more than $\alpha$ to pay for itself, so smaller trees win; for large enough $\alpha$ only the root survives.

~~~dg-fig 04.02-ralpha.svg
Each candidate subtree's penalised cost is a straight line in α: it starts at its training error and rises with slope equal to its number of leaves. The best tree is whichever line is lowest, and it switches at α = 0.03, 0.05, 0.07 and 0.21.
~~~

Big trees start low and climb steeply; small trees start high and climb slowly. The lowest line switches at a handful of break points, so there are only a few distinct "best trees", not one per value of $\alpha$.

## Step 5: weakest-link pruning

Breiman's key result is that these best trees are **nested**: each comes from the previous one by collapsing more nodes, so there is no need to search all subtrees. For each internal node $t$, ask at what penalty collapsing $t$ into a leaf breaks even:

$$g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$$

- $R(t)$ is the error if node $t$ were a leaf (predicting its majority class).
- $T_t$ is the branch hanging below $t$, $R(T_t)$ its error and $|T_t|$ its number of leaves.
- The numerator is the error the branch removes; the denominator is the number of leaves you save by collapsing it.

So $g(t)$ is "error removed per extra leaf". The node with the smallest $g(t)$ is the **weakest link**. Collapse it, recompute $g$ for its ancestors, and repeat until only the root is left. Each collapse gives the next tree in the sequence, and its $g$ value is the $\alpha$ at which that tree becomes the best.

<div class="ex">
<div class="lab">Worked example</div>

**weakest-link pruning on a five-leaf tree**

A tree trained on 100 samples of classes A and B. The root (56 A, 44 B) splits into L (44 A, 11 B) and R (12 A, 33 B). L splits into leaves L1 (40 A, 0 B) and L2 (4 A, 11 B). R splits into leaf R1 (2 A, 28 B) and node R2 (10 A, 5 B), which splits into leaves R2a (9 A, 1 B) and R2b (1 A, 4 B). Errors are counted as misclassified samples out of 100.

Leaf errors (minority counts): L1 0, L2 4, R1 2, R2a 1, R2b 1. Total 8, so $T_0$ has $R = 0.08$ with 5 leaves.

R2: as a leaf 5 errors; its branch $1 + 1 = 2$; collapsing saves 1 leaf. $g(R2) = (0.05 - 0.02)/1 = 0.03$.

R: as a leaf 12; branch $2 + 1 + 1 = 4$ with 3 leaves. $g(R) = (0.12 - 0.04)/2 = 0.04$.

L: as a leaf 11; branch $0 + 4 = 4$ with 2 leaves. $g(L) = (0.11 - 0.04)/1 = 0.07$. Root: as a leaf 44; whole tree 8 with 5 leaves. $g(\text{root}) = (0.44 - 0.08)/4 = 0.09$.

Round 1: the smallest is $g(R2) = 0.03$, so collapse R2. Tree $T_1$ has 4 leaves and error $0 + 4 + 2 + 5 = 11$, so $R = 0.11$.

Recompute R2's ancestors. R's branch is now R1 and R2: error $2 + 5 = 7$, 2 leaves, so $g(R) = (0.12 - 0.07)/1 = 0.05$. Root: $g = (0.44 - 0.11)/3 = 0.11$. L is unchanged at 0.07.

Round 2: collapse R at 0.05. $T_2$ has leaves L1, L2, R: error $0 + 4 + 12 = 16$, $R = 0.16$. Now $g(L) = 0.07$ and $g(\text{root}) = (0.44 - 0.16)/2 = 0.14$.

Round 3: collapse L at 0.07. $T_3$ has leaves L and R: error $11 + 12 = 23$, $R = 0.23$. $g(\text{root}) = (0.44 - 0.23)/1 = 0.21$. Round 4: collapse the root at 0.21, leaving one leaf with $R = 0.44$.

The sequence is $T_0$ (best for $\alpha < 0.03$), $T_1$ (0.03 to 0.05), $T_2$ (0.05 to 0.07), $T_3$ (0.07 to 0.21), root (above 0.21). Spot check at $\alpha = 0.06$: $T_0$ costs $0.08 + 5 \times 0.06 = 0.38$, $T_1$ $0.11 + 0.24 = 0.35$, $T_2$ $0.16 + 0.18 = 0.34$, $T_3$ $0.23 + 0.12 = 0.35$, root $0.44 + 0.06 = 0.50$. $T_2$ is lowest, as predicted. These are exactly the lines in the previous figure.

</div>

Note that R's value only dropped below L's after R2 was gone, which is why $g$ must be recomputed after every collapse.

## Step 6: choose α with cross-validation

Pruning gives a short list of candidate trees. Training error cannot choose among them (it always prefers $T_0$), so use $k$-fold cross-validation on the training set (module 06): for each candidate $\alpha$, train on $k-1$ folds, score on the held-out fold, and average.

~~~dg-fig 04.02-path.svg
Breast-cancer data. Left: the number of leaves falls in steps as ccp_alpha grows. Right: training accuracy falls steadily, but 10-fold CV accuracy rises slightly and then drops. The circle marks the best CV score (6 leaves); the diamond is the one-standard-error pick (5 leaves).
~~~

A common refinement is the **one-standard-error rule**: compute the standard error of the best CV score (fold-score spread divided by $\sqrt{k}$), then pick the *largest* $\alpha$ (smallest tree) whose mean score is within one standard error of the best. Differences inside that band are statistically invisible.

~~~dg-flow Cost-complexity pruning: grow big, generate a nested sequence of subtrees, let cross-validation pick one, then refit.
Grow full tree T0 -(weakest link)-> Subtrees T1 ⊃ T2 ⊃ … -(CV score per α)-> *Pick α -(refit on all training data)-> Final tree
~~~

## Pruning in code

scikit-learn exposes the whole pruning path and a `ccp_alpha` parameter. Here is the full recipe on the breast-cancer dataset:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)                 # 569 tumours, 30 features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print(full.get_n_leaves(), full.get_depth(), full.score(X_tr, y_tr), full.score(X_te, y_te))
# 16 6 1.0 0.906

# 1. candidate alphas, one per subtree in the nested sequence
path = full.cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas[:-1]                             # the last one prunes to the root
print(len(alphas))                                        # 11, from 0.0 to 0.0265

# 2. score each alpha with 10-fold cross-validation on the training set
cv = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=0), X_tr, y_tr, cv=10)
      for a in alphas]
mean = np.array([s.mean() for s in cv])
se = np.array([s.std(ddof=1) / np.sqrt(len(s)) for s in cv])
best = int(np.argmax(mean))
one_se = max(i for i in range(len(alphas)) if mean[i] >= mean[best] - se[best])
print(alphas[best], mean[best], alphas[one_se])           # 0.0088 0.942 0.009

# 3. refit with the chosen alpha; touch the test set once
final = DecisionTreeClassifier(ccp_alpha=alphas[one_se], random_state=0).fit(X_tr, y_tr)
print(final.get_n_leaves(), final.get_depth(), final.score(X_te, y_te))
# 5 3 0.918

```

The pruned tree has 5 leaves instead of 16 and is slightly more accurate on the test set (0.918 against 0.906); `sklearn.tree.export_text` prints it in a form a doctor can check. Note that `ccp_alpha` is on the scale of weighted impurity, so useful values are small (here 0.002 to 0.03): take the grid from the pruning path rather than guessing. Pre-pruning knobs are tuned with the same cross-validation loop (`GridSearchCV`).

## Reading the result

- **Compare training and validation scores.** A large gap means the tree is still too big; both low and close means too small.
- **Look at leaf sizes.** `tree_.n_node_samples` at the leaves (where `children_left == -1`) shows how many samples each prediction rests on. Leaves of 1-3 samples on noisy data are a warning.
- **Check stability.** Refit on a few bootstrap resamples and compare the top splits. If the root feature keeps changing, do not interpret the tree, however well it scores.

## Complexity depends on the job

- **Single interpretable tree**: prune hard. A tree with 8-20 leaves can be read and checked by a domain expert.
- **Random forest member**: grow deep (default `max_depth=None`), perhaps with a small `min_samples_leaf`. Averaging (lesson 04.03) removes variance, so each tree should have low bias; pruning them is wasted effort.
- **Boosting member**: keep shallow (depth 3-8, or 15-63 leaves). Each tree fits a small correction, and deep trees make boosting overfit quickly (lesson 04.05).

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Using the default unlimited tree as a final model | Training accuracy 100%, test much lower, hundreds of tiny leaves | Set `min_samples_leaf` or prune with a CV-chosen `ccp_alpha`
| Tuning depth or $\alpha$ on the test set | Test score looks great, next month's data does not | Tune with cross-validation on training data; use the test set once
| Guessing a `ccp_alpha` grid like 0.1, 1, 10 | Every value prunes to the root, accuracy equals the majority class | Use `cost_complexity_pruning_path` to get the real break points
| Relying only on `max_depth` with skewed data | Sparse regions still get 1-sample leaves | Prefer `min_samples_leaf` (or combine both)
| Reading a single tree's splits as stable facts | Different root feature on each resample | Check stability; use an ensemble plus importance tools (04.06) for explanation

## Going deeper

**Why the best subtrees are nested.** For a fixed $\alpha$, the optimal subtree can be found bottom-up: at each node, keep the branch if its penalised cost $R(T_t) + \alpha|T_t|$ is lower than the leaf cost $R(t) + \alpha$, otherwise collapse. Rearranged, keep the branch exactly when $\alpha < g(t)$ (with $g$ computed on the already-optimised branch). Raising $\alpha$ can only turn "keep" into "collapse", never the reverse, so the optimal trees for growing $\alpha$ shrink monotonically. For each $\alpha$ there is a unique smallest minimising subtree.

**Repeated alphas.** Tied weakest links are collapsed together in theory; scikit-learn collapses one node per step, so its path can list equal or nearly equal values (0.0025 and 0.0049 appear twice above).

**Other schemes.** Reduced-error pruning collapses any node whose removal does not hurt accuracy on a held-out set (simple, but wastes data). C4.5's pessimistic pruning charges each leaf an upper confidence bound on its error, so small leaves pay extra (no validation data, but heuristic).

**The penalty view.** $R(T) + \alpha|T|$ has the same shape as other penalised objectives in this course: ridge adds $\lambda\|w\|^2$ to squared error (02.02). Here the penalty counts leaves, a discrete $L_0$-like penalty, which is why it needs the combinatorial weakest-link trick instead of gradients.

## Key terms

| Term | Meaning
| --- | --- |
| Capacity (of a tree) | How flexible it is, measured by the number of leaves $|T|$
| Pre-pruning | Stopping rules that refuse splits while the tree grows (`max_depth`, `min_samples_leaf`, ...)
| Post-pruning | Growing the tree fully, then collapsing branches back into leaves
| Horizon effect | A stopping rule rejecting a split whose value only appears in the splits below it
| Cost-complexity criterion | $R_\alpha(T) = R(T) + \alpha|T|$: training error plus a price per leaf
| Weakest link | The internal node with the smallest error reduction per extra leaf, $g(t)$
| Pruning path | The nested sequence of subtrees and the $\alpha$ at which each becomes optimal
| One-standard-error rule | Choose the simplest model whose CV score is within one standard error of the best

<div class="take">
<div class="lab">Key takeaway</div>

an unrestricted tree memorises noise, so its size must be controlled. Use `min_samples_leaf`-style stopping rules for a quick fix, or, for a single tree you want to trust, grow it fully and pick a cost-complexity $\alpha$ from the pruning path by cross-validation; inside ensembles the target changes, with deep trees for bagging and shallow ones for boosting.

</div>
