---
id: "04.01"
course: "machine-learning"
module: "04"
slug: "decision-trees"
title: "Decision trees"
keyIdeas: ["A decision tree asks a sequence of questions of the form \"is feature $x_j \\le \\theta$?\". Geometrically it cuts the feature space into axis-aligned rectangles and predicts one value in each: the majority class or the mean target of the training examples there.", "**Impurity** measures how mixed a node is. Gini $1-\\sum_k p_k^2$ and entropy $-\\sum_k p_k \\log_2 p_k$ are both zero for a pure node and peak at an even mix; they choose the same split most of the time.", "A split is scored by its **impurity decrease**: parent impurity minus the size-weighted impurity of the two children. Misclassification error is a poor score because it is flat (linear) and often ignores useful splits.", "CART grows the tree greedily: at every node, try every feature and every midpoint threshold, keep the best, and recurse. Regression trees do the same with squared error, using running sums to score each threshold in constant time.", "Trees need no feature scaling and capture interactions automatically, but they are unstable, draw staircase boundaries, cannot extrapolate beyond the training targets, and overfit when grown fully, which is why the rest of the module limits and combines them."]
question: "You have a two-class problem where the label is 1 exactly when $x_1$ and $x_2$ have the same sign, with both features uniform on $[-1, 1]$. Will a CART tree with `min_impurity_decrease=0.01` learn it? What happens without that rule, and why?"
answer: "No. The label is an XOR pattern of the signs. Any single threshold on $x_1$ leaves each child with about half positives (whatever the cut, the region on each side is still split evenly by the sign of $x_2$), and the same holds for any threshold on $x_2$. So the expected Gini decrease of every root split is close to zero; only sampling noise makes some cuts look slightly better than others. A rule requiring a decrease of at least 0.01 will usually stop at the root and predict the majority class, giving about 50% accuracy (55% on one sample of 400 points, simply because one class happened to be more common). Without the rule, the tree splits on whichever threshold looks best through noise. If that cut lands near zero on one feature, each child contains one clean half of the pattern, and a split near zero on the other feature becomes perfectly informative, so a depth-2 tree recovers the rule exactly. The lesson is that greedy splitting judges each split only by its immediate gain, so an early-stopping threshold can block splits whose value appears only one level deeper. This is one argument for growing a large tree and pruning it back afterwards (lesson 04.02), and for ensembles that try many different trees."
sources: [{"title": "scikit-learn User Guide: Decision Trees", "url": "https://scikit-learn.org/stable/modules/tree.html", "note": "Section 1.10.7 gives the exact impurity formulas and complexity; read the tips on practical use."}, {"title": "The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free PDF. Section 9.2 covers CART regression and classification trees and why misclassification error is a poor split criterion."}, {"title": "An Introduction to Statistical Learning", "url": "https://www.statlearning.com/", "note": "Free PDF. Chapter 8.1 is a gentle walk through regression and classification trees with figures."}]
practice: ["decision-tree-classifier-cart", "decision-tree-regressor"]
tags: ["CART", "Gini impurity", "information gain"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- What a decision tree is: yes/no questions that cut the feature space into rectangles, with one prediction per rectangle.
- How Gini impurity and entropy measure how mixed a group is, and how a split is scored by the weighted impurity decrease.
- How CART grows a tree greedily: try every feature and threshold, keep the best, repeat.
- How regression trees work (leaf = mean, score = drop in squared error), by hand and with a fast scan.
- How to fit and read a tree in scikit-learn, and the failure modes that motivate the rest of this module.

*Before you start:* you need proportions, averages and sums of squares, plus the ideas of training loss and overfitting from module 01. Lesson 02.03 (logistic regression) makes a useful contrast: one straight boundary instead of boxes. No calculus is needed.

## Where this module starts

Module 02 built models out of weighted sums. This module takes a different route: instead of adding up features, a model asks questions about them. **Decision trees**, and the ensembles built from many trees, are the most widely used models for tabular data (spreadsheets, logs, customer records).

This lesson grows a single tree. Lesson 04.02 limits its size, 04.03 averages many trees (random forests), 04.04 and 04.05 build trees that fix each other's mistakes (boosting), and 04.06 explains what an ensemble has learned. All of them reuse the splitting machinery from this lesson.

## The idea in one picture

A small lender wants to predict whether an applicant will repay a loan. It has ten past applicants:

| Income (thousand / year) | 25 | 30 | 32 | 36 | 38 | 40 | 44 | 55 | 63 | 70
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age (years) | 24 | 50 | 41 | 58 | 38 | 44 | 30 | 25 | 33 | 28
| Outcome | defaulted | repaid | defaulted | repaid | defaulted | defaulted | repaid | repaid | repaid | repaid

Six repaid, four defaulted. A loan officer might say: "Above about 42 thousand income, they repay. Below that, the older applicants repaid and the younger ones did not." That is a decision tree, and it fits these ten people perfectly.

~~~dg-fig 04.01-boxes.svg
The same rule drawn as a map. The first question draws a vertical cut at income 42; the second draws a horizontal cut at age 47, but only on the left side. Each of the three rectangles predicts one class.
~~~

~~~dg-flow The tree behind the picture. Every applicant answers the questions from the top and lands in exactly one leaf.
[Higher income] Applicant -(income ≤ 42? no)-> *Leaf: repaid
[Lower income, older] Applicant -(income ≤ 42? yes)-> Age question -(age ≤ 47? no)-> Leaf: repaid
[Lower income, younger] Applicant -(income ≤ 42? yes)-> Age question -(age ≤ 47? yes)-> Leaf: defaulted
~~~

A new applicant with income 35 and age 30 answers "yes" (35 ≤ 42), then "yes" (30 ≤ 47), and lands in the "defaulted" leaf. Vocabulary for the rest of the module:

- The first question is the **root**. Every question is a **split** on one feature against a **threshold**, such as "income ≤ 42".
- A question that leads to more questions is an **internal node**. An end point that outputs a prediction is a **leaf**.
- The **depth** of a tree is the number of questions on its longest path. This tree has depth 2 and 3 leaves.
- In the map view, the tree is a **partition**: it cuts the feature space into non-overlapping **axis-aligned rectangles** (every cut is parallel to an axis, because each question looks at one feature only).

The rest of the lesson answers one question: how does a computer find these splits on its own?

## Step 1: what a tree predicts

Assume the splits are given. To predict, send the example down to its leaf and summarise the training examples that landed there: the majority class for **classification** (a leaf with 9 repaid and 1 defaulted says "repaid", probability 0.9), or their average target for **regression**.

Three consequences follow straight away:

- Only comparisons $x_j \le \theta$ are made, and logs, standardising or ranks keep the order of values, so the splits do not change. **Trees need no feature scaling.**
- A question low in the tree applies only to examples that reached it. "Age matters, but only for low incomes" is an **interaction**, captured without adding a product column.
- A leaf outputs an average of training targets, so a regression tree **cannot extrapolate** beyond the targets it saw.

## Step 2: measure how mixed a node is

A good split produces children that are each as close to "all one class" as possible. We need a number for how mixed a group is, called **impurity**. Write $p_k$ for the share of class $k$ in the node (for our root: $p_{\text{repaid}} = 0.6$, $p_{\text{defaulted}} = 0.4$). The two standard choices are

$$G = 1 - \sum_k p_k^2 \qquad\qquad H = -\sum_k p_k \log_2 p_k.$$

$G$ is **Gini impurity**. In words: draw two examples from the node at random (with replacement); $G$ is the probability that their labels differ. $H$ is **entropy**, measured in bits: the average number of yes/no questions you need to learn the label of a random example from the node.

Number check at our root: $G = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$. $H = -(0.6 \log_2 0.6 + 0.4 \log_2 0.4) = 0.442 + 0.529 = 0.971$ bits.

~~~dg-fig 04.01-impurity.svg
For two classes, impurity depends only on the class share. Gini and entropy are smooth hills: zero for a pure node, highest for a 50/50 mix. Misclassification error is a tent made of two straight lines, which turns out to matter (Step 3).
~~~

Both are zero for a pure node and largest for an even mix (for two classes, 0.5 for Gini and 1 bit for entropy). **Misclassification error** $1 - \max_k p_k$ has the same endpoints but makes a bad splitting score, as Step 3 shows.

## Step 3: score a split

A split sends $n_L$ of the node's $n$ examples to the left child and $n_R$ to the right. Each child has its own impurity. We average them, weighting each child by its size, and compare with the parent:

$$\Delta = I(\text{parent}) - \frac{n_L}{n}\, I(L) - \frac{n_R}{n}\, I(R).$$

Here $I$ is whichever impurity you chose (Gini or entropy), $I(L)$ and $I(R)$ are the children's impurities, and $\Delta$ is the **impurity decrease**: how much purer the split makes things on average. With entropy, $\Delta$ has its own name, **information gain**. The weights matter: a split that peels off one pure example achieves little, and the weighting makes sure it scores little.

<div class="ex">
<div class="lab">Worked example</div>

**Gini and entropy on two candidate splits**

The root holds our 10 applicants: 6 repaid, 4 defaulted. From Step 2, Gini = 0.48 and entropy = 0.971 bits.

Split A, "income ≤ 42". The right side (income > 42) gets 4 applicants, all repaid, so its impurity is 0. The left side gets 6 applicants: 2 repaid, 4 defaulted.

Left Gini: $1 - \big((2/6)^2 + (4/6)^2\big) = 1 - (1/9 + 4/9) = 4/9 = 0.444$. Weighted: $0.6 \times 0.444 = 0.267$ (plus $0.4 \times 0 = 0$ for the pure side). Decrease: $0.48 - 0.267 = 0.213$.

Left entropy: $-(1/3 \log_2 1/3 + 2/3 \log_2 2/3) = 0.918$ bits. Weighted: $0.6 \times 0.918 = 0.551$. Information gain: $0.971 - 0.551 = 0.420$ bits.

Split B, "age ≤ 35.5". Left: 4 repaid, 1 defaulted, Gini $1 - (0.64 + 0.04) = 0.32$. Right: 2 repaid, 3 defaulted, Gini $1 - (0.16 + 0.36) = 0.48$. Weighted: $0.5 \times 0.32 + 0.5 \times 0.48 = 0.40$. Decrease: $0.48 - 0.40 = 0.08$.

With entropy, split B's children have $0.722$ and $0.971$ bits, weighted $0.846$, so the gain is $0.971 - 0.846 = 0.125$ bits.

Both criteria prefer A by a wide margin. A creates one perfectly pure child, which is exactly what a curved (concave) impurity rewards.

</div>

Why not just count mistakes? Take a parent with 400 of each class. Split 1 gives children (300, 100) and (100, 300); split 2 gives (200, 400) and (200, 0). Both make 200 mistakes, so misclassification error ties them at 0.25. Gini scores split 1 at 0.375 and split 2 at $0.75 \times 0.444 = 0.333$, preferring the split that creates a pure node. Worse, when both children keep the parent's majority class, the misclassification decrease is exactly zero however much purer one child becomes. The straight sides of the tent are the reason; a curved hill always rewards children that differ from each other.

## Step 4: search every feature and every threshold

Now the computer can score any split, so it tries them all. Only thresholds between consecutive distinct sorted values matter (any cut between 40 and 44 sends the same people left), and CART uses the midpoints: 9 candidates on income, 9 on age.

~~~dg-fig 04.01-scan.svg
Every candidate split at the root, scored by the weighted Gini of its children (lower is better). Income at 42 drops to 0.267; nothing on age gets below 0.40. The tree takes income ≤ 42 as its root.
~~~

This is the **CART** procedure (Classification And Regression Trees), used by scikit-learn:

1. At the current node, sort the examples by each feature in turn.
2. Score every midpoint threshold by impurity decrease.
3. Split on the (feature, threshold) pair with the largest decrease.
4. Repeat inside each child until a stopping rule fires: pure node, too few examples, or maximum depth.

On our data, step 3 picks income ≤ 42. The right child is pure, so it becomes a leaf. The left child (2 repaid, 4 defaulted, Gini 0.444) gets its own search; "age ≤ 47" separates it perfectly (decrease 0.444), and the tree is done.

This search is **greedy**: each split is the best one *right now*, with no look-ahead. Finding the smallest accurate tree is NP-hard (Hyafil and Rivest, 1976), so greedy growth is the practical compromise; its blind spot appears in "What goes wrong".

## Step 5: regression trees

For a numeric target only two things change. A leaf predicts the **mean** of its training targets, and impurity becomes the **sum of squared errors** (SSE) around that mean. The best split is the one with the smallest total SSE in the two children, which is the same as the largest drop in SSE (equivalently, the largest reduction in variance).

<div class="ex">
<div class="lab">Worked example</div>

**one regression split by hand**

Five houses: size index $x = (1, 2, 3, 4, 5)$, price $y = (10, 12, 11, 30, 32)$. Parent mean $19$; parent SSE $= 81 + 49 + 64 + 121 + 169 = 484$.

Threshold 2.5: left $\{10, 12\}$, mean 11, SSE $1 + 1 = 2$; right $\{11, 30, 32\}$, mean 24.33, SSE 268.67. Total 270.67.

Threshold 3.5: left $\{10, 12, 11\}$, mean 11, SSE $1 + 1 + 0 = 2$; right $\{30, 32\}$, mean 31, SSE $1 + 1 = 2$. Total 4.

Thresholds 1.5 and 4.5 leave one large mixed group behind, with totals 382.75 and 272.75.

Best: $x \le 3.5$, with SSE dropping from 484 to 4. The stump predicts 11 for small houses and 31 for large ones. Asked about a house at $x = 9$, it still answers 31: a tree never predicts outside the range of the leaf means it learned.

</div>

Grow the tree deeper and the steps get narrower, so the prediction can follow any curve, including noise.

~~~dg-fig 04.01-regression.svg
Left: a regression tree is a staircase. Depth 1 has two steps; depth 8 has 73 and starts tracing the noise. Right: past the edge of the training data, a tree repeats its last leaf forever while a line keeps its trend.
~~~

Other losses only swap the leaf statistic: absolute error uses the median of the leaf, and Poisson deviance (for counts) uses a mean on a log scale.

## Decision trees in code

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

income = np.array([25, 30, 32, 36, 38, 40, 44, 55, 63, 70.])
age    = np.array([24, 50, 41, 58, 38, 44, 30, 25, 33, 28.])
y      = np.array(["D", "R", "D", "R", "D", "D", "R", "R", "R", "R"])
X = np.c_[income, age]                    # shape (10, 2); no scaling needed

tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0).fit(X, y)
print(export_text(tree, feature_names=["income", "age"]))
# |--- income <= 42.00
# |   |--- age <= 47.00
# |   |   |--- class: D
# |   |--- age >  47.00
# |   |   |--- class: R
# |--- income >  42.00
# |   |--- class: R
print(tree.predict([[35, 52], [35, 30]]))   # ['R' 'D']
print(tree.predict_proba([[35, 30]]))       # [[1. 0.]]  columns: tree.classes_ = ['D', 'R']
print(tree.get_depth(), tree.get_n_leaves())  # 2 3
print(tree.feature_importances_)            # [0.444 0.556]

reg = DecisionTreeRegressor(max_depth=1).fit(np.array([[1], [2], [3], [4], [5.]]),
                                             np.array([10, 12, 11, 30, 32.]))
print(reg.tree_.threshold[0], reg.predict([[2.5], [9.0]]))   # 3.5 [11. 31.]
```

Both trees found the splits we derived by hand. Use `criterion="entropy"` for information gain, and `"squared_error"` (default) or `"absolute_error"` for regression. Set `random_state`: ties between splits are broken through a random feature order.

## Reading the fitted tree

**Print or plot it.** `export_text` or `plot_tree` show every split and the class counts in each node. For a small tree this is the whole model, and a domain expert can check whether the rules make sense.

**Check its size, then compare training and validation scores.** `get_depth()` and `get_n_leaves()` measure complexity. A tree grown without limits splits until every leaf is pure, so it scores near 100% on training data by construction; only held-out data tells you anything, and a large gap is the signature of overfitting (04.02).

**Treat probabilities with care.** `predict_proba` returns the leaf's class shares. Deep trees have pure leaves, so nearly every probability is 0 or 1: overconfident.

**Read importances as a summary.** `feature_importances_` sums each feature's impurity decreases, weighted by node size, normalised to 1. The root removed $10 \times 0.213 = 2.13$ units and the age split $6 \times 0.444 = 2.67$, so age gets 0.556 although income was chosen first (more in 04.06).

## What goes wrong

| Problem | Symptom | Why it happens | Fix
| --- | --- | --- | --- |
| Overfitting | Training accuracy near 100%, validation much lower | Splitting continues until leaves are pure, so the tree memorises noise | Limit depth or leaf size, or prune (04.02)
| Instability | Small changes in the data change the root and the whole tree | The best and second-best splits are often close; a different winner changes everything below it | Average many trees (04.03)
| Staircase boundaries | Many splits needed for a simple diagonal rule | Each question looks at one feature, so boundaries are axis-aligned | Add a feature such as $x_1 - x_2$, or use an ensemble
| No extrapolation | Flat predictions beyond the training range | Leaves output averages of seen targets | Detrend first, or use a linear model for trends
| Greedy blind spots | Tree refuses to split on a pattern that needs two splits | Each split is judged only by its own immediate gain | Grow deeper and prune instead of stopping early

~~~dg-fig 04.01-failures.svg
Left: the true boundary is the diagonal $x_2 = x_1$; a depth-3 tree approximates it with a few coarse steps and a depth-8 tree with more. Right: two fully grown trees fitted to two random resamples of the same noisy data draw very different regions. Both effects motivate ensembles.
~~~

A concrete greedy blind spot: let the label be 1 when $x_1$ and $x_2$ have the same sign (XOR). Every single cut leaves both sides about half positive, so no root split looks useful. With `min_impurity_decrease=0.01`, a tree on 400 such points never splits and scores 55%; without that rule, a depth-2 tree reaches 100%, because the first split's value only shows one level deeper.

## Going deeper

**Cost of growing a tree.** Sorting a node's $n$ examples on one feature costs $O(n \log n)$; a left-to-right sweep with running class counts then scores each threshold in $O(1)$. One node costs $O(d\, n \log n)$ for $d$ features, about $O(d\, n \log^2 n)$ over a balanced tree.

**The prefix-sum trick for regression.** For any contiguous block of sorted examples, $\text{SSE} = \sum y^2 - (\sum y)^2 / m$, where $m$ is the block size. With running sums of $y$ and $y^2$, every threshold costs $O(1)$:

```python
def best_split(x, y):
    # Best threshold on one feature for squared error, O(n log n).
    o = np.argsort(x); x, y = x[o], y[o]
    n = len(y); cs, cs2 = np.cumsum(y), np.cumsum(y * y)
    nl = np.arange(1, n)                                  # left size if we cut after i
    sse_l = cs2[:-1] - cs[:-1] ** 2 / nl
    sse_r = (cs2[-1] - cs2[:-1]) - (cs[-1] - cs[:-1]) ** 2 / (n - nl)
    cost = np.where(x[1:] != x[:-1], sse_l + sse_r, np.inf)   # only between distinct values
    i = int(np.argmin(cost))
    return (x[i] + x[i + 1]) / 2, cost[i]
# best_split on the five houses returns (3.5, 4.0)
```

**Histograms instead of sorting.** LightGBM and scikit-learn's `HistGradientBoosting*` bucket each feature into at most 255 bins, so a split search scans bins instead of examples: $O(\text{bins})$ per feature after one pass over the data (04.05).

**Why concave impurity never makes things worse.** Gini and entropy are strictly concave functions of the class proportions, and the parent's proportions are the size-weighted average of the children's. Jensen's inequality then gives $I(\text{parent}) \ge \frac{n_L}{n} I(L) + \frac{n_R}{n} I(R)$, with equality only when both children have exactly the parent's proportions. So every split that changes the mix has a positive decrease. Misclassification error is piecewise linear, so the inequality is often an equality and useful splits score zero.

**Gini versus entropy in practice.** The two curves have almost the same shape, and they disagree on the chosen split only in a small fraction of cases. Gini avoids a logarithm and is the scikit-learn default.

**Categorical features.** A feature with $K$ levels can be split into two groups in $2^{K-1} - 1$ ways. For two classes or squared error, sorting the levels by their mean target and treating them as ordered finds the optimal grouping in $O(K \log K)$, a classic CART result.

**Other tree families.** C4.5 (Quinlan) uses information gain divided by the split's own entropy (gain ratio), so that many-valued features are not favoured, and allows multi-way splits. Oblique trees split on combinations $w^\top x \le \theta$ to avoid staircases, at a much higher search cost.

## Key terms

| Term | Meaning
| --- | --- |
| Decision tree | A model that predicts by asking a sequence of threshold questions and reading the answer in a leaf
| Split / threshold | A question "feature $j \le \theta$?" that sends each example left or right
| Leaf | An end node; predicts the majority class (classification) or mean target (regression) of its training examples
| Impurity | A score of how mixed a node's labels are; zero when the node is pure
| Gini impurity | $1 - \sum_k p_k^2$: the chance two random examples from the node have different labels
| Entropy / information gain | $-\sum_k p_k \log_2 p_k$, in bits; the impurity decrease measured with entropy
| Impurity decrease | Parent impurity minus the size-weighted impurity of the children
| CART | Greedy top-down growth: at each node, choose the feature and threshold with the largest impurity decrease
| Depth | Number of questions on the longest root-to-leaf path

<div class="take">
<div class="lab">Key takeaway</div>

a decision tree grows greedily from the top: at each node it tries every feature and threshold, keeps the split with the largest weighted drop in impurity (Gini, entropy or squared error), and stops at leaves that predict a majority class or a mean. The result needs no scaling and captures interactions, but it is unstable, axis-bound, cannot extrapolate and overfits when grown fully, which is what the rest of this module fixes.

</div>
