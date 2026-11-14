---
id: "04.06"
course: "machine-learning"
module: "04"
slug: "interpreting-tree-models"
title: "Interpreting tree models"
keyIdeas: ["Separate three questions: which features the model relies on (global importance), how the prediction moves with a feature (PD and ICE curves), and why one prediction came out as it did (SHAP).", "Impurity importance (MDI) sums each feature's weighted impurity decrease over all splits. It is free, but computed on training data and biased towards continuous and high-cardinality features, so an ID column can look important.", "Permutation importance shuffles one column of held-out data and measures the drop in score. It is model-agnostic and reflects generalisation, but correlated features share or hide importance; permute correlated groups together.", "Partial dependence averages predictions while forcing one feature to each value. ICE curves show the per-row lines behind it and reveal interactions the average hides; both extrapolate into impossible rows when features are correlated.", "SHAP values split one prediction minus the average prediction into additive per-feature contributions; TreeSHAP computes them exactly and fast for tree ensembles.", "Every one of these tools describes the model, not the world: an important feature is one the model uses, which is not evidence of causation."]
question: "A random forest predicting house prices ranks `listing_id` (a unique integer) second by `feature_importances_`, while `sqft_living` and `sqft_above` (correlation 0.88) both rank low by permutation importance. Explain both observations and what you would report."
answer: "Two different mechanisms are at work.\n\n`listing_id` ranks high under impurity importance (MDI) because MDI is computed on the training data and adds up every impurity decrease a feature produces. A column with a distinct value per row offers hundreds of candidate thresholds, and deep trees can always find some that isolate individual noisy houses, effectively memorising them. MDI credits those splits even though they do not generalise. To confirm, compute permutation importance on a validation set: shuffling the ID there should barely change the score (importance near zero, within its error bar), because the memorised IDs do not reappear in new data. I would drop the column and refit; validation performance should not get worse.\n\nThe two square-footage features look weak under permutation importance because they carry mostly the same information. Shuffling `sqft_living` leaves `sqft_above` intact, so the forest falls back on it and the score barely drops, and vice versa. Each single-feature number therefore understates the shared signal, and the split between the two can be very uneven. Shuffling one also creates implausible rows, such as a huge living area with a tiny above-ground area, so part of the measured importance reflects the model's behaviour on impossible inputs.\n\nWhat I would report: (1) cluster correlated features (for example by Spearman correlation) and give permutation importance for the group, by shuffling both columns together; it will show that house size is among the most important signals. (2) Optionally keep one representative per cluster, or use drop-column importance, if the stakeholders need a per-column number. (3) State clearly that the size signal is strong but that dividing it between the two correlated columns is not well defined, and that importance describes what the model uses, not a causal effect of square footage on price."
sources: [{"title": "scikit-learn User Guide: Permutation feature importance", "url": "https://scikit-learn.org/stable/modules/permutation_importance.html", "note": "Algorithm, its relation to MDI, and the correlated-feature caveat."}, {"title": "scikit-learn User Guide: Partial dependence and ICE plots", "url": "https://scikit-learn.org/stable/modules/partial_dependence.html", "note": "Definitions, the brute vs recursion methods for trees, and interpretation limits."}, {"title": "Strobl et al. (2007): Bias in random forest variable importance measures", "url": "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25", "note": "Shows MDI's preference for features with many categories or values."}, {"title": "Lundberg, Erion, Lee: Consistent Individualized Feature Attribution for Tree Ensembles", "url": "https://arxiv.org/abs/1802.03888", "note": "The TreeSHAP algorithm and why gain-based importances can be inconsistent."}]
practice: ["permutation-feature-importance", "random-forest"]
tags: ["feature importance", "partial dependence", "SHAP"]
tools: ["scikit-learn", "SHAP"]
---

## What you will learn

- Three separate questions people ask about a tree model ("which features?", "how does a feature act?", "why this prediction?") and which tool answers each.
- How impurity importance (MDI) is computed, by hand on a two-split tree, and why it favours ID-like columns.
- How permutation importance works, why it should run on held-out data, and how correlated features fool it.
- How to read partial dependence and ICE curves, and when they ask the model about people who cannot exist.
- What a SHAP value is, how to compute one by hand for a two-feature model, and what TreeSHAP adds.

*Before you start:* you need decision trees and impurity (04.01), random forests (04.03) and gradient boosting (04.05). Validation sets come from module 01.

## Why this lesson closes the module

The module went from one readable tree (04.01) to boosted ensembles of hundreds of trees (04.05). Each step traded readability for accuracy, yet these models are deployed for credit, pricing and churn, where somebody will ask "why?". This lesson gives you the standard toolkit for answering, and the known ways each tool misleads. All of them describe **the fitted model**, not the world.

## The idea in one picture

Take a small house-price problem: 1,000 houses, price in thousands, five columns. Four are real (`size_m2`, `age_years`, `dist_km` to the centre, `has_garden`). The fifth, `listing_id`, is a random number from 0 to 999 that carries no information at all. We fit a random forest of 200 trees and ask it: which features matter?

~~~dg-fig 04.06-mdi-vs-perm.svg
The same forest, two importance measures. Impurity importance gives the useless listing_id 11% of the credit, more than the garden; permutation importance on validation data gives it essentially zero.
~~~

| Feature | Impurity importance (training) | Permutation importance (validation)
| --- | --- | --- |
| size_m2 | 0.517 | 0.845
| dist_km | 0.174 | 0.179
| age_years | 0.160 | 0.072
| has_garden | 0.038 | 0.067
| listing_id | 0.112 | 0.009

Both agree that size matters most. But the left column says a random ID matters three times as much as a garden, and the right says it matters not at all. They measure different things:

- **Impurity importance** (MDI, *mean decrease in impurity*) counts how much each feature helped the trees fit the *training* data, including fitting noise.
- **Permutation importance** measures how much the model's score on *new* data drops when that feature's information is destroyed.

The forest really did split on `listing_id`, to memorise individual noisy houses. That helped on the training set (training $R^2 = 0.92$) and did nothing for new houses (validation $R^2 = 0.53$).

## Step 1: three questions, three tools

People blur these together. Keep them apart:

| Question | Scope | Tool
| --- | --- | --- |
| Which features does the model rely on? | **global**, whole dataset | MDI, permutation importance, mean absolute SHAP
| How does the prediction move as one feature changes? | **effect curve** | partial dependence (PD), ICE
| Why did *this* prediction come out this way? | **local**, one row | SHAP values

A **global** answer is one ranking for the whole model; a **local** answer is specific to one row, since two customers can get the same prediction for different reasons.

## Step 2: impurity importance (MDI)

Every split in a tree makes its children purer than its parent. MDI simply adds up those improvements per feature.

For one split at node $t$, the weighted impurity decrease is

$$\Delta(t) = \frac{n_t}{n}\Big(I(t) - \frac{n_L}{n_t} I(L) - \frac{n_R}{n_t} I(R)\Big),$$

where $n$ is the number of training samples, $n_t$ how many reach node $t$, $L$ and $R$ the two children, and $I(\cdot)$ the impurity (Gini or entropy for classification, variance for regression, as in 04.01). The factor $n_t/n$ means a split near the root, which affects many samples, counts more than one deep in the tree.

The importance of feature $j$ is the sum of $\Delta(t)$ over every node that splits on $j$, averaged over the trees of the forest, then rescaled so all importances add up to 1. scikit-learn exposes this as `feature_importances_`, and it costs nothing: the numbers are collected while the trees grow.

<div class="ex">
<div class="lab">Worked example</div>

**MDI by hand on a two-split tree**

Ten training customers, 5 churn and 5 stay. Root Gini: $1 - 0.5^2 - 0.5^2 = 0.5$.

Split 1 on `tenure`: the left child gets 4 customers (all stay, Gini 0); the right child gets 6 (5 churn, 1 stays). Right Gini: $1 - (5/6)^2 - (1/6)^2 = 1 - 26/36 = 10/36 \approx 0.278$.

Decrease at the root: $\frac{10}{10}\big(0.5 - \frac{4}{10}\cdot 0 - \frac{6}{10}\cdot 0.278\big) = 0.5 - 0.167 = 0.333$. This goes to `tenure`.

Split 2 on `customer_id` inside the right child: it isolates the one stayer, so both grandchildren have Gini 0. Decrease: $\frac{6}{10}(0.278 - 0) = 0.167$. This goes to `customer_id`.

Total decrease $0.333 + 0.167 = 0.5$ (the whole root impurity, because the leaves are pure). Normalised: `tenure` gets $0.333/0.5 = 0.67$, `customer_id` gets $0.33$.

A meaningless ID earned a third of the importance just by separating one customer.

</div>

That example shows the two structural problems with MDI:

1. **Cardinality bias.** A column with many distinct values offers many thresholds, and some reduce training impurity by luck. A binary feature like `has_garden` offers one threshold and is under-credited (0.038, below the ID). Strobl et al. (2007) documented this bias.
2. **Training data.** MDI rewards splits that fit noise; the deeper the trees, the more credit flows to junk.

Treat MDI as a quick first look, never the final word.

## Step 3: permutation importance

The idea fits in one sentence: if the model relies on feature $j$, scrambling feature $j$ should hurt.

~~~dg-flow Permutation importance for one feature. Repeat for each feature and several shuffles; the model is never retrained.
Validation set -(score)-> Baseline s -(shuffle column j)-> Corrupted set -(score)-> *Importance s − s_j
~~~

In symbols: let $s$ be the model's score on a held-out set and $s_{j,k}$ the score after the $k$-th random shuffle of column $j$. The importance is

$$\mathrm{PI}_j = s - \frac{1}{K}\sum_{k=1}^{K} s_{j,k},$$

with $K$ repeats (5 to 30) whose standard deviation is the error bar. Shuffling keeps the column's distribution but breaks its link to the target and to the rest of the row.

Shuffling `listing_id` on new houses changes nothing, because the memorised IDs do not reappear there: $0.009 \pm 0.010$, indistinguishable from zero. (A drop in $R^2$ can exceed the baseline, as for size, because a corrupted model can do worse than predicting the mean.)

<div class="ex">
<div class="lab">Worked example</div>

**reading permutation importances for a churn model**

Validation AUC is 0.84. Shuffling `tenure_months` drops it to 0.71, so its importance is $0.84 - 0.71 = 0.13$ (± 0.01 over 10 shuffles).

Shuffling `contract_type` gives 0.80: importance $0.04 \pm 0.01$. Shuffling `account_id` gives 0.84: importance $0.00 \pm 0.002$.

MDI on the same forest ranked `account_id` third (50,000 distinct values, used to memorise). Permutation importance shows it carries no signal that generalises: drop it.

</div>

Cost: $d \times K$ prediction passes over the validation set for $d$ features; nothing is retrained, and any model and metric work.

## Step 4: the correlated-feature trap

Permutation importance has one big weakness. Suppose we add a noisy copy of size in square feet (`size_ft2`, correlation 0.99 with `size_m2`) and refit.

~~~dg-fig 04.06-correlated.svg
Size alone drops validation R² by 0.85 when shuffled. With a near-copy present, shuffling either column alone hurts much less, because the forest falls back on the other. Shuffling both together recovers the true 0.84.
~~~

Shuffling `size_m2` leaves `size_ft2`, so the score drops by 0.64 instead of 0.85; shuffling `size_ft2` alone barely matters (0.01). The two add to 0.65, less than the 0.84 from shuffling both. Correlated features *share* importance, often unevenly. Worse, shuffling one creates rows the model never saw, such as a 150 m² house listed at 500 ft², so the importance partly measures behaviour on impossible inputs.

Fixes:

- **Group permutation:** cluster features by Spearman correlation and permute each cluster as a block.
- **Keep one representative** per correlated cluster and refit.
- **Drop-column importance:** retrain without the feature; honest, but one refit per feature.

## Step 5: partial dependence and ICE

Importance says *how much*; effect curves say *how*: steadily, flattening, or with a jump?

The **partial dependence** of the prediction $f$ on a feature $x_S$ at value $v$ is

$$\mathrm{PD}(v) = \frac{1}{n}\sum_{i=1}^{n} f\big(x_S = v,\ x_C^{(i)}\big).$$

In words: overwrite feature $S$ with $v$ in every row, keep the other features $x_C^{(i)}$, predict, and average; repeat over a grid of $v$. An **ICE curve** (*individual conditional expectation*) is the same for one row, before averaging. PD is the average of the ICE lines.

<div class="ex">
<div class="lab">Worked example</div>

**PD and ICE by hand**

A toy model: price $= 150$ if size $< 100$ m², otherwise $200$, plus $30$ more if the house is large *and* has a garden. Four houses: (80 m², no garden), (120, garden), (90, garden), (110, no garden).

PD at size $= 70$: every row becomes a small house, so all four predictions are 150. $\mathrm{PD}(70) = 150$.

PD at size $= 130$: the two rows without a garden predict 200, the two with a garden predict 230. $\mathrm{PD}(130) = (200 + 230 + 230 + 200)/4 = 215$.

The PD curve says "going from small to large adds 65". The ICE lines say more: houses without a garden jump by 50, houses with a garden by 80. The average hides an **interaction** between size and garden.

</div>

~~~dg-fig 04.06-pd-ice.svg
Left: ICE lines for 30 real houses; garden homes (green) climb more steeply, so the PD average (thick) is a compromise. Right: for energy use, heated homes fall with temperature and air-conditioned homes rise, so the PD line is nearly flat although shuffling temperature wrecks the model.
~~~

The right panel is the warning case: energy use depends strongly on temperature, but in opposite directions for heated and air-conditioned homes, so the average cancels out. Always draw ICE lines under a PD curve; if they cross or fan out, colour them by a suspect feature to find the interaction.

PD shares the correlated-feature trap from step 4, in a sharper form:

~~~dg-fig 04.06-extrapolation.svg
Age and years employed lie in a narrow band. To compute PD at age 25, every row is given age 25, including people with 30+ years of work: points far outside the data, where the trees' answers are arbitrary.
~~~

A tree answers such rows from whatever leaf they fall into, so the PD curve there can be meaningless. **Accumulated local effects** (ALE) plots fix most of this by adding up small prediction differences computed only within narrow bins of real data.

## Step 6: SHAP values for one prediction

Now the local question: the forest priced one validation house at 110 thousand, while its average prediction is 200. Which features account for the 90 thousand gap?

SHAP borrows **Shapley values** from cooperative game theory. The features are players who jointly "earn" the gap between this prediction and the average one; a feature's fair share is its extra contribution when it joins, averaged over every joining order. Formally, for $d$ features $F$ and a value function $v(S)$ = the expected prediction when only the features in $S$ are known,

$$\phi_j = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(d-|S|-1)!}{d!}\Big(v(S\cup\{j\}) - v(S)\Big).$$

Here $\phi_j$ is feature $j$'s SHAP value, $S$ runs over subsets of the other features, and the fraction is the share of orders in which exactly $S$ arrives before $j$. The key property is **additivity**: $\text{base value} + \sum_j \phi_j = \text{prediction}$, where the base value $v(\varnothing)$ is the average prediction.

<div class="ex">
<div class="lab">Worked example</div>

**SHAP values by hand for the toy model**

Explain the house (120 m², garden), which the toy model prices at 230. Use the four houses above as background data. With $d = 2$ both weights in the formula are $1/2$.

$v(\varnothing)$: average prediction over the background: $(150 + 230 + 150 + 200)/4 = 182.5$.

$v(\{\text{size}\})$: set size to 120 in every background row, keep their gardens: $(200 + 230 + 230 + 200)/4 = 215$.

$v(\{\text{garden}\})$: set garden to yes, keep their sizes (80, 120, 90, 110): $(150 + 230 + 150 + 230)/4 = 190$.

$v(\{\text{size}, \text{garden}\}) = 230$, the prediction itself.

$\phi_{\text{size}} = \frac12(215 - 182.5) + \frac12(230 - 190) = 16.25 + 20 = 36.25$.

$\phi_{\text{garden}} = \frac12(190 - 182.5) + \frac12(230 - 215) = 3.75 + 7.5 = 11.25$.

Check additivity: $182.5 + 36.25 + 11.25 = 230$. The garden's 30-point interaction bonus was split fairly between the two features that create it.

</div>

With 50 features there are about $10^{15}$ subsets, so the formula cannot be used directly. **TreeSHAP** (Lundberg et al.) exploits the tree structure to compute exact values in polynomial time, $O(TLD^2)$ for $T$ trees with at most $L$ leaves and depth $D$. That makes it fast enough for forests, XGBoost and LightGBM.

~~~dg-fig 04.06-shap.svg
A SHAP waterfall for one validation house. Starting from the average prediction (200), the small size subtracts 87 and the long distance 37; being young and having a garden add some back, ending at this house's prediction (110).
~~~

Averaging $|\phi_j|$ over many rows gives a global ranking; plotting feature value against $\phi_j$ (a **dependence plot**) is a per-row alternative to PD.

## Interpreting tree models in code

Everything above, on the house data from the first figure:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, partial_dependence
import shap

rng = np.random.default_rng(0); n = 1000
size = rng.uniform(40, 160, n); age = rng.uniform(0, 60, n); dist = rng.uniform(0.5, 20, n)
garden = (rng.random(n) < 0.4).astype(float); listing_id = rng.permutation(n).astype(float)
price = (50 + 2.2*size - 1.2*age - 6*dist + 25*garden + 0.25*size*garden
         + rng.normal(0, 80, n))                     # thousand; noisy on purpose
X = np.c_[size, age, dist, garden, listing_id]
names = ["size_m2", "age_years", "dist_km", "has_garden", "listing_id"]
X_tr, X_va, y_tr, y_va = train_test_split(X, price, test_size=0.3, random_state=0)

rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_tr, y_tr)
print(round(rf.score(X_tr, y_tr), 2), round(rf.score(X_va, y_va), 2))   # 0.92 0.53

# 1. impurity importance (training data, free)
print(dict(zip(names, rf.feature_importances_.round(3).tolist())))
# {'size_m2': 0.517, 'age_years': 0.16, 'dist_km': 0.174, 'has_garden': 0.038, 'listing_id': 0.112}

# 2. permutation importance (validation data)
res = permutation_importance(rf, X_va, y_va, n_repeats=10, random_state=0)
for j in res.importances_mean.argsort()[::-1]:
    print(f"{names[j]:11s} {res.importances_mean[j]:.3f} +- {res.importances_std[j]:.3f}")
# size_m2 0.845 +- 0.082 | dist_km 0.179 +- 0.027 | age_years 0.072 +- 0.021
# has_garden 0.067 +- 0.011 | listing_id 0.009 +- 0.010

# 3. partial dependence (average of ICE lines) for size
pd_ = partial_dependence(rf, X_va, features=[0], kind="both", grid_resolution=5)
print(pd_["grid_values"][0].round(0), pd_["average"][0].round(0))
# [ 49.  76. 102. 129. 156.] [ 85. 162. 210. 254. 338.]

# 4. SHAP values for one house (TreeSHAP)
ex = shap.TreeExplainer(rf)
phi = ex.shap_values(X_va[:1])[0]                   # one value per feature
base = float(np.ravel(ex.expected_value)[0])        # average prediction
print(round(base, 1), phi.round(1), round(base + phi.sum(), 1), rf.predict(X_va[:1]).round(1))
# 199.9 [-115.6   29.1   -8.    12.6   11.2] 129.2 [129.2]
```

Base plus SHAP values equals the prediction exactly. `listing_id` gets $+11.2$ for this house: not a bug, the forest really used the ID here, and SHAP reports what the model did; permutation importance says the habit does not help on new data. `PartialDependenceDisplay.from_estimator(rf, X_va, [0], kind="both")` draws PD with ICE; for trees, `method="recursion"` computes the average PD from the tree structure, but not ICE.

## Reading the result

A checklist before you report an interpretation:

- **Is the model any good?** Explaining a model with validation $R^2$ near zero explains noise.
- **Do MDI and permutation agree?** MDI high but permutation low usually means high-cardinality junk.
- **Are the error bars small?** Importance within two standard deviations of zero means unimportant.
- **Check correlations** (matrix or cluster dendrogram) before reading any single-feature number.
- **Do ICE lines run parallel to the PD line?** Crossing or fanning lines mean an interaction.
- **Is the PD supported by data?** Draw a rug of the feature and distrust sparse regions.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Trusting MDI alone | IDs, timestamps or noise columns rank high | permutation importance on held-out data
| Permutation on training data | memorised junk looks important again | always use a validation or test set
| Correlated features scored one at a time | both members of a pair look weak | permute groups, or keep one per cluster
| PD without ICE | flat PD for a feature with high importance | plot ICE lines, colour by suspect interacting feature
| PD in empty regions | wild wiggles at the ends of the curve | restrict the grid to percentiles 5 to 95, use ALE
| Reading importance as causation | "lower the price of X to change Y" proposals | run an experiment; importance is about the model

The last row matters most. If `has_umbrella` predicts rain, shuffling it hurts the model, but handing out umbrellas will not make it rain.

## Going deeper

**Why MDI is biased.** Each node takes the *maximum* impurity decrease over candidate splits, and the maximum of many noisy candidates is biased upward. A feature with $m$ distinct values offers $m-1$ candidates, a binary one offers 1, so even with no real relationship high-cardinality features win more splits. Computing decreases on out-of-bag data, or conditional-inference trees (Strobl et al.), removes most of the bias.

**Conditional permutation.** Shuffle $x_j$ only within groups of rows with similar values of its correlates. Rows stay realistic, and the answer is "what does $x_j$ add *beyond* its correlates?"

**Gain is not consistent.** Lundberg et al. showed a pair of trees where a model is changed so that it depends *more* on a feature, yet that feature's gain-based importance can go *down*. Shapley values are the unique attribution satisfying additivity, a symmetry rule, a null-player rule and consistency, which is why mean $|\phi_j|$ is a safer global importance.

**Two flavours of TreeSHAP.** The default "path-dependent" version estimates $v(S)$ from how many training samples went down each branch, roughly following the data's correlations. The "interventional" version (pass background data) breaks correlations, like the hand example. With correlated features they differ; they answer slightly different questions.

## Key terms

| Term | Meaning
| --- | --- |
| MDI (impurity importance) | Sum of weighted impurity decreases from splits on a feature, from training data
| Permutation importance | Drop in held-out score when one feature's column is shuffled
| Cardinality bias | Features with many distinct values get extra MDI credit by chance
| Partial dependence (PD) | Average prediction when one feature is forced to each value
| ICE curve | One row's prediction as one feature is varied; PD is their average
| ALE plot | Effect curve built from local differences within real data, robust to correlation
| SHAP value | A feature's fair share of the gap between one prediction and the average prediction
| TreeSHAP | Exact polynomial-time algorithm for SHAP values of tree ensembles

<div class="take">
<div class="lab">Key takeaway</div>

ask which question you are answering, then use permutation importance on held-out data for "which features", PD with ICE lines for "how", and SHAP for "why this row"; check correlated features before trusting any single number, and remember all of them describe the model, not cause and effect in the world.

</div>
