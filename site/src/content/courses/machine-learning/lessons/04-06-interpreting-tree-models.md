---
id: "04.06"
course: "machine-learning"
module: "04"
slug: "interpreting-tree-models"
title: "Interpreting tree models"
keyIdeas: ["Impurity-based importance (MDI) sums each feature's weighted impurity decrease over all splits. It is free, but computed on training data and biased towards continuous and high-cardinality features.", "Permutation importance shuffles one column of held-out data and measures the drop in score. It is model-agnostic and reflects generalisation, but correlated features share or hide importance.", "Partial dependence averages predictions while forcing one feature to each value: it shows the average effect, but extrapolates into impossible feature combinations when features are correlated.", "ICE curves show the per-sample lines behind a PD plot and reveal interactions that averaging hides; SHAP (TreeSHAP) gives exact per-prediction attributions for tree ensembles.", "Importance measures describe the model, not the world: an important feature is one the model uses, which is not evidence of causation."]
question: "A random forest predicting house prices ranks `listing_id` (a unique integer) second by `feature_importances_`, while `sqft_living` and `sqft_above` (correlation 0.88) both rank low by permutation importance. Explain both observations and what you would report."
answer: "`listing_id` ranks high under MDI because it is a continuous-looking feature with a distinct value per row. Deep trees can always find thresholds on it that reduce training impurity, effectively memorising individual houses, and MDI credits those splits even though they do not generalise. Its permutation importance on held-out data should be near zero, and I would drop it. The two square-footage features both look weak under permutation because they carry largely the same information: shuffling one leaves the other available, so the score barely drops. Shuffling also creates implausible pairs, such as a large living area with a tiny above-ground area. I would report importance for the group: permute both columns together, or cluster correlated features and keep one representative. I would state clearly that the size signal is strong, but that splitting it between the two features is not well defined."
sources: [{"title": "scikit-learn User Guide: Permutation feature importance", "url": "https://scikit-learn.org/stable/modules/permutation_importance.html", "note": "Algorithm, its relation to MDI, and the correlated-feature caveat."}, {"title": "scikit-learn User Guide: Partial dependence and ICE plots", "url": "https://scikit-learn.org/stable/modules/partial_dependence.html", "note": "Definitions, the brute vs recursion methods for trees, and interpretation limits."}, {"title": "Strobl et al. (2007): Bias in random forest variable importance measures", "url": "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25", "note": "Shows MDI's preference for features with many categories or values."}, {"title": "Lundberg, Erion, Lee: Consistent Individualized Feature Attribution for Tree Ensembles", "url": "https://arxiv.org/abs/1802.03888", "note": "The TreeSHAP algorithm and why gain-based importances can be inconsistent."}]
practice: ["permutation-feature-importance", "random-forest"]
tags: ["feature importance", "partial dependence", "SHAP"]
tools: ["scikit-learn", "SHAP"]
---

## Three questions, three tools

Trees and ensembles are often deployed where someone asks "why?". This last lesson of the module separates three questions that people tend to blur together:

- **Which features does the model rely on?** Global importance: MDI or permutation importance.
- **How does the prediction change with a feature?** Effect curves: partial dependence and ICE.
- **Why did this one prediction come out this way?** Local attribution: SHAP values.

Each tool answers only its own question, and each has a known way of misleading you.

## Impurity-based importance (MDI)

Every split in a tree reduces impurity by some amount, weighted by the fraction of samples reaching that node. Mean decrease in impurity (MDI) adds these up per feature, averages over the trees, and normalises to sum to 1. scikit-learn exposes it as `feature_importances_`.

It is free and often reasonable, but it has two structural biases:

1. **Cardinality bias**: a feature with many distinct values offers many candidate thresholds, so by chance alone it finds splits that reduce training impurity. A random numeric column or a customer ID can get substantial MDI.
2. **Training-data scores**: MDI measures how much a feature helped fit the training set, including fitting noise. Deep forests that memorise give importance to features that do not generalise.

Strobl et al. (2007) documented the bias; the practical rule is to treat MDI as a quick first look and confirm with permutation importance.

## Permutation importance

The idea: if the model relies on feature $j$, destroying the information in $j$ should hurt. Take a held-out set, record the score $s$, shuffle column $j$ (breaking its relation with the target and other features while keeping its distribution), and record the new score $s_j$. The importance is $s - s_j$, averaged over several shuffles.

~~~dg-flow Permutation importance for one feature. Repeat for each feature and several shuffles; the model is never retrained.
Validation set -(score)-> Baseline s -(shuffle column j)-> Corrupted set -(score)-> *Importance s − s_j
~~~

```python
from sklearn.inspection import permutation_importance
res = permutation_importance(model, X_val, y_val, scoring="roc_auc",
                             n_repeats=10, random_state=0, n_jobs=-1)
order = res.importances_mean.argsort()[::-1]   # report mean ± std per feature
```

Cost is $d \times$ `n_repeats` predictions over the validation set; no retraining. It works for any model and any metric.

The main failure is **correlated features**. If `height_cm` and `height_in` are both present, shuffling one leaves the other intact, so the model barely suffers and both look unimportant. Worse, shuffling one creates rows such as 190 cm alongside 60 in, which the model never saw, so the importance partly measures behaviour on impossible inputs. Remedies: cluster correlated features (for example by Spearman correlation) and permute or drop whole groups, or use drop-column importance (retrain without the feature), which is honest but costs one fit per feature.

<div class="ex">
<div class="lab">Worked example</div>

**reading permutation importances**

A churn model has validation AUC 0.84. Shuffling `tenure_months` drops it to 0.71 (importance 0.13 ± 0.01); shuffling `contract_type` gives 0.80 (0.04 ± 0.01); shuffling `account_id` gives 0.84 (0.00 ± 0.002).

MDI on the same forest ranked `account_id` third, because it has 50,000 distinct values and the deep trees used it to memorise. Permutation importance on held-out data correctly shows it carries no generalisable signal: drop it.

If `tenure_months` and `total_charges` were 0.9 correlated, their individual importances would understate the pair; permuting both together would show the joint effect.

</div>

## Partial dependence and ICE

Importance says how much; effect curves say how. The partial dependence of the prediction on a feature $x_S$ is

$$\mathrm{PD}(v) = \frac{1}{n}\sum_{i=1}^{n} f\big(x_S = v,\ x_C^{(i)}\big)$$

that is, set feature $S$ to the value $v$ in every row, keep the other features $x_C$ as they are, and average the predictions. Plotting over a grid of $v$ shows the model's average response.

An individual conditional expectation (ICE) curve is the same computation for a single row, before averaging. Plotting a few hundred ICE lines under the PD line reveals heterogeneity: if half the lines rise and half fall, the PD line is flat even though the feature matters a lot through an interaction.

Partial dependence shares the correlated-feature trap: forcing `age = 20` onto rows with `years_employed = 35` asks the model about inputs that do not exist, and trees answer such questions arbitrarily. Accumulated local effects (ALE) plots average differences only within narrow bins of the actual data and avoid most of this. Either way, a PD curve is a description of the fitted function, not of what would happen if you intervened on the feature.

## Local attributions: SHAP

Shapley values from game theory split one prediction, relative to the average prediction, into additive contributions per feature, with consistency guarantees. They are expensive in general, but TreeSHAP (Lundberg et al.) computes them exactly for tree ensembles in polynomial time, $O(T L D^2)$ for $T$ trees, $L$ leaves and depth $D$, so it is practical for XGBoost, LightGBM and forests. Averaging $\lvert\phi_j\rvert$ over rows gives a global importance that, unlike MDI, does not favour high-cardinality features by construction; a SHAP dependence plot (feature value against $\phi_j$) is a per-row alternative to PD.

| Method | Question | Data used | Main pitfall |
| --- | --- | --- | --- |
| MDI | Global reliance | Training | High-cardinality bias, training noise |
| Permutation | Global reliance | Held-out | Correlated features, off-manifold rows |
| PD / ICE | Average and per-row effect | Any | Extrapolation with correlated features |
| TreeSHAP | Per-prediction attribution | Any | Still describes the model, not causation |

<div class="take">
<div class="lab">Key takeaway</div>

prefer permutation importance on held-out data over impurity importance, check correlated features before trusting either, and read PD/ICE and SHAP as descriptions of the model's behaviour, not causal effects.

</div>
