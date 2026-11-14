---
id: "07.04"
course: "machine-learning"
module: "07"
slug: "feature-selection"
title: "Feature selection"
keyIdeas: ["Selection is worth doing for cost, latency, robustness and interpretability as much as for accuracy; a well-regularised model often loses little from extra weak features.", "**Filter** methods score each feature alone (correlation, mutual information, F-test, chi-squared): fast, but blind to interactions and redundancy.", "**Wrapper** methods (forward selection, recursive feature elimination) search subsets by retraining the model; they are expensive and can overfit the validation data.", "**Embedded** methods select during training: L1 penalties drive weights to zero, trees rank features by split gain. Correlated features share or steal importance, so cluster them before pruning.", "Selection is part of training. Choosing features on the full dataset and then cross-validating gives wildly optimistic scores; selection must be repeated inside each fold."]
question: "A genomics team has 120 patients and 20,000 gene-expression features. They pick the 50 genes with the largest t-statistics between cases and controls using all 120 patients, train an SVM on those 50 genes, and report 10-fold CV accuracy of 94%. Explain why the estimate is unreliable, what an honest protocol looks like, and what else you would report about the selected genes."
answer: "The 50 genes were chosen using the labels of all 120 patients, including those later used as validation folds. With 20,000 candidates and 120 samples, many genes separate these particular labels by chance, so the selection step has already fitted the validation labels; CV then only evaluates the SVM on a feature set tailored to the data it is scored on, and 94% can arise even from pure noise. Honest protocol: make selection part of the model. Inside each of the 10 folds, compute t-statistics on the training 108 patients only, select 50 genes, train the SVM, score the held-out 12. If the number of genes or the SVM's C is also tuned, wrap that in nested CV. Report the mean and spread of fold accuracies, and report the stability of the selected genes: how often each gene is chosen across folds (or bootstrap resamples). An unstable list means the specific genes should not be over-interpreted even if accuracy holds up."
sources: [{"title": "An introduction to variable and feature selection (Guyon and Elisseeff, JMLR 2003)", "url": "https://www.jmlr.org/papers/v3/guyon03a.html", "note": "The standard survey of filter, wrapper and embedded methods, with a practical checklist."}, {"title": "scikit-learn: Feature selection", "url": "https://scikit-learn.org/stable/modules/feature_selection.html", "note": "Variance threshold, univariate selection, RFE, SelectFromModel and sequential selection."}, {"title": "The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free PDF; section 7.10.2 is the wrong-and-right-way-to-cross-validate demonstration."}, {"title": "scikit-learn: Permutation importance with multicollinear features", "url": "https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html", "note": "Why correlated features look unimportant and how clustering them fixes it."}]
practice: ["permutation-feature-importance", "anova-f-statistic", "pearson-correlation-matrix", "lasso-regression"]
tags: ["feature selection", "selection bias"]
tools: ["scikit-learn"]
---

## Why select at all

Removing features can improve accuracy when many are pure noise relative to the number of rows, but that is not the only, or even the main, reason. Each feature in production is a dependency: a pipeline to compute it, data to keep fresh, a column that can break, drift or arrive late. Fewer features mean cheaper serving, fewer failure modes, faster training and a model that is easier to explain. Strongly regularised models (ridge, boosted trees with sensible settings) are quite tolerant of weak features, so measure whether selection helps before adding it to the pipeline.

The three families differ in how much they involve the model.

~~~dg-layers Three families of feature selection, from cheapest to most model-aware.
Filter | score each feature alone against the target; model-free, fast
Wrapper | search subsets by retraining the model; slow, captures interactions
*Embedded | the model selects while training: L1 weights, tree importances
~~~

## Filter methods

A filter ranks features with a univariate statistic and keeps the top $k$ or those above a threshold.

- **Variance threshold**: drop constant and near-constant columns. Almost always worth doing.
- **Correlation** with a numeric target (Pearson for linear relationships, Spearman for monotone ones).
- **ANOVA F-test**: for a numeric feature and a class label, compares between-class to within-class variance.
- **Chi-squared**: for non-negative count features against a class label.
- **Mutual information**: captures any dependence, linear or not, at the cost of a noisier estimate.

Filters are cheap and model-agnostic, which makes them a good first pass on thousands of columns. Their blind spots: a feature that matters only in combination with another (the XOR pattern, or price that matters only relative to income) scores zero alone, and ten nearly identical copies of one useful feature all score high, so the top $k$ can be all redundant.

## Wrapper methods

Wrappers treat the subset as a hyperparameter. **Forward selection** starts empty and adds the feature that most improves the CV score; **backward elimination** starts full and removes the least useful. **Recursive feature elimination** (RFE) trains the model, drops the features with the smallest coefficients or importances, and repeats. They account for interactions and redundancy because the model is retrained, but they need on the order of $p^2$ fits for sequential search over $p$ features, and searching over many subsets overfits the validation folds the same way a large hyperparameter search does (06.05).

## Embedded methods

- **L1 regularisation** (lasso, L1 logistic regression) sets many coefficients exactly to zero; the strength $\lambda$ controls how many survive. With groups of correlated features, lasso tends to pick one arbitrarily and drop the others; elastic net keeps groups together. Lesson 02.02 covers the geometry.
- **Tree importances**: impurity-based importance is free but biased towards high-cardinality and continuous features. **Permutation importance** on validation data (lesson 04.06) measures how much the score drops when a feature is shuffled, which is more faithful.
- **Correlated features** confuse both: permuting one of two near-duplicates hurts little because the model reads the other, so both look unimportant, and dropping both would hurt a lot. Cluster features by correlation (for example, hierarchical clustering on Spearman correlation), keep one per cluster, then evaluate importance.

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(VarianceThreshold(1e-6), StandardScaler(),
                     SelectFromModel(LogisticRegression(penalty="l1", C=0.1, solver="liblinear")),
                     LogisticRegression(max_iter=1000))
# Selection happens inside fit, so cross_val_score(pipe, X, y) repeats it in every fold.
```

## Selection bias: the classic mistake

Selecting features using all rows and *then* cross-validating the model on those features leaks the labels of the validation folds into the choice of features. With many candidate features and few rows the effect is dramatic. *The Elements of Statistical Learning* (section 7.10.2, "The wrong and right way to do cross-validation") gives the canonical demonstration.

<div class="ex">
<div class="lab">Worked example</div>

**perfect accuracy on pure noise**

The setting from ESL: 50 samples, two balanced classes, 5,000 features of pure noise, so no model can beat 50% error on new data. Wrong way: pick the 100 features most correlated with the label using all 50 rows, then run 5-fold CV of a 1-nearest-neighbour classifier on those 100. Among 5,000 noise columns, plenty correlate with these particular labels by chance, and CV reports an error of about 3%.

Right way: inside each fold, select the top 100 using only the 40 training rows, then score the 10 held-out rows. CV error rises to about 50%, the truth. The only difference is whether the validation labels took part in selection.

</div>

The rule generalises: any step that looks at labels (selection, target encoding, resampling, tuning) belongs inside the training fold. Putting the selector in a pipeline, as in the code above, makes that automatic.

## A practical recipe

1. Remove constant, duplicate and leaky columns (identifiers, post-outcome fields, anything with suspiciously high single-feature AUC).
2. Cluster highly correlated features and keep a representative of each cluster.
3. Fit a strong embedded model and compute permutation importance on validation data, ideally with repeats and across folds to see how stable the ranking is.
4. Drop features that are consistently near zero and confirm with CV that the score does not fall by more than its noise. Weigh what is left against the cost of computing each feature in production.

<div class="take">
<div class="lab">Key takeaway</div>

select features for robustness and cost as much as accuracy, prefer embedded methods with permutation checks, handle correlated groups explicitly, and always run selection inside the training folds.

</div>
