---
id: "07.04"
course: "machine-learning"
module: "07"
slug: "feature-selection"
title: "Feature selection"
keyIdeas: ["Feature selection is worth doing for cost, latency, robustness and interpretability as much as for accuracy; a well-regularised model often loses little from extra weak features, so measure before adding it.", "**Filter** methods score each feature alone (variance, correlation, F-test, chi-squared, mutual information). They are fast, but blind to interactions (the XOR pattern) and to redundancy (near-copies all score high).", "**Wrapper** methods (forward selection, backward elimination, recursive feature elimination) search subsets by retraining the model. They see interactions but cost many fits and can overfit the validation folds.", "**Embedded** methods select while training: an L1 penalty drives weights to exactly zero, and tree models rank features. Correlated features share or steal importance, so cluster them before pruning and prefer permutation importance on validation data.", "Selection is part of training. Choosing features on all rows and then cross-validating can report near-perfect accuracy on pure noise; put the selector in a pipeline so it is repeated inside every fold."]
question: "A genomics team has 120 patients and 20,000 gene-expression features. They pick the 50 genes with the largest t-statistics between cases and controls using all 120 patients, train an SVM on those 50 genes, and report 10-fold CV accuracy of 94%. Explain why the estimate is unreliable, what an honest protocol looks like, and what else you would report about the selected genes."
answer: "The 50 genes were chosen using the labels of all 120 patients, including those later used as validation folds. With 20,000 candidates and 120 samples, many genes separate these particular labels by chance (the largest of 20,000 noise correlations over 120 samples is around $\\sqrt{2 \\ln 20000 / 120} \\approx 0.41$), so the selection step has already fitted the validation labels. CV then only evaluates the SVM on a feature set tailored to the data it is scored on, and 94% can arise even from pure noise. Honest protocol: make selection part of the model. Inside each of the 10 folds, compute t-statistics on the 108 training patients only, select 50 genes, train the SVM, and score the 12 held-out patients. If the number of genes or the SVM's C is also tuned, wrap that in nested CV so the outer folds never influence those choices. A quick sanity check is to rerun the whole protocol with shuffled labels; it should give about 50%. Report the mean and spread of fold accuracies, and report the stability of the selected genes: how often each gene is chosen across folds or bootstrap resamples. An unstable list means the specific genes should not be over-interpreted even if accuracy holds up, and correlated genes may substitute for each other, so a gene's absence from the list does not mean it is irrelevant."
sources: [{"title": "An introduction to variable and feature selection (Guyon and Elisseeff, JMLR 2003)", "url": "https://www.jmlr.org/papers/v3/guyon03a.html", "note": "The standard survey of filter, wrapper and embedded methods, with a practical checklist."}, {"title": "scikit-learn: Feature selection", "url": "https://scikit-learn.org/stable/modules/feature_selection.html", "note": "Variance threshold, univariate selection, RFE, SelectFromModel and sequential selection."}, {"title": "The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free PDF; section 7.10.2 is the wrong-and-right-way-to-cross-validate demonstration."}, {"title": "scikit-learn: Permutation importance with multicollinear features", "url": "https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html", "note": "Why correlated features look unimportant and how clustering them fixes it."}]
practice: ["permutation-feature-importance", "anova-f-statistic", "pearson-correlation-matrix", "lasso-regression"]
tags: ["feature selection", "selection bias"]
tools: ["scikit-learn"]
---

## What you will learn

- Why you would remove features at all, and why accuracy is often not the main reason.
- The three families of selection methods (filter, wrapper, embedded): how each one works, what it costs and what it misses.
- How to score a feature by hand with a correlation, and why a feature that scores zero alone can still be essential.
- How an L1 penalty and permutation importance pick features, and why correlated features confuse both.
- The most expensive mistake in this topic: selecting features before cross-validation. You will see it report near-perfect accuracy on pure noise.

*Before you start:* you need cross-validation and the idea of overfitting the validation set (module 06, especially 06.05), lasso and the L1 penalty (02.02), and permutation importance (04.06). The previous lessons in this module (07.01 to 07.03) covered how to make each column usable; this one decides which columns to keep.

## From fixing columns to choosing them

Lessons 07.01 to 07.03 took each column and made it model-ready: scaled, encoded, imputed. After that work a typical table has dozens or thousands of usable columns. **Feature selection** is the step that decides which of them the model actually gets to see.

Every column you keep must be computed, monitored and explained in production; every column you drop is information the model never sees. Selection is the trade between the two.

## The idea in one picture

A subscription business wants to predict which of its 500 customers will cancel next month (**churn**; 35% of them did). The data team has eight candidate columns. The quickest way to rank them is to score each one on its own: compute the correlation between the column and the 0/1 churn label, and look at its size.

| Column | abs. correlation with churn | What it is
| --- | --- | --- |
| support_calls | 0.41 | calls to support last quarter
| tenure_months | 0.39 | months as a customer
| tenure_weeks | 0.39 | the same tenure, measured in weeks
| monthly_bill | 0.12 | current monthly price
| discount_pct | 0.08 | discount on the plan
| user_id_hash | 0.07 | a random identifier
| signup_weekday | 0.03 | day of the week the customer signed up
| country | 0.00 | every customer is in "DE"

~~~dg-fig 07.04-filter.svg
Scoring each column alone. Three columns carry real signal, the constant and random columns score near zero, but the duplicate tenure_weeks scores exactly as high as the column it copies.
~~~

Reading the picture:

- The **constant** column (country) scores exactly zero. It cannot help any model, so dropping it is free.
- The random columns (user_id_hash, signup_weekday) score near zero, but not exactly zero. With finite data, noise always correlates a little with the label by chance. Remember this; it becomes the main danger later.
- tenure_weeks scores as high as tenure_months because it *is* tenure_months. Keeping the top three by score would keep two copies of the same fact and miss monthly_bill. Scoring one column at a time cannot see **redundancy**.

Simple scores are fast and useful, but they judge each column in isolation.

## Step 1: decide whether selection is worth it

There are four reasons to drop features, and only one of them is accuracy.

- **Accuracy.** With many columns per row, noise columns let the model fit chance patterns.
- **Cost and latency.** Each feature is a pipeline to compute and keep fresh.
- **Robustness.** Each feature can break, drift or arrive late.
- **Interpretability.** 12 inputs can be explained to a stakeholder; 1,200 cannot.

To see the accuracy effect, take 200 rows with 200 columns, of which only 8 carry signal. Keep the top $k$ columns by a simple score, fit a logistic regression, and measure accuracy with cross-validation for each $k$.

~~~dg-fig 07.04-curve.svg
Training accuracy only goes up as columns are added. Cross-validated accuracy peaks near the number of truly useful columns (here 6 to 8) and then slides from 0.99 to 0.93 as noise columns pour in.
~~~

Training accuracy climbs to 100% because each extra column is one more way to fit the training labels. The drop in held-out accuracy is real but not dramatic, and that is typical. A well-**regularised** model (ridge, boosted trees with sensible settings) tolerates weak features quite well. So measure whether selection helps before adding it to the pipeline, and weigh the cost and robustness reasons separately.

The three families of methods differ in how much they involve the model:

~~~dg-layers Three families of feature selection, from cheapest to most model-aware.
Filter | score each feature alone against the target; model-free, fast
Wrapper | search subsets by retraining the model; slow, captures interactions
*Embedded | the model selects while training: L1 weights, tree importances
~~~

## Step 2: filter methods score one column at a time

A **filter** computes a statistic for each column against the target and keeps the top $k$, or every column above a threshold. No model is trained, so filters handle thousands of columns in seconds. The common scores:

- **Variance threshold**: drop constant and near-constant columns. It ignores the label and is almost always worth doing.
- **Correlation** with a numeric target: Pearson for straight-line relationships, Spearman (correlation of ranks) for any monotone one.
- **ANOVA F-test**: for a numeric feature and a class label. It compares the gap between class averages with the spread inside each class. For two classes, F is the square of the t-statistic.
- **Chi-squared**: for non-negative counts (word counts, clicks) against a class label.
- **Mutual information (MI)**: how much knowing the feature reduces uncertainty about the label. It is zero only under independence, so it catches any dependence, but its estimate is noisier.

The Pearson correlation between a feature $x$ and a target $y$ over $n$ rows is

$$r = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sqrt{\sum_i (x_i - \bar x)^2}\,\sqrt{\sum_i (y_i - \bar y)^2}},$$

where $\bar x$ and $\bar y$ are the averages. The top asks "do the two move together?", and the bottom rescales so that $r$ always lies between $-1$ and $+1$. A filter usually ranks by $|r|$, because a strong negative relationship is just as useful as a positive one.

<div class="ex">
<div class="lab">Worked example</div>

**scoring two columns by hand**

Five customers. Churn label $y = (0, 0, 1, 1, 1)$, so $\bar y = 0.6$ and $y - \bar y = (-0.6, -0.6, 0.4, 0.4, 0.4)$. Then $\sum (y - \bar y)^2 = 0.36 + 0.36 + 0.16 + 0.16 + 0.16 = 1.2$.

Support calls $x = (0, 1, 1, 3, 5)$: $\bar x = 10/5 = 2$, deviations $(-2, -1, -1, 1, 3)$, $\sum (x - \bar x)^2 = 4 + 1 + 1 + 1 + 9 = 16$.

Cross term: $(-2)(-0.6) + (-1)(-0.6) + (-1)(0.4) + (1)(0.4) + (3)(0.4) = 1.2 + 0.6 - 0.4 + 0.4 + 1.2 = 3.0$.

So $r = 3.0 / \sqrt{16 \times 1.2} = 3.0 / \sqrt{19.2} = 3.0 / 4.382 = 0.68$. Customers who call support more often churn more.

Signup weekday $z = (2, 4, 5, 1, 3)$: $\bar z = 3$, deviations $(-1, 1, 2, -2, 0)$. Cross term: $0.6 - 0.6 + 0.8 - 0.8 + 0 = 0$, so $r = 0$. A filter with any positive threshold keeps support calls and drops weekday.

</div>

Filters have two blind spots, and both come from looking at one column at a time.

**Redundancy.** Ten near-copies of one useful feature all score high, so the top $k$ can be nothing but copies, as with tenure_weeks above.

**Interactions.** A feature can be useless alone and essential together with another. The classic case is the **XOR pattern**: the label depends on whether two features have the same sign.

~~~dg-fig 07.04-xor.svg
Each feature alone has a correlation with the label close to zero, so any filter would drop both. Together they split the classes perfectly by quadrant.
~~~

Real versions are milder but common: a loan amount matters relative to income. Build the combined feature yourself (amount / income) or use a method that trains a model.

## Step 3: wrapper methods search subsets with the model

A **wrapper** treats the set of features as a hyperparameter and tries candidate subsets by actually training and validating the model.

- **Forward selection** starts with no features. In each round it tries adding each remaining feature, keeps the one that most improves the cross-validated score, and repeats.
- **Backward elimination** starts with all features and removes, in each round, the one whose removal hurts least.
- **Recursive feature elimination (RFE)** trains once, drops the features with the smallest weights or importances, retrains on the rest, and repeats. It is cheaper because it ranks with the model's own numbers instead of trying every candidate.

~~~dg-flow Forward selection: grow the set one feature at a time, judged by cross-validation.
Start: no features -(try each remaining feature)-> Cross-validate each candidate -(keep the best one)-> *Add it to the set -(score still improving?)-> Repeat or stop
~~~

Because the model is retrained, wrappers see interactions and redundancy: once tenure_months is in, tenure_weeks adds nothing and is never chosen. The price is compute: with $p$ features, a full forward pass costs about $p + (p-1) + \dots + 1 \approx p^2/2$ fits, each times the number of folds. And because many subsets are tried on the same folds, the winner partly reflects luck, as in a large hyperparameter search (06.05). Confirm its score on data the search never touched.

## Step 4: embedded methods select while training

**Embedded** methods get selection as a by-product of fitting the model once.

**L1 regularisation.** Lasso (and L1 logistic regression) adds $\lambda \sum_j |w_j|$ to the loss, where $w_j$ is the weight of feature $j$ and $\lambda$ is the penalty strength. Lesson 02.02 showed why the corners of the L1 diamond push many weights to *exactly* zero. A feature with weight zero is a feature the model does not use, so $\lambda$ becomes a dial for how many features survive.

~~~dg-fig 07.04-lasso.svg
The lasso path on scikit-learn's diabetes data (10 standardised features). With a strong penalty (left) only bmi and s5 are non-zero; as the penalty weakens, features enter one by one: 4 are kept at λ = 10, 7 at λ = 1, 9 at λ = 0.1.
~~~

At $\lambda \approx 30$ only body-mass index and the blood-serum measure s5 are used; by $\lambda = 10$, blood pressure and s3 have joined. Pick $\lambda$ by cross-validation (`LassoCV`), or pick the largest $\lambda$ whose score is within noise of the best if you want fewer features.

One quirk: with a group of strongly correlated features, lasso tends to keep one, more or less arbitrarily, and zero the rest. A different sample can pick a different member. **Elastic net** (a mix of L1 and L2 penalties) keeps correlated groups together.

**Tree importances.** Tree ensembles report an **impurity-based importance** for free: how much each feature's splits reduced the loss. It is biased towards continuous and high-cardinality features, because they offer more possible split points to exploit noise. **Permutation importance** on validation data (04.06) is more faithful: shuffle one column, and measure how much the validation score drops. If the score does not drop, the model was not relying on that column.

**Correlated features confuse both kinds of importance.** In the next figure, a random forest predicts a label from income and savings. Alone, income has a permutation importance of 0.28 (shuffling it costs 28 points of accuracy). Then a near-copy of income (correlation 0.99) is added. Shuffling either copy now costs only about 0.1, because the forest simply reads the other one.

~~~dg-fig 07.04-corr.svg
Adding a near-copy of income splits its importance between the two copies, so each looks only moderately important. Yet dropping both costs 0.22 accuracy.
~~~

"Drop low-importance features" could remove both copies and lose 22 points. Group first: cluster features by correlation (for example, hierarchical clustering on Spearman correlation), keep one representative per cluster, and only then compute importance.

## Step 5: selection must happen inside the folds

Selection looks at the labels. Anything that looks at labels is part of training. If you select features using all rows and *then* cross-validate, the labels of every validation fold have already helped choose the features, and the CV score is no longer honest. *The Elements of Statistical Learning* (section 7.10.2, "The wrong and right way to do cross-validation") gives the canonical demonstration.

<div class="ex">
<div class="lab">Worked example</div>

**near-perfect accuracy on pure noise**

The setting: 50 samples, two balanced classes, 5,000 features of pure random noise. No model can beat 50% error on new data, because there is nothing to learn.

Wrong way: using all 50 rows, pick the 100 features most correlated with the label. Then run 5-fold CV of a 1-nearest-neighbour classifier on those 100 features. Among 5,000 noise columns, many correlate with *these particular* labels by chance, so the folds look easy. ESL reports a CV error of about 3%; our 30 repeats in the figure average 1%.

Right way: inside each fold, select the top 100 using only the 40 training rows, then score the 10 held-out rows. The CV error averages 50%, the truth.

The only difference between the two runs is whether the validation labels took part in choosing the features.

</div>

~~~dg-fig 07.04-leak.svg
Thirty repeats of the pure-noise experiment. Selecting on all rows first reports errors between 0% and 4%; selecting inside each training fold scatters around the true 50%.
~~~

Every step that looks at labels (selection, target encoding, resampling, tuning) belongs inside the training fold. A scikit-learn `Pipeline` makes this automatic, because `cross_val_score` refits the whole pipeline on each fold's training rows (lesson 07.05).

## Feature selection in code

The data below is the 200-row, 200-column set from the curve figure, plus a pure-noise set for the leak.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = make_classification(n_samples=200, n_features=200, n_informative=8, n_redundant=0,
                           n_clusters_per_class=1, class_sep=1.5, flip_y=0.02,
                           shuffle=False, random_state=2)   # columns 0-7 carry the signal
cv = StratifiedKFold(5, shuffle=True, random_state=0)
clf = LogisticRegression(max_iter=5000)

# Filter: top 8 columns by F-test, refit inside every fold
filt = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=8), clf)
print(cross_val_score(filt, X, y, cv=cv).mean())          # 0.98

# Embedded: L1 logistic regression chooses columns, a plain model is fit on them
# (scikit-learn < 1.8: write penalty="l1" instead of l1_ratio=1.0)
emb = make_pipeline(VarianceThreshold(1e-6), StandardScaler(),
                    SelectFromModel(LogisticRegression(l1_ratio=1.0, C=0.1, solver="liblinear")),
                    clf)
print(cross_val_score(emb, X, y, cv=cv).mean())           # 0.99
emb.fit(X, y)
kept = emb[2].get_support()                               # mask over the 200 columns
print(kept.sum(), np.flatnonzero(kept))                   # 4 [0 2 5 7]

# No selection at all
print(cross_val_score(make_pipeline(StandardScaler(), clf), X, y, cv=cv).mean())  # 0.93

# The leak, on pure noise (60 rows, 2,000 columns, random labels)
rng = np.random.default_rng(0)
Xn, yn = rng.normal(size=(60, 2000)), rng.integers(0, 2, 60)
top = SelectKBest(f_classif, k=20).fit(Xn, yn).get_support()       # uses ALL labels
print(cross_val_score(clf, Xn[:, top], yn, cv=cv).mean())           # 0.95  (wrong)
print(cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), clf),
                      Xn, yn, cv=cv).mean())                         # 0.48  (right)
```

Notice that the L1 selector kept only 4 of the 8 real columns and still scored 0.99: the selected set does not have to be "the true features" to predict well. For wrappers, `RFECV` and `SequentialFeatureSelector` drop into the same pipeline slot.

## Reading the result

A selection is only as trustworthy as these checks:

- **Score inside CV, with its spread.** Compare the pipeline with and without selection. A gain smaller than the fold-to-fold spread is no gain.
- **Stability.** Record which features each fold or bootstrap resample selects. Chosen in 10 of 10 folds is solid; a list that changes every fold should not be over-interpreted.
- **Suspiciously strong single features.** Near-perfect single-feature AUC usually means a leak (an identifier, a field filled in after the outcome).
- **The production bill.** Note what each kept feature costs to compute.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Selecting on all rows, then cross-validating | CV score far above test score; great results even on shuffled labels | put the selector in a pipeline; nested CV if $k$ or $\lambda$ is tuned
| Top-$k$ filter on redundant columns | selected set is several copies of one signal | remove near-duplicates or cluster first
| Filter drops an interaction | model without the pair underperforms a model with it | build the combined feature, or use a wrapper or tree model
| Dropping low-importance correlated features | each copy looks weak; removing all of them hurts badly | cluster by correlation, keep one per cluster
| Trusting impurity importance | IDs and continuous noise rank high | use permutation importance on validation data
| Wrapper search over many subsets | best subset's CV score not reproduced on fresh data | hold out a final test set; prefer cheap methods
| Selecting when the model is already well regularised | extra complexity, no measurable gain | compare with and without selection first

## A practical recipe

1. Remove constant, duplicate and leaky columns.
2. Cluster highly correlated features; keep one per cluster.
3. Fit a strong embedded model and compute permutation importance on validation data, across folds.
4. Drop features consistently near zero, confirm with CV that the score holds, and weigh the rest against production cost.

## Going deeper

**Why noise wins on wide data.** If a feature is pure noise, its sample correlation with the label over $n$ rows is roughly normal with standard deviation $1/\sqrt{n}$. The largest of $p$ such values is about $\sqrt{2 \ln p / n}$. For the ESL setting, $\sqrt{2 \ln 5000 / 50} = \sqrt{17.0/50} \approx 0.58$: the best of 5,000 noise columns looks like a strong predictor. That is why selection bias grows with the number of candidates and shrinks with the number of rows.

**Exact subset search is hopeless.** There are $2^p$ subsets of $p$ features; for $p = 50$ that is about $10^{15}$. Best-subset selection for least squares is NP-hard in general. Forward, backward and RFE are greedy shortcuts, and lasso is the convex relaxation: it replaces "count the non-zero weights" (the $L_0$ norm) with "add up their sizes" (the $L_1$ norm).

**When does lasso find the right features?** Recovering exactly the true set requires the useful and useless features not to be too correlated (the *irrepresentable condition*) and the true weights to be large enough relative to noise. With correlated features this fails, which is the group-picking quirk seen above. **Stability selection** reruns lasso on many half-samples and keeps features chosen in a large fraction of runs, which controls false selections better than a single fit.

**Relevance is not one thing.** Guyon and Elisseeff separate *strongly relevant* features (removing them always loses information), *weakly relevant* ones (a substitute exists, like the two incomes) and irrelevant ones. Methods such as mRMR (minimum redundancy, maximum relevance) penalise overlap with features already chosen.

**Selection is not causation.** A selected feature is predictive here; it may be a proxy for the real cause. Do not read the list as "the drivers".

## Key terms

| Term | Meaning
| --- | --- |
| Feature selection | Choosing which columns the model may use
| Filter method | Scores each feature alone against the target (correlation, F-test, chi-squared, MI)
| Wrapper method | Searches feature subsets by retraining and validating the model
| Embedded method | Selection that happens during training, such as L1 weights or tree importances
| Recursive feature elimination | Train, drop the weakest features, retrain, repeat
| Mutual information | How much a feature reduces uncertainty about the label; zero only under independence
| Permutation importance | Drop in validation score when one feature's values are shuffled
| Redundancy | Two or more features carrying the same information
| Selection bias | Optimistic scores caused by choosing features with labels that are later used for evaluation
| Selection stability | How consistently the same features are chosen across folds or resamples

<div class="take">
<div class="lab">Key takeaway</div>

select features for cost and robustness as much as accuracy, prefer embedded methods checked with permutation importance, cluster correlated features before pruning, and always run selection inside the training folds so validation labels never help choose the features.

</div>
