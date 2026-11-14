---
id: "07.03"
course: "machine-learning"
module: "07"
slug: "missing-data"
title: "Missing data"
keyIdeas: ["Why a value is missing matters more than how many: **MCAR** (unrelated to anything), **MAR** (explained by observed columns) and **MNAR** (depends on the missing value itself) call for different handling.", "Dropping rows is unbiased only under MCAR and wastes data; in prediction it also fails at serving time, when you must score rows with gaps.", "Mean or median imputation is simple but shrinks variance and weakens correlations. Adding a **missingness indicator** lets the model use the fact that a value was missing, which is often predictive.", "Model-based imputers (k-NN, iterative/MICE) use other columns; gradient-boosted trees can handle NaN natively by learning which branch missing values take.", "An imputer is learned state like a scaler: fit it on training data, and never fill gaps with information from the future or from validation rows."]
question: "An insurance model uses \"years since last claim\", which is blank for customers who have never claimed (about 55% of rows). The current pipeline fills blanks with the column mean, about 4.2 years. Classify the missingness, explain what the mean fill does to the model, and propose a better treatment for (a) a logistic regression and (b) a gradient-boosted tree."
answer: "The blank is not really missing: it is structurally determined by \"never claimed\", a strongly MNAR, informative state, and those customers are likely the lowest risk. Mean filling puts 55% of customers at 4.2 years, mixing them with people who claimed about four years ago, so the model can no longer distinguish \"never claimed\" from \"claimed a while ago\"; it shrinks the feature's spread and blurs its relationship with risk. (a) For logistic regression, add an explicit indicator never_claimed, and fill the numeric column with a constant (0 or a capped large value) after deciding how the indicator and the number should interact; possibly bin years-since-claim and give \"never\" its own bin. (b) For a boosted tree, leave the blanks as NaN so the tree learns a default direction for them, or encode \"never\" as a large sentinel such as 99, which a tree can isolate with one split; an indicator is optional. In both cases, fit any imputation on training data only."
sources: [{"title": "scikit-learn: Imputation of missing values", "url": "https://scikit-learn.org/stable/modules/impute.html", "note": "Simple, k-NN and iterative imputers, missing indicators, and NaN support in estimators."}, {"title": "Flexible Imputation of Missing Data, 2nd ed. (van Buuren)", "url": "https://stefvanbuuren.name/fimd/", "note": "Free online book: chapter 1 on MCAR/MAR/MNAR and why ad hoc methods fail, chapter 2 on multiple imputation."}, {"title": "scikit-learn: Imputing missing values before building an estimator", "url": "https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html", "note": "Compares imputation strategies and indicators on real data."}, {"title": "pandas: Working with missing data", "url": "https://pandas.pydata.org/docs/user_guide/missing_data.html", "note": "NaN versus NA, sentinel replacement, and fill methods with limits."}]
practice: ["missing-value-imputation", "handle-missing-values", "replace-values", "null-handling", "boolean-masking"]
tags: ["missing data", "imputation"]
tools: ["scikit-learn", "pandas"]
---

## Missingness is information

A missing value is a fact about the data-collection process, and that process often correlates with the label. A lab test is missing because the doctor did not order it, usually because the patient looked healthy. An income field is blank because the applicant chose not to share it. A sensor reading is missing because the device was offline. Before choosing an imputation, find out *why* values are missing; ML Toolkit's pandas lessons cover the mechanics of finding and filling them.

Little and Rubin's classification gives the vocabulary:

- **MCAR** (missing completely at random): missingness is unrelated to any value, observed or not. A random subset of forms was lost. Rare in practice.
- **MAR** (missing at random): missingness depends only on observed columns. Older customers skip the app-usage survey, and age is recorded.
- **MNAR** (missing not at random): missingness depends on the unobserved value itself. High earners are more likely to leave income blank.

~~~dg-layers The three mechanisms, from easiest to hardest.
MCAR | deletion is unbiased, just wasteful; any simple imputation is fine
MAR | impute from the observed columns that explain the gaps
*MNAR | the gap carries signal: keep an indicator, model it explicitly
~~~

For inference (estimating a coefficient or a mean), the mechanism decides whether results are biased, and statisticians use multiple imputation to carry the uncertainty through. For prediction, the goal is simpler: make the model accurate on future rows that will have the same kind of gaps. That shifts the advice towards keeping the missingness pattern visible to the model.

## Deletion

**Listwise deletion** (drop any row with a gap) is unbiased under MCAR, but with 20 columns each 5% missing independently, only $0.95^{20} \approx 36\%$ of rows survive. Worse for production: you cannot refuse to score a customer because one field is empty, so a model trained only on complete rows will meet inputs it never saw. Dropping a **column** that is mostly missing and carries no signal is often fine; check its indicator's predictive value first.

## Simple imputation and indicators

- **Mean / median** for numeric columns (median is robust to skew), **most frequent** or an explicit "missing" category for categoricals, a **constant** where it has meaning (0 purchases when there is no purchase record).
- Imputing a single value shrinks the column's variance and pulls its correlation with other features towards zero; with 30% missing, a third of the rows sit at one point.
- A **missingness indicator** (a 0/1 column "was missing") restores the information that imputation hides. Under MNAR it is often the most predictive part of the feature. scikit-learn's `SimpleImputer(add_indicator=True)` adds one per column that had gaps in training.

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
simple = SimpleImputer(strategy="median", add_indicator=True)
knn = KNNImputer(n_neighbors=5)                        # scale features first: k-NN uses distances
mice = IterativeImputer(max_iter=10, random_state=0)   # regresses each column on the others, round-robin
```

## Model-based imputation

**k-NN imputation** fills a gap with the average of that column among the $k$ most similar rows (by the observed columns). **Iterative imputation** (MICE-style) models each column with gaps as a regression on the others and cycles until the fills stabilise. Both exploit correlations between features, which helps under MAR. They cost more, they are more learned state to fit and ship, and they do not fix MNAR: if the missing values are systematically different, no amount of modelling from the observed columns recovers that.

## Native handling in trees

Gradient-boosted tree libraries (XGBoost, LightGBM, scikit-learn's `HistGradientBoosting`) accept NaN directly. At each split they try sending missing values left and then right, and keep whichever reduces the loss more, so the model learns a "default direction" per split. That is an implicit indicator, learned where it matters, and is usually as good as or better than imputing. If missing values appear only at serving time for a feature that never had gaps in training, the default direction was never learned (scikit-learn sends such values to the child with more training samples), so monitor missingness rates (lesson 09.03).

## Traps

1. **Sentinel values**: -999, 0, "N/A", "unknown" or empty strings that mean missing must be converted to real missing values first, or the model learns a fake spike at -999 and your imputer never fires.
2. **Leakage through the imputer**: means computed on all data, or on validation rows, are leakage just as with scalers. Fit the imputer inside the training fold.
3. **Time series**: forward-fill within each entity (station, user) from past values, with a limit on how stale a value may be. Backward fill and interpolation across the gap use future values that would not exist at prediction time.
4. **Training-serving mismatch**: the notebook filled with the median, the serving code fills with 0. Ship the fitted imputer with the model (lesson 07.05).

<div class="ex">
<div class="lab">Worked example</div>

**a lab test missing for 40% of patients**

Predicting 30-day readmission, a lab value is missing for 40% of patients, mostly those the doctor considered low risk. Readmission is 6% among patients without the test and 15% among those with it.

Mean imputation: validation AUC 0.74; the imputed patients sit at the average lab value, which looks moderately risky, the opposite of the truth. Median plus missingness indicator: AUC 0.79, and the indicator alone is one of the top three features. Native NaN handling in a boosted model: AUC 0.80. The gap was MNAR (the decision to test depended on how sick the patient looked), so the fact of missingness was signal worth keeping.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

ask why values are missing, keep a missingness indicator (or use native NaN handling) when the gap may carry signal, fit imputers on training data only, and fill time series only from the past.

</div>
