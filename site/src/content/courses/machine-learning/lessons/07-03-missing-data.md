---
id: "07.03"
course: "machine-learning"
module: "07"
slug: "missing-data"
title: "Missing data"
keyIdeas: ["Why a value is missing matters more than how many: **MCAR** (pure accident), **MAR** (explained by observed columns) and **MNAR** (depends on the missing value itself) call for different handling.", "Dropping incomplete rows is unbiased only under MCAR and wastes data fast, since only $(1-p)^d$ of rows are complete; in prediction it also fails at serving time, when you must score rows with gaps.", "Mean or median imputation shrinks the variance by the factor $n_{obs}/n$ and weakens correlations. A **missingness indicator** keeps the fact that a value was missing, which is often one of the most predictive features.", "Model-based imputers (k-NN, iterative/MICE) use the other columns and help under MAR but cannot fix MNAR; gradient-boosted trees handle NaN natively by learning a default direction at each split.", "An imputer is learned state like a scaler: convert sentinels to NaN first, fit it on training data only (inside a Pipeline), and in time series fill only from the past."]
question: "An insurance model uses \"years since last claim\", which is blank for customers who have never claimed (about 55% of rows). The current pipeline fills blanks with the column mean, about 4.2 years. Classify the missingness, explain what the mean fill does to the model, and propose a better treatment for (a) a logistic regression and (b) a gradient-boosted tree."
answer: "The blank is not really missing: it is structurally determined by \"never claimed\", a strongly informative state that behaves like MNAR (whether the value exists depends on the claim history itself), and those customers are likely the lowest risk. Mean filling puts 55% of customers at 4.2 years, mixing them with people who claimed about four years ago, so the model can no longer tell \"never claimed\" from \"claimed a while ago\". It also creates a huge spike at 4.2, shrinks the column's variance to roughly 45% of the observed variance ($n_{obs}/n$), and blurs the feature's relationship with risk. (a) For logistic regression, add an explicit indicator never_claimed and fill the numeric column with a constant (0, or a capped large value); with the indicator present the model learns its own offset for the never-claimed group, so the fill value matters little. A good alternative is to bin years-since-claim (0-1, 1-3, 3-5, 5+) and give \"never\" its own bin, which also captures a non-linear effect. (b) For a boosted tree, leave the blanks as NaN so each split learns a default direction for them, or encode \"never\" as a large sentinel such as 99, which a tree can isolate with one split; an explicit indicator is optional. In both cases put the treatment inside a Pipeline so any statistics are fitted on training folds only, compare the options on the same validation folds, and expect the never_claimed flag to rank among the most important features."
sources: [{"title": "scikit-learn: Imputation of missing values", "url": "https://scikit-learn.org/stable/modules/impute.html", "note": "Simple, k-NN and iterative imputers, missing indicators, and NaN support in estimators."}, {"title": "Flexible Imputation of Missing Data, 2nd ed. (van Buuren)", "url": "https://stefvanbuuren.name/fimd/", "note": "Free online book: chapter 1 on MCAR/MAR/MNAR and why ad hoc methods fail, chapter 2 on multiple imputation."}, {"title": "scikit-learn: Imputing missing values before building an estimator", "url": "https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html", "note": "Compares imputation strategies and indicators on real data."}, {"title": "pandas: Working with missing data", "url": "https://pandas.pydata.org/docs/user_guide/missing_data.html", "note": "NaN versus NA, sentinel replacement, and fill methods with limits."}]
practice: ["missing-value-imputation", "handle-missing-values", "replace-values", "null-handling", "boolean-masking"]
tags: ["missing data", "imputation"]
tools: ["scikit-learn", "pandas"]
---

## What you will learn

- Why the reason a value is missing matters more than how many values are missing, and the three standard names for those reasons: MCAR, MAR and MNAR.
- Why simply dropping incomplete rows is wasteful, often biased, and impossible at prediction time.
- What mean and median imputation do to a column (a spike, a smaller spread, weaker correlations), with the exact formula for how much the spread shrinks.
- How a **missingness indicator**, model-based imputers (k-NN, iterative) and native NaN handling in boosted trees keep or recover information.
- How to do all of this in pandas and scikit-learn without leaking validation or future data into your fills.

*Before you start:* you should know what fitting a transformer on training data means (lesson 07.01 on scaling), how one-hot encoding handles categories (07.02), and how train/validation splits and AUC work (module 06). Basic pandas (`isna`, `replace`) helps.

## From encodings to gaps

Lessons 07.01 and 07.02 assumed every cell held a value. Real tables have holes: a blood test not ordered, a form field skipped, a sensor offline for an hour. Most scikit-learn models refuse NaN, so you must decide what to do. Done carelessly this throws away one of the most predictive signals in the data; done well it recovers almost all of it.

## The idea in one picture

Six customers of a bank filled in an optional income field. Four answered; two left it blank. Suppose (we can only know this in a simulation) the two blanks belong to the two richest customers, who prefer not to say.

| Customer | A | B | C | D | E | F
| --- | --- | --- | --- | --- | --- | --- |
| Income given (thousand) | 30 | 40 | 50 | 60 | blank | blank
| True income (hidden) | 30 | 40 | 50 | 60 | 90 | 110

The average of the answers is 45. The true average of all six is about 63. Any analysis that looks only at the answers is badly wrong, and filling the blanks with 45 makes E and F look like ordinary customers when they are the two richest. The *pattern* of who left the field blank was the most informative thing in the column.

The figure shows the same effect on 160 simulated people under three different reasons for the gaps.

~~~dg-fig 07.03-mechanisms.svg
Hollow circles are incomes that went missing. When gaps are random (left), the recorded mean matches the true mean. When older people skip the question (middle) or high earners do (right), the recorded rows are no longer representative and the recorded mean drops well below 56.
~~~

Three terms, from Little and Rubin's classification:

- **MCAR** (missing completely at random): whether a value is missing has nothing to do with any value, recorded or not. A random batch of paper forms was lost. The recorded rows are a fair sample. This is rare in practice.
- **MAR** (missing at random): missingness depends only on columns you *do* observe. Older customers skip the app-usage survey, and age is recorded. Among people of the same age, the answers you have are representative, so you can repair the gaps using age.
- **MNAR** (missing not at random): missingness depends on the missing value itself. High earners are more likely to leave income blank. Nothing in the observed columns fully explains the gap, so the gap itself carries signal.

~~~dg-layers The three mechanisms, from easiest to hardest.
MCAR | deletion is unbiased, just wasteful; any simple imputation is fine
MAR | impute from the observed columns that explain the gaps
*MNAR | the gap carries signal: keep an indicator, model it explicitly
~~~

The names are confusing ("missing at random" is not the harmless case). Remember: MCAR = pure accident, MAR = explained by something you can see, MNAR = explained by the hidden value itself.

## Step 1: ask why values are missing

A missing value is a fact about how the data was collected, and that process often correlates with the label. A lab test is missing because the doctor did not order it, usually because the patient looked healthy. A sensor reading is missing because the device was offline, perhaps because it overheated. Before choosing any technique:

1. Measure how much is missing per column: `df.isna().mean()`.
2. Compare the label rate for rows with and without the value. A 6% vs 15% gap means missingness is predictive.
3. Ask whoever collects the data. Data can show that gaps are *not* MCAR, but never that they are not MNAR, because the evidence is the very values you lack.

Goals differ too. For **inference** (estimating an average or a coefficient), the mechanism decides whether the answer is biased. For **prediction**, you only need accuracy on future rows with the same kinds of gaps, which favours keeping the missingness pattern visible to the model.

## Step 2: why dropping rows rarely works

**Listwise deletion** means dropping every row that has a gap anywhere. It is unbiased under MCAR, but it is wasteful in a way that grows fast with the number of columns. If each column is missing independently with probability $p$, a row survives only if all $d$ of its columns are present:

$$P(\text{row complete}) = (1-p)^d.$$

Here $p$ is the missing rate per column and $d$ is the number of columns. With $p = 0.05$ and $d = 20$, $0.95^{20} \approx 0.36$: two thirds of your data is gone even though 95% of the cells are filled.

~~~dg-fig 07.03-dropping.svg
Complete rows shrink geometrically with the number of columns. Even a small missing rate per column leaves few complete rows in a wide table.
~~~

Under MAR or MNAR, deletion is also biased: the survivors are the younger, poorer or healthier rows. And in prediction it fails outright at serving time. You cannot refuse to score a customer because one field is empty, so a model trained only on complete rows will meet inputs it never saw. Dropping a whole **column** that is mostly missing and carries no signal is often fine; check first whether its missingness indicator predicts the label.

## Step 3: simple imputation and what it costs

**Imputation** means filling a gap with a guessed value. The simple versions use one number per column:

- **Mean** or **median** for numeric columns. The median is robust to skew (one billionaire does not move it).
- **Most frequent** value, or better an explicit `"missing"` category, for categorical columns.
- A **constant** where the gap has a real meaning: 0 purchases when there is no purchase record.

Simple imputation is fast and safe from crashes, but it distorts the column. Every filled row sits at the same point, which creates a spike and pulls the spread in. If $n$ rows exist, $n_{obs}$ of them are observed, and you fill the rest with the observed mean, the squared deviations of the filled rows are zero, so

$$\sigma^2_{\text{filled}} = \frac{n_{obs}}{n}\,\sigma^2_{obs}.$$

Here $\sigma^2_{obs}$ is the variance of the observed values (dividing by the count) and $\sigma^2_{\text{filled}}$ is the variance after filling. With 30% missing, the variance drops to 70% of its value. The correlation with other features also weakens, roughly by a factor $\sqrt{n_{obs}/n}$ under MCAR, because the filled rows show no relationship at all.

~~~dg-fig 07.03-meanfill.svg
Filling 30% of a column with its mean. Left: the filled points form a flat line that ignores x, so the correlation falls from 0.81 to 0.64. Right: the histogram gains a tall spike and its standard deviation falls from 8.1 to 6.9.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**mean-filling the income column by hand**

Recorded incomes: 30, 40, 50, 60. Their mean is $(30+40+50+60)/4 = 45$.

Deviations from 45: $-15, -5, 5, 15$. Squares: $225 + 25 + 25 + 225 = 500$. Observed variance: $500/4 = 125$, standard deviation $\approx 11.2$.

Fill E and F with 45. The column is now 30, 40, 50, 60, 45, 45. Its mean is still $270/6 = 45$, and the two new deviations are 0, so the sum of squares is still 500. Variance: $500/6 \approx 83.3$, which is exactly $\tfrac{4}{6} \times 125$, as the formula says. Standard deviation $\approx 9.1$.

Compare with the truth: the true incomes 30, 40, 50, 60, 90, 110 have mean $380/6 \approx 63.3$. Deviations $-33.3, -23.3, -13.3, -3.3, 26.7, 46.7$; squares $1111.1 + 544.4 + 177.8 + 11.1 + 711.1 + 2177.8 = 4733.3$; variance $4733.3/6 \approx 788.9$, standard deviation $\approx 28.1$.

So the filled column has a mean 18 thousand too low and a spread three times too narrow, and E and F, the richest customers, now sit in the middle. Under MNAR, filling cannot repair this because nothing in the data says the blanks are high.

Add a column `income_missing` = 0, 0, 0, 0, 1, 1 and the picture changes: a model can learn that "missing" means "probably rich" directly from the label.

</div>

## Step 4: keep the gap visible with an indicator

A **missingness indicator** is a 0/1 column that records whether the value was missing before you filled it. It restores the information imputation hides. Under MNAR it is often the most predictive part of the feature: in the bank example it is a near-perfect "high income" flag.

The indicator also makes the fill value matter less: in a linear model, the indicator's weight corrects whatever constant you filled. scikit-learn's `SimpleImputer(add_indicator=True)` adds one indicator per column that had gaps *in training*. A column that was complete in training gets no indicator, even if it later has gaps.

## Step 5: model-based imputation

Simple fills ignore the other columns. Model-based imputers use them, which helps under MAR, where the observed columns explain the gaps.

- **k-NN imputation** fills a gap with the average of that column among the $k$ most similar rows, measured on the columns that are present. Because it uses distances, scale the features first (lesson 07.01).
- **Iterative imputation** (the MICE idea: multiple imputation by chained equations) starts from simple fills, then models each column with gaps as a regression on all the others, re-predicts its missing entries, and cycles through the columns for several rounds until the fills settle.

Both preserve correlations far better than a mean fill, but cost more, add learned state to ship, and do **not** fix MNAR: if missing values are systematically different, the observed columns cannot reveal it. Pair them with an indicator.

## Step 6: let trees handle NaN themselves

Gradient-boosted tree libraries (XGBoost, LightGBM, scikit-learn's `HistGradientBoosting`) accept NaN directly. At each split they try sending missing rows left, then right, and keep whichever reduces the loss more: a learned "default direction" per split. This acts as an indicator learned exactly where it matters, and the same feature's gaps may go left at one node and right at another.

One trap: if a feature had no missing values in training but has some at serving time, no default direction was ever learned. scikit-learn then sends those rows to the child with more training samples, which is an arbitrary choice. Monitor missingness rates in production (lesson 09.03).

## Comparing the options on one problem

To see the size of the effects, here is a simulated version of a classic hospital case. We predict 30-day readmission from age and one lab value. Patients who looked healthy were less often tested, so the lab value is missing for 42% of patients (MNAR). Readmission is 5.3% among untested patients and 13.3% among tested ones. Each approach was trained on 14,000 patients and scored on 6,000 others.

~~~dg-fig 07.03-compare.svg
Test AUC on the simulated readmission data. Methods that discard the fact of missingness (orange) lose most of the gap to the no-missing ceiling; methods that keep it (blue) recover nearly all of it.
~~~

**Mean fill** (0.674) puts every untested patient at the average lab value of *tested* patients, who are sicker, so the healthiest group looks moderately ill. **Iterative imputation** (0.677) predicts the lab from age, but age cannot reveal that the doctor chose not to test. Adding an **indicator** (0.700 and 0.701) lets the model learn "untested means lower risk" directly, and a boosted tree's **native NaN handling** (0.696) gets nearly the same benefit with no imputation code. With no gaps at all the model reaches 0.708, so the indicator closes about three quarters of the gap that mean filling opens.

## Missing data in code

Real gaps often hide behind **sentinel values**: -999, 0, "N/A", "unknown" or an empty string that someone used to mean missing. Convert them first, or the model learns a fake spike at -999 and your imputer never fires. Then fit the imputer on training rows only.

```python
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "age":    [34, 51, 29, 62, 45, 38],
    "income": [30, 40, -999, 50, 60, -999],      # -999 is a sentinel meaning "not given"
    "city":   ["Oslo", None, "Bergen", "Oslo", "unknown", "Bergen"],
})
df["income"] = df["income"].replace(-999, np.nan)   # sentinels -> real missing values
df["city"] = df["city"].replace("unknown", np.nan)
print(df.isna().mean().round(2).to_dict())          # {'age': 0.0, 'income': 0.33, 'city': 0.33}

train, test = df.iloc[:4], df.iloc[4:]
imp = SimpleImputer(strategy="median", add_indicator=True)
print(imp.fit_transform(train[["age", "income"]]))
# [[34. 30.  0.]
#  [51. 40.  0.]
#  [29. 40.  1.]      <- income filled with the training median 40, indicator = 1
#  [62. 50.  0.]]
print(imp.statistics_)                              # [42.5 40. ]  medians learned from train only
print(imp.transform(test[["age", "income"]]))
# [[45. 60.  0.]
#  [38. 40.  1.]]     <- test gap filled with the TRAINING median
```

Only `income` got an indicator, because only it had gaps in training. The other imputers have the same interface, and the boosted tree needs none:

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (unlocks the class)
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

knn = KNNImputer(n_neighbors=5)                       # scale features first: k-NN uses distances
mice = IterativeImputer(max_iter=10, random_state=0, add_indicator=True)
linear = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       StandardScaler(), LogisticRegression(max_iter=1000))
tree = HistGradientBoostingClassifier()               # accepts NaN as is
```

Putting the imputer inside a `Pipeline` means cross-validation refits it on each training fold automatically, which is the leakage-safe default (lesson 07.05). For categorical columns, `SimpleImputer(strategy="constant", fill_value="missing")` before one-hot encoding makes "missing" a category of its own.

## Checking the result

- **Compare against a baseline.** Score at least mean fill, median plus indicator, and a model with native NaN support on the same validation folds. If indicators help a lot, missingness is informative (MAR or MNAR), and you should keep it visible.
- **Look at the indicator's importance.** A missingness flag among the top features is a finding worth explaining to domain experts, not a bug.
- **Check missingness at serving time.** Compute the missing rate per column in production and compare with training. A jump usually means a broken upstream pipeline, not a change in the world.

## What goes wrong

Time series need special care. Forward fill (repeat the last known value) uses only the past. Interpolation and backward fill draw on readings that arrive after the gap, which would not exist when the model has to predict.

~~~dg-fig 07.03-timeseries.svg
A sensor goes offline from hour 9 to 15. Forward fill holds the last past reading. Linear interpolation draws a line to the hour-16 reading, quietly using the future, and makes offline evaluation look better than production will be.
~~~

| Mistake | Symptom | Fix
| --- | --- | --- |
| Sentinels left in (-999, 0, "unknown") | Odd spike in the histogram; imputer reports no gaps | Replace sentinels with NaN before anything else
| Imputer fitted on all data | Validation score a bit too good; small drift in production | Fit inside the training fold, use a Pipeline
| Interpolation or backward fill in time series | Great backtest, poor live performance | Forward fill within each entity, with a staleness limit
| Mean fill on an informative gap | Missing rows get wrong-looking predictions; indicator would rank high | Add an indicator or use native NaN handling
| Different fill in notebook and serving code | Same input, different predictions | Ship the fitted imputer with the model (07.05)
| New gaps in a column that was complete in training | No indicator exists; trees send rows an arbitrary way | Monitor missing rates; retrain with realistic gaps

For time series, forward-fill **within each entity** (each station or user, via `groupby(...).ffill(limit=...)`), and set a limit on how stale a value may be.

## Going deeper

**Why the variance formula is exact.** Filling with the observed mean $\bar x_{obs}$ keeps the mean at $\bar x_{obs}$, and each filled row contributes $(\bar x_{obs} - \bar x_{obs})^2 = 0$ to the sum of squares. The sum is unchanged while the count grows from $n_{obs}$ to $n$, giving the factor $n_{obs}/n$. The covariance with another feature shrinks by about the same factor, while each standard deviation shrinks only by its square root, so the correlation falls by roughly $\sqrt{n_{obs}/n}$ under MCAR. In the figure, $0.81 \times \sqrt{0.7} \approx 0.68$; the simulated value was 0.64, close given sampling noise.

**Single versus multiple imputation.** A single fill treats guesses as measurements, so for inference the confidence intervals come out too narrow. **Multiple imputation** draws $m$ plausible completed datasets (with noise reflecting each guess's uncertainty), analyses each, and pools them with Rubin's rules: the estimate is the average, and its variance is the average within-dataset variance plus $(1 + 1/m)$ times the variance *between* datasets. For prediction, one good imputation plus indicators is usually enough, since held-out data measures success directly.

**What can and cannot be tested.** You can test MCAR against the alternative that gaps depend on observed columns (Little's MCAR test does this, and a classifier predicting the indicator from other columns is an informal version). You cannot test MAR against MNAR from the data, because the difference lies in the values you do not see. Handling MNAR for inference needs explicit assumptions: **selection models** describe how the chance of being missing depends on the value, and **pattern-mixture models** describe how the values differ between missing and observed groups. Sensitivity analysis, re-running under several assumed shifts, shows how much the conclusions depend on them.

**Structural gaps are not really missing.** "Spouse income" is blank for single people: a defined state. Encode it explicitly (an indicator or its own category) rather than imputing a number.

## Key terms

| Term | Meaning
| --- | --- |
| MCAR | Missing completely at random: gaps unrelated to any value, recorded or not
| MAR | Missing at random: gaps explained by columns you do observe
| MNAR | Missing not at random: gaps depend on the missing value itself
| Listwise deletion | Dropping every row that has any gap
| Imputation | Filling a gap with an estimated value
| Missingness indicator | A 0/1 column recording that a value was missing before filling
| Iterative imputation (MICE) | Repeatedly regressing each incomplete column on the others to refine fills
| Sentinel value | A placeholder such as -999 or "unknown" that secretly means missing
| Default direction | The branch a tree split sends missing values to, chosen to reduce the loss

<div class="take">
<div class="lab">Key takeaway</div>

ask why values are missing before filling them; convert sentinels to NaN, fit imputers on training data only, keep a missingness indicator (or use native NaN handling in trees) whenever the gap may carry signal, and fill time series only from the past.

</div>
