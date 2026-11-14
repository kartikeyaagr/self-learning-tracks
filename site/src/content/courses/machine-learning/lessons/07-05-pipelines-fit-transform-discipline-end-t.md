---
id: "07.05"
course: "machine-learning"
module: "07"
slug: "pipelines-fit-transform-discipline-end-to-end"
title: "Pipelines: fit/transform discipline end to end"
keyIdeas: ["Every preprocessing step with learned state follows one contract: `fit` learns from training data only, `transform` applies the frozen state anywhere. A **pipeline** chains these steps with the model into one estimator that holds everything needed to turn a raw row into a prediction.", "Because the pipeline is one object, cross-validation and hyperparameter search clone it and refit *every* step inside each fold. On pure noise, selecting features before CV reported about 0.93 accuracy; selecting inside the pipeline reported the true 0.5.", "`ColumnTransformer` routes columns to different steps (scale numerics, one-hot encode categoricals) and keeps the output layout fixed, and `step__parameter` names make preprocessing choices tunable like any hyperparameter.", "Leakage can still enter before the pipeline: aggregates over all rows, features built with future data, duplicates across splits, target-derived columns. The pipeline only protects what is inside it.", "Ship the fitted pipeline as one versioned artefact and run the same object in batch and online serving; re-implementing preprocessing by hand in serving code is the main source of training-serving skew."]
question: "A team computes, before any splitting, each user's \"average order value over all their orders\" and a global StandardScaler, then uses a scikit-learn Pipeline containing only the model inside GridSearchCV with a time-based split. Validation looks excellent. Identify each leak, say which ones moving steps into the pipeline would fix, and which need a different fix."
answer: "There are two leaks. (1) The global StandardScaler was fitted on all rows, including the validation periods, so validation statistics (their mean and spread) shape the inputs the model trains on. Moving the scaler into the pipeline fixes this completely: GridSearchCV then clones the pipeline and refits the scaler on each training fold only, transforming the validation fold with the training fold's numbers. The effect on the score is usually small for plain scaling, but it is a real leak and it costs nothing to remove. (2) \"Average order value over all their orders\" includes orders made **after** each row's prediction time, and possibly the very order being predicted, so it leaks the future and possibly the target. Putting a group-mean transformer into the pipeline only partly helps: it would restrict the statistic to training-fold rows, but within the training fold it still averages each user's later orders into earlier rows, so the model learns from information it will never have at serving time. The proper fix is point-in-time feature construction: for each row, average only that user's orders strictly before its timestamp (an expanding mean shifted by one, or a point-in-time join against an order history table), and use exactly the same definition in serving. After both fixes, re-run the time-based CV and expect lower, more honest scores. A shuffled-label run of the corrected setup is a cheap extra check: its score should fall to chance."
sources: [{"title": "scikit-learn: Pipelines and composite estimators", "url": "https://scikit-learn.org/stable/modules/compose.html", "note": "Pipeline, ColumnTransformer, parameter naming for search, caching and feature names."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Inconsistent preprocessing and data leakage, with the pipeline fix shown side by side."}, {"title": "scikit-learn: Model persistence", "url": "https://scikit-learn.org/stable/model_persistence.html", "note": "Pickle, joblib and skops, version compatibility and the security caveats of loading pickles."}, {"title": "Hidden technical debt in machine learning systems (Sculley et al., NeurIPS 2015)", "url": "https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "Pipeline jungles, glue code and training-serving skew as system-level costs."}]
practice: ["training-serving-skew", "normalize-columns", "point-in-time-feature-join", "record-schema-validation", "transform-pipeline"]
tags: ["pipelines", "data leakage", "training-serving skew"]
tools: ["scikit-learn", "pandas"]
---

## What you will learn

- The **fit/transform contract** that every preprocessing step with learned state follows, and why breaking it once is enough to inflate your scores.
- How a scikit-learn `Pipeline` chains imputation, encoding, scaling, selection and the model into one object that behaves like a single model.
- Why passing that one object to cross-validation or `GridSearchCV` removes the most common leaks automatically, and how to tune preprocessing choices with `step__parameter` names.
- How `ColumnTransformer` sends numeric and categorical columns down different paths and keeps the output layout fixed.
- What a pipeline cannot protect you from (row-building leaks), and how to ship the fitted pipeline so training and serving run identical code.

*Before you start:* this lesson ties the module together, so it helps to have read scaling (07.01), categorical encoding (07.02), missing data (07.03) and feature selection (07.04). You also need cross-validation and the idea of data leakage from module 06 and lesson 01.03.

## From four lessons to one habit

Lessons 07.01 to 07.04 each ended with the same warning in different words: learn the medians, the means, the category list, the chosen columns *from the training rows only*. Selection in 07.04 was the most dangerous case, because choosing columns while looking at every row can make pure noise look predictive.

Remembering that warning step by step does not scale to ten steps, five folds and a hyperparameter search. This lesson turns it into a structure, the **pipeline**, so the right thing becomes the default.

## The idea in one picture

Take one numeric column, yearly income in thousands, with a gap. Four rows are for training and two arrive later as test rows:

| Row | 1 | 2 | 3 | 4 | test 1 | test 2
| --- | --- | --- | --- | --- | --- | --- |
| Income | 20 | 30 | missing | 50 | 90 | missing

The preprocessing is two steps: fill gaps with the median, then standardise to mean 0 and standard deviation 1. Each step has **state**: numbers it must learn from data before it can do its job. The imputer learns one median; the scaler learns one mean and one standard deviation.

Done correctly, the steps learn from rows 1 to 4 only. Done carelessly, they learn from all six rows. The two choices give different numbers for every row:

| Learned from | Median | Mean | Std | Test 1 (90) becomes | Train row 1 (20) becomes
| --- | --- | --- | --- | --- | --- |
| Training rows only | 30 | 32.5 | 10.90 | 5.28 | -1.15
| All six rows (leak) | 40 | 45.0 | 22.17 | 2.03 | -1.13

In the honest version, the test income of 90 looks like what it is: extreme, over five standard deviations from the training mean. In the leaky version it has already pulled the mean up and widened the spread, so it looks unremarkable at 2.03. That is **data leakage**: information from the evaluation rows reaching the training process.

With cross-validation the same question arises once per fold. The figure shows who learns from whom.

~~~dg-fig 07.05-folds.svg
Top row: a scaler fitted once before cross-validation learns from every row, including the rows it is later validated on. Rows below: inside a pipeline, each fold's scaler learns only from its own 16 training rows (blue) and merely transforms its 4 validation rows (green).
~~~

The rest of the lesson is about getting the bottom five rows of that picture for free, every time.

## Step 1: the contract every step follows

scikit-learn gives every step with learned state the same three methods. Steps that follow them are called **transformers**.

- `fit(X, y)` learns the state from training data and stores it in attributes whose names end in an underscore: `mean_`, `scale_`, `statistics_`, `categories_`.
- `transform(X)` applies that frozen state to any data, without learning anything new.
- `fit_transform(X, y)` does both on the training data in one call.

For the scaler, the frozen state is $\mu_{\text{train}}$ and $\sigma_{\text{train}}$, and transforming any value $x$ means

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}},$$

where $x$ is the raw value, $\mu_{\text{train}}$ is the training mean, $\sigma_{\text{train}}$ is the training standard deviation and $z$ is what the model receives. The subscript is the whole point: the test row, the validation fold and tomorrow's live request are all scaled with the *training* numbers.

<div class="ex">
<div class="lab">Worked example</div>

**the income column by hand**

Training rows: 20, 30, missing, 50. The imputer learns the median of the observed values 20, 30, 50, which is **30**. The filled training column is 20, 30, 30, 50.

The scaler learns the mean: $(20 + 30 + 30 + 50)/4 = 130/4 = 32.5$.

Deviations from the mean: $-12.5, -2.5, -2.5, 17.5$. Squares: $156.25, 6.25, 6.25, 306.25$, which sum to $475$. scikit-learn divides by $n$, so the variance is $475/4 = 118.75$ and $\sigma = \sqrt{118.75} = 10.90$.

Transform test 1: $(90 - 32.5)/10.90 = 57.5/10.90 = 5.28$. Transform test 2: missing becomes the stored median 30, then $(30 - 32.5)/10.90 = -0.23$.

Leaky version for comparison: the observed values across all six rows are 20, 30, 50, 90, whose median is 40. Filled, the six values are 20, 30, 40, 50, 90, 40, with mean $270/6 = 45$ and standard deviation 22.17. Test 1 becomes $(90 - 45)/22.17 = 2.03$. Every learned number changed because two rows that should have been unseen were seen.

</div>

The mirror-image mistake is calling `fit_transform` on the test set or a batch of live requests: the same raw input then gets a different value depending on its neighbours.

~~~dg-fig 07.05-batch.svg
The training data has mean 50.5 and standard deviation 15.2, so a customer earning 60 is scaled to +0.63. Refit the scaler on a low-income serving batch and the same customer becomes +3.08; refit on a high-income batch and they become -1.84.
~~~

The model learned what +0.63 means; feeding it +3.08 for the same person is a silent bug. Rule: `fit` and `fit_transform` are only ever called on training data.

## Step 2: chain the steps into a pipeline

A **pipeline** is a list of named transformers followed by a final estimator (the model). It has the same methods as a model, and it calls the right method on each step:

- `pipe.fit(X, y)` runs `fit_transform` on step 1, feeds the result to `fit_transform` on step 2, and so on, then fits the model on the final output.
- `pipe.predict(X)` runs only `transform` on every step, then `predict` on the model.

~~~dg-flow The same fitted object serves both paths: fit on training rows only, then transform everywhere else.
[Fit] Raw training rows -(fit + transform)-> Impute -(fit + transform)-> Encode and scale -(fit)-> *Model
[Predict] Raw new rows -(transform)-> Impute -(transform)-> Encode and scale -(predict)-> Prediction
~~~

The fitted pipeline *is* the model: the median, the mean, the category list, the chosen columns and the weights live in one object, so they can never get out of sync.

## Step 3: let cross-validation refit everything

Without a pipeline, the natural notebook order is: clean, impute, scale, encode, select, *then* split and cross-validate. Every step before the split saw the validation rows.

With a pipeline you split first and hand the whole pipeline to `cross_val_score` or `GridSearchCV`. For each fold, scikit-learn makes a fresh unfitted copy (a **clone**), fits every step on that fold's training rows, and scores on the held-out rows: the bottom half of the fold figure.

How much does it matter? For scaling and median imputation, usually a little. For anything that looks at the label, a lot. The next figure runs the classic test from lesson 07.04: 60 rows, labels that are coin flips, and 2000 columns of pure noise. One version picks the 20 columns most correlated with the label using all rows, then cross-validates. The other puts the selection inside the pipeline.

~~~dg-fig 07.05-leak.svg
Left: over 20 repeats, selecting features before cross-validation reports about 0.93 accuracy on data with no signal; cross-validating the pipeline reports about 0.53, close to the true 0.5. Right: the more candidate columns the selector can search, the bigger the fake gain.
~~~

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 2000))          # 2000 columns of pure noise
y = rng.integers(0, 2, 60)               # labels are coin flips

X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)          # selection saw every row
print(cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean().round(2))   # 0.95

pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
print(cross_val_score(pipe, X, y, cv=5).mean().round(2))                        # 0.5
```

Same data, model and folds; only *where* `fit` was called differs. The same applies to target encoding (07.02) and to resampling for imbalanced classes (06.06), whose samplers must run during `fit` but never during `predict`, so they need imbalanced-learn's `Pipeline`.

## Step 4: different columns, different treatment

Real tables mix numeric and categorical columns, and each needs its own steps. `ColumnTransformer` routes named groups of columns to their own sub-pipelines and glues the outputs side by side in a fixed order:

~~~dg-flow ColumnTransformer splits the table by column, processes each group, and concatenates the results in a fixed order.
Raw table -(split by column)-> tenure, charges: median impute + indicator, then scale; plan: one-hot, ignore unseen -(concatenate)-> *Feature matrix
~~~

The fixed order means the model's third weight always multiplies the same feature, however the incoming columns are ordered.

## Step 5: tune preprocessing like any hyperparameter

Every step's settings become searchable through names of the form `step__substep__parameter`, joined by double underscores. `"pre__num__impute__strategy"` means: in the step named `pre`, the group named `num`, the step named `impute`, set `strategy`. Choosing between mean and median imputation by cross-validation is then as natural as choosing the regularisation strength $C$, and because the search refits the pipeline per fold, the comparison is fair.

## The whole thing in code

A small churn table with a numeric column that has 15% gaps, a second numeric column and one categorical column:

```python
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

# df: columns tenure (15% missing), charges, plan in {basic, plus, pro}, churn in {0, 1}; 2000 rows
X, y = df.drop(columns="churn"), df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)  # split FIRST

num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler())])
pre = ColumnTransformer([("num", num, ["tenure", "charges"]),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan"])])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean().round(3))   # 0.762

grid = GridSearchCV(model, {"pre__num__impute__strategy": ["mean", "median"],
                            "clf__C": [0.01, 0.1, 1]}, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
print(grid.best_params_)          # {'clf__C': 1, 'pre__num__impute__strategy': 'median'}

best = grid.best_estimator_       # refitted on all of X_tr with the best settings
print(best.named_steps["pre"].get_feature_names_out())
# ['num__tenure' 'num__charges' 'num__missingindicator_tenure'
#  'cat__plan_basic' 'cat__plan_plus' 'cat__plan_pro']
print(best.named_steps["pre"].named_transformers_["num"]["impute"].statistics_)   # [14.  60.15]

new = pd.DataFrame({"tenure": [np.nan], "charges": [75.0], "plan": ["enterprise"]})
print(best.predict_proba(new)[:, 1].round(3))   # [0.336]  gap -> 14, unseen plan -> all zeros
```

The last line scored a raw, messy row: the missing tenure became the stored median of 14 months, the unseen plan became three zeros instead of a crash, and scaling used the training numbers. The test set `X_te` is touched once, at the end, for a final score (0.768 here).

Two habits pay off: list columns by explicit name rather than by dtype, so a column whose type changes upstream fails loudly; and while debugging, call `set_output(transform="pandas")` so intermediate outputs are labelled DataFrames.

## Checking that it worked

- **Inspect the learned state.** `statistics_`, `mean_`, `categories_` should match numbers you can compute by hand from the training split alone, as in the worked example.
- **Read the output columns.** `get_feature_names_out()` tells you which model weight belongs to which input. If a name surprises you, a column went to the wrong group.
- **Run the shuffled-label test.** Permute `y_tr` and cross-validate the pipeline again. Scores should collapse to chance (AUC near 0.5). If they stay high, something outside the pipeline still sees the labels.
- **Compare CV with a final holdout.** A cross-validated score far above the held-out test score is the classic sign of leakage or of an over-tuned search.

## What a pipeline cannot protect

The pipeline only sees the rows and columns you hand it. Leakage that happens while *building* those rows is invisible to it:

1. **Aggregate features over all data.** "Average spend of this customer's city" computed over the full table includes validation rows, and their labels if spend relates to the target. Compute such aggregates inside the pipeline with a custom transformer, or as of each row's timestamp.
2. **Future information.** Features joined without a point-in-time cut (lesson 09.02 and the ML Toolkit SQL lessons).
3. **Duplicates and groups** split across folds (lesson 01.03). The pipeline refits per fold, but if the same customer sits in both halves, it still memorises them.
4. **Target-derived columns** created upstream, like a status flag set after the outcome.

When you need custom logic, write a transformer that follows the contract, so it slots into a pipeline and is refit in every fold:

```python
from sklearn.base import BaseEstimator, TransformerMixin
class GroupMean(BaseEstimator, TransformerMixin):
    # Adds the training-set mean of `col` per `key`; unseen keys get the global mean.
    def __init__(self, key, col): self.key, self.col = key, col
    def fit(self, X, y=None):
        self.means_ = X.groupby(self.key)[self.col].mean(); self.global_ = X[self.col].mean(); return self
    def transform(self, X):
        return X.assign(**{f"{self.col}_by_{self.key}": X[self.key].map(self.means_).fillna(self.global_)})
# fit on cities A: (10, 20), B: (40, 60); transform city A -> 15.0, unseen city C -> 32.5
```

Note the conventions: `__init__` only stores its arguments (so `clone` and grid search work), learned state ends in an underscore, and `fit` returns `self`.

## From notebook to serving

**Training-serving skew** is when serving computes different features than training did for the same input. The usual cause is preprocessing re-implemented by hand in the serving service, drifting apart one small decision at a time.

<div class="ex">
<div class="lab">Worked example</div>

**AUC 0.86 offline, 0.78 in production**

A churn model scored AUC 0.86 on a clean time-based test set but 0.78 live. A skew report comparing logged serving features with offline features for the same customers found two differences.

First, the notebook filled missing tenure with the training median (14 months); the service filled it with 0, which the model reads as "brand new customer, high churn risk". On the model in the figure below, that moves a customer's churn probability from 0.33 to 0.71.

Second, the service rebuilt the plan-type mapping from a database query whose ordering had changed, so integer codes pointed at the wrong plans.

Fix: the team put imputation and one-hot encoding into a single scikit-learn pipeline, persisted it with the model, and served that object behind the same API. The live AUC moved to 0.85, and the skew report now runs in CI before every release.

</div>

~~~dg-fig 07.05-skew.svg
Churn risk falls with tenure. The same customer with missing tenure gets risk 0.33 when filled with the training median of 14 months, and 0.71 when hand-written serving code fills it with 0.
~~~

The pipeline gives you one artefact to ship instead of two implementations:

- **Persist the fitted pipeline**, not just the model weights, and load the same object in batch scoring and in the online service. Pickle and joblib files execute code when loaded, so load only artefacts you produced; skops offers a safer format.
- **Pin library versions** with the artefact; a pickled scikit-learn object may not load or behave identically under another version.
- **Validate inputs** against the schema the pipeline was trained on (column names, types, allowed ranges) before transforming; fail loudly on missing columns.
- **Monitor** unseen-category rates, imputation rates and feature distributions after the pipeline, since those are where upstream changes show up first (lesson 09.03).

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Scaling, imputing or encoding before the split | CV slightly better than the final test | move the step inside the pipeline
| Feature selection or target encoding on all rows | CV far above chance on weak or noisy data; shuffled-label test still scores high | selection and encoding inside the pipeline, refit per fold
| `fit_transform` on test or serving batches | predictions for one row change with its batch | only `transform` outside training
| Oversampling before CV | inflated minority recall; synthetic copies in validation folds | imbalanced-learn `Pipeline` with the sampler as a step
| Aggregates or joins built over the full table | great offline, poor live | point-in-time features or a custom transformer
| Re-implementing preprocessing in serving | live metric below offline for the same period | ship the fitted pipeline as one artefact
| Selecting columns by dtype | a type change silently reroutes a column | explicit column lists plus schema validation
| Reporting `GridSearchCV.best_score_` as the final number | final holdout score lower than the search promised | report a separate test score, or use nested CV

## Going deeper

**Why the leak grows with the search.** With $n$ rows, the sample correlation between a pure-noise column and the label has a standard deviation of about $1/\sqrt{n}$, which is 0.13 for $n = 60$. Among $p$ such columns, the largest absolute correlation grows roughly like $\sqrt{2\ln p / n}$. Simulating $p = 2000$ gives a largest correlation around 0.45 and a twentieth-largest around 0.33, although the true value of every one is zero. A selector run on all rows keeps exactly these lucky columns, and their luck is shared with the validation rows. Inside the pipeline, each fold picks columns that were lucky on its training rows, and that luck does not transfer, so the score falls back to chance. This is the right-hand panel of the leak figure.

**`fit_transform` is not always `fit` then `transform`.** For most transformers the two are identical. `TargetEncoder` (07.02) is the important exception: its `fit_transform` uses internal cross-fitting so each training row's encoding comes from other rows, while `transform` applies the full-data means. The pipeline calls `fit_transform` during training, so it gets the safe behaviour automatically. Calling `fit(X).transform(X)` yourself would encode each training row with its own label.

**Tuning is also learning.** `GridSearchCV` picks the best of many configurations using the same folds it scores on, so `best_score_` is mildly optimistic, and more so as the grid grows. For an unbiased estimate, wrap the search in an outer cross-validation (**nested CV**), or keep a test set the search never touches, as the code above does.

**Stateless steps are safe anywhere.** A log transform learns nothing, so applying it before the split leaks nothing; putting it in the pipeline (`FunctionTransformer`) still means serving gets it for free.

**Time-ordered data.** A pipeline plus `TimeSeriesSplit` fixes *which rows* each step learns from, not *what each row contains*.

## Key terms

| Term | Meaning
| --- | --- |
| Transformer | A step that learns state in `fit` and applies it in `transform`
| Learned state | Numbers a step stores after fitting, such as a median, mean or category list
| Pipeline | Transformers plus a final model, fitted and applied as one object
| ColumnTransformer | Routes column groups to different steps and joins the outputs in a fixed order
| Data leakage | Information from evaluation rows (or the future) reaching training
| Clone | A fresh unfitted copy of an estimator, made per fold during cross-validation
| Nested CV | An outer CV loop around a hyperparameter search, giving an unbiased score
| Training-serving skew | Different feature values at serving time than at training time for the same input

<div class="take">
<div class="lab">Key takeaway</div>

put every step that learns from data inside one pipeline so that cross-validation and search refit it per fold, keep row-building steps point-in-time correct because the pipeline cannot see them, and ship the fitted pipeline itself so training and serving run identical code.

</div>
