---
id: "07.05"
course: "machine-learning"
module: "07"
slug: "pipelines-fit-transform-discipline-end-to-end"
title: "Pipelines: fit/transform discipline end to end"
keyIdeas: ["Every preprocessing step with learned state follows the same contract: `fit` learns from training data only, `transform` applies the frozen state anywhere. A **pipeline** chains these steps with the model into one estimator.", "Because the pipeline is one object, cross-validation and hyperparameter search refit *every* step inside each fold, which removes the commonest leaks automatically.", "`ColumnTransformer` routes columns to different steps (scale numerics, encode categoricals) and keeps the output layout fixed, so serving receives exactly the columns training produced.", "Leakage can still enter before the pipeline: aggregates computed over all rows, features built with future data, duplicates across splits. The pipeline only protects what is inside it.", "Ship the fitted pipeline as one versioned artefact and run the same object in batch and online serving; re-implementing preprocessing in serving code is the main source of training-serving skew."]
question: "A team computes, before any splitting, each user's \"average order value over all their orders\" and a global StandardScaler, then uses a scikit-learn Pipeline containing only the model inside GridSearchCV with a time-based split. Validation looks excellent. Identify each leak, say which ones moving steps into the pipeline would fix, and which need a different fix."
answer: "Two leaks. (1) The global StandardScaler was fitted on all rows, including validation periods, so validation statistics shape the training inputs. Moving the scaler into the pipeline fixes it: GridSearchCV then refits it on each training fold only. The effect is usually small, but it is a real leak. (2) \"Average order value over all their orders\" includes orders made **after** each row's prediction time, and possibly the very order being predicted, so it leaks the future and possibly the target. Putting a groupby transformer into the pipeline only partly helps: it would restrict the statistic to training-fold rows, but within the training fold it still averages each user's later orders into earlier rows. The proper fix is point-in-time feature construction: compute, for each row, the average over that user's orders strictly before its timestamp (an expanding mean shifted by one, or a point-in-time join), and use the same definition in serving. Then re-run the time-based CV and expect lower, more honest scores."
sources: [{"title": "scikit-learn: Pipelines and composite estimators", "url": "https://scikit-learn.org/stable/modules/compose.html", "note": "Pipeline, ColumnTransformer, parameter naming for search, caching and feature names."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Inconsistent preprocessing and data leakage, with the pipeline fix shown side by side."}, {"title": "scikit-learn: Model persistence", "url": "https://scikit-learn.org/stable/model_persistence.html", "note": "Pickle, joblib and skops, version compatibility and the security caveats of loading pickles."}, {"title": "Hidden technical debt in machine learning systems (Sculley et al., NeurIPS 2015)", "url": "https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "Pipeline jungles, glue code and training-serving skew as system-level costs."}]
practice: ["training-serving-skew", "normalize-columns", "point-in-time-feature-join", "record-schema-validation", "transform-pipeline"]
tags: ["pipelines", "data leakage", "training-serving skew"]
tools: ["scikit-learn", "pandas"]
---

## The contract

Every lesson in this module ended with the same instruction: learn the statistics on the training fold only. A scaler learns $\mu$ and $\sigma$, an imputer learns medians, an encoder learns a vocabulary or target means, a selector learns which columns to keep. scikit-learn makes this a contract with three methods:

- `fit(X, y)` learns state from training data and stores it in attributes ending in an underscore (`mean_`, `categories_`).
- `transform(X)` applies that frozen state to any data, without learning anything.
- `fit_transform(X, y)` does both on training data, sometimes differently from calling them in sequence (the cross-fitted `TargetEncoder` of lesson 07.02 is the important case).

A **pipeline** is a sequence of transformers followed by an estimator. Calling `fit` runs `fit_transform` on each step in order and fits the final model on the result; `predict` runs `transform` on each step and then the model. The fitted pipeline *is* the model: everything needed to turn a raw row into a prediction, in one object.

~~~dg-flow The same fitted object serves both paths: fit on training rows only, then transform everywhere else.
[Fit] Raw training rows -(fit + transform)-> Impute -(fit + transform)-> Encode and scale -(fit)-> *Model
[Predict] Raw new rows -(transform)-> Impute -(transform)-> Encode and scale -(predict)-> Prediction
~~~

## Why it removes most leaks

Without a pipeline, the natural notebook order is: clean, impute, scale, encode, select, then split and cross-validate. Every step before the split saw the validation rows. With a pipeline, you split first and pass the whole pipeline to `cross_val_score` or `GridSearchCV`; each fold clones it and fits every step on that fold's training rows only. The same holds for anything else that learns from data: feature selection (07.04), target encoding (07.02), and resampling (06.06), which needs imbalanced-learn's pipeline because samplers must run during `fit` but never during `predict`.

A side benefit is that hyperparameters of preprocessing steps become searchable alongside the model's, using `step__parameter` names: `"pre__num__impute__strategy": ["mean", "median"]`. Choosing the imputation strategy by CV is then as natural as choosing $C$.

## Mixed column types

Real tables mix numeric, categorical and text columns that need different treatment. `ColumnTransformer` applies a different sub-pipeline to each column group and concatenates the results in a fixed order:

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20))])
pre = ColumnTransformer([("num", num, make_column_selector(dtype_include="number")),
                         ("cat", cat, make_column_selector(dtype_include=object))])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
grid = GridSearchCV(model, {"pre__num__impute__strategy": ["mean", "median"], "clf__C": [0.1, 1, 10]},
                    cv=cv, scoring="average_precision").fit(X_train, y_train)
```

Two habits help. Select columns by explicit name lists in production code, not by dtype, so that a column that changes type upstream fails loudly rather than silently moving groups. And call `set_output(transform="pandas")` or `get_feature_names_out()` while debugging, so you can see which output column came from which input.

## What a pipeline cannot protect

The pipeline only sees the rows and columns you hand it. Leakage that happens while building those rows is invisible to it:

1. **Aggregate features over all data**: "average spend of this customer's city" computed over the full table includes validation rows (and their labels, if spend relates to the target). Compute such aggregates inside the pipeline with a custom transformer, or as of each row's timestamp.
2. **Future information**: features joined without a point-in-time cut (lesson 09.02 and the ML Toolkit SQL lessons).
3. **Duplicates and groups** split across folds (lesson 01.03).
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
```

## From notebook to serving

Training-serving skew is when the features computed at serving time differ from those at training time, for the same input. The usual cause is duplicated logic: preprocessing in a notebook, re-implemented by hand in the serving service, drifting apart one small decision at a time. The pipeline gives you one artefact to ship instead:

- **Persist the fitted pipeline**, not just the model weights, and load the same object in batch scoring and in the online service. Pickle and joblib files execute code when loaded, so load only artefacts you produced; skops offers a safer format.
- **Pin library versions** with the artefact; a pickled scikit-learn object is not guaranteed to load or behave identically under another version.
- **Validate inputs** against the schema the pipeline was trained on (column names, types, allowed ranges) before transforming; fail loudly on missing columns.
- **Monitor** unseen-category rates, imputation rates and feature distributions after the pipeline, since those are where upstream changes show up first (lesson 09.03).

<div class="ex">
<div class="lab">Worked example</div>

**AUC 0.86 offline, 0.78 in production**

A churn model scored AUC 0.86 on a clean time-based test set but 0.78 live. A training-serving skew report comparing logged serving features with offline features for the same customers found two differences. The notebook filled missing tenure with the training median (14 months); the service filled it with 0, which the model reads as "brand new customer, high churn risk". And the service rebuilt the plan-type mapping from a database query whose ordering had changed, so integer codes pointed at the wrong plans.

Fix: the team put imputation and one-hot encoding into a single scikit-learn pipeline, persisted it with the model, and served that object behind the same API. The live AUC moved to 0.85, and the skew report now runs in CI before every release.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

put every step that learns from data inside one pipeline so CV refits it per fold, keep row-building steps point-in-time correct, and ship the fitted pipeline itself so training and serving run identical code.

</div>
