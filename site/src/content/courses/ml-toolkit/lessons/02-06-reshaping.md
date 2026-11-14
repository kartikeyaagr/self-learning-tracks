---
id: "02.06"
course: "ml-toolkit"
module: "02"
slug: "reshaping"
title: "Reshaping"
keyIdeas: ["**Long** data has one row per observation (entity, variable, value); **wide** data has one row per entity and one column per variable. Logs and databases store long; most models consume wide.", "`pivot` reshapes long to wide and fails on duplicate entries; `pivot_table` aggregates duplicates (mean by default) and can fill gaps. `crosstab` is a pivot table of counts.", "`melt` goes from wide to long, which is what you want for plotting, storing many metrics, or applying one groupby across many columns.", "`stack` and `unstack` move levels between a MultiIndex on rows and on columns. After pivoting, fix the column set with `reindex(columns=train_columns, fill_value=0)` so serving produces exactly the training layout."]
question: "Your fraud features include `n_tx_<category>`, built per user by `pivot_table` over the last 30 days of transactions with `aggfunc=\"count\", fill_value=0`, and `avg_amount_<category>` built the same way with `aggfunc=\"mean\", fill_value=0`. In production, the scoring service rebuilds these columns from the live 30-day window and passes `wide.values` to the model. What can go wrong, and how do you fix both the features and the serving path?"
answer: "Three problems. First, the column set depends on which categories happen to occur in the window: the live data may lack categories seen in training (columns disappear) or contain new ones (columns appear), and because the service passes `wide.values` by position, every feature after the first difference is shifted onto the wrong model weight with no error. Fix it by saving the training column list and doing `wide.reindex(columns=TRAIN_COLUMNS, fill_value=...)` before scoring, mapping unseen categories to an `other` bucket before pivoting, and selecting by name rather than position. Second, `fill_value=0` for `avg_amount_<category>` conflates 'no transactions in this category' with 'transactions averaging zero'; leave these as NaN (and impute consistently) or rely on the count column to disambiguate. Counts filled with 0 are correct. Third, column order from `pivot_table` is sorted by category value, which can differ between environments if category labels are normalised differently (case, whitespace); apply the same normalisation function in both places. A test that builds features for a fixed sample in both the training pipeline and the serving code and compares them column by column catches all three, which is the core of a training-serving skew check."
sources: [{"title": "pandas docs: Reshaping and pivot tables", "url": "https://pandas.pydata.org/docs/user_guide/reshaping.html", "note": "pivot, pivot_table, melt, stack, unstack, crosstab, wide_to_long and get_dummies."}, {"title": "Wickham: Tidy Data (2014)", "url": "https://www.jstatsoft.org/article/view/v059i10", "note": "Why long, tidy tables make transformation easy and how messy layouts map onto them."}, {"title": "pandas docs: MultiIndex / advanced indexing", "url": "https://pandas.pydata.org/docs/user_guide/advanced.html", "note": "How hierarchical row and column indexes work, which stack and unstack move between."}]
practice: ["pivot-tables", "melt-wide-to-long", "unstack-long-to-wide", "cross-tabulation", "training-serving-skew"]
tags: ["reshaping", "pivot", "tidy data"]
tools: ["pandas"]
---

## Long and wide

The same information can be laid out in two shapes. Long (or "tidy") data has one row per observation:

| user | metric | value
| --- | --- | --- |
| u1 | clicks | 12
| u1 | spend | 30.5
| u2 | clicks | 3

Wide data has one row per entity and one column per variable:

| user | clicks | spend
| --- | --- | --- |
| u1 | 12 | 30.5
| u2 | 3 | NaN

Long is the natural shape for event logs, for databases, and for adding a new variable without changing a schema. Wide is the shape of a feature matrix: one row per example, one column per feature. Hadley Wickham's *Tidy Data* paper gives the vocabulary: each variable a column, each observation a row, each type of observational unit a table. Feature engineering is often a deliberate move from tidy long tables to a wide design matrix.

~~~dg-flow Moving between the two shapes.
Long events (user, category, amount) -(pivot_table index=user, columns=category)-> *Wide features (one row per user) -(melt id_vars=user)-> Long again
~~~

## pivot and pivot_table

`pivot` rearranges without computing anything. If two rows share the same index and column pair it raises `ValueError: Index contains duplicate entries, cannot reshape`, which is useful: it tells you your assumption of uniqueness was wrong.

```python
wide = long.pivot(index="user", columns="metric", values="value")
```

`pivot_table` aggregates duplicates instead. The default `aggfunc` is `"mean"`, which is rarely what you want for counts or money, so always state it:

```python
spend = tx.pivot_table(index="user_id", columns="category", values="amount",
                       aggfunc="sum", fill_value=0)
spend.columns = [f"spend_{c}" for c in spend.columns]   # flat, prefixed names
spend = spend.reset_index()
```

`margins=True` adds row and column totals for reports. With several `values` or several `aggfunc`s the columns become a MultiIndex; flatten them with `["_".join(map(str, c)) for c in df.columns]` before handing the frame to a model or writing Parquet.

`pd.crosstab(a, b)` counts co-occurrences of two categorical Series, which makes it the quickest confusion matrix: `pd.crosstab(y_true, y_pred, normalize="index")` gives per-class recall along the diagonal.

## melt

`melt` is the inverse: it turns columns into rows.

```python
long = wide.melt(id_vars=["user_id"], value_vars=["q1", "q2", "q3"],
                 var_name="quarter", value_name="revenue")
```

Use it when the same operation should run over many columns, for example normalising 50 sensor channels per device: melt, `groupby(["device", "channel"])`, transform, then pivot back. It is also the format plotting libraries such as seaborn expect. `pd.wide_to_long` handles columns that encode two things, like `score_2023` and `score_2024`.

## stack and unstack

With a MultiIndex, `unstack(level)` moves an index level into the columns and `stack()` moves the innermost column level into the index. A groupby over two keys followed by `unstack` is a pivot:

```python
counts = events.groupby(["user_id", "event_type"]).size().unstack(fill_value=0)
```

`unstack` without arguments uses the innermost level. In pandas 2.1 `stack` gained a new implementation behind `future_stack=True`; in pandas 3.0 that implementation is the default, it no longer drops rows that are all missing, and the old `dropna` and `sort` arguments have no effect and must stay unspecified (both are deprecated). If you relied on `stack()` removing `NaN` rows, add `.dropna()` explicitly.

## Wide features at training and serving

A pivot creates one column per category value *that occurs in the data*. That has two consequences for ML:

- A serving batch with no `"travel"` transactions produces no `spend_travel` column, and a new category produces a column the model has never seen. Fix the layout to the training columns: `X = wide.reindex(columns=TRAIN_COLUMNS, fill_value=0)`.
- A rare category in a long tail creates hundreds of almost-empty columns. Group rare values into `"other"` before pivoting, or keep only the top-k categories learned on training data.

`fill_value=0` is correct for counts and sums (no transactions means zero spend), but wrong for averages or last-known values, where "no data" is not zero. Use `NaN` and let lesson 3's imputation handle it, or add a count column so the model can tell zero from absent.

<div class="ex">
<div class="lab">Worked example</div>

**turning a training log into a comparison table**

A training run logs long records `(step, split, metric, value)`, about 40,000 rows. To compare train and validation loss per step, `log.pivot_table(index="step", columns=["split", "metric"], values="value", aggfunc="last")` gives a MultiIndex column for every split and metric pair.

`pivot` would have failed because a resumed run logged steps 3,000 to 3,100 twice; `aggfunc="last"` keeps the latest record for each step. Flattening the columns to `train_loss`, `val_loss`, `val_acc` gives a wide table where `(t["val_loss"] - t["train_loss"]).rolling(10).mean()` tracks the generalisation gap directly.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

store and transform data long, model it wide, state the aggregation in every `pivot_table`, and pin the wide column set to the training layout so serving cannot add, drop or reorder features.

</div>
