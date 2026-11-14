---
id: "02.04"
course: "ml-toolkit"
module: "02"
slug: "groupby-and-aggregation"
title: "GroupBy and aggregation"
keyIdeas: ["`groupby` is split-apply-combine: split rows by key, apply a function to each group, and combine the results. The function's output shape decides what comes back.", "`agg` reduces each group to one row; `transform` returns a result with the original index, so you can attach a group statistic to every row as a feature; `apply` is the general, slow fallback.", "Named aggregation, `agg(total=(\"amount\", \"sum\"), n=(\"amount\", \"size\"))`, gives flat, readable column names and is the standard way to build a per-entity feature table.", "A group statistic that includes the target or the current row (target encoding, a user's mean label) leaks; compute it on training data only or out of fold."]
question: "You build a click-through model on impression logs with columns `user_id`, `ad_id`, `ts` and `clicked`. A teammate adds `ad_ctr = df.groupby(\"ad_id\")[\"clicked\"].transform(\"mean\")` and `user_n_impr = df.groupby(\"user_id\")[\"ad_id\"].transform(\"size\")` on the full log, then does a random 80/20 split. Validation AUC jumps from 0.71 to 0.93. Explain what each feature leaks, and how you would compute them correctly."
answer: "`ad_ctr` is a target encoding computed over all rows, including the row's own `clicked` value and the validation rows. Rare ads are the worst case: an ad with one impression gets `ad_ctr` equal to its own label, and an ad with a few impressions gets a value strongly correlated with each row's label, so the model reads the answer off the feature. `user_n_impr` uses the full log, including impressions that happen after each row's timestamp, which leaks future activity (users who keep getting impressions are by construction active later), and it is computed partly from validation rows. The random split makes it worse because the same users and ads appear on both sides. Correct versions: split by time first (train on earlier days, validate on later ones); compute `ad_ctr` from training-period rows only, with smoothing such as `(clicks + m * global_ctr) / (impressions + m)`, and map it onto validation rows by `ad_id`, filling unseen ads with the global rate; within the training set use out-of-fold or leave-one-out encoding so a row never sees its own label. For `user_n_impr`, count only impressions strictly before each row's `ts`, e.g. `df.sort_values(\"ts\").groupby(\"user_id\").cumcount()`, which is also exactly what can be computed at serving time."
sources: [{"title": "pandas docs: Group by: split-apply-combine", "url": "https://pandas.pydata.org/docs/user_guide/groupby.html", "note": "Aggregation, named aggregation, transformation, filtration and apply, with performance notes."}, {"title": "Wickham: The split-apply-combine strategy for data analysis (2011)", "url": "https://www.jstatsoft.org/article/view/v040i01", "note": "The paper that named the pattern pandas groupby implements."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Data leakage from statistics computed on the full dataset."}]
practice: ["groupby-basics", "multi-level-groupby", "top-n-per-group", "aggregation-functions", "apply-custom-transforms"]
tags: ["groupby", "aggregation", "target encoding"]
tools: ["pandas"]
---

## Split, apply, combine

Most feature tables are groupbys: one row per user, item or store, with columns such as "number of orders" and "mean basket size". pandas implements this as split-apply-combine.

~~~dg-flow df.groupby("user")["amount"].sum()
Rows -(split by user)-> one group per user -(apply sum)-> *one value per group -(combine)-> Series indexed by user
~~~

```python
g = orders.groupby("user_id")
g["amount"].sum()              # Series indexed by user_id
g.size()                       # rows per group, counts missing values too
g["amount"].count()            # non-missing values per group
g[["amount", "items"]].mean()  # DataFrame, one column per input column
orders.groupby(["user_id", "channel"])["amount"].sum()   # MultiIndex result
```

Grouping keys become the index of the result; `as_index=False` or a final `reset_index()` turns them back into columns, which is usually what you want before a merge (lesson 5). Missing keys are dropped by default; pass `dropna=False` to keep a `NaN` group.

## agg and named aggregation

`agg` takes one function, a list, or a dict per column. For feature tables use **named aggregation**: each keyword becomes a column, and the value is a `(column, function)` pair.

```python
user_feats = orders.groupby("user_id").agg(
    n_orders=("order_id", "nunique"),
    total_spend=("amount", "sum"),
    mean_spend=("amount", "mean"),
    max_items=("items", "max"),
    last_order=("ts", "max"),
    share_mobile=("channel", lambda s: (s == "mobile").mean()),
).reset_index()
```

Built-in names such as `"sum"`, `"mean"`, `"nunique"`, `"first"` run in optimised code; a lambda runs in Python once per group, which is fine for thousands of groups and slow for millions (lesson 8). The older dict-of-lists form produces MultiIndex columns that then need flattening.

## agg, transform, apply, filter

| method | function returns | result shape | use it for
| --- | --- | --- | --- |
| `agg` | a scalar per group | one row per group | feature tables, reports
| `transform` | same length as the group (or a scalar, broadcast) | original index | per-row features: group mean, rank, z-score
| `apply` | anything | depends | complex logic you cannot express otherwise
| `filter` | a bool per group | subset of original rows | keep groups with at least n rows

`transform` is the one people under-use. It lets you compare each row to its group without a merge:

```python
orders["user_mean"] = orders.groupby("user_id")["amount"].transform("mean")
orders["amount_vs_user"] = orders["amount"] / orders["user_mean"]
orders["rank_in_user"] = orders.groupby("user_id")["ts"].rank(method="first")
orders["order_idx"] = orders.groupby("user_id").cumcount()      # 0, 1, 2, ...
```

Top-n per group is a sort followed by `head`: `df.sort_values("score", ascending=False).groupby("user_id").head(10)`. This is faster and clearer than `apply` with `nlargest`.

`apply` passes each group as a DataFrame to your function. In pandas 2.2, if that DataFrame still contains the grouping columns, pandas warns that this is deprecated; pass `include_groups=False` and use the group key from `.name` if you need it. In pandas 3.0 `include_groups=True` is no longer allowed, so the grouping columns are always left out of the frame passed to `apply`.

Grouping by a `category` column used to return every category, including those with no rows (`observed=False`). That default was deprecated in pandas 2.1 and is `observed=True` in 3.0, so only categories that actually occur appear unless you ask otherwise.

## Group statistics and leakage

A group statistic is a feature computed from other rows, so you must ask which rows. Two common traps:

- **Target encoding.** `df.groupby("merchant")["is_fraud"].transform("mean")` gives every row the fraud rate of its merchant, *including its own label*. For a merchant with one transaction the feature equals the label. Compute the encoding on the training fold only, or out of fold, and add smoothing toward the global mean for rare groups.
- **Using the future.** A user's mean spend over the whole table includes orders made after the prediction time. For time-ordered data use expanding or shifted statistics (lesson 7) or compute features as of a cutoff date.

```python
s = df.groupby("merchant")["is_fraud"].transform("sum")
n = df.groupby("merchant")["is_fraud"].transform("size")
df["merchant_rate_loo"] = (s - df["is_fraud"]) / (n - 1)    # leave-one-out; NaN for singletons
```

At serving time the same feature must come from the training-period table, mapped by key: `test["merchant_rate"] = test["merchant"].map(train_rates).fillna(global_rate)`.

<div class="ex">
<div class="lab">Worked example</div>

**a per-user feature table**

A recommendation model needs one row per user from 12 million interaction rows. `interactions.groupby("user_id").agg(n=("item_id", "size"), n_items=("item_id", "nunique"), last=("ts", "max"))` runs in a few seconds because every aggregation is built in.

Adding `top_genre=("genre", lambda s: s.mode().iat[0])` makes the same call take minutes: the lambda runs once for each of 900,000 users. Replacing it with `interactions.groupby(["user_id", "genre"]).size()` followed by sorting by count and taking the first row per user brings it back to seconds.

Both versions are computed only from interactions before the training cutoff, so the table can be joined to labels from after it without leakage.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

use `agg` with named aggregations to build one-row-per-entity features, `transform` to attach group statistics to every row, `apply` only as a last resort, and always ask whether a group statistic can see the label or the future.

</div>
