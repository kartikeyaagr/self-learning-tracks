---
id: "02.04"
course: "ml-toolkit"
module: "02"
slug: "groupby-and-aggregation"
title: "GroupBy and aggregation"
keyIdeas: ["`groupby` is **split-apply-combine**: rows are split by a key, a function is applied to each group, and the results are combined, with the keys as the index unless you pass `as_index=False`.", "Named aggregation, `agg(total=(\"amount\", \"sum\"), n=(\"order_id\", \"nunique\"))`, builds a one-row-per-entity feature table with flat, readable column names.", "`agg` returns one row per group, `transform` returns one value per original row (so a group statistic can be attached as a column), `filter` keeps or drops whole groups, and `apply` is the slow general fallback.", "Built-in aggregation names run in compiled code and stay fast with many groups; a lambda runs once per group in Python and was about 100 times slower at 100,000 groups here.", "A group statistic that includes the row's own label (target encoding) or later rows (whole-history counts) leaks. Encode from training rows only, out of fold, smoothed toward the global rate, and use only earlier rows for time-ordered data."]
question: "You build a click-through model on impression logs with columns `user_id`, `ad_id`, `ts` and `clicked`. A teammate adds `ad_ctr = df.groupby(\"ad_id\")[\"clicked\"].transform(\"mean\")` and `user_n_impr = df.groupby(\"user_id\")[\"ad_id\"].transform(\"size\")` on the full log, then does a random 80/20 split. Validation AUC jumps from 0.71 to 0.93. Explain what each feature leaks, and how you would compute them correctly."
answer: "Both features are group statistics computed on the whole log before the split, so they see rows they should not. `ad_ctr` is a target encoding: each row's value includes its own `clicked` label and the labels of validation rows. Rare ads are the worst case: an ad with one impression gets `ad_ctr` exactly equal to its label, and an ad with a handful of impressions gets a value strongly tied to each row's own label, so the model reads the answer off the feature (on pure-noise labels such a feature can still score an AUC far above 0.5). `user_n_impr` counts impressions from the full log, including impressions that happen after each row's timestamp, so it leaks future activity (users who keep getting impressions are by construction active later), and it is partly computed from validation rows. The random split makes both worse because the same users and ads appear on both sides. Correct versions: split by time first (train on earlier days, validate on later ones). Compute `ad_ctr` from training-period rows only, smoothed toward the global rate, for example `(clicks + m * global_ctr) / (impressions + m)` with m around 10 to 100 tuned on validation, and map it onto validation and live rows by `ad_id` with `.map(rates).fillna(global_ctr)` so unseen ads get the global rate. Within the training set, use out-of-fold encoding (each row's rate computed from the other folds, as `sklearn.preprocessing.TargetEncoder` does) so no row sees its own label. For `user_n_impr`, sort by `ts` and count only earlier impressions with `df.sort_values(\"ts\").groupby(\"user_id\").cumcount()`, which is exactly the number available at serving time. Expect validation AUC to drop back towards 0.71 or a little above; that lower number is the honest one."
sources: [{"title": "pandas docs: Group by: split-apply-combine", "url": "https://pandas.pydata.org/docs/user_guide/groupby.html", "note": "Aggregation, named aggregation, transformation, filtration and apply, with performance notes."}, {"title": "Wickham: The split-apply-combine strategy for data analysis (2011)", "url": "https://www.jstatsoft.org/article/view/v040i01", "note": "The paper that named the pattern pandas groupby implements."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Data leakage from statistics computed on the full dataset."}]
practice: ["groupby-basics", "multi-level-groupby", "top-n-per-group", "aggregation-functions", "apply-custom-transforms"]
tags: ["groupby", "aggregation", "target encoding"]
tools: ["pandas"]
---

## What you will learn

- What `groupby` actually does, pictured as three moves: split the rows by a key, apply a function to each piece, combine the answers.
- How to build a one-row-per-user feature table with **named aggregation**, and why built-in function names are up to a hundred times faster than lambdas.
- The difference between `agg`, `transform`, `filter` and `apply`, told by the shape of what each one returns.
- How to attach a group statistic (a user's average spend, a rank, a z-score) to every row without a merge.
- Why a group statistic that can see the label or the future leaks, and how to compute target encodings safely (out of fold, smoothed, mapped from training data).

*Before you start:* you should be able to build a DataFrame and select columns (02.01, 02.02) and know why fitted cleaning steps must learn on training rows only (02.03). Lesson 01.05 on leakage and validation is useful background for the last third.

Lesson 02.03 ended with a clean table, one row per event. Models rarely want that table directly: a churn model wants one row per *customer*, a fraud model wants each transaction compared with the merchant's usual behaviour. Both come from grouping rows that share a key and summarising each group, which in pandas is `groupby`.

## The idea in one picture

Here is a tiny order log: eight orders from three users.

```python
import pandas as pd

orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "user_id":  ["ana", "ben", "ana", "cy", "ben", "ana", "cy", "ben"],
    "amount":   [20, 35, 50, 10, 15, 20, 40, 25],
    "channel":  ["web", "mobile", "mobile", "web", "web", "web", "mobile", "mobile"],
})
orders.groupby("user_id")["amount"].sum()
# user_id
# ana    90
# ben    75
# cy     50
```

The question "how much has each user spent?" is answered in one line, and the figure shows what happened inside it.

~~~dg-fig 02.04-split-apply-combine.svg
Split-apply-combine. The eight rows are sorted into one bucket per user (split), each bucket's amounts are summed (apply), and the three sums are stacked into a Series whose index is the user (combine).
~~~

Three terms carry the rest of the lesson:

- The **key** is the column (or columns) whose value decides which group a row belongs to; here `user_id`.
- A **group** is the set of rows sharing one key value. Ana's group has three rows: 20, 50 and 20.
- An **aggregation** is a function that turns many values into one: sum, mean, count, max.

This pattern, named **split-apply-combine** by Hadley Wickham (2011), is what SQL calls `GROUP BY`. All groupby methods split the same way; they differ only in what the apply step returns.

## Step 1: split, and look at the groups

`orders.groupby("user_id")` does no arithmetic yet. It returns a lazy object that knows which rows belong to each key, and you can inspect it:

```python
g = orders.groupby("user_id")
g.ngroups            # 3
g.size()             # ana 3, ben 3, cy 2   (rows per group)
g.groups["cy"]       # row labels 3 and 6
g.get_group("cy")    # the two rows of cy as a DataFrame
```

Selecting a column, `g["amount"]`, narrows the apply step to it and gives a Series back; a list of columns gives a DataFrame.

You can group by several keys at once. The result is indexed by every combination that occurs, using a **MultiIndex** (an index with several levels):

```python
orders.groupby(["user_id", "channel"])["amount"].sum()
# ana  mobile 50, web 40 | ben  mobile 60, web 15 | cy  mobile 40, web 10
```

Two defaults to know. The keys become the *index* of the result; before a merge (02.05) pass `as_index=False` or call `.reset_index()`. And rows with a missing key are silently dropped: if ana's first order had no `user_id`, her total would print 70. `dropna=False` keeps a visible `NaN` group.

## Step 2: apply a reduction with agg

The most common apply step is a **reduction**: every group becomes one number. In symbols, if $G_u$ is the set of row positions in user $u$'s group and $x_i$ is the amount in row $i$, the per-group mean is

$$\bar x_u = \frac{1}{|G_u|}\sum_{i \in G_u} x_i,$$

where $|G_u|$ is the number of rows in the group. For ana, $G_{\text{ana}}$ holds three rows, so $\bar x_{\text{ana}} = (20 + 50 + 20)/3 = 30$.

`agg` accepts one function name, a list of names, or (best for features) **named aggregation**: each keyword argument becomes an output column, and its value is a `(column, function)` pair.

```python
user_feats = orders.groupby("user_id").agg(
    n_orders=("order_id", "nunique"),
    total=("amount", "sum"),
    mean_spend=("amount", "mean"),
    max_spend=("amount", "max"),
    share_mobile=("channel", lambda s: (s == "mobile").mean()),
).reset_index()
#   user_id  n_orders  total  mean_spend  max_spend  share_mobile
# 0     ana         3     90        30.0         50      0.333333
# 1     ben         3     75        25.0         35      0.666667
# 2      cy         2     50        25.0         40      0.500000
```

That is a complete **feature table**: one row per entity, flat column names you chose, ready to join onto labels. The older dictionary form, `agg({"amount": ["sum", "mean"]})`, produces two-level column names such as `("amount", "sum")` that you then have to flatten; named aggregation avoids that step.

Built-in names worth memorising: `"size"`, `"count"`, `"sum"`, `"mean"`, `"median"`, `"std"`, `"min"`, `"max"`, `"nunique"`, `"first"`, `"last"`. Two of them look alike but differ:

| aggregation | counts | ben's group if one amount is missing
| --- | --- | --- |
| `size` | all rows, missing or not | 3
| `count` | non-missing values only | 2
| `sum` | skips missing values | 40 instead of 75
| `mean` | skips missing values (divides by count) | 20.0 instead of 25.0

pandas quietly skips missing values. If missingness means something, count it as its own feature.

## Step 3: same groups, different output shapes

The apply step does not have to reduce. What it returns decides the shape of the combined result, and that is the cleanest way to tell the four methods apart.

~~~dg-fig 02.04-agg-transform-filter.svg
One split, three outputs. agg gives one row per group; transform gives one value per original row (the group mean repeated for each member) on the original index; filter keeps or drops whole groups and returns the original rows.
~~~

| method | your function returns | result | use it for
| --- | --- | --- | --- |
| `agg` | one value per group | one row per group, indexed by key | feature tables, reports
| `transform` | one value per row (or a scalar, broadcast) | same index and length as the input | per-row features: group mean, rank, z-score
| `filter` | `True` or `False` per group | the original rows of passing groups | keep users with at least 3 orders
| `apply` | anything | depends on what you return | logic none of the above can express

`transform` is the one people forget, and it saves a lot of merging. Because its output lines up with the original rows, you can assign it straight back as a column:

```python
g = orders.groupby("user_id")
orders["user_mean"] = g["amount"].transform("mean")          # 30, 25, 30, 25, 25, 30, 25, 25
orders["amount_vs_user"] = orders["amount"] / orders["user_mean"]
orders["order_idx"] = g.cumcount()                           # 0, 1, 2 within each user
orders["rank_in_user"] = g["amount"].rank(method="first", ascending=False)
```

`filter` keeps whole groups: `g.filter(lambda d: len(d) >= 3)` returns the six rows of ana and ben and drops cy's two.

<div class="ex">
<div class="lab">Worked example</div>

**a per-row "compared with my usual" feature, by hand**

Ana's orders are 20, 50 and 20. Step 1, the group mean: $(20 + 50 + 20)/3 = 90/3 = 30$.

Step 2, `transform("mean")` writes 30 on each of ana's three rows, so the new column has the same length as the table.

Step 3, the ratio `amount / user_mean`: $20/30 = 0.667$, $50/30 = 1.667$, $20/30 = 0.667$. Her 50 order is two thirds above her own usual spend.

Step 4, the group z-score uses the group standard deviation as well. Deviations from the mean are $-10, 20, -10$; their squares sum to $100 + 400 + 100 = 600$; dividing by $n - 1 = 2$ gives variance 300 and standard deviation $\sqrt{300} = 17.32$. So the z-scores are $-10/17.32 = -0.577$, $20/17.32 = 1.155$ and $-0.577$.

pandas agrees: `(orders.amount - g.amount.transform("mean")) / g.amount.transform("std")` gives $-0.577, 1.155, -0.577$ on ana's rows. Note that pandas' `std` divides by $n - 1$ by default, NumPy's by $n$.

</div>

## Step 4: when you need apply, and when you do not

`apply` hands each group to your function as a DataFrame and accepts any return value. The price: your Python function runs once per group, with overhead each time. Since pandas 3.0 the grouping columns are always left out of the frame that `apply` passes in (pandas 2.2 warned about this and offered `include_groups=False`), so read the key from the group's `.name` if you need it.

Most uses of `apply` and of lambdas inside `agg` can be rewritten with built-ins. Top-n per group is a sort followed by `head`:

```python
# each user's biggest order
orders.sort_values("amount", ascending=False).groupby("user_id").head(1)
# rows: ana 50, cy 40, ben 35
```

"Most frequent channel per user" is a count per (user, channel) followed by keeping the largest:

```python
top = (orders.groupby(["user_id", "channel"]).size().rename("n").reset_index()
       .sort_values(["user_id", "n"], ascending=[True, False])
       .drop_duplicates("user_id"))
```

Each runs a few vectorised passes instead of thousands of tiny Python calls, and the difference is large:

~~~dg-fig 02.04-timing.svg
Measured on one machine with one million rows (timings are machine-dependent; the trend is what matters). Built-in "mean" stays around a hundredth of a second however many groups there are; a lambda's cost grows with the number of groups, about 118 times slower at 100,000 groups. A mode lambda took about 6 seconds where size + sort + drop_duplicates took 0.2.
~~~

Rule of thumb: a lambda is fine for a few thousand groups and a problem at hundreds of thousands. Lesson 02.08 covers more speed tricks.

## Step 5: group statistics and leakage

A group statistic is a feature computed *from other rows*, so you must ask which rows. Two traps catch nearly everyone.

**Trap 1: the statistic includes the row's own label.** **Target encoding** replaces a category (merchant, ad, postcode) with the average label in that category. Written naively it is one line: `df.groupby("merchant")["is_fraud"].transform("mean")`. For a merchant with one transaction the feature *equals* the label, and with a few it is strongly tied to it. The model reads the answer off the feature, shines in validation, and fails in production, where the label is not yet known.

To see how strong this effect is, the next figure uses labels that are pure coin flips: no feature can genuinely predict them, so any honest feature should score an AUC of 0.5.

~~~dg-fig 02.04-leakage.svg
Fraud labels that are pure noise, encoded four ways. The naive group mean scores AUC 1.0 when every merchant has one row and is still 0.69 at ten rows per merchant: all of that is leakage. Leave-one-out, out-of-fold and train-only encodings stay near 0.5, which is the truth.
~~~

The fixes, in the order you should reach for them:

1. **Split first, then encode.** Compute the rates on the training rows only, and map them onto validation, test and live rows by key. Keys never seen in training get the global rate.
2. **Inside the training set, encode out of fold.** Split the training rows into, say, 5 folds; each row's encoding comes from the other 4 folds. `sklearn.preprocessing.TargetEncoder` does exactly this inside `fit_transform`.
3. **Smooth toward the global mean.** A rate from 3 rows is mostly noise. Blend it with the overall rate.

The smoothed rate for a group with $s$ positives in $n$ rows is

$$\text{rate} = \frac{s + m\,p}{n + m},$$

where $p$ is the global positive rate and $m$ is a pseudo-count: how many "imaginary average rows" you add to every group. When $n$ is much smaller than $m$, the rate stays close to $p$; when $n$ is much larger, it approaches the raw $s/n$.

~~~dg-fig 02.04-smoothing.svg
220 ads that all truly convert at 5%. Left: raw rates of small ads scatter between 0% and 50% purely by chance. Right: with m = 20 the small ads are pulled toward the global rate, while large ads keep their own rate.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**three ways to encode a merchant, by hand**

Training data: merchant "kiosk" has four transactions with labels $1, 0, 0, 0$; merchant "atm7" has one transaction with label $1$. Assume the global fraud rate is $p = 0.02$ and the pseudo-count is $m = 10$.

Naive mean: kiosk gets $1/4 = 0.25$ on all four rows; atm7 gets $1/1 = 1.0$, which is exactly its label. That single row is a perfect leak.

Leave-one-out: kiosk has sum $s = 1$ and size $n = 4$. The fraud row gets $(1 - 1)/(4 - 1) = 0$; each clean row gets $(1 - 0)/3 = 0.333$. atm7 gets $(1 - 1)/0 =$ undefined (NaN), so it needs a fallback.

Smoothed: kiosk gets $(1 + 10 \times 0.02)/(4 + 10) = 1.2/14 = 0.0857$; atm7 gets $(1 + 0.2)/(1 + 10) = 1.2/11 = 0.109$. One fraud out of one is no longer read as "100% fraud".

At serving time a new transaction at kiosk is given 0.0857, and one at a merchant never seen in training is given the global 0.02.

</div>

In code, with the training table `train` and anything else as `test`:

```python
p, m = train["is_fraud"].mean(), 10
stats = train.groupby("merchant")["is_fraud"].agg(["sum", "size"])
rates = (stats["sum"] + m * p) / (stats["size"] + m)        # one value per merchant
test["merchant_rate"] = test["merchant"].map(rates).fillna(p)
```

**Trap 2: the statistic includes the future.** A user's mean spend over the whole log includes orders placed after the moment you are predicting for. For time-ordered data, sort by time and use only earlier rows:

```python
o = orders.sort_values("ts")
gu = o.groupby("user_id")["amount"]
o["n_prev"] = o.groupby("user_id").cumcount()                 # orders before this one
o["mean_prev"] = (gu.cumsum() - o["amount"]) / o["n_prev"]    # NaN on a user's first order
# ana: NaN, 20.0, 35.0
```

That is exactly what serving can compute, the test of a leak-free feature. Lesson 02.07 extends this with time windows.

## Checking the result

Groupby bugs are quiet: the code runs and the numbers are plausible. A few checks catch most of them.

- **Row counts.** A per-user feature table must have exactly `orders["user_id"].nunique()` rows (plus one if you kept a `NaN` group). A `transform` result must have `len(orders)` rows. If not, a key has stray whitespace, mixed types (`"42"` versus `42`) or missing values.
- **Totals survive.** `user_feats["total"].sum()` should equal `orders["amount"].sum()`. If it is lower, rows were dropped (missing keys).
- **Spot-check one group by hand.** Filter one user's rows and recompute one feature.
- **Size distribution.** `g.size().describe()` tells you how many groups are tiny. Rates, means and standard deviations from groups of one or two rows are mostly noise and deserve smoothing or a minimum-count rule.
- **Compare train and validation scores** for target-style features: AUC 0.95 in training and 0.6 in validation means a leak.

## What goes wrong

| symptom | cause | fix
| --- | --- | --- |
| Feature looks amazing in validation, useless in production | target encoding or count computed on all rows | split first; encode from training rows, out of fold, with smoothing
| Model relies on "number of events per user" that is not known at prediction time | statistic included future rows | sort by time; `cumcount`, shifted or expanding statistics
| Totals smaller than expected | rows with missing keys dropped silently | `dropna=False`, then decide what the `NaN` group means
| Merge after groupby fails or duplicates rows | key is in the index, not a column | `as_index=False` or `reset_index()`
| Column names like `("amount", "sum")` | dict-of-lists `agg` | named aggregation
| Feature job takes minutes | lambda or `apply` over many groups | built-in names; sort + `head` or `drop_duplicates`
| Grouping by a category column returns empty groups, or stops returning them | `observed` default changed (False before 3.0, True since) | pass `observed=True` or `observed=False` explicitly
| "Top channel" changes between runs | ties broken arbitrarily | add an explicit tie-break column to the sort

On the `observed` row: grouping by a `category` column used to return every declared category, even empty ones. That default was deprecated in pandas 2.1 and is `observed=True` in 3.0, so spell the argument out.

## Going deeper

**How the split is done.** pandas first **factorises** the key: it maps every distinct key value to an integer code $0, \dots, K-1$, where $K$ is the number of groups. Built-in reductions then run one compiled pass over the column, adding each value into slot `codes[i]` of an output array of length $K$. That is roughly $O(n)$ work for $n$ rows, plus the cost of factorising (a hash table, also about $O(n)$), and sorting the $K$ group keys when `sort=True` (the default; pass `sort=False` to skip it). A Python-level function instead pays a fixed overhead per group, so its cost grows like $K$ times that overhead, which is the rising dashed lines in the timing figure.

**Which aggregations combine across chunks.** Sum, count, min and max can be computed per chunk and combined; mean is sum divided by count; variance can be combined from per-chunk count, mean and sum of squared deviations. Median, `nunique` and mode cannot be combined exactly from per-chunk summaries. In Spark or Dask, decomposable aggregations scale easily; the others need a shuffle by key.

**Leave-one-out is subtler than it looks.** In the noise experiment, leave-one-out scored 0.5 on its own, because each row's value depends only on the other rows. But within one merchant, the fraud rows always get a slightly *lower* value than the clean rows (0 versus 0.333 in the example). A flexible model that also sees the merchant identity or its row count can learn that pattern and recover the label. Out-of-fold encoding with five or so folds breaks the link more completely, which is why scikit-learn's `TargetEncoder` uses cross fitting.

**Smoothing as Bayesian shrinkage.** The smoothed rate $(s + mp)/(n + m)$ is the posterior mean of a Beta prior with mean $p$ and strength $m$, updated with $s$ successes in $n$ trials. A group with $n = m$ rows is trusted exactly as much as the global rate. Tune $m$ on validation data, or let `TargetEncoder(smooth="auto")` estimate it.

## Key terms

| Term | Meaning
| --- | --- |
| Split-apply-combine | Partition rows by a key, run a function on each part, stitch the results together
| Key | The column or columns whose values define the groups
| Aggregation (reduction) | A function turning a group's values into one value, such as sum or mean
| Named aggregation | `agg(new_name=("column", "func"))`, giving flat, chosen output column names
| `transform` | Group operation whose result has the original index, one value per row
| `filter` | Keeps or drops whole groups by a true/false test on each group
| Target encoding | Replacing a category with the average label of that category
| Smoothing | Blending a group's rate with the global rate, weighted by group size
| Out-of-fold | Computing a row's feature only from folds that do not contain it

<div class="take">
<div class="lab">Key takeaway</div>

think of every groupby as split, apply, combine, and pick the method by the shape you need back: `agg` for one row per entity, `transform` for one value per row, `filter` to keep whole groups, `apply` only when nothing else fits. Prefer built-in aggregation names, and before trusting any group statistic ask whether it could see the row's own label or the future.

</div>
