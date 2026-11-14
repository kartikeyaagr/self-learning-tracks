---
id: "02.06"
course: "ml-toolkit"
module: "02"
slug: "reshaping"
title: "Reshaping"
keyIdeas: ["**Long** data has one row per observation (entity, variable, value); **wide** data has one row per entity and one column per variable. Logs and databases store long; most models consume wide.", "`pivot` reshapes long to wide and fails on duplicate entries; `pivot_table` aggregates duplicates with `aggfunc` (mean by default, so always state it) and can fill empty cells with `fill_value`. `crosstab` is a pivot table of counts.", "Choose `fill_value` by meaning: 0 is right for sums and counts, wrong for averages or last-known values, where an empty cell means 'no data', not zero.", "`melt` goes from wide to long, which is what you want for plotting, storing many metrics, or applying one groupby across many columns. The result has (rows) times (melted columns) rows.", "`stack` and `unstack` move levels between a MultiIndex on rows and on columns; `groupby([a, b]).agg(...).unstack()` is a pivot. In pandas 3, `stack` keeps all-NaN rows, so add `.dropna()` if you relied on the old behaviour.", "A pivot only creates columns for values present in the data. Save the training columns and apply `reindex(columns=train_columns, fill_value=...)` at serving so the model sees exactly the training layout."]
question: "Your fraud features include `n_tx_<category>`, built per user by `pivot_table` over the last 30 days of transactions with `aggfunc=\"count\", fill_value=0`, and `avg_amount_<category>` built the same way with `aggfunc=\"mean\", fill_value=0`. In production, the scoring service rebuilds these columns from the live 30-day window and passes `wide.values` to the model. What can go wrong, and how do you fix both the features and the serving path?"
answer: "Three problems. First, the column set depends on which categories happen to occur in the window: the live data may lack categories seen in training (columns disappear) or contain new ones (columns appear), and because the service passes `wide.values` by position, every feature after the first difference is shifted onto the wrong model weight with no error. For example, if a live user has a purchase in a new category that sorts alphabetically before 'food', its count lands on the weight learned for food. Fix it by saving the training column list and doing `wide.reindex(columns=TRAIN_COLUMNS, fill_value=...)` before scoring, mapping unseen and rare categories to an `other` bucket before pivoting (using the category list learned on training data), and selecting by name rather than position. Second, `fill_value=0` for `avg_amount_<category>` conflates 'no transactions in this category' with 'transactions averaging zero'; leave these as NaN (and impute consistently in both pipelines) or rely on the count column to disambiguate. Counts filled with 0 are correct, because no transactions really means a count of zero. Third, column order from `pivot_table` is sorted by category value, which can differ between environments if category labels are normalised differently (case, whitespace), so 'Food' and 'food' become different columns; apply the same normalisation function in both places. A test that builds features for a fixed sample of users in both the training pipeline and the serving code and compares them column by column (names, order, values, including the reindexed fills) catches all three, and an `assert list(X.columns) == TRAIN_COLUMNS` right before `predict` stops a bad deploy. That comparison is the core of a training-serving skew check."
sources: [{"title": "pandas docs: Reshaping and pivot tables", "url": "https://pandas.pydata.org/docs/user_guide/reshaping.html", "note": "pivot, pivot_table, melt, stack, unstack, crosstab, wide_to_long and get_dummies."}, {"title": "Wickham: Tidy Data (2014)", "url": "https://www.jstatsoft.org/article/view/v059i10", "note": "Why long, tidy tables make transformation easy and how messy layouts map onto them."}, {"title": "pandas docs: MultiIndex / advanced indexing", "url": "https://pandas.pydata.org/docs/user_guide/advanced.html", "note": "How hierarchical row and column indexes work, which stack and unstack move between."}]
practice: ["pivot-tables", "melt-wide-to-long", "unstack-long-to-wide", "cross-tabulation", "training-serving-skew"]
tags: ["reshaping", "pivot", "tidy data"]
tools: ["pandas"]
---

## What you will learn

- The difference between **long** and **wide** tables, and why logs are stored one way while models eat the other.
- How `pivot` and `pivot_table` turn long data into wide features, and why you should always name the `aggfunc`.
- How `melt` goes back from wide to long, and when that is the shape you actually want.
- How `stack` and `unstack` move levels of a MultiIndex between rows and columns (and what changed in pandas 3).
- How to keep a pivoted feature matrix identical between training and serving.

*Before you start:* you should be comfortable with DataFrames and the index (02.01), selecting columns (02.02) and `groupby` (02.04). Lesson 02.05 combined tables side by side and end to end; this lesson keeps one table and changes its **shape** instead.

A model that scores users needs one row per *user*, but the joined table from 02.05 still has one row per transaction. Getting from "one row per event" to "one row per entity" is the job of reshaping.

## The idea in one picture

Here is a tiny card-transaction log. Each row is one purchase:

```python
import pandas as pd
tx = pd.DataFrame({"user": ["u1","u1","u1","u2","u2","u3","u3"],
                   "category": ["food","travel","food","games","food","travel","travel"],
                   "amount": [12, 80, 8, 15, 20, 45, 55]})
```

You want a feature table with one row per user and one column of total spend per category. That is one call:

```python
tx.pivot_table(index="user", columns="category", values="amount",
               aggfunc="sum", fill_value=0)
# category  food  games  travel
# user
# u1          20      0      80
# u2          20     15       0
# u3           0      0     100
```

~~~dg-fig 02.06-longwide.svg
The same seven purchases in two shapes. Colours follow the category. The two darker food rows of u1 merge into one cell (12 + 8 = 20); cells with no purchases become 0 because we asked for fill_value=0.
~~~

Three things happened, and they are the whole lesson in miniature:

- The values of one column (`category`) became **column names**. That is the defining move of a pivot.
- Rows that landed in the same cell (u1 bought food twice) were combined by an **aggregation function**, here `sum`.
- Cells with no matching rows did not exist in the log at all. Pandas has to invent something for them: `NaN` by default, or the `fill_value` you give.

**Long** data (sometimes called **tidy** data) has one row per observation, with columns that say who, what and how much. **Wide** data has one row per entity and one column per variable. Neither is "correct"; they suit different jobs.

## Step 1: know which shape you have and which you need

Quick test: a new kind of measurement adds *rows* to a long table and a *column* to a wide one.

| Shape | One row is | Good for | Typical source
| --- | --- | --- | --- |
| Long | one observation (entity, variable, value) | storage, logs, appending new variables, groupby over many metrics, plotting | event streams, databases, metric loggers
| Wide | one entity | model input (a feature matrix), side-by-side comparison, reports | the output of feature engineering

Wickham's *Tidy Data* paper gives the vocabulary: each variable a column, each observation a row, each kind of observational unit its own table. Feature engineering deliberately moves from tidy long tables to a wide **design matrix**.

~~~dg-flow Moving between the two shapes.
Long events (user, category, amount) -(pivot_table index=user, columns=category)-> *Wide features (one row per user) -(melt id_vars=user)-> Long again
~~~

## Step 2: pivot, when every cell has exactly one value

`DataFrame.pivot` rearranges without computing anything. You say which column supplies the new row labels (`index`), which supplies the new column labels (`columns`), and which fills the cells (`values`):

```python
long = pd.DataFrame({"user": ["u1", "u1", "u2"],
                     "metric": ["clicks", "spend", "clicks"],
                     "value": [12, 30.5, 3]})
long.pivot(index="user", columns="metric", values="value")
# metric  clicks  spend
# user
# u1        12.0   30.5
# u2         3.0    NaN
```

One row per distinct `index` value, one column per distinct `columns` value: 2 by 2 cells, three filled from the log and one `NaN`, because the long table had no row for u2's spend. Wide tables show every gap.

`pivot` refuses when two rows want the same cell, because it has no rule for combining them:

```python
tx.pivot(index="user", columns="category", values="amount")
# ValueError: Index contains duplicate entries, cannot reshape
```

Treat that error as information: your belief "one row per user and category" is false. Decide whether the repeats are bugs (deduplicate, 02.03) or real events (use `pivot_table` with a deliberate aggregation).

## Step 3: pivot_table, when cells need aggregating

`pivot_table` is a `groupby` on the (index, columns) pair followed by a reshape. Every group of rows that lands in one cell is reduced to a single number by `aggfunc`.

The trap is the default. If you do not pass `aggfunc`, pandas uses `"mean"`:

```python
tx.pivot_table(index="user", columns="category", values="amount")
# category  food  games  travel
# user
# u1        10.0    NaN    80.0
# u2        20.0   15.0     NaN
# u3         NaN    NaN    50.0
```

u3 spent 100 on travel, but the table says 50. Nothing errors; the feature is quietly half of what you meant. For money and counts, the mean is rarely what you want, so always write `aggfunc` out loud.

~~~dg-fig 02.06-aggfunc.svg
The value one wide cell receives depends entirely on aggfunc. For u3's two travel purchases (45 and 55) the cell can be 100, 50, 2, 45 or 55. The default, mean, gives half the total spend.
~~~

Pick by the question the feature answers: `"sum"` for totals, `"count"` or `"size"` for how often (`count` skips missing values), `"mean"` or `"median"` for a typical event, `"last"` or `"first"` for the latest or earliest record after sorting by time, `"max"` or `"min"` for extremes.

<div class="ex">
<div class="lab">Worked example</div>

**compute a pivot_table by hand**

Start from the seven rows of `tx`. Group them by (user, category): (u1, food) holds 12 and 8; (u1, travel) holds 80; (u2, games) holds 15; (u2, food) holds 20; (u3, travel) holds 45 and 55. That is 5 non-empty groups.

The output grid has 3 users times 3 categories = 9 cells. 5 cells get a value and 9 - 5 = 4 cells are empty: (u1, games), (u2, travel), (u3, food), (u3, games).

With `aggfunc="sum"`: (u1, food) = 12 + 8 = 20 and (u3, travel) = 45 + 55 = 100. Single-row groups keep their value: 80, 15, 20.

With `aggfunc="mean"`: (u1, food) = 20 / 2 = 10 and (u3, travel) = 100 / 2 = 50. With `aggfunc="count"` both are 2.

Row totals with `sum`: u1 = 20 + 0 + 80 = 100, u2 = 20 + 15 + 0 = 35, u3 = 100. Grand total 235, which matches `tx.amount.sum()` = 12 + 80 + 8 + 15 + 20 + 45 + 55 = 235. A sum pivot never creates or loses money; that equality is a cheap test to put in your pipeline.

</div>

`margins=True` prints exactly that check:

```python
tx.pivot_table(index="user", columns="category", values="amount",
               aggfunc="sum", fill_value=0, margins=True)
# category  food  games  travel  All
# user
# u1          20      0      80  100
# u2          20     15       0   35
# u3           0      0     100  100
# All         40     15     180  235
```

## Step 4: tidy up the columns before a model sees them

A pivot leaves the column labels in an awkward state: their names come from data values (`food`), the column index has a name (`category`), and the users sit in the index rather than in a column. Before saving or modelling, flatten everything:

```python
spend = tx.pivot_table(index="user", columns="category", values="amount",
                       aggfunc="sum", fill_value=0)
spend.columns = [f"spend_{c}" for c in spend.columns]   # flat, prefixed names
spend = spend.reset_index()
#   user  spend_food  spend_games  spend_travel
# 0   u1          20            0            80
# 1   u2          20           15             0
# 2   u3           0            0           100
```

If you pass a list of aggregations, the columns become a two-level **MultiIndex**, one level for the function and one for the category:

```python
m = tx.pivot_table(index="user", columns="category", values="amount",
                   aggfunc=["sum", "count"], fill_value=0)
m.columns.tolist()[:2]        # [('sum', 'food'), ('sum', 'games')]
m.columns = ["_".join(map(str, c)) for c in m.columns]
m.columns.tolist()            # ['sum_food', 'sum_games', 'sum_travel',
                              #  'count_food', 'count_games', 'count_travel']
```

**Choosing fill_value.** `fill_value=0` is correct for sums and counts: no transactions really does mean zero spend. It is wrong for averages and last-known values. "Average food purchase = 0" claims the user bought food for free, while the truth is "no food purchases". Leave such cells as `NaN` and handle them with the imputation methods from 02.03, or keep a count column next to the average so the model can tell zero from absent.

## Step 5: melt, from wide back to long

`melt` is the inverse move. It takes several columns and stacks them into two: one holding the old column *name*, one holding the *value*.

```python
rev = pd.DataFrame({"shop": ["A", "B"], "q1": [10, 7], "q2": [12, 9], "q3": [15, 8]})
long = rev.melt(id_vars=["shop"], value_vars=["q1", "q2", "q3"],
                var_name="quarter", value_name="revenue")
#   shop quarter  revenue
# 0    A      q1       10
# 1    B      q1        7
# 2    A      q2       12
# ...                       (6 rows in total)
```

~~~dg-fig 02.06-melt.svg
melt keeps the id column (shop), turns the three quarter column names into values of a new quarter column, and produces 2 rows times 3 columns = 6 rows. pivot puts it back.
~~~

`id_vars` are repeated on every new row; `value_vars` are folded into rows (default: every other column); `var_name` and `value_name` name the two new columns. The row count is always (rows in the wide table) times (number of `value_vars`). Here $2 \times 3 = 6$. If you melt 10,000 devices with 50 sensor columns, expect 500,000 rows.

Why go back to long? To run one operation over many columns (to standardise 50 sensor channels, melt, `groupby(["device", "channel"])`, `transform`, pivot back), for plotting (seaborn expects long data), and for storage that accepts new metrics without a schema change.

For columns that encode two things at once, such as `score_2023` and `score_2024`, `pd.wide_to_long` splits the name into a stub and a suffix:

```python
sc = pd.DataFrame({"id": [1, 2], "score_2023": [70, 80], "score_2024": [75, 82]})
pd.wide_to_long(sc, stubnames="score", i="id", j="year", sep="_")
#          score
# id year
# 1  2023     70
# 2  2023     80
# 1  2024     75
# 2  2024     82
```

## Step 6: stack and unstack, the MultiIndex view

`pivot` and `melt` work on ordinary columns. `stack` and `unstack` do the same job when the labels already live in a MultiIndex, which is exactly what a `groupby` on two keys produces:

```python
g = tx.groupby(["user", "category"])["amount"].sum()
# user  category
# u1    food         20
#       travel       80
# u2    food         20
#       games        15
# u3    travel      100

g.unstack(fill_value=0)       # same table as the sum pivot_table above
g.unstack(0)                  # move level 0 (user) instead: users become columns
```

- `unstack(level)` moves one row-index level into the columns. With no argument it takes the innermost level.
- `stack()` does the reverse: it moves the innermost column level into the row index, turning a DataFrame into a Series with a longer index.

~~~dg-fig 02.06-stack.svg
unstack turns the five-row Series into a 3 by 3 frame; the four missing pairs appear as NaN. In pandas 3, stack() brings all nine pairs back, NaN included, so you need .dropna() to return to the original five rows.
~~~

**The pandas 3 change.** pandas 2.1 added a new implementation of `stack` behind `future_stack=True`. In pandas 3.0 that implementation is the default. It no longer drops rows that are entirely missing, and the old `dropna` and `sort` arguments have no effect and should not be passed. If older code relied on `stack()` silently removing `NaN` rows, add `.dropna()` explicitly; otherwise a round trip through wide and back adds rows (nine instead of five in the figure).

## Step 7: crosstab, a pivot table of counts

`pd.crosstab(a, b)` counts how often each pair of values from two Series appears together. It is a shortcut for a count pivot, and it is the quickest confusion matrix you can make:

```python
y_true = pd.Series(["cat", "cat", "cat", "dog", "dog", "bird"], name="true")
y_pred = pd.Series(["cat", "dog", "cat", "dog", "dog", "cat"], name="pred")
pd.crosstab(y_true, y_pred)
# pred  cat  dog
# true
# bird    1    0
# cat     2    1
# dog     0    2
pd.crosstab(y_true, y_pred, normalize="index").round(2)
# pred   cat   dog
# bird  1.00  0.00
# cat   0.67  0.33
# dog   0.00  1.00
```

With `normalize="index"` each row sums to 1, so the diagonal is per-class recall: 2 of 3 cats found (0.67). The model never predicted "bird", so there is no `bird` column at all: the missing-column problem again.

## Wide features at training and serving

A pivot creates one column per category value *that occurs in the data you pivoted*. At training time that might be food, games and travel. At serving time you pivot a different slice of data, and the column set can differ.

Suppose a new user u9 arrives with one food purchase of 5 and one purchase of 40 in a category the model never saw, `crypto`:

```python
live = pd.DataFrame({"user": ["u9", "u9"], "category": ["food", "crypto"], "amount": [5, 40]})
lw = live.pivot_table(index="user", columns="category", values="amount",
                      aggfunc="sum", fill_value=0)
lw.columns = [f"spend_{c}" for c in lw.columns]
lw.values                        # [[40  5]]   (crypto first, alphabetically)

TRAIN_COLUMNS = ["spend_food", "spend_games", "spend_travel"]
lw.reindex(columns=TRAIN_COLUMNS, fill_value=0)
#       spend_food  spend_games  spend_travel
# user
# u9             5            0             0
```

~~~dg-fig 02.06-skew.svg
If the serving code passes wide.values by position, crypto's 40 is multiplied by the food weight and food's 5 by the games weight. Reindexing to the saved training columns restores the exact layout.
~~~

A scikit-learn model fitted on a DataFrame complains if names differ, but a raw array or an exported model only sees positions. The rules:

- Save the training column list with the model. At serving, call `wide.reindex(columns=TRAIN_COLUMNS, fill_value=...)` before predicting. Missing columns are added, unseen ones dropped, order fixed.
- Map unseen or rare categories to `"other"` *before* pivoting, using the category list learned on training data, so new values still contribute somewhere.
- Use the same `fill_value` logic as training: 0 for sums and counts, `NaN` for averages.
- Normalise category labels (case, whitespace) with the same function in both places; `"Food"` and `"food "` become different columns.

**Long tails make wide tables huge and empty.** Real category fields follow a long tail: a few common values and hundreds of rare ones. Each value becomes a column, and most users have zero in most of them.

~~~dg-fig 02.06-tail.svg
Measured on 2,000 simulated users with 30,000 purchases over 400 long-tailed categories. Keeping every category gives 400 columns that are 97% zeros; keeping the top 20 (rest merged into "other") gives 21 columns that are 67% zeros.
~~~

Keep the top-$k$ categories by training frequency and fold the rest into `"other"`, or use sparse or embedding features.

<div class="ex">
<div class="lab">Worked example</div>

**turning a training log into a comparison table**

A training run logs long records `(step, split, metric, value)`, about 40,000 rows, with split in {train, val} and metric in {loss, acc}. To compare train and validation loss per step, call `log.pivot_table(index="step", columns=["split", "metric"], values="value", aggfunc="last")`. Because `columns` is a list, the result has a two-level column MultiIndex such as `("val", "loss")`.

Plain `pivot` would have failed with "Index contains duplicate entries": a resumed run logged steps 3,000 to 3,100 twice. Sorting the log by write time and using `aggfunc="last"` keeps the latest record for each step. The default `mean` would have silently averaged the old and new values.

Flatten the columns with `"_".join` to get `train_loss`, `val_loss`, `val_acc`. Now `(t["val_loss"] - t["train_loss"]).rolling(10).mean()` tracks the generalisation gap directly, one row per step.

</div>

## Checking the result

A reshape is easy to get subtly wrong, so add cheap assertions:

- **Shape.** After a pivot, rows should equal the number of distinct index values and columns the number of distinct column values: `assert wide.shape == (tx.user.nunique(), tx.category.nunique())`.
- **Totals.** For a sum pivot, `wide.to_numpy().sum()` must equal the original column's sum (235 in the example). For a count pivot, the total must equal the row count.
- **Uniqueness.** Before `pivot`, `tx.duplicated(["user", "category"]).any()` tells you whether it will fail.
- **Column contract.** At serving, `assert list(X.columns) == TRAIN_COLUMNS` right before `predict`.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| `ValueError: Index contains duplicate entries` | `pivot` met two rows for one cell | deduplicate, or `pivot_table` with an explicit `aggfunc`
| Totals are about half what you expect | `pivot_table` used its default `mean` | always pass `aggfunc="sum"` (or whatever you mean)
| Average features of 0 for users who never bought | `fill_value=0` on a mean | keep `NaN`, impute deliberately, add a count column
| Model scores drift with no error after deploy | serving pivot has different or reordered columns | `reindex(columns=TRAIN_COLUMNS, fill_value=...)`
| Hundreds of nearly empty columns | pivoting a long-tailed category | top-$k$ plus `"other"`, or sparse/embedding features
| Parquet or model rejects column names | MultiIndex or non-string column labels | flatten with `"_".join(map(str, c))`
| Row count grows after a stack round trip | pandas 3 `stack` keeps all-NaN rows | add `.dropna()`
| Melt produced far more rows than expected | `value_vars` defaulted to every non-id column | list `value_vars` explicitly

## Going deeper

**Pivoting is groupby plus unstack.** `df.pivot_table(index=a, columns=b, values=v, aggfunc=f)` is equivalent to `df.groupby([a, b])[v].agg(f).unstack(b)`. Seeing it this way explains its cost: a hash-based grouping that is roughly linear in the number of rows, followed by building a dense grid of size (distinct $a$) times (distinct $b$). The grid, not the grouping, is what blows up memory. With 1 million users and 2,000 categories the dense result has 2 billion cells, about 16 GB as float64, even if only 5 million cells are non-zero. In that regime keep the data long, or build a sparse matrix directly from the integer codes of the two keys (`scipy.sparse.coo_matrix((values, (row_codes, col_codes)))`).

**Lossy moves.** `pivot` and `melt` are inverses only when every (index, columns) pair is unique. An aggregating `pivot_table` is lossy (45 and 55 cannot be recovered from 100), and filled cells look like real ones afterwards.

**Categorical dtype.** If `category` is a pandas `category` dtype whose categories come from training data, `pivot_table(..., observed=False)` emits every declared category, another way to pin the column set.

**get_dummies is a per-row pivot.** `pd.get_dummies(tx["category"])` makes one indicator column per value for each *row*; summing them within each user gives the count pivot, with the same unseen-category problem.

## Key terms

| Term | Meaning
| --- | --- |
| Long (tidy) data | One row per observation: who, which variable, what value
| Wide data | One row per entity, one column per variable; the shape of a feature matrix
| `pivot` | Long to wide with no computation; fails on duplicate cells
| `pivot_table` | Long to wide with an aggregation per cell; default `aggfunc` is mean
| `aggfunc` / `fill_value` | How duplicate rows are combined / what to put in cells with no rows
| `melt` | Wide to long: column names become values of a new column
| `stack` / `unstack` | Move a level from columns to rows / from rows to columns of a MultiIndex
| `crosstab` | A count table of two categorical Series, e.g. a confusion matrix
| Training-serving skew | Features built differently at serving than at training, such as different pivot columns

<div class="take">
<div class="lab">Key takeaway</div>

store and transform data long, model it wide, state the aggregation in every `pivot_table`, choose `fill_value` by what an empty cell means, and pin the wide column set to the training layout with `reindex` so serving cannot add, drop or reorder features.

</div>
