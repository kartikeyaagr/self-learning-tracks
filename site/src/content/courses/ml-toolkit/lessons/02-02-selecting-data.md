---
id: "02.02"
course: "ml-toolkit"
module: "02"
slug: "selecting-data"
title: "Selecting data"
keyIdeas: ["`.loc` selects by **label** and its slices include the end; `.iloc` selects by **position** and its slices exclude the end. After any sort or filter the two disagree, so pick the one that matches what you mean.", "Plain `df[...]` should get only a column name, a list of names (which also fixes column order for a feature matrix) or a boolean mask.", "Combine conditions with `&`, `|` and `~`, each condition in parentheses; Python's `and`/`or` raise an error on a Series. `isin`, `between` and `query` keep longer filters readable.", "Write with a single `df.loc[row_selector, column] = value`. Chained forms like `df[mask][\"col\"] = v` write into a temporary object and never reach `df`.", "Copy-on-Write is opt-in in pandas 2.x and the only behaviour in pandas 3.0: every selection behaves like a copy, `SettingWithCopyWarning` is gone, chained writes raise `ChainedAssignmentError`, and defensive `.copy()` calls are no longer needed.", "Attach model predictions as a NumPy array (assigned by position) or as a Series built with the frame's own index; a fresh `pd.Series(preds)` aligns by label and scrambles them."]
question: "You sort a validation DataFrame by `timestamp`, filter it to rows with `segment == \"new_users\"`, call `preds = model.predict(val[FEATURES])` and then write `val.loc[:, \"pred\"] = preds` followed by `metrics = val.groupby(\"day\").apply(...)`. Separately, a teammate computes `val_sorted.loc[0:99]` to look at \"the first 100 rows\". Which of these steps are correct, which are not, and why?"
answer: "Assigning a NumPy array with `val.loc[:, \"pred\"] = preds` is correct: an array has no index, so pandas places it by position in the frame's current row order, which is exactly the order in which `val[FEATURES]` was passed to the model. The risky variant would be building `pd.Series(preds)` without an index and assigning that. Its labels run 0..m-1, while `val` still carries the original labels from before sorting and filtering, so pandas aligns by label: some rows receive another row's prediction and rows whose labels are not in 0..m-1 get NaN, with no error raised. Either keep the raw array, wrap it as `pd.Series(preds, index=val.index)`, or call `reset_index(drop=True)` before predicting. Because `FEATURES` is an explicit list, the column order matches training, and a missing column would raise a `KeyError`. The teammate's `val_sorted.loc[0:99]` is wrong. `.loc` slices by label and includes the end. After sorting, the index is no longer monotonic, so pandas finds the positions where labels 0 and 99 happen to sit and returns everything between them: possibly thousands of rows, an empty frame if label 99 comes before label 0, or a `KeyError` if either label was removed by the filter. The positional intent is `val_sorted.iloc[:100]` or `val_sorted.head(100)`. Finally, if the filtered frame is modified later, write with a single `.loc` call on the object you want to change rather than through a chained selection such as `val[mask][\"pred\"] = ...`, because under Copy-on-Write (pandas 3.0) chained writes are discarded with a `ChainedAssignmentError` warning."
sources: [{"title": "pandas docs: Indexing and selecting data", "url": "https://pandas.pydata.org/docs/user_guide/indexing.html", "note": "loc, iloc, boolean indexing, query and the section on returning a view versus a copy."}, {"title": "pandas docs: Migration guide for pandas 3.0 (Copy-on-Write)", "url": "https://pandas.pydata.org/docs/user_guide/migration.html", "note": "The CoW rule, chained assignment, read-only NumPy arrays and how to migrate code."}, {"title": "pandas docs: Essential basic functionality", "url": "https://pandas.pydata.org/docs/user_guide/basics.html", "note": "Reindexing and alignment, which explain why a Series assignment can land on the wrong rows."}]
practice: ["loc-vs-iloc", "boolean-indexing", "multi-column-selection", "column-selection", "resetting-index"]
tags: ["indexing", "boolean masks", "Copy-on-Write"]
tools: ["pandas"]
---

## What you will learn

- The four selection tools (`df[...]`, `.loc`, `.iloc`, boolean masks) and the one question each answers.
- Why `.loc` slices include their end and `.iloc` slices do not, and why the two disagree after any sort or filter.
- How to build row filters from conditions with `&`, `|` and `~`, plus the readable helpers `isin`, `between` and `query`.
- How to pick a feature matrix whose columns and order cannot drift between training and serving.
- How to write values so the change really lands, and what Copy-on-Write (the default in pandas 3.0) changed about that.

*Before you start:* you should know what a DataFrame, its index and its columns are, and how two Series align on labels (lesson 02.01). Comfort with NumPy slicing and boolean arrays (module 01) helps, because pandas reuses both ideas.

Lesson 02.01 loaded data into a DataFrame and showed that every row carries a **label** in the index and every column carries a name. Almost every later pipeline step starts by picking part of that table: the training rows, the feature columns, the bad values to fix. This lesson is about saying precisely which part you mean.

## The idea in one picture

Here is a tiny customer table. The index labels are 0 to 5, in order, because we just built it.

```python
import pandas as pd
df = pd.DataFrame({"plan":    ["free", "pro", "team", "pro", "free", "team"],
                   "country": ["NL", "DE", "NL", "FR", "NL", "DE"],
                   "age":     [23, 35, 41, 31, 52, 19],
                   "spend":   [0, 48, 120, 30, 0, 95]})
top = df.sort_values("spend", ascending=False)   # biggest spenders first
```

After the sort, the rows are in a new order but they keep their old labels. Reading down the table, the labels are now 2, 5, 1, 3, 0, 4. So every row has two different "numbers": its **label** (the value in the index, which stays attached to the row) and its **position** (where it sits right now, counting from 0).

~~~dg-fig 02.02-loc-iloc.svg
The sorted table twice. The grey "pos" column is where each row sits now; the coloured "label" column is its index value. `.loc[5:3]` walks from label 5 to label 3 and includes it (3 rows); `.iloc[1:3]` takes positions 1 and 2 and stops before 3 (2 rows).
~~~

- `top.loc[5:3]` means "from the row labelled 5 to the row labelled 3, inclusive". That is positions 1, 2 and 3: three rows.
- `top.iloc[1:3]` means "positions 1 up to but not including 3", like a Python list. That is two rows.

Both start at the same row yet return different data. Most pandas selection bugs come from mixing up labels and positions.

## Step 1: columns with plain brackets

Square brackets directly on the frame, `df[...]`, are the shortest tool, and you should use them for exactly three things.

| what you pass | what you get | example
| --- | --- | --- |
| one column name | a Series | `df["spend"]`
| a list of names | a DataFrame, columns in the list's order | `df[["age", "spend"]]`
| a boolean Series | the rows where it is True | `df[df["age"] >= 40]`

In the second row the inner brackets are a Python list: `df[["spend"]]` is a (6, 1) DataFrame, while `df["spend"]` is a (6,) Series. scikit-learn wants a 2-D `X`, so this matters for a single feature.

Do not pass integers or slices to plain brackets: `df[0]` looks for a *column* named 0, and `df[1:3]` is a confusing positional special case. Use `.loc` or `.iloc`.

**Picking a feature matrix.** For a model, write the feature list out once and select with it:

```python
FEATURES = ["age", "spend"]            # fixed list, fixed order
X = df[FEATURES].to_numpy()            # shape (6, 2), int64
df[["age", "income"]]                  # KeyError: "['income'] not in index"
```

Once data becomes an array, a model knows columns only by position: weight 0 goes with column 0. If column order differs between training and serving, weights silently hit the wrong features. An explicit list fixes which columns go in and their order, and a missing column fails loudly with a `KeyError`.

## Step 2: `.loc`, selection by label

`.loc` is the general tool. It takes a row selector, a comma, and a column selector:

$$\texttt{df.loc[}\;\text{rows}\;,\;\text{columns}\;\texttt{]}$$

Each selector can be a label, a list of labels, a slice `a:b`, a boolean mask, or `:` for "everything".

```python
df.loc[2, "spend"]                       # 120         one cell
df.loc[[1, 3], ["plan", "age"]]          # rows 1 and 3, two columns
df.loc[:, "country":"spend"]             # columns country, age, spend (end included)
df.loc[df["age"] >= 40, ["plan", "spend"]]   # rows 2 and 4: a mask plus columns
```

Why does a label slice include its end? Labels are not always numbers. With a date index, `df.loc["2024-01-01":"2024-01-31"]` should give all of January; excluding the end would force you to know the label after 31 January, which may not exist.

On a sorted index, a label slice is found with binary search and includes every label in the range, even ones that do not exist exactly. On an unsorted index (like `top`), pandas instead finds the *positions* of the two end labels and takes everything between them. That gives the surprising results in the picture, and two more:

- `top.loc[0:3]` is an **empty** frame: label 0 sits at position 4, label 3 at position 3, and you cannot walk forwards from 4 to 3.
- `top.loc[0:99]` raises `KeyError: Cannot get right slice bound for non-monotonic index with a missing label 99`.

## Step 3: `.iloc`, selection by position

`.iloc` has the same `[rows, columns]` shape, but every selector is an integer position, and slices behave exactly like Python lists and NumPy arrays: start included, end excluded, negatives count from the back.

```python
df.iloc[0]               # first row, whatever its label
df.iloc[:2, -2:]         # first two rows, last two columns
df.iloc[-2:, [0, 2]]     # last two rows, columns 0 and 2
```

Use `.iloc` when the meaning is genuinely positional: "the first 1,000 rows" or "the last 20% for a time-ordered validation split". `df.head(n)` is shorthand for `df.iloc[:n]`.

A pandas 3.0 change: on a Series, `s[0]` used to fall back to position when there was no label 0. That fallback was deprecated in 2.1 and removed in 3.0, so with string labels `s[0]` raises `KeyError`, and with integer labels it means the label. For "first element", write `s.iloc[0]`.

<div class="ex">
<div class="lab">Worked example</div>

**labels versus positions, by hand**

Start from `top`, the table sorted by spend. Reading down, the labels are $(2, 5, 1, 3, 0, 4)$ at positions $(0, 1, 2, 3, 4, 5)$ and the spends are $(120, 95, 48, 30, 0, 0)$.

`top.iloc[0]`: position 0 holds label 2, so this is the team customer from NL with spend 120. `top.loc[0]`: label 0 sits at position 4, so this is the free customer aged 23 with spend 0. Same number, different rows.

`top.loc[5:3]`: label 5 is at position 1 and label 3 is at position 3. The index is not sorted, so pandas takes positions 1 to 3 inclusive: labels 5, 1, 3, spends 95, 48, 30, for a total of $95 + 48 + 30 = 173$.

`top.iloc[1:3]`: positions 1 and 2 only, so labels 5 and 1, total $95 + 48 = 143$. The difference, 30, is exactly the end row that `.loc` included.

To get "the top three spenders" without thinking about labels at all, use `top.iloc[:3]` or `top.head(3)`: spends 120, 95 and 48. If you prefer labels that match positions again, call `top.reset_index(drop=True)`; after that, `.loc[0:2]` and `.iloc[:3]` select the same three rows.

</div>

## Step 4: filtering rows with boolean masks

A **boolean mask** is a Series of True and False values with the same index as the frame. Comparing a column to something produces one; passing it to `df[...]` or `.loc` keeps the rows where it is True.

Python's `and`, `or` and `not` cannot combine masks: they ask "is this whole Series true?", which has no single answer, so pandas raises `ValueError: The truth value of a Series is ambiguous`. Use the element-wise operators instead:

| operator | meaning | row kept when
| --- | --- | --- |
| `a & b` | and | both conditions are True
| `a|b` | or | at least one is True
| `~a` | not | the condition is False

~~~dg-fig 02.02-mask.svg
Each condition is a column of True and False lined up on the row labels. `&` keeps rows 1, 2 and 3 (both True); `|` keeps rows 1 to 5 (at least one True). Only row 0, a 23-year-old on the free plan, fails both.
~~~

The filter from the figure:

```python
adult = df["age"] >= 25             # F T T T T F  -> 4 True
paid  = df["plan"] != "free"        # F T T T F T  -> 4 True
df[adult & paid]                    # rows 1, 2, 3
df[adult | paid]                    # rows 1, 2, 3, 4, 5
df[adult & ~paid]                   # row 4: an adult on the free plan
```

Every condition needs its own parentheses when you write them inline: `(df["age"] >= 25) & (df["plan"] != "free")`. In Python, `&` binds more tightly than `>=` and `!=`, so without parentheses the expression is read as `df["age"] >= (25 & df["plan"]) != "free"`, which fails with a `TypeError` or the "truth value is ambiguous" error. Naming each condition, as above, avoids the problem and documents the filter.

Three helpers keep longer filters readable:

```python
df[df["country"].isin(["NL", "FR"])]          # rows 0, 2, 3, 4: membership in a set
df[df["spend"].between(30, 95)]               # rows 1, 3, 5: both ends included by default
min_age = 25
df.query("age >= @min_age and plan in ['pro', 'team']")   # rows 1, 2, 3
```

Inside a `query` string, `and`, `or` and `in` are allowed because pandas parses the string itself, and `@name` refers to a Python variable.

**Missing values in a mask.** A nullable `boolean` mask holding `pd.NA` treats NA as False, silently dropping those rows; an object mask containing `NaN` raises `ValueError: Cannot mask with non-boolean array containing NA / NaN values`. Decide what "unknown" means with `mask.fillna(False)` or `fillna(True)`.

**Masks align by label.** A mask built from a different frame is matched on index labels, not positions. If its labels do not cover the frame, pandas raises `IndexingError: Unalignable boolean Series`.

## Step 5: writing values so they stick

The safe write pattern is always **one indexing call that names both the rows and the column**:

```python
df.loc[df["age"] < 18, "age"] = pd.NA            # fix invalid values in place
df["spend_log"] = np.log1p(df["spend"])          # new column for every row
df = df.assign(is_paid=df["plan"] != "free")     # functional style, returns a new frame
```

The unsafe pattern is **chained assignment**: two indexing steps in a row, where the second one writes.

~~~dg-flow Chained assignment: the first step builds a new object, and the write lands on that object instead of on df.
df[mask] -(new object)-> temp -(["col"] = v)-> *temp modified -> df unchanged
~~~

`df[mask]` creates a temporary DataFrame; `["income"] = cap` writes into it, and it is thrown away at the end of the line. `df` never sees the change. The same holds for `df["income"][mask] = cap`.

~~~dg-fig 02.02-chained.svg
3,000 simulated incomes with a cap of 150k (92 customers are above it). Left: the chained write is lost, so the maximum is still 322k. Right: a single `.loc` write caps every one of them.
~~~

Before pandas 3.0, whether `df[mask]` shared memory with `df` (a **view**) or held its own data (a **copy**) depended on internal details such as the column dtypes. A chained write therefore *sometimes* changed `df` and sometimes did not. pandas printed the vague `SettingWithCopyWarning`.

## Copy-on-Write

**Copy-on-Write** (CoW) replaces that guesswork with one rule: *any object derived from a DataFrame or Series behaves like an independent copy*. Memory is still shared behind the scenes and only copied when one side is written to.

- pandas 2.0 to 2.2: CoW is opt-in with `pd.options.mode.copy_on_write = True`; 2.2 also accepts `"warn"`, which flags every place where your code's behaviour will change.
- pandas 3.0: CoW is always on. `SettingWithCopyWarning` no longer exists. A chained assignment never modifies the original, and pandas emits a `ChainedAssignmentError` warning to tell you the write was lost.
- Modifying a subset never touches the parent: after `train = df[df["split"] == "train"]`, `train["x"] = 0` changes only `train`. Defensive `.copy()` calls added to silence warnings can go.
- Arrays from `df.to_numpy()` or `s.values` may be read-only views. Writing into one raises `ValueError: assignment destination is read-only`; take `np.array(arr, copy=True)` first if you need to modify it in place.

<div class="ex">
<div class="lab">Worked example</div>

**the clipping that never happened**

A preprocessing script contains `df[df["income"] > cap]["income"] = cap`. In pandas 1.x it printed `SettingWithCopyWarning` and did nothing; in 3.0 it prints `ChainedAssignmentError` and still does nothing. Either way the model trains on uncapped incomes.

In the simulated data of the figure, 92 of 3,000 customers earn more than the 150k cap, and the largest income is 322k. A linear model fits squared errors, so those few extreme rows pull its weights much more than their 3% share of the data suggests.

The fix is one call, `df.loc[df["income"] > cap, "income"] = cap`, or `df["income"] = df["income"].clip(upper=cap)`. The maximum is then exactly 150k.

A one-line test, `assert df["income"].max() <= cap`, would have caught it in every version.

</div>

## Selecting data in code

The patterns of this lesson on the six-row table, with printed results in the comments.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"plan": ["free", "pro", "team", "pro", "free", "team"],
                   "country": ["NL", "DE", "NL", "FR", "NL", "DE"],
                   "age": [23, 35, 41, 31, 52, 19],
                   "spend": [0, 48, 120, 30, 0, 95]})

df["spend"].shape, df[["spend"]].shape        # (6,)  (6, 1)
df.loc[2, "spend"], df.iloc[2, 3]             # 120 120
df.select_dtypes("number").columns.tolist()   # ['age', 'spend']

top = df.sort_values("spend", ascending=False)
top.index.tolist()                            # [2, 5, 1, 3, 0, 4]
len(top.loc[5:3]), len(top.iloc[1:3])         # 3 2
top.head(3)["spend"].tolist()                 # [120, 95, 48]

mask = (df["age"] >= 25) & (df["plan"] != "free")
df.loc[mask].index.tolist()                   # [1, 2, 3]
df.loc[mask, "spend"].sum()                   # 198

df["spend_capped"] = df["spend"].clip(upper=50)
df["spend_capped"].tolist()                   # [0, 48, 50, 30, 0, 50]
```

Note the habits: the mask is built once and named, and every write is a single statement on `df` itself.

## Checking the result

A selection that runs is not necessarily the one you meant. Quick checks:

1. **Shape.** Print `sub.shape` and compare the row count with what you expected. `mask.sum()` tells you how many rows a filter will keep before you apply it, and `mask.mean()` gives the fraction.
2. **Index.** If `sub.index[:5]` shows labels like 4012, 17, 9333, positional code (`.iloc`, arrays) and label code (`.loc`, Series) now disagree.
3. **Writes.** After a write, assert the property you wanted (`df["income"].max() <= cap`, `df["age"].min() >= 0`). In tests, turn warnings into errors with `python -W error` or `pytest -W error` so a `ChainedAssignmentError` fails the build.
4. **Attaching predictions** is where label/position confusion costs the most:

~~~dg-fig 02.02-align.svg
400 rows, sorted by x and then filtered to x > 2, so their labels are scrambled. Left: predictions attached as a NumPy array go to the right rows. Right: the same numbers wrapped in `pd.Series(preds)` carry labels 0, 1, 2, ..., so they are matched by label: 19% of rows get NaN and the rest get another row's prediction.
~~~

A NumPy array has no index, so `val["pred"] = preds` places it by position, in the order rows went to the model. A Series is aligned by label, so give it the right labels, `pd.Series(preds, index=val.index)`, or call `val.reset_index(drop=True)` before predicting.

## What goes wrong

| mistake | symptom | fix
| --- | --- | --- |
| `top.loc[0:99]` to get "the first 100 rows" | wrong rows, an empty frame, or a `KeyError` after a sort or filter | `top.iloc[:100]` or `top.head(100)`
| `s[0]` for the first element | `KeyError` in pandas 3.0 on a string index, or the wrong element on an integer index | `s.iloc[0]`
| `and` / `or` between conditions | `ValueError: truth value of a Series is ambiguous` | `&`, `|`, `~`, each condition in parentheses
| nullable mask with `pd.NA` | rows silently dropped | `mask.fillna(False)` or `fillna(True)` on purpose
| chained write `df[mask]["col"] = v` | `ChainedAssignmentError`, values unchanged | `df.loc[mask, "col"] = v`
| `pd.Series(preds)` assigned to a filtered frame | shuffled predictions, some NaN, metrics far worse than expected | assign the array, or `pd.Series(preds, index=X.index)`
| feature list built from `df.columns` | new or reordered columns reach the model | a fixed `FEATURES` list, `df[FEATURES]`

## Going deeper

**How lookups are found.** For a unique index, pandas builds a hash table on first lookup, so `df.loc[label]` costs about the same at a hundred rows or ten million. On a sorted index, label slices use binary search, which is why they accept bounds that are not present; an unsorted index must find the exact end labels, hence the `KeyError`. If you slice by label often, call `df.sort_index()` once.

**Vectorised filtering versus loops.** A mask is computed for the whole column at once in compiled code. Row-by-row approaches pay Python overhead on every row.

~~~dg-fig 02.02-timing.svg
Measured on one machine (timings are machine-dependent; the two slowest bars are extrapolated from 100,000 rows). Left: filtering 1,000,000 rows with a mask, `isin` or `query` takes 120 to 150 ms, versus about 7.6 s for a row-wise `apply` and about 30 s for an `iterrows` loop. Right: under Copy-on-Write, selecting 10 columns takes 0.2 ms, while forcing a real copy takes about 59 ms.
~~~

The three vectorised forms are close, so choose between them for readability. Here the text comparison `plan != "free"` dominated their cost; numeric conditions are faster still.

**Copy-on-Write under the hood.** The right panel shows the "lazy" part of CoW. `df[cols]` creates a new DataFrame object that points at the same column arrays and records that they are shared. Nothing is copied until either side writes; at that moment only the touched column is copied. You can watch this with `np.shares_memory`: right after `sub = df[["age", "spend"]]`, the `age` arrays of `sub` and `df` share memory; after `sub.loc[0, "age"] = 99` they no longer do, and `df` still holds 23. It is also why `to_numpy()` may return a read-only array: that memory may still be shared.

## Key terms

| Term | Meaning
| --- | --- |
| Label | The value in the index attached to a row; it moves with the row when you sort or filter
| Position | Where a row sits right now, counting from 0
| `.loc` | Selection by label or boolean mask; slices include the end
| `.iloc` | Selection by integer position; slices exclude the end
| Boolean mask | A True/False Series aligned on the index that keeps the True rows
| Chained assignment | Two indexing steps where the second writes, so the value lands on a temporary object
| Copy-on-Write | pandas rule that every derived object acts as a copy, with memory copied only when written
| Alignment | Matching values from two objects by index label rather than by position

<div class="take">
<div class="lab">Key takeaway</div>

decide whether you mean labels or positions and say it with `.loc` or `.iloc`; combine conditions with parenthesised `&`, `|` and `~`; and make every write a single `df.loc[rows, column] = value`, which under Copy-on-Write is the only form that reaches the original.

</div>
