---
id: "02.02"
course: "ml-toolkit"
module: "02"
slug: "selecting-data"
title: "Selecting data"
keyIdeas: ["`.loc` selects by **label** and its slices include the end; `.iloc` selects by **position** and its slices exclude the end. Plain `df[...]` takes a column name, a list of names or a boolean mask, and nothing else should be relied on.", "Combine boolean conditions with `&`, `|` and `~`, each condition in parentheses; Python's `and`/`or` raise an error on Series. `isin`, `between` and `query` keep longer filters readable.", "Write with a single `df.loc[row_selector, column] = value`. Chained forms like `df[mask][\"col\"] = v` write into a temporary object and never reach `df` under Copy-on-Write.", "Copy-on-Write is opt-in in pandas 2.x and the only behaviour in pandas 3.0: every selection behaves like a copy, `SettingWithCopyWarning` is gone, and defensive `.copy()` calls are no longer needed for correctness."]
question: "You sort a validation DataFrame by `timestamp`, filter it to rows with `segment == \"new_users\"`, call `preds = model.predict(val[FEATURES])` and then write `val.loc[:, \"pred\"] = preds` followed by `metrics = val.groupby(\"day\").apply(...)`. Separately, a teammate computes `val_sorted.loc[0:99]` to look at \"the first 100 rows\". Which of these steps are correct, which are not, and why?"
answer: "Assigning a NumPy array with `val.loc[:, \"pred\"] = preds` is correct: an array has no index, so pandas assigns it by position in the frame's current row order, which is the order `val[FEATURES]` was passed to the model. The risky variant would be building `pd.Series(preds)` without an index and assigning that: its labels run 0..n-1 while `val`'s index holds the original labels from before sorting and filtering, so alignment would put predictions on the wrong rows or leave NaNs. Either keep raw arrays, or wrap with `pd.Series(preds, index=val.index)`. The teammate's `val_sorted.loc[0:99]` is wrong: `.loc` slices by label, inclusive of the end, and after sorting the labels are no longer in order, so it returns every row positioned between label 0 and label 99 wherever they fell, possibly thousands of rows or none. The positional intent is `val_sorted.iloc[:100]` or `val_sorted.head(100)`. If the filtered frame is also modified later, write with `.loc` on `val` itself rather than through a chained selection, because under Copy-on-Write chained writes are discarded."
sources: [{"title": "pandas docs: Indexing and selecting data", "url": "https://pandas.pydata.org/docs/user_guide/indexing.html", "note": "loc, iloc, boolean indexing, query and the section on returning a view versus a copy."}, {"title": "pandas docs: Migration guide for pandas 3.0 (Copy-on-Write)", "url": "https://pandas.pydata.org/docs/user_guide/migration.html", "note": "The CoW rule, chained assignment, read-only NumPy arrays and how to migrate code."}, {"title": "pandas docs: Essential basic functionality", "url": "https://pandas.pydata.org/docs/user_guide/basics.html", "note": "Reindexing and alignment, which explain why a Series assignment can land on the wrong rows."}]
practice: ["loc-vs-iloc", "boolean-indexing", "multi-column-selection", "column-selection", "resetting-index"]
tags: ["indexing", "boolean masks", "Copy-on-Write"]
tools: ["pandas"]
---

## Three ways in

Lesson 1 showed that a DataFrame has two sets of labels, the index and the columns. Selection is about saying which of them you mean.

| syntax | selects by | slice end | typical use
| --- | --- | --- | --- |
| `df["col"]`, `df[["a", "b"]]` | column name(s) | n/a | pick columns
| `df[mask]` | boolean Series aligned on the index | n/a | filter rows
| `df.loc[rows, cols]` | labels (or boolean mask) | **included** | the general tool, also for writing
| `df.iloc[rows, cols]` | integer positions | excluded | first n rows, positional splits

```python
df.loc["2024-01-01":"2024-01-31", ["clicks", "spend"]]   # both dates included
df.iloc[:1000, :5]                                         # rows 0..999, first 5 columns
df.loc[df["country"] == "NL", "spend"]                    # mask plus column
```

The two disagree whenever the index is not `0..n-1`, which happens after almost any filter, sort or concat. `df.loc[0]` means "the row labelled 0", which may be the 4,000th row after a sort or may not exist after a filter. On a Series, `s[0]` used to fall back to position when the label was missing; that fallback was deprecated in pandas 2.1 and removed in 3.0, so write `s.iloc[0]` when you mean position.

## Boolean masks

A mask is a boolean Series with the same index as the frame. Build each condition separately and combine with element-wise operators:

```python
active = (df["last_seen"] >= "2024-06-01") & (df["plan"] != "free")
risky = df["country"].isin(["XX", "YY"]) | ~df["email_verified"]
df[active & ~risky]
df.query("age >= @min_age and plan in ['pro', 'team']")   # @ refers to a Python variable
```

The parentheses are required because `&` binds tighter than `>=` in Python. Writing `and` raises `ValueError: The truth value of a Series is ambiguous`. A mask containing missing values (from a nullable boolean column) cannot be used directly; decide what missing means with `.fillna(False)`.

A mask from another frame is aligned by index, not by position. After filtering, call `reset_index(drop=True)` before you pair rows with a NumPy array (for example predictions from `model.predict(X)`), or keep the index and wrap predictions as `pd.Series(preds, index=X.index)`.

## Picking a feature matrix

`X = df[FEATURES]` with an explicit list fixes both which columns go to the model and their **order**. Most estimators work on positions once data becomes an array, so a column order that differs between training and serving silently feeds the wrong values into each weight. Keep `FEATURES` in code, never build it from `df.columns` minus a few names, and when scoring, select `df[FEATURES]` so a missing column fails loudly with a `KeyError`.

## Writing values safely

The safe form is always one indexing call that names both rows and columns:

```python
df.loc[df["age"] < 0, "age"] = pd.NA          # fix invalid values in place
df["spend_log"] = np.log1p(df["spend"])        # new column on the whole frame
df = df.assign(is_weekend=df["ts"].dt.dayofweek >= 5)   # functional style
```

The unsafe form is **chained assignment**: two indexing steps, the second one writing.

~~~dg-flow Chained assignment: the write lands on an intermediate object, not on df.
df[mask] -(new object)-> temp -(["col"] = v)-> *temp modified -> df unchanged
~~~

Before Copy-on-Write, whether `df[mask]` returned a view or a copy depended on the internal memory layout, so a chained write sometimes changed `df` and sometimes did not. pandas raised `SettingWithCopyWarning` because it could not tell which case you were in. The same uncertainty applied to `sub = df[mask]; sub["x"] = 1`, where you usually did not want `df` to change but could not be sure.

## Copy-on-Write in pandas 2.x and 3.0

Copy-on-Write (CoW) removes the ambiguity with one rule: **any object derived from another behaves as a copy**. Internally pandas still shares memory until one side is written to, and only then copies, so selection stays cheap.

- In pandas 2.0 to 2.2 CoW is opt-in with `pd.options.mode.copy_on_write = True`; pandas 2.2 also offers `"warn"` to list the places where behaviour will change.
- In pandas 3.0 CoW is always on. `SettingWithCopyWarning` no longer exists, and a chained assignment never modifies the original; pandas emits a `ChainedAssignmentError` warning to tell you the write was lost.
- Modifying a subset is fine and never touches the parent: `train = df[df.split == "train"]; train["x"] = 0` changes only `train`. Calls to `.copy()` added just to silence warnings can be dropped.
- Arrays from `df.to_numpy()` or `s.values` may be read-only views under CoW; copy them with `np.array(..., copy=True)` before modifying in place.

<div class="ex">
<div class="lab">Worked example</div>

**the clipping that never happened**

A preprocessing script contains `df[df["income"] > cap]["income"] = cap`. In pandas 1.x it printed `SettingWithCopyWarning` and did nothing; in 3.0 it prints `ChainedAssignmentError` and still does nothing. Either way the model trains on uncapped incomes, and the large values dominate a linear model's fit.

The fix is one call, `df.loc[df["income"] > cap, "income"] = cap`, or simply `df["income"] = df["income"].clip(upper=cap)`. A unit test that asserts `df["income"].max() <= cap` after preprocessing would have caught it in both versions.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

use `.loc` for labels, `.iloc` for positions, parenthesised `&`/`|`/`~` for masks, and a single `.loc[rows, col] = value` for every write. Under Copy-on-Write that is not just good style; it is the only form that works.

</div>
