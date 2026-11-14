---
id: "02.08"
course: "ml-toolkit"
module: "02"
slug: "fast-memory-safe-pandas"
title: "Fast, memory-safe pandas"
keyIdeas: ["Column-wise operations run in compiled code; `apply(axis=1)` and `iterrows` run Python once per row and are typically 10 to 1000 times slower. Reach for arithmetic, `np.where`/`np.select`, `map`, `cut` and the `.str`/`.dt` accessors first.", "Measure memory with `df.memory_usage(deep=True)`. Text columns in `object` dtype dominate it; `category` (or the Arrow-backed `str` default in pandas 3.0) and downcast numerics usually cut a frame several-fold.", "Categories are part of the feature contract: build them from the training set and apply the same categories at serving so the integer codes mean the same thing.", "For data larger than memory, read only the columns you need, stream with `chunksize` and combine partial aggregates correctly; method chaining with `assign` and `pipe` keeps the pipeline readable and free of half-modified intermediates."]
question: "A preprocessing step takes 40 minutes on 20 million rows. It reads a CSV, then runs `df[\"bucket\"] = df.apply(lambda r: \"high\" if r[\"score\"] > 0.8 and r[\"country\"] in EU else \"low\", axis=1)`, then `df[\"device\"] = df[\"device\"].astype(\"category\")` and finally writes `df[\"device\"].cat.codes` as a model feature. The serving service applies the same code to each incoming batch. Identify the performance and correctness problems and rewrite the logic."
answer: "Performance: `apply(axis=1)` calls a Python function and builds a row Series 20 million times; it is almost certainly where the 40 minutes go. The vectorised form is `df[\"bucket\"] = np.where((df[\"score\"] > 0.8) & df[\"country\"].isin(EU), \"high\", \"low\")`, which runs in well under a second; store it as a category too. Reading the CSV with explicit `dtype` and `usecols` (or switching to Parquet) and casting low-cardinality text to `category` will also cut memory and parse time. Correctness: `astype(\"category\")` infers categories from whatever data it sees and sorts them, so in serving each batch gets its own category list and `cat.codes` can map the same device to different integers between training and serving, or between batches. The model then receives meaningless codes without any error. Fix it by building `DEVICE_DTYPE = pd.CategoricalDtype(train_categories)` once on the training data, saving it with the model, and applying `astype(DEVICE_DTYPE)` in both pipelines; unseen devices become NaN (code -1), which should be handled explicitly, for example by mapping to an 'other' category that exists in training. Putting both steps in one function used by training and serving, with a test comparing their outputs on the same rows, guards against the skew."
sources: [{"title": "pandas docs: Scaling to large datasets", "url": "https://pandas.pydata.org/docs/user_guide/scale.html", "note": "Loading fewer columns, efficient dtypes, chunking and when to use other libraries."}, {"title": "pandas docs: Categorical data", "url": "https://pandas.pydata.org/docs/user_guide/categorical.html", "note": "CategoricalDtype, codes, memory savings and how categories behave in operations."}, {"title": "pandas docs: Enhancing performance", "url": "https://pandas.pydata.org/docs/user_guide/enhancingperf.html", "note": "Why apply is slow and how Cython, Numba and eval speed up hot paths."}, {"title": "pandas docs: PyArrow functionality", "url": "https://pandas.pydata.org/docs/user_guide/pyarrow.html", "note": "Arrow-backed dtypes and the pyarrow engine for I/O."}]
practice: ["apply-custom-transforms", "training-serving-skew", "change-data-types", "data-types-overview"]
tags: ["performance", "categoricals", "memory"]
tools: ["pandas"]
---

## Why apply is slow

A pandas column is a NumPy or Arrow array, so an expression like `df["price"] * df["qty"]` is one loop in compiled code (the NumPy module's vectorisation lesson explains why). `df.apply(f, axis=1)` instead builds a Series for each row and calls the Python function on it; on a million rows that is a million Python calls and a million small objects.

~~~dg-layers Rough speed ladder for a per-row computation, fastest at the top.
*Vectorised column ops | arithmetic, comparisons, np.where, .str and .dt accessors
Built-in groupby / map / merge | compiled loops keyed by group or lookup
Series.apply / map with a function | one Python call per element
DataFrame.apply(axis=1), iterrows | one Python call plus one Series per row
~~~

Most row-wise logic has a vectorised form:

```python
df["fee"] = np.where(df["plan"] == "pro", 0.0, df["amount"] * 0.03)

df["tier"] = np.select(                      # branches checked in order
    [df["spend"] >= 1000, df["spend"] >= 100],
    ["gold", "silver"], default="bronze")

df["region"] = df["country"].map(COUNTRY_TO_REGION)       # NaN if missing from the dict
df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 120], right=False)
df["domain"] = df["email"].str.split("@").str[1]
df["hour"] = df["ts"].dt.hour
```

When a Python function is genuinely needed, apply it to the unique values and map back: `df["parsed"] = df["ua"].map({u: parse(u) for u in df["ua"].unique()})`. With 10 million rows but 5,000 distinct user agents, that is 5,000 calls instead of 10 million. Lesson 4's named aggregations follow the same rule: prefer built-in function names to lambdas.

## Measuring and reducing memory

`df.info(memory_usage="deep")` and `df.memory_usage(deep=True)` report true memory, including the Python string objects behind `object` columns, which the shallow figure omits. Typical savings:

| change | how | effect
| --- | --- | --- |
| repeated strings to category | `astype("category")` | integer codes plus one copy of each value; often 10x smaller
| text to Arrow strings | pandas 3.0 default, or `dtype_backend="pyarrow"` | compact buffers, faster `.str` methods
| float64 to float32 | `pd.to_numeric(s, downcast="float")` | half the memory; fine for most features
| int64 to smaller ints | `pd.to_numeric(s, downcast="integer")` | 2 to 8x smaller, check the value range
| fewer columns | `usecols=` / `columns=` at read time | the cheapest saving of all

Parquet stores dtypes (including categories) and supports reading a subset of columns, so `pd.read_parquet(path, columns=FEATURES)` is far cheaper than reading a CSV and dropping columns afterwards.

## Categoricals as a contract

A `category` column stores integer codes and a list of categories. Many tools use the codes directly: LightGBM and XGBoost's categorical support read them, and `s.cat.codes` is a quick ordinal encoding. That makes the category list part of the model:

```python
cats = pd.CategoricalDtype(sorted(train["city"].dropna().unique()))
train["city"] = train["city"].astype(cats)
serve["city"] = serve["city"].astype(cats)     # unseen cities become NaN, codes stay aligned
```

If serving instead calls `astype("category")` on its own batch, categories are inferred from that batch, so code 3 may mean "Lyon" in training and "Oslo" in production: a silent train/serve skew. Save the dtype (or the category list) with the model. Grouping by a categorical with many unused categories used to produce empty groups; since pandas 3.0 `observed=True` is the default.

## Data that does not fit

For a file larger than memory, stream it and combine partial results:

```python
totals = None
for chunk in pd.read_csv("events.csv", usecols=["user_id", "amount"],
                         dtype={"user_id": "string", "amount": "float32"},
                         chunksize=1_000_000):
    part = chunk.groupby("user_id")["amount"].agg(["sum", "count"])
    totals = part if totals is None else totals.add(part, fill_value=0)
totals["mean"] = totals["sum"] / totals["count"]
```

Aggregate sums and counts per chunk, never means: the mean of chunk means is wrong when chunks hold different numbers of rows per user. Distinct counts and medians do not combine this way at all. When chunking gets complicated, move the work to a tool designed for it: the pandas scaling guide lists options, and DuckDB, Polars or the SQL module's techniques handle out-of-core group-bys directly.

## Method chaining

Chaining builds a pipeline as one expression, with each step returning a new frame:

```python
feats = (
    pd.read_parquet("orders.parquet", columns=["user_id", "ts", "amount", "plan"])
      .query("amount > 0")
      .assign(month=lambda d: d["ts"].dt.to_period("M"),
              log_amount=lambda d: np.log1p(d["amount"]))
      .pipe(add_user_features, cutoff=CUTOFF)
      .astype({"plan": PLAN_DTYPE})
)
```

`assign` with a lambda refers to the frame *at that step*, `pipe` slots in your own functions, and nothing is half-modified if a step fails. Under Copy-on-Write (lesson 2) these steps share memory until something is written, so chaining no longer costs a full copy per step. Wrapping the chain in a function gives you one entry point to call from both training and serving.

<div class="ex">
<div class="lab">Worked example</div>

**a 9 GB frame that fits in 1 GB**

An interaction log of 60 million rows has `user_id` and `item_id` as text, `event` as text with 4 distinct values, and `ts` and `dwell` as float64. `memory_usage(deep=True)` reports about 9 GB, almost all of it Python string objects.

Converting `event` to `category` (1 byte per row), `user_id` and `item_id` to integer ids via `pd.factorize` (int32), `dwell` to float32 and `ts` to `datetime64` brings it to about 1.1 GB. The id mappings from `factorize` are saved next to the model, for the same reason categories are.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

vectorise first, apply to unique values when you must use Python, measure memory with `deep=True`, fix categories on the training set, and stream large files with aggregates that combine correctly.

</div>
