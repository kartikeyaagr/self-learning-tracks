---
id: "02.08"
course: "ml-toolkit"
module: "02"
slug: "fast-memory-safe-pandas"
title: "Fast, memory-safe pandas"
keyIdeas: ["Column-wise operations run in compiled code, while `apply(axis=1)` and `iterrows` call Python once per row; on 200,000 rows the vectorised fee rule here ran about 75 times faster than `apply` and 400 times faster than `iterrows`. Reach for `np.where`, `np.select`, `map`, `cut`, `.str` and `.dt` first.", "When a Python function is unavoidable, run it on `unique()` values and `map` the results back: thousands of calls instead of millions.", "Measure memory with `memory_usage(deep=True)`. Text columns dominate it; `category`, integer ids from `factorize` and downcast numbers shrank a one-million-row log from 180 MB to 11 MB.", "A category code is only a position in the category list, so the list is part of the model: build one `CategoricalDtype` on training data, save it, and apply it at serving, handling unseen values explicitly.", "For data larger than memory, read only needed columns with explicit dtypes, stream with `chunksize`, and combine sums and counts rather than means; method chaining with `assign` and `pipe` keeps the pipeline readable and reusable."]
question: "A preprocessing step takes 40 minutes on 20 million rows. It reads a CSV, then runs `df[\"bucket\"] = df.apply(lambda r: \"high\" if r[\"score\"] > 0.8 and r[\"country\"] in EU else \"low\", axis=1)`, then `df[\"device\"] = df[\"device\"].astype(\"category\")` and finally writes `df[\"device\"].cat.codes` as a model feature. The serving service applies the same code to each incoming batch. Identify the performance and correctness problems and rewrite the logic."
answer: "Performance: `apply(axis=1)` calls a Python function and builds a row Series 20 million times; it is almost certainly where the 40 minutes go. The vectorised form is `df[\"bucket\"] = np.where((df[\"score\"] > 0.8) & df[\"country\"].isin(EU), \"high\", \"low\")`, which runs in well under a second; store it as a category too. Reading the CSV with explicit `dtype` and `usecols` (or switching to Parquet with `columns=`) and casting low-cardinality text such as `country` and `device` to `category` will also cut memory and parse time; check the gain with `memory_usage(deep=True)`. Correctness: `astype(\"category\")` infers categories from whatever data it sees and sorts them, so in serving each batch gets its own category list and `cat.codes` can map the same device to different integers between training and serving, or between batches (a batch containing only 'ios' and 'web' numbers them 0 and 1 whatever they were in training). The model then receives meaningless codes without any error. Fix it by building `DEVICE_DTYPE = pd.CategoricalDtype(sorted(train[\"device\"].unique()))` once on the training data, saving it with the model, and applying `astype(DEVICE_DTYPE)` in both pipelines. Unseen devices should be handled explicitly before the cast, for example `s.where(s.isin(DEVICE_DTYPE.categories), \"other\")` with an 'other' category that exists in training (pandas 3.0 warns when casting values outside the categories, and will raise in future). Putting the whole transformation in one function used by training and serving, with a test that asserts identical dtypes and category lists and compares outputs on the same rows, guards against the skew."
sources: [{"title": "pandas docs: Scaling to large datasets", "url": "https://pandas.pydata.org/docs/user_guide/scale.html", "note": "Loading fewer columns, efficient dtypes, chunking and when to use other libraries."}, {"title": "pandas docs: Categorical data", "url": "https://pandas.pydata.org/docs/user_guide/categorical.html", "note": "CategoricalDtype, codes, memory savings and how categories behave in operations."}, {"title": "pandas docs: Enhancing performance", "url": "https://pandas.pydata.org/docs/user_guide/enhancingperf.html", "note": "Why apply is slow and how Cython, Numba and eval speed up hot paths."}, {"title": "pandas docs: PyArrow functionality", "url": "https://pandas.pydata.org/docs/user_guide/pyarrow.html", "note": "Arrow-backed dtypes and the pyarrow engine for I/O."}]
practice: ["apply-custom-transforms", "training-serving-skew", "change-data-types", "data-types-overview"]
tags: ["performance", "categoricals", "memory"]
tools: ["pandas"]
---

## What you will learn

- Why `apply(axis=1)` and `iterrows` are slow, and the vectorised replacements (`np.where`, `np.select`, `map`, `cut`, `.str`, `.dt`) that cover most row-wise logic.
- The "apply to unique values, then map" trick for the cases where you really need a Python function.
- How to measure true memory with `deep=True`, and how `category`, integer ids and downcast numbers can shrink a frame more than tenfold.
- Why a category list is part of your model's contract, and how inferring it at serving time silently corrupts predictions.
- How to process a file bigger than memory in chunks, combining partial results correctly, and how method chaining keeps the pipeline tidy.

*Before you start:* you should know dtypes and how to load data (02.01), Copy-on-Write (02.02) and `groupby` aggregation (02.04). The NumPy module's lesson on vectorisation explains why a loop in compiled code beats a Python loop; here we apply that idea to DataFrames.

## Closing the module

Lesson 02.07 built time-series features. Everything in this module so far assumed the data fits in memory and speed does not matter. On the full training set, 50 million rows, a step that took one second on a sample can take an hour, or the kernel dies out of memory.

This last lesson has two big ideas. First, **let compiled code do the looping**: express the rule once for the whole column. Second, **store each column in the smallest type that keeps its meaning**. A corollary: once text becomes codes, the code-to-value mapping is part of your model and must be saved with it.

## The idea in one picture

Take an interaction log with one million rows: a user id like `"u0000012345"`, an event type with four possible values, a two-letter country code, a dwell time in seconds and a click count. Loaded with default types, the columns look like this:

```python
log.dtypes        # user_id: str, event: str, country: str, dwell: float64, clicks: int64
log.memory_usage(deep=True, index=False) / 1e6
# user_id 60.0   event 53.2   country 51.0   dwell 8.0   clicks 8.0     -> 180 MB in total
```

The three text columns take 164 of the 180 MB, even though `event` has only four distinct values and `country` has eight. After converting each column to a tighter type, the same information takes 11 MB.

~~~dg-fig 02.08-memory.svg
Measured memory per column for a one-million-row log, before and after conversion. Text columns dominate; once they become integer codes, the whole frame is about 16 times smaller.
~~~

Two words to pin down before we go on:

- A **dtype** is the storage type of a column: how many bytes each value takes and how to read them (`int64`, `float32`, `str`, `category`, ...).
- **Vectorised** code runs one operation over a whole column inside compiled C, instead of calling Python code once per row.

## Step 1: see why row-by-row code is slow

A pandas column is a NumPy (or Arrow) array, so `df["amount"] * 0.03` is one tight compiled loop.

`df.apply(f, axis=1)` instead builds a fresh Series for each row, calls your Python function on it, and collects the result; `iterrows` does the same. On a million rows that is a million Python calls and a million small objects. You pay for interpreter overhead, not arithmetic.

To see the size of the gap, take one rule: customers on the `pro` plan pay no fee, everyone else pays 3% of the amount. Written five ways and timed on 200,000 rows:

~~~dg-fig 02.08-speed.svg
The same fee rule timed five ways on 200,000 rows (log scale). Each step down the ladder is roughly 5 to 10 times faster; timings are machine-dependent, the ratios much less so.
~~~

On this machine `iterrows` took about 3 seconds, `apply(axis=1)` about 0.6 seconds, a plain list comprehension over `zip` 40 ms, `np.where` 8 ms, and `np.where` on a `category` column under 1 ms. Vectorised is roughly 75 times faster than `apply` and 400 times faster than `iterrows`. The last bar is faster again because pandas compares one-byte category codes rather than strings: small dtypes save time as well as memory.

~~~dg-layers Rough speed ladder for a per-row computation, fastest at the top.
*Vectorised column ops | arithmetic, comparisons, np.where, .str and .dt accessors
Built-in groupby / map / merge | compiled loops keyed by group or lookup
Series.apply / map with a function | one Python call per element
DataFrame.apply(axis=1), iterrows | one Python call plus one Series per row
~~~

## Step 2: rewrite row logic as column logic

Almost every `if/else` you would write inside a row function has a vectorised form. Here is a small frame and six common patterns:

```python
df = pd.DataFrame({
    "plan": ["free", "pro", "basic", "free"],  "amount": [120.0, 80.0, 1500.0, 40.0],
    "country": ["DE", "FR", "US", "NL"],        "age": [17, 34, 52, 29],
    "email": ["a@x.com", "b@y.org", "c@x.com", "d@z.net"],
    "ts": pd.to_datetime(["2024-03-01 09:15", "2024-03-01 18:40",
                          "2024-03-02 07:05", "2024-03-02 23:59"]),
})
REGION = {"DE": "EU", "FR": "EU", "NL": "EU"}

df["fee"] = np.where(df["plan"] == "pro", 0.0, df["amount"] * 0.03)     # one if/else
df["tier"] = np.select([df["amount"] >= 1000, df["amount"] >= 100],     # if/elif/else,
                       ["gold", "silver"], default="bronze")            # checked in order
df["region"] = df["country"].map(REGION)                                # lookup; NaN if missing
df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 120], right=False)
df["domain"] = df["email"].str.split("@").str[1]                        # string methods
df["hour"] = df["ts"].dt.hour                                           # datetime parts
#     fee    tier region   age_bin domain  hour
# 0   3.6  silver     EU   [0, 18)  x.com     9
# 1   0.0  bronze     EU  [18, 35)  y.org    18
# 2  45.0    gold    NaN  [35, 60)  x.com     7
# 3   1.2  bronze     EU  [18, 35)  z.net    23
```

How to read them:

- `np.where(cond, a, b)` picks `a` where the condition holds and `b` elsewhere, for all rows at once.
- `np.select` is the `elif` chain: the first true condition wins. Row 0 (amount 120) fails `>= 1000`, passes `>= 100`, so it gets `silver`.
- `map(dict)` is a compiled hash lookup. Missing keys become NaN (the US row); fill them deliberately.
- `pd.cut` bins numbers; `right=False` puts age 18 in `[18, 35)`.

Combine conditions with `&`, `|`, `~` (each comparison in brackets) and use `isin` instead of `in`.

## Step 3: when you really need Python, call it once per distinct value

Some logic has no vectorised form: a user-agent parser, a tokenizer, a messy business rule. But real columns repeat themselves: ten million log rows may hold only 5,000 distinct user agents. Run the slow function on the distinct values, then `map` the results back:

```python
lookup = {u: parse(u) for u in df["ua"].unique()}   # 5,000 Python calls
df["browser"] = df["ua"].map(lookup)                 # compiled hash lookup for every row
```

~~~dg-fig 02.08-unique.svg
Measured on this machine: calling a parser on every row grows in step with the row count, while parsing 5,000 unique values and mapping the dict stays far cheaper (about 7 times at one million rows, more as rows grow).
~~~

Lesson 02.04's advice to prefer `"mean"` over a lambda in aggregations is the same idea.

## Step 4: measure memory honestly

Plain `df.memory_usage()` counts only 8 bytes per row for a Python-object text column: the pointer, not the string. Ask for the deep figure:

```python
s = pd.Series(["view", "click", "view", "buy"] * 250_000)   # 1,000,000 rows, dtype str
s.memory_usage(index=False)               # 8,000,000   (pointers only)
s.memory_usage(deep=True, index=False)    # 53,000,000  (pointers + string objects)
c = s.astype("category")
c.memory_usage(deep=True, index=False)    # 1,000,159
c.cat.categories.tolist(), c.cat.codes.dtype   # ['buy', 'click', 'view'], int8
```

`df.info(memory_usage="deep")` prints the same deep total for a whole frame.

Why 53 bytes? Each row holds an 8-byte pointer to a Python string object, and in recent CPython a short ASCII string object costs about 41 bytes of overhead plus one byte per character. `"view"` is therefore $8 + 41 + 4 = 53$ bytes, to store four letters.

A note on pandas 3.0. Text columns now default to the new `str` dtype. If `pyarrow` is installed, that dtype uses compact Arrow buffers (the characters plus an offset per row), which is much smaller and makes `.str` methods faster. Without `pyarrow`, as on the machine that produced these numbers, it falls back to Python objects and costs the same as the old `object` dtype. Either way, a low-cardinality column is smaller still as a `category`.

## Step 5: shrink each column to the type it needs

A **category** column stores two things: a small integer **code** per row, and one list of the distinct values, called the **categories**. The code is just the position in that list.

~~~dg-fig 02.08-category.svg
Left: a text column keeps a pointer and a whole string object for every row. Right: a category column keeps a one-byte code per row and stores each distinct string once.
~~~

The main conversions, and what each one buys:

| change | how | effect
| --- | --- | --- |
| repeated strings to category | `astype("category")` or a fixed `CategoricalDtype` | 1 byte per row (up to 127 categories) plus one copy of each value
| id strings to integer ids | `codes, uniques = pd.factorize(s)` | 4 bytes per row as int32; keep `uniques` to map back
| text to Arrow strings | pandas 3.0 `str` with pyarrow, or `dtype_backend="pyarrow"` | compact buffers, faster `.str` methods
| float64 to float32 | `pd.to_numeric(s, downcast="float")` | half the memory; about 7 significant digits
| int64 to smaller ints | `pd.to_numeric(s, downcast="integer")` | 2 to 8 times smaller; picks the smallest type that fits
| fewer columns | `usecols=` (CSV) / `columns=` (Parquet) at read time | the cheapest saving of all

```python
pd.to_numeric(pd.Series([3, 17, 42, 250]), downcast="integer").dtype   # int16 (250 > 127)
pd.to_numeric(pd.Series([0.5, 1.25, 30.1]), downcast="float").tolist()
# [0.5, 1.25, 30.100000381469727]      float32 keeps ~7 digits
```

Parquet stores dtypes, including categories, and reads only the columns you ask for, so `pd.read_parquet(path, columns=FEATURES)` beats reading a whole CSV and dropping columns. With CSV, pass `usecols=` and `dtype=` so the wide version is never built.

<div class="ex">
<div class="lab">Worked example</div>

**shrink the one-million-row log by hand**

Before. Bytes per row with default dtypes: `user_id` is an 11-character string, so $8 + 41 + 11 = 60$ bytes. `event` averages about 45 bytes of string object (`view` is 45, `click` 46) plus the 8-byte pointer, about 53. `country` is 2 characters: $8 + 41 + 2 = 51$. `dwell` (float64) and `clicks` (int64) are 8 each.

Total per row: $60 + 53 + 51 + 8 + 8 = 180$ bytes. For $10^6$ rows: $180 \times 10^6$ bytes = 180 MB, which matches the measured figure.

After. `user_id` through `pd.factorize` becomes int32: 4 bytes. `event` (4 values) and `country` (8 values) become categories with int8 codes: 1 byte each. `dwell` becomes float32: 4 bytes. `clicks` lies between 0 and 49, so it fits in int8: 1 byte.

Total per row: $4 + 1 + 1 + 4 + 1 = 11$ bytes, so 11 MB for a million rows. The ratio is $180 / 11 \approx 16$.

Scale it up. At 60 million rows: $180 \times 6 \times 10^7 \approx 10.8$ GB before, $11 \times 6 \times 10^7 = 660$ MB after.

The 50,000 distinct user strings returned by `factorize` take about 3 MB. Save them next to the model, because the integer ids mean nothing without them.

</div>

## Step 6: treat categories as a contract

Because a code is only a position in the category list, the list itself carries meaning. Many tools read the codes directly: LightGBM's and XGBoost's categorical support use them, and `s.cat.codes` is a quick ordinal encoding. So the category list is part of the model, just like the weights.

The trap is `astype("category")` with no explicit list. It infers the categories from whatever data it is given and sorts them. Train on six rows, serve on a batch of three:

~~~dg-fig 02.08-skew.svg
Training assigns Berlin 0, Lyon 1, Oslo 2, Paris 3. Applying the saved dtype at serving keeps those codes; inferring categories from the batch renumbers them, so the model reads Paris as Lyon and Oslo as Berlin.
~~~

The fix is to build the dtype once from the training data, save it, and apply it everywhere:

```python
train = pd.Series(["Paris", "Berlin", "Oslo", "Lyon", "Paris", "Berlin"])
serve = pd.Series(["Paris", "Oslo", "Rome"])

CITY = pd.CategoricalDtype(sorted(train.unique()))     # save this with the model
CITY.categories.tolist()                                # ['Berlin', 'Lyon', 'Oslo', 'Paris']

serve.astype("category").cat.codes.tolist()             # [1, 0, 2]   wrong: renumbered
known = serve.where(serve.isin(CITY.categories))        # unseen 'Rome' -> NaN, explicitly
known.astype(CITY).cat.codes.tolist()                   # [3, 2, -1]  aligned; -1 = missing
```

Unseen values (Rome) become NaN with code -1; decide how the model sees them, for example via an `"other"` category that exists in training. In pandas 3.0, casting values outside the categories emits a deprecation warning (it will raise in future), which is why the snippet masks them first with `where`. The same contract covers `factorize` uniques and `pd.cut` bin edges. Also new in pandas 3.0: `groupby` on a categorical defaults to `observed=True`, so unused categories no longer produce empty groups.

## Step 7: data that does not fit in memory

When even the shrunk frame is too big, read the file in pieces. `read_csv(..., chunksize=N)` returns an iterator of DataFrames of up to N rows. Aggregate each chunk, then combine the partial results.

The rule for combining: carry **sums and counts**, never means. A user's orders can be spread unevenly across chunks, and a plain average of chunk averages gives each chunk equal weight regardless of how many rows it held.

~~~dg-fig 02.08-chunks.svg
One user's ten orders split across two chunks: one order of 100, then nine of 10. Averaging the two chunk means gives 55; dividing the summed total by the summed count gives the correct 19.
~~~

```python
totals = None
for chunk in pd.read_csv("events.csv", usecols=["user_id", "amount"],
                         dtype={"user_id": "str", "amount": "float32"},
                         chunksize=10):            # use ~1_000_000 on real files
    part = chunk.groupby("user_id")["amount"].agg(["sum", "count"])
    totals = part if totals is None else totals.add(part, fill_value=0)
totals["mean"] = totals["sum"] / totals["count"]
#            sum  count  mean
# user_id
# u1       190.0     10  19.0       (averaging the chunk means would give 55.0)
# u2        80.0     10   8.0
```

`add(..., fill_value=0)` aligns on the user index, so users missing from some chunks are fine. Sums, counts, minimums and maximums combine this way (variance too, if you carry sums of squares). Exact medians and distinct counts do not. When chunking gets that complicated, use a tool built for out-of-core work: DuckDB or Polars run such group-bys directly.

## Step 8: method chaining

Long feature scripts often accumulate half-modified frames: `df2 = df[...]`, `df2["x"] = ...`, `df3 = df2.merge(...)`. **Method chaining** writes the pipeline as one expression in which each step returns a new frame:

```python
PLAN = pd.CategoricalDtype(["free", "basic", "pro"])

def add_user_counts(d, col):
    return d.assign(n_orders=d.groupby(col)[col].transform("size"))

feats = (
    orders
      .query("amount > 0")
      .assign(month=lambda d: d["ts"].dt.to_period("M"),
              log_amount=lambda d: np.log1p(d["amount"]))
      .pipe(add_user_counts, col="user_id")
      .astype({"plan": PLAN})
)
```

`assign` with a lambda sees the frame *at that step* (only rows that survived the `query`); `pipe` slots in your own function. Nothing is left half-modified if a step fails, and under Copy-on-Write (02.02) the steps share memory until something is written. Wrap the chain in one function called from both training and serving.

## Checking the result

Speed and memory changes are only improvements if the output is unchanged. After any rewrite:

- **Compare outputs** of old and new versions on a sample with `pd.testing.assert_frame_equal`, passing `rtol` where you moved to float32.
- **Measure, do not guess:** `%timeit` or `time.perf_counter()`; find the step that takes 90% of the time first.
- **Check memory with `deep=True`** before and after, per column.
- **Check dtypes at the boundary:** assert serving dtypes and category lists equal training's.
- **Check value ranges** after downcasting, with headroom for future data.

## What goes wrong

| mistake | symptom | fix
| --- | --- | --- |
| `apply(axis=1)` / `iterrows` on millions of rows | a step takes minutes; CPU busy in Python | `np.where`, `np.select`, `map`, `cut`, `.str`, `.dt`
| Python function on every row of a repetitive column | slow even with `Series.map(func)` | run it on `unique()` and `map` the dict
| Shallow memory figure | "the frame is only 8 MB" but the process uses 200 MB | `memory_usage(deep=True)`, `info(memory_usage="deep")`
| `astype("category")` inferred at serving | no error, accuracy quietly drops | save one `CategoricalDtype` from training; mask unseen values
| Downcast to a type that is too small | int8 values wrap around: 100 + 120 gives -36 | check min/max; leave headroom for future data
| float32 for large ids or money totals | ids above 16,777,216 merge; cents drift | keep ids as integers; keep totals in float64
| Mean of chunk means | aggregates differ from the full-data answer | carry sums and counts, divide at the end
| Reading every column then dropping | slow load, memory spike | `usecols=` / `columns=` and `dtype=` at read time

The wrap-around is silent: $100 + 120 = 220$ exceeds the int8 maximum of 127, so int8 stores $220 - 256 = -36$. Downcast for storage; do arithmetic that can grow in a wider type.

## Going deeper

**Why vectorised code wins.** A compiled loop over a contiguous array pays a fixed setup cost once and then a few nanoseconds per element; the CPU's caches and SIMD instructions work at full speed because the values sit next to each other in memory. A Python-level loop pays a fixed cost of roughly 0.1 to several microseconds *per element* for interpreter dispatch, boxing each value into a Python object, and (for `apply(axis=1)`) building a Series. The ratio between those per-element costs is the 10x to 1000x you see. Both are linear in $n$; vectorising changes the constant, not the complexity.

**When categories do not help.** A category stores each distinct value once plus a code per row, so its size is about $n \cdot b + u \cdot s$, where $n$ is the row count, $b$ the code width in bytes, $u$ the number of distinct values and $s$ the bytes per stored value. If nearly every value is unique (free text, ids used once), $u \approx n$ and the category is *larger* than the plain column. Use categories for low-cardinality columns; use `factorize` for ids you need as integers; leave free text as Arrow strings.

**Code width grows with cardinality.** Up to 127 categories fit int8 codes, up to 32,767 need int16, and beyond that int32. Code -1 is reserved for missing.

**Other speed tools.** `df.eval` and `df.query` can skip intermediate arrays when `numexpr` is installed, and Numba compiles a numeric loop that truly needs state across rows. Use them after vectorising, not instead.

## Key terms

| Term | Meaning
| --- | --- |
| Vectorised operation | One operation applied to a whole column in compiled code, not a Python call per row
| dtype | The storage type of a column, which sets bytes per value and allowed operations
| Deep memory usage | Memory including the Python objects behind text columns, from `deep=True`
| Category / categorical | A column stored as small integer codes plus one list of distinct values
| CategoricalDtype | A fixed category list (and order) that you build once and reuse
| Downcasting | Moving numbers to a smaller type, such as float64 to float32
| Chunking | Reading and processing a large file in pieces of N rows
| Method chaining | Writing a pipeline as one expression of steps that each return a new frame

<div class="take">
<div class="lab">Key takeaway</div>

vectorise first and apply Python only to unique values, measure memory with `deep=True` and store each column in the smallest type that keeps its meaning, fix category lists on the training data and ship them with the model, and stream large files with sums and counts that combine correctly.

</div>
