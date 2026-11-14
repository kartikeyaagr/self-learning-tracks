---
id: "02.01"
course: "ml-toolkit"
module: "02"
slug: "series-dataframes-and-loading-data"
title: "Series, DataFrames and loading data"
keyIdeas: ["A `Series` is a 1-D array of values plus an **index** of labels; a `DataFrame` is a set of named columns that share one index. Arithmetic between pandas objects aligns on labels first, not on position.", "Every column has exactly one dtype. Integers with a missing value silently become `float64` unless you use a nullable dtype (`Int64`), and text is `object` in pandas 2.x but the dedicated `str` dtype by default in pandas 3.0.", "`read_csv` guesses types, missing-value markers and dates. For anything that feeds a model, pass `usecols`, `dtype`, `na_values`/`keep_default_na` and `parse_dates` explicitly so training and serving read the file the same way.", "A first inspection is always the same handful of calls: `shape`, `info()`, `head()`/`sample()`, `describe(include=\"all\")`, `isna().mean()` and `nunique()`. They catch most loading bugs before any modelling starts."]
question: "A colleague's training script reads `events.csv` with a bare `pd.read_csv(path)`. It worked for months, but after a new export the `item_id` column comes out as `float64` instead of `int64`, a later join on `item_id` against an items table (where it is `int64`) now matches fewer rows, and some ids print as `1.2345678e+07`. What probably changed in the file, why did pandas react this way, and how would you make loading robust?"
answer: "The new export almost certainly contains at least one empty or missing `item_id`. NumPy's `int64` has no missing value, so pandas promotes the whole column to `float64` to hold `NaN`. Floats print large integers in scientific notation and, beyond 2^53, cannot represent every integer exactly, so some ids can change value; and the missing rows can never match. The fix is to stop relying on inference: pass `dtype={\"item_id\": \"Int64\"}` (nullable integer, missing becomes `pd.NA`) or read ids as `\"string\"` since they are labels, list the expected missing markers in `na_values`, and restrict to known columns with `usecols`. Then decide explicitly what a missing id means (drop the row, or route it to an 'unknown' bucket) and count how many rows are affected, rather than letting the join silently drop them. The same schema dictionary should be used wherever this file is read, including serving."
sources: [{"title": "pandas docs: Intro to data structures", "url": "https://pandas.pydata.org/docs/user_guide/dsintro.html", "note": "Series, DataFrame, index alignment and how construction from dicts and arrays works."}, {"title": "pandas API: read_csv", "url": "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html", "note": "Every parsing option, especially dtype, na_values, keep_default_na, parse_dates and dtype_backend."}, {"title": "pandas docs: Essential basic functionality", "url": "https://pandas.pydata.org/docs/user_guide/basics.html", "note": "dtypes, astype, describe, info and the head/tail inspection helpers."}]
practice: ["inspect-dataframe-shape", "data-types-overview", "create-dataframe-from-dict", "head-and-tail-operations", "change-data-types"]
tags: ["DataFrame", "dtypes", "read_csv"]
tools: ["pandas"]
---

## Orientation: pandas between raw data and arrays

The NumPy module gave you fast homogeneous arrays. Real ML data rarely arrives that way: it comes as CSV exports, Parquet files and SQL results with mixed types, missing values, string categories and timestamps. pandas is the layer where you turn that into a clean, keyed feature table, which you then hand to NumPy, scikit-learn or a PyTorch `Dataset` (module 4). This module goes in the order of a typical pipeline: load and inspect (this lesson), select (2), clean (3), aggregate (4), join (5), reshape (6), handle time (7), and keep it fast (8). The SQL module (3) covers the same operations when the data lives in a database.

~~~dg-layers The pandas data model, from the bottom up.
NumPy or Arrow array | one dtype per column, contiguous values
Index | row labels used for alignment and lookup
Series | one array + one index + a name
*DataFrame | named columns sharing one index, each with its own dtype
~~~

## Series, index and alignment

A `Series` pairs values with labels. The labels are not decoration: binary operations between two Series match labels, and labels present in only one side produce `NaN`.

```python
import pandas as pd
a = pd.Series([1.0, 2.0, 3.0], index=["u1", "u2", "u3"])
b = pd.Series([10.0, 20.0], index=["u3", "u1"])
a + b        # u1 21.0, u2 NaN, u3 13.0  (matched by label, not position)
a.values + b.reindex(a.index).values   # explicit, if you really want arrays
```

Alignment is what makes pandas safe for keyed data, and it is also a classic ML bug: after filtering or sorting, a column's index no longer runs 0..n-1, so assigning a NumPy array of predictions works by position while assigning a Series works by label. When the two disagree you get shuffled predictions with no error. Lesson 2 shows when to call `reset_index(drop=True)`.

A `DataFrame` is a dict-like collection of Series sharing one index. `df.columns` is itself an Index, `df.index` holds the row labels, and `df.dtypes` lists one dtype per column.

## Dtypes you will meet

| dtype | holds | missing marker | note
| --- | --- | --- | --- |
| `int64` | integers | none | a single NaN forces the column to `float64`
| `Int64`, `Float64`, `boolean` | nullable numbers and booleans | `pd.NA` | keep integers integral when values are missing
| `float64` | floats | `NaN` | default for any numeric column with gaps
| `object` | arbitrary Python objects | `None` or `NaN` | pandas 2.x default for text; slow and memory hungry
| `str` | text | `NaN` | pandas 3.0 default for text, Arrow-backed when pyarrow is installed
| `datetime64[ns]` (or other units) | timestamps | `NaT` | only if parsed; otherwise the column stays text
| `category` | a small set of repeated values | `NaN` | integer codes plus a lookup table (lesson 8)

The switch from `object` to a real string dtype in pandas 3.0 changes `df.dtypes` output and means `select_dtypes(include="object")` only still finds text columns through a deprecated compatibility path; select them with the builtin `include=[str]` and check dtype-based column selection when you upgrade.

## Loading with read_csv

`read_csv` will infer almost everything, which is convenient in a notebook and dangerous in a pipeline. Inference runs per file (and, with the default C parser and `low_memory=True`, per internal chunk), so the same column can come out as `int64` on one day's export and `float64` or `object` on the next. Pin down what you know:

```python
SCHEMA = {"user_id": "string", "country": "category", "age": "Int64", "spend": "float64"}

df = pd.read_csv(
    "users.csv",
    usecols=[*SCHEMA, "signup_ts"],   # read only what you need
    dtype=SCHEMA,
    parse_dates=["signup_ts"],         # or pd.to_datetime(..., format=...) afterwards
    na_values=["", "?", "-999"],      # extra markers for missing
    keep_default_na=False,             # do not treat "NA", "null", "nan" as missing
)
```

Other options worth knowing: `index_col` to make a column the index, `nrows` to read a sample, `chunksize` to stream a large file (lesson 8), `date_format` (pandas 2.0+) to fix the timestamp format, and `dtype_backend="pyarrow"` or `engine="pyarrow"` for Arrow-backed columns and a faster multithreaded parser. `keep_default_na=False` matters more than it looks: by default the strings `"NA"`, `"N/A"`, `"null"`, `"nan"` and several others become missing, which silently deletes Namibia's country code. Note that with `keep_default_na=False` empty fields are only treated as missing if you list `""` in `na_values`, as above.

Keep the schema dictionary in code and use it in both the training job and the serving path. A model that saw `user_id` as text in training and as an integer at serving will not crash; it will just stop matching keys.

## The first five minutes with a new table

```python
df.shape                       # rows, columns
df.info()                      # dtypes, non-null counts, memory
df.head(); df.sample(5, random_state=0)
df.describe(include="all").T   # ranges, counts, top categories
df.isna().mean().sort_values(ascending=False)   # missing rate per column
df.nunique().sort_values()     # constants and ID-like columns
df["label"].value_counts(dropna=False, normalize=True)
```

Read the output with questions in mind. Is a column you expected to be numeric `object`? Then some rows contain text such as `"1,234"` or `"unknown"`. Does an ID column have as many unique values as rows? Is any column constant? Is the label's class balance plausible? Is there a column whose name suggests it was computed after the outcome (for example `refund_amount` in a churn table)? That last question is how leakage gets caught early.

<div class="ex">
<div class="lab">Worked example</div>

**postcodes that lost their zeros**

A CSV holds Dutch and US addresses. Without a dtype, `read_csv` parses the `zip` column as `int64`, turning `"02139"` into `2139`, and the US rows no longer match the geography table they are joined to later.

`df.dtypes` shows `zip int64`, which is the clue: postcodes are labels, not quantities. Re-reading with `dtype={"zip": "string"}` keeps the leading zeros, and the join match rate goes from 61% to 99.8%.

The same rule applies to phone numbers, product codes and any ID: read them as text, because arithmetic on them is meaningless and inference will damage them.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

treat a DataFrame as labelled columns with one dtype each, read files with an explicit schema rather than letting pandas guess, and run the same short inspection on every new table before you trust it.

</div>
