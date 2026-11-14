---
id: "02.01"
course: "ml-toolkit"
module: "02"
slug: "series-dataframes-and-loading-data"
title: "Series, DataFrames and loading data"
keyIdeas: ["A `Series` is a 1-D array of values plus an **index** of labels; a `DataFrame` is a set of named columns that share one index. Arithmetic between pandas objects aligns on labels first, not on position, and labels on only one side give `NaN`.", "Every column has exactly one dtype. Integers with a missing value silently become `float64` unless you use a nullable dtype (`Int64`), and text is `object` in pandas 2.x but the dedicated `str` dtype by default in pandas 3.0.", "`read_csv` guesses types, missing-value markers and dates. For anything that feeds a model, pass `usecols`, `dtype`, `na_values`/`keep_default_na` and `parse_dates` explicitly, and share that schema between training and serving.", "Sentinels like `-999` and default NA strings like `\"NA\"` quietly corrupt statistics: one unlisted `-999` turned a mean age of 39 into -168.6 in the six-row example.", "A first inspection is always the same handful of calls: `shape`, `info()`, `head()`/`sample()`, `describe(include=\"all\")`, `isna().mean()` and `nunique()`. They catch most loading bugs, ID-like and constant columns, and many leaks before any modelling starts."]
question: "A colleague's training script reads `events.csv` with a bare `pd.read_csv(path)`. It worked for months, but after a new export the `item_id` column comes out as `float64` instead of `int64`, a later join on `item_id` against an items table (where it is `int64`) now matches fewer rows, and some ids print as `1.2345678e+07`. What probably changed in the file, why did pandas react this way, and how would you make loading robust?"
answer: "The new export almost certainly contains at least one empty or missing `item_id`. NumPy's `int64` has no bit pattern for 'missing', so pandas promotes the whole column to `float64` to hold `NaN`. That has three effects. Floats print large integers in scientific notation, which is the `1.2345678e+07` you see. Beyond $2^{53}$ a float64 cannot represent every integer exactly, so some large ids can change value and stop matching. And the rows with a missing id can never match anything, so the join loses them. Nothing raised an error because inference simply reacted to the new file. The fix is to stop relying on inference. Pass `dtype={\"item_id\": \"Int64\"}` (nullable integer, missing becomes `pd.NA`) or, better, read ids as `\"string\"` since they are labels, not quantities. List the expected missing markers in `na_values` (with `keep_default_na=False` if codes like 'NA' are valid values), and restrict to known columns with `usecols` so a renamed column fails loudly. Read the items table with the same dtype for the key so both sides of the join agree. Then decide explicitly what a missing id means (drop the row, or route it to an 'unknown' bucket) and log how many rows are affected, instead of letting the join drop them silently. Put the schema dictionary in one shared module and use it everywhere this file is read, including the serving path, and add a quick check after loading (dtypes match the schema, missing rate of `item_id` below a threshold) so the next format change is caught at load time rather than as a quiet drop in model quality."
sources: [{"title": "pandas docs: Intro to data structures", "url": "https://pandas.pydata.org/docs/user_guide/dsintro.html", "note": "Series, DataFrame, index alignment and how construction from dicts and arrays works."}, {"title": "pandas API: read_csv", "url": "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html", "note": "Every parsing option, especially dtype, na_values, keep_default_na, parse_dates and dtype_backend."}, {"title": "pandas docs: Essential basic functionality", "url": "https://pandas.pydata.org/docs/user_guide/basics.html", "note": "dtypes, astype, describe, info and the head/tail inspection helpers."}]
practice: ["inspect-dataframe-shape", "data-types-overview", "create-dataframe-from-dict", "head-and-tail-operations", "change-data-types"]
tags: ["DataFrame", "dtypes", "read_csv"]
tools: ["pandas"]
---

## What you will learn

- What a `Series` and a `DataFrame` really are: arrays of values plus **labels**, and why those labels change how arithmetic behaves.
- How pandas stores each column with exactly one dtype, which dtypes you will meet, and how a single missing value can silently turn integers into floats.
- How to load a CSV with an explicit schema (`usecols`, `dtype`, `na_values`, `keep_default_na`, `parse_dates`) so that training and serving read the file the same way.
- A short, fixed inspection routine that catches most loading bugs in the first five minutes.
- How to hand a clean table on to NumPy, scikit-learn or PyTorch without losing rows or shuffling them.

*Before you start:* you should be comfortable with NumPy arrays, dtypes and vectorised operations (module 01, especially 01.01 Arrays, dtypes and memory layout). No pandas experience is assumed.

## Where this module starts

The NumPy module gave you fast arrays where every element has the same type. Real ML data almost never arrives like that. It comes as CSV exports, Parquet files and SQL query results, with numbers, text categories, timestamps and gaps all mixed in one table. pandas is the layer where you turn that mess into a clean, keyed feature table, and then hand it to NumPy, scikit-learn or a PyTorch `Dataset` (module 04, lesson 04.05).

This module follows the order of a typical data pipeline: load and inspect (this lesson), select (02.02), clean (02.03), aggregate (02.04), join (02.05), reshape (02.06), handle time (02.07), and keep it fast (02.08).

The SQL module (03) covers the same operations when the data still lives in a database. Everything in this lesson is checked against pandas 3.0; where pandas 2.x behaves differently, the text says so.

~~~dg-layers The pandas data model, from the bottom up.
NumPy or Arrow array | one dtype per column, contiguous values
Index | row labels used for alignment and lookup
Series | one array + one index + a name
*DataFrame | named columns sharing one index, each with its own dtype
~~~

## The idea in one picture

Here is a small export of six users. It looks harmless:

```text
user_id,country,age,spend,signup_ts,churned
00017,NL,34,120.5,2024-01-03,0
00018,US,,80.0,2024-01-05,1
00021,NA,29,,2024-01-09,0
00025,DE,41,310.2,2024-02-11,0
00031,US,-999,45.0,2024-02-14,1
00032,NL,52,99.9,2024-03-01,0
```

Read it with a bare `pd.read_csv("users.csv")` and pandas makes four guesses, three of them wrong for an ML pipeline. `user_id` becomes the integer 17 (the leading zeros are gone). Namibia's country code `NA` becomes missing. `age` becomes `float64`, because one cell is empty, and the sentinel `-999` is kept as a real age. `signup_ts` stays as plain text. Nothing raises an error.

Read the same file with a schema and you get the table you meant:

```python
import pandas as pd

SCHEMA = {"user_id": "string", "country": "category", "age": "Int64",
          "spend": "float64", "churned": "int8"}
df = pd.read_csv("users.csv", usecols=[*SCHEMA, "signup_ts"], dtype=SCHEMA,
                 parse_dates=["signup_ts"], na_values=["", "-999"],
                 keep_default_na=False)
print(df.dtypes)
# user_id      string
# country      category
# age          Int64
# spend        float64
# signup_ts    datetime64[us]
# churned      int8
```

~~~dg-fig 02.01-anatomy.svg
Four rows of the users table. The grey column is the index (row labels), the coloured blocks are columns, and each column carries exactly one dtype, written underneath. Missing values show as <NA> in nullable columns and NaN in float columns.
~~~

The figure shows the three parts every pandas table has:

- The **index**: a label for each row. Here the user ids serve as labels; by default pandas uses 0, 1, 2, and so on.
- The **columns**: an ordered set of names, itself stored as an Index.
- The **values**: one array per column, each with a single **dtype** (data type) such as `int64`, `float64`, `category` or `datetime64`.

A **DataFrame** is a set of named columns that share one index. Pick one column, `df["age"]`, and you get a **Series**: one array, one index, one name. Most of pandas is just careful bookkeeping around these labels.

## Step 1: a Series is values plus labels

A Series looks like a NumPy array with names attached, and you can build one directly:

```python
s = pd.Series([0.2, 0.7, 0.1], index=["u1", "u2", "u3"], name="p_churn")
s["u2"]          # 0.7   look up by label
s.iloc[1]        # 0.7   look up by position
s.index          # Index(['u1', 'u2', 'u3'], dtype='str')
s.to_numpy()     # array([0.2, 0.7, 0.1])   the bare values
```

When you combine two Series, pandas first lines them up by label, then does the arithmetic. A label on only one side gives a missing result (`NaN`).

```python
a = pd.Series([1.0, 2.0, 3.0], index=["u1", "u2", "u3"])
b = pd.Series([10.0, 20.0], index=["u3", "u1"])
a + b                       # u1 21.0, u2 NaN, u3 13.0
a.add(b, fill_value=0)      # u1 21.0, u2 2.0,  u3 13.0
a.values + b.reindex(a.index).values   # explicit array maths, if you really want it
```

~~~dg-fig 02.01-align.svg
Addition matches labels, not positions. b lists u3 first, yet 1.0 is added to 20.0 (both labelled u1). u2 exists only in a, so the sum there is NaN.
~~~

This is **alignment**: per-user tables combine correctly without being sorted identically.

<div class="ex">
<div class="lab">Worked example</div>

**aligning two Series by hand**

Take `a` = {u1: 1.0, u2: 2.0, u3: 3.0} and `b` = {u3: 10.0, u1: 20.0}.

Step 1, collect the union of labels in sorted order: u1, u2, u3.

Step 2, pair values label by label. u1: $1.0 + 20.0 = 21.0$. u2: $2.0 + \text{(nothing)} = \text{NaN}$. u3: $3.0 + 10.0 = 13.0$.

Step 3, compare a positional sum: $1.0 + 10.0 = 11.0$ and $2.0 + 20.0 = 22.0$, both wrong.

With `fill_value=0` the missing side counts as 0, so u2 becomes $2.0 + 0 = 2.0$. Use that only when "absent" really means zero, such as a user with no purchases.

</div>

## Step 2: a DataFrame is columns sharing one index

The most common way to build a DataFrame by hand is from a dict of columns:

```python
d = pd.DataFrame({"age": [34, 29, 41], "country": ["NL", "NA", "DE"]},
                 index=["u1", "u2", "u3"])
d.dtypes            # age int64, country str
d.shape             # (3, 2)
type(d["age"])      # pandas.Series, sharing d's index
```

Think of it as a dict-like collection of Series. Different columns can have different dtypes; that is the main thing a DataFrame gives you over a 2-D NumPy array.

Alignment applies to rows as well. When you assign a Series as a new column, pandas matches its index to the frame's index. When you assign a NumPy array, it is placed by position. Those two rules give one of the classic ML bugs:

```python
t = pd.DataFrame({"x": [5, 3, 9, 1]}).sort_values("x")   # index is now 3, 1, 0, 2
t["p_arr"] = np.array([10, 30, 50, 90])   # by position: 10, 30, 50, 90
t["p_ser"] = pd.Series([10, 30, 50, 90])  # by label:    90, 30, 10, 50
```

Both lines run without a warning, yet they give different columns. If predictions come back as a Series with a stale index after filtering or sorting, they get attached to the wrong rows. Lesson 02.02 shows when to call `reset_index(drop=True)` to avoid this.

## Step 3: every column has exactly one dtype

Because each column is one array, it has exactly one dtype. Pandas picks it when you build or load the data, and that choice decides what you can compute, how missing values look and how much memory the column takes.

| dtype | holds | missing marker | note
| --- | --- | --- | --- |
| `int64` | integers | none | a single NaN forces the column to `float64`
| `Int64`, `Float64`, `boolean` | nullable numbers and booleans | `pd.NA` | keep integers integral when values are missing
| `float64` | floats | `NaN` | default for any numeric column with gaps
| `object` | arbitrary Python objects | `None` or `NaN` | pandas 2.x default for text; slow and memory hungry
| `str` | text | `NaN` | pandas 3.0 default for text, Arrow-backed when pyarrow is installed
| `string` | text | `pd.NA` | the opt-in string dtype, with nullable `pd.NA` semantics
| `datetime64[ns]` (or other units) | timestamps | `NaT` | only if parsed; otherwise the column stays text
| `category` | a small set of repeated values | `NaN` | integer codes plus a lookup table (lesson 02.08)

The first row is the trap. NumPy's `int64` has no way to say "missing", so as soon as one value is absent pandas promotes the whole column to `float64`, where `NaN` lives.

```python
pd.Series([1, 2, 3]).dtype                    # int64
pd.Series([1, None, 3]).dtype                 # float64  (1.0, NaN, 3.0)
pd.Series([1, None, 3], dtype="Int64")        # 1, <NA>, 3   dtype: Int64
```

~~~dg-fig 02.01-promote.svg
The same id column read three ways. With no gaps it is int64. One empty cell makes the default reader turn every value into a float. The nullable Int64 dtype keeps the integers and marks the gap as <NA>.
~~~

For measurements such as age, a float is harmless. For identifiers it is not: floats print large ids as `1.2345678e+07`, and very large values cannot even be stored exactly.

~~~dg-fig 02.01-floatgap.svg
The gap between neighbouring float64 numbers grows with their size. Past 2⁵³ (about 9 × 10¹⁵) the gap exceeds 1, so two different integer ids can round to the same float.
~~~

Text changed in pandas 3.0: 2.x stored it as `object` (pointers to Python strings), 3.0 uses the dedicated `str` dtype, Arrow-backed when pyarrow is installed. So `df.dtypes` looks different after an upgrade, and `select_dtypes(include="object")` only still finds text through a deprecated compatibility path. Select text columns with `include=[str]` instead. The separate opt-in `"string"` dtype (used for `user_id` above) is also text, but marks gaps as `pd.NA` rather than `NaN`.

Dtype also decides memory. The figure below was measured here on one million values.

~~~dg-fig 02.01-memory.svg
Memory for one million values, measured with memory_usage(deep=True) on pandas 3.0.5 without pyarrow; exact numbers depend on the machine and versions. Two-letter country codes cost 51 MB as Python strings and 1 MB as a category.
~~~

Without pyarrow, `str` holds Python string objects and costs the same as `object`; with pyarrow it is far more compact. `category` stores each distinct value once plus a small integer code per row. Lesson 02.08 returns to this.

## Step 4: load with an explicit schema

`read_csv` will infer almost everything: column types, missing markers, even dates if asked. That is convenient in a notebook and dangerous in a pipeline. Inference runs separately for every file (and, with the default C parser and `low_memory=True`, per internal chunk of a large file). So the same column can come out as `int64` from Monday's export and `float64` or text from Tuesday's, and downstream code keeps running on the wrong thing.

The fix is to write down what you know and pass it in:

```python
SCHEMA = {"user_id": "string", "country": "category", "age": "Int64",
          "spend": "float64", "churned": "int8"}

df = pd.read_csv(
    "users.csv",
    usecols=[*SCHEMA, "signup_ts"],   # read only the columns you need
    dtype=SCHEMA,                      # no type guessing
    parse_dates=["signup_ts"],         # text -> datetime64
    na_values=["", "-999"],            # extra markers that mean "missing"
    keep_default_na=False,             # do not treat "NA", "null", "nan" as missing
)
```

What each argument buys you:

- `usecols` skips columns you do not need and fails loudly if an expected column disappears.
- `dtype` fixes each column's type: ids as text, counts as nullable integers, small sets as categories.
- `na_values` lists the strings that mean "missing" in this particular file: empty cells, `-999`, `?`.
- `keep_default_na=False` turns off pandas' built-in list of missing markers. By default `"NA"`, `"N/A"`, `"null"`, `"nan"` and several others become missing, which silently deletes Namibia's country code. With this switched off, empty fields are only treated as missing if you list `""` in `na_values`, as above.
- `parse_dates` turns timestamp text into `datetime64`. pandas 3.0 picks microsecond resolution (`datetime64[us]`) when parsing strings; 2.x used nanoseconds. Add `date_format="%Y-%m-%d"` (pandas 2.0+) when you know the format, which is both faster and stricter.

Also useful: `index_col`, `nrows=1000` for a quick sample, `thousands=","`, `chunksize` for streaming (02.08), and `engine="pyarrow"` or `dtype_backend="pyarrow"` for a faster parser and Arrow-backed columns.

Keep the schema dictionary in code, in one module, and import it in both the training job and the serving path. A model that saw `user_id` as text in training and as an integer at serving time will not crash; it will just stop matching keys.

<div class="ex">
<div class="lab">Worked example</div>

**what the sentinel does to a simple average**

The `age` column holds 34, empty, 29, 41, -999, 52.

Bare `read_csv`: the empty cell becomes NaN and is skipped, but -999 is kept as an age. Count = 5. Sum = $34 + 29 + 41 - 999 + 52 = -843$. Mean = $-843 / 5 = -168.6$. pandas prints exactly `-168.6`, and `describe()` shows a minimum age of `-999.0`.

With `na_values=["", "-999"]`: both gaps are missing. Count = 4. Sum = $34 + 29 + 41 + 52 = 156$. Mean = $156 / 4 = 39.0$.

Missing rate: 2 of 6 rows, so `df["age"].isna().mean()` is $2/6 \approx 0.333$, not the $1/6 \approx 0.167$ the bare read reports.

Any scaler or model fitted on the first version learned from an age of -999.

</div>

## Step 5: the first five minutes with a new table

Before any modelling, run the same short inspection on every new table. It costs a minute and catches most loading bugs.

```python
df.shape                                   # (6, 6)  rows, columns
df.info()                                  # dtypes, non-null counts, memory
df.head(); df.sample(5, random_state=0)    # look at real rows, not only the top
df.describe(include="all").T               # ranges, counts, top categories
df.isna().mean().sort_values(ascending=False)   # missing rate per column
df.nunique().sort_values()                 # constants and ID-like columns
df["churned"].value_counts(dropna=False, normalize=True)   # 0: 0.667, 1: 0.333
```

`sample` matters because exports are often sorted, so `head()` can hide problems further down.

~~~dg-fig 02.01-inspect.svg
A profile of a 2,000-row table built from two one-liners. Left: isna().mean() shows coupon_code is 92% empty. Right: nunique() as a share of rows flags user_id (unique on every row, an identifier) and source_system (a single value, useless as a feature).
~~~

## Reading the inspection output

Read the output with specific questions in mind:

- **Is a column you expected to be numeric shown as text?** Then some rows contain things like `"1,234"` or `"unknown"`. Find them with `pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()`.
- **Is an integer column `float64`?** It has gaps. Decide whether it should be `Int64`, or whether the gaps are a loading bug.
- **As many distinct values as rows?** An identifier: keep it as a key, never as a numeric feature. **Constant?** Drop it.
- **Are the ranges plausible?** Ages of -999, prices of 0, dates in 1970 or 2099 are sentinels in disguise.
- **Is the label balance plausible?** 50% churn in a file from a product with 2% churn means a filter happened somewhere.
- **Does any column sound like it was computed after the outcome?** A `refund_amount` column in a churn table is a leak: it is only known once the customer has left. This question, asked early, catches most leakage.

When the table looks right, hand it on. `df[["age", "spend"]].to_numpy(dtype="float64", na_value=np.nan)` gives a plain float array with `NaN` for the gaps (nullable columns need the `na_value` argument). Most scikit-learn estimators also accept a DataFrame directly and remember the column names.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| ids lose leading zeros (`02139` becomes `2139`) | id parsed as a number | `dtype={"zip": "string"}`
| integer column suddenly `float64`, ids print as `1.2e+07` | one missing value in the new export | `Int64` or `"string"`, and count the missing rows
| a country, name or code turns into NaN | default NA strings such as `"NA"` | `keep_default_na=False` plus explicit `na_values`
| impossible values like `-999` in `describe()` | sentinel not declared missing | add it to `na_values`
| date column has dtype `str` or `object` | dates not parsed | `parse_dates` plus `date_format`
| numeric column is text | stray strings such as `"1,234"` or `"unknown"` | `thousands=","`, `pd.to_numeric(errors="coerce")`, then inspect what failed
| predictions attached to the wrong rows | array assigned by position, Series by stale label | `reset_index(drop=True)` or assign `.to_numpy()` deliberately
| `select_dtypes("object")` misses text after upgrading | pandas 3.0 `str` dtype | `select_dtypes(include=[str])`
| training and serving disagree on a column type | each path infers separately | one shared schema dict

<div class="ex">
<div class="lab">Worked example</div>

**postcodes that lost their zeros**

A CSV holds Dutch and US addresses. Without a dtype, `read_csv` parses the `zip` column as `int64`, turning `"02139"` into `2139`, and the US rows no longer match the geography table they are joined to later.

`df.dtypes` shows `zip int64`, which is the clue: postcodes are labels, not quantities. Re-reading with `dtype={"zip": "string"}` keeps the leading zeros, and the join match rate goes from 61% to 99.8%.

The same rule applies to phone numbers, product codes and any ID: read them as text, because arithmetic on them is meaningless and inference will damage them.

</div>

## Going deeper

**The cost of alignment.** Alignment exists because keyed series rarely cover exactly the same labels. The cost: each binary operation first checks whether the indexes are identical and, if not, reindexes both sides to their union. When objects are already aligned and speed matters, use `.to_numpy()` deliberately.

**Where the NaN-for-integers rule comes from.** Classic pandas columns are NumPy arrays, and NumPy integers use every bit pattern for a number, leaving none free to mean "missing". `NaN` is a special float bit pattern, so the only missing marker available was a float one. The nullable dtypes (`Int64`, `boolean`, `string`) fix this by storing a separate boolean **mask** next to the values: the values array stays integer and the mask records which entries are missing. That is why `Int64` costs 9 bytes per value instead of 8 in the memory figure: one extra byte of mask per row.

**Why big ids break as floats.** A float64 has 53 bits of precision (52 stored, one implied). Every integer up to $2^{53} = 9{,}007{,}199{,}254{,}740{,}992$ is exactly representable, but beyond that the spacing between floats is 2, then 4, and so on. `float(2**53 + 1) == 2**53` is `True`. Snowflake-style ids from many systems are larger than $2^{53}$, so one missing value in such a column can make two different users share an id.

**Copy-on-Write.** In pandas 3.0 anything derived from a DataFrame (a column, a slice) behaves as a copy, so chained assignment such as `df["a"][0] = 1` no longer modifies `df`. Lesson 02.02 shows the `.loc` form.

## Key terms

| Term | Meaning
| --- | --- |
| Series | A 1-D array of values with an index of labels and a name
| DataFrame | Named columns that share one index; each column can have its own dtype
| Index | The row labels (or column names) used for lookup and alignment
| Alignment | Matching values by label before any arithmetic or assignment
| dtype | The single data type of one column, such as `int64`, `str`, `category`
| Nullable dtype | `Int64`, `boolean`, `string`: a values array plus a missing-value mask, with `pd.NA`
| Sentinel | A placeholder such as `-999` or `"?"` that means missing but looks like data
| Schema | The agreed list of columns and dtypes, kept in code and used by every reader

<div class="take">
<div class="lab">Key takeaway</div>

treat a DataFrame as labelled columns with one dtype each, read files with an explicit schema rather than letting pandas guess, and run the same short inspection on every new table before you trust it.

</div>
