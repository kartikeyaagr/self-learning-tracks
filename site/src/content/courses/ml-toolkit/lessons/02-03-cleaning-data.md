---
id: "02.03"
course: "ml-toolkit"
module: "02"
slug: "cleaning-data"
title: "Cleaning data"
keyIdeas: ["pandas marks missing values as `NaN` (floats), `NaT` (datetimes) or `pd.NA` (nullable dtypes), and `isna()` detects all of them. Sentinels such as `-999`, `\"unknown\"` or `\"\"` are not missing until you convert them.", "Any cleaning step that learns a number from the data (a median to impute, a cap for outliers, a category list) is part of the model: fit it on the training split only and apply the same fitted values to validation, test and serving.", "Convert types deliberately with `pd.to_numeric` and `pd.to_datetime` using `errors=\"coerce\"`, then count what was coerced to missing; a silent jump in that count is a data-quality alert.", "Define duplicates by a business key, not by whole-row equality, and decide which copy to keep (`sort_values` then `drop_duplicates(keep=\"last\")`). Duplicates that straddle train and test inflate offline metrics."]
question: "You inherit a fraud dataset of card transactions. `amount` is `object` dtype, about 3% of `merchant_country` values are `\"\"` or `\"N/A\"`, there are 40,000 rows sharing a `transaction_id` with another row, and a notebook caps `amount` at the 99.5th percentile and fills missing `merchant_country` with the mode, both computed on the whole table before a random train/test split. List what you would check and change, in order, and explain the leakage risks."
answer: "First fix types: inspect why `amount` is text (currency symbols, thousands separators, words like 'refund'), clean with `.str` methods and convert with `pd.to_numeric(errors=\"coerce\")`, then count and examine the values that became NaN rather than dropping them blindly. Next make missing explicit: map `\"\"` and `\"N/A\"` to NaN (and check whether `read_csv` already turned `\"N/A\"` into NaN by default), and add a `merchant_country_missing` indicator because missingness may itself be a fraud signal. Then the duplicates: check whether rows sharing a `transaction_id` are exact copies (a double load, so drop) or versions of one transaction (an authorisation and a settlement, say), in which case keep the authoritative one with `sort_values` plus `drop_duplicates(subset=\"transaction_id\", keep=...)`. If duplicates remain and the split is random, the two copies of a transaction can land on either side, inflating test scores, so deduplicate before splitting and prefer a time-based split for fraud. Finally move the fitted steps after the split: compute the 99.5th percentile cap and the mode of `merchant_country` on the training rows only and apply those values to test and serving. Computing them on the whole table leaks test-set information into training features; the effect is usually small for a mode but can matter for caps when the test period has a different amount distribution."
sources: [{"title": "pandas docs: Working with missing data", "url": "https://pandas.pydata.org/docs/user_guide/missing_data.html", "note": "NaN, NaT and pd.NA, nullable dtypes, fillna, ffill and interpolate."}, {"title": "pandas docs: Working with text data", "url": "https://pandas.pydata.org/docs/user_guide/text.html", "note": "The str accessor, regex extraction and the string dtypes."}, {"title": "pandas docs: Duplicate labels", "url": "https://pandas.pydata.org/docs/user_guide/duplicates.html", "note": "How duplicate index labels behave and how to detect or disallow them."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Why preprocessing must be fitted on training data only, with examples of leakage."}]
practice: ["handle-missing-values", "drop-duplicates", "replace-values", "change-data-types", "rename-columns"]
tags: ["missing values", "data cleaning", "leakage"]
tools: ["pandas"]
---

## A cleaning pipeline has two kinds of steps

Some cleaning is pure transformation: trimming whitespace, lower-casing, parsing a date. It needs no statistics and behaves identically on any row. Other steps *learn* something: the median for imputation, the 99th percentile for capping, the set of known categories. Those are fitted parameters, exactly like model weights, and they create leakage if you compute them on data that includes the validation or test rows.

~~~dg-flow Stateless steps run anywhere; fitted steps learn on train only and are replayed everywhere.
Raw table -> Parse types -> Normalise strings -> Deduplicate -> *Fit stats on train -(medians, caps, categories)-> Apply to train, val, test, serving
~~~

## Missing values

Start by making missing values visible. Replace sentinels first, then measure:

```python
df = df.replace({"age": {-1: np.nan, 999: np.nan}, "city": {"": np.nan, "unknown": np.nan}})
df.isna().mean().sort_values(ascending=False)
```

Then choose a strategy per column:

- **Drop rows** (`dropna(subset=["label"])`) when the label itself is missing; such rows cannot train a supervised model.
- **Drop columns** that are mostly empty and carry no signal, but check first whether *being* missing is predictive.
- **Fill with a constant** (`0`, `"missing"`) when missing has a meaning, for example no purchases yet.
- **Fill with a statistic** fitted on train: `med = train["age"].median(); df["age"] = df["age"].fillna(med)`.
- **Fill within a group** or along time: `df["temp"] = df.groupby("station")["temp"].ffill()` for sensor gaps (lesson 7 discusses forward fill and time order).
- **Add an indicator** `df["age_missing"] = df["age"].isna()` before filling, so the model can still see that the value was absent.

`fillna(method="ffill")` was deprecated in pandas 2.1; use `.ffill()` and `.bfill()`. Backward fill uses future values and is a leakage risk in any time-ordered data.

## Type conversion

Columns that should be numeric often arrive as text because of a few bad rows. `astype(float)` raises on the first bad value; `to_numeric` with `errors="coerce"` converts what it can and marks the rest missing, which you should then count:

```python
raw = df["price"].str.replace(",", "", regex=False).str.strip()
df["price"] = pd.to_numeric(raw, errors="coerce")
bad = df["price"].isna() & raw.notna()
print(bad.sum(), raw[bad].unique()[:10])     # what failed to parse, and why

df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True)
df["n_items"] = df["n_items"].astype("Int64")   # integers with gaps stay integers
```

Pass an explicit `format` to `to_datetime`. Since pandas 2.0 the format is inferred from the first non-missing value and applied to all rows, so a file mixing `03/04/2024` (day first) with `2024-04-03` will either error or coerce half the rows to `NaT`.

## Strings

The `.str` accessor applies vectorised string operations and propagates missing values:

```python
df["email"] = df["email"].str.strip().str.lower()
df["plan"] = df["plan"].str.replace(r"\s+", "_", regex=True)
df[["area", "number"]] = df["phone"].str.extract(r"^\((\d{3})\)\s*(\d{3}-\d{4})$")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
```

Normalising strings before grouping or joining matters more than it seems: `"Pro"`, `"pro "` and `"PRO"` become three categories with a third of the data each. In pandas 3.0 text columns use the `str` dtype by default, and `.str` methods run on Arrow kernels when pyarrow is installed, which is considerably faster than on `object` columns.

## Duplicates

Whole-row duplicates (`df.duplicated().sum()`) usually mean a pipeline loaded the same file twice. Key duplicates are subtler: two rows for the same `(user_id, event_id)` with different timestamps or values. Decide which version is authoritative and keep it explicitly:

```python
df = (df.sort_values("updated_at")
        .drop_duplicates(subset=["user_id", "event_id"], keep="last"))
```

For ML there is a second concern: near-identical examples split across train and test. The model memorises the training copy and scores the test copy perfectly. Split by a group key (user, session, document) when such duplicates are likely.

## Outliers

Distinguish **errors** (negative ages, a timestamp in 1970, a price of 10^9) from **rare but real** values (a genuinely large order). Errors become missing; real extremes are usually capped (winsorised) rather than dropped, using thresholds learned on train:

```python
lo, hi = train["amount"].quantile([0.01, 0.99])
for part in (train, val, test):
    part["amount"] = part["amount"].clip(lo, hi)
```

Never drop outlier rows from the validation or test set; the model will see them in production. Tree models are fairly insensitive to extremes, while linear models and neural networks benefit from clipping or a log transform such as `np.log1p`.

<div class="ex">
<div class="lab">Worked example</div>

**median imputation that leaked**

A churn model imputes missing `tenure_days` with `df["tenure_days"].median()` computed on the full dataset before the train/test split. Missing tenure is concentrated in recent sign-ups, who churn more, so the imputed value encodes a bit of information from the test rows.

Moving the median calculation after the split, `med = train["tenure_days"].median()`, and adding a `tenure_missing` flag changes the test AUC from 0.842 to 0.836. The lower number is the honest one. In scikit-learn, a `SimpleImputer` inside a `Pipeline` enforces this automatically during cross-validation.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

make missing values and bad types visible first, keep stateless cleaning separate from fitted cleaning, and fit every statistic on the training split only, then replay it unchanged everywhere else.

</div>
