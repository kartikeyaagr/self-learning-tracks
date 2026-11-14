---
id: "02.03"
course: "ml-toolkit"
module: "02"
slug: "cleaning-data"
title: "Cleaning data"
keyIdeas: ["pandas marks missing values as `NaN` (floats and default text), `NaT` (datetimes) or `pd.NA` (nullable dtypes such as `Int64`), and `isna()` detects all of them. Sentinels such as `-1`, `999` or `\"unknown\"` are not missing until you convert them, and they can hide most of a column's true missing rate.", "Convert types deliberately with `pd.to_numeric` and `pd.to_datetime` using `errors=\"coerce\"` and an explicit date `format`, then count what was coerced to missing; a sudden jump in that count is a data-quality alert.", "Normalise text with the `.str` accessor (`strip`, `lower`) before grouping or encoding, otherwise one category splits into several thin ones.", "Define duplicates by a business key, not by whole-row equality, and keep the authoritative copy explicitly (`sort_values` then `drop_duplicates(keep=\"last\")`). Copies that straddle train and test inflate offline metrics.", "Any cleaning step that learns a number from the data (a median to impute, a cap for outliers, a category list) is part of the model: fit it on the training split only, add a missing indicator where absence may matter, and apply the same fitted values to validation, test and serving."]
question: "You inherit a fraud dataset of card transactions. `amount` is `object` dtype, about 3% of `merchant_country` values are `\"\"` or `\"N/A\"`, there are 40,000 rows sharing a `transaction_id` with another row, and a notebook caps `amount` at the 99.5th percentile and fills missing `merchant_country` with the mode, both computed on the whole table before a random train/test split. List what you would check and change, in order, and explain the leakage risks."
answer: "Work from stateless fixes to fitted ones. (1) Types first: look at why `amount` is text by listing values that fail `pd.to_numeric` (currency symbols, thousands separators, words such as 'refund', a stray header row). Strip them with `.str.replace` and convert with `pd.to_numeric(errors=\"coerce\")`, then count `new.isna() & raw.notna()` and inspect every value that became NaN instead of dropping those rows blindly; a refund might need a negative sign rather than NaN. (2) Make missing explicit: `read_csv` already turns empty fields and `\"N/A\"` into NaN by default, so check what is really in the column, map any remaining sentinels to NaN, and add a `merchant_country_missing` indicator, because a missing merchant country may itself be a fraud signal. (3) Duplicates: check whether rows sharing a `transaction_id` are exact copies (a double load, so `drop_duplicates()`) or different versions of one transaction (an authorisation and a later settlement, say). In the second case decide which record is authoritative and keep it with `sort_values` on the timestamp plus `drop_duplicates(subset=\"transaction_id\", keep=\"last\")`. If copies survive and the split is random, the two halves of one transaction can land in train and test, and the model scores the test copy from memory, inflating metrics. Deduplicate before splitting, and for fraud prefer a time-based split, since that is how the model will be used. (4) Only now the fitted steps, after the split: compute the 99.5th-percentile cap and the mode of `merchant_country` on the training rows only and apply those exact values to test and serving, ideally inside a scikit-learn `Pipeline` (`SimpleImputer(strategy=\"most_frequent\", add_indicator=True)` and a clipping step) so cross-validation refits them per fold. Computing them on the whole table leaks test information into training features. For the mode the effect is usually tiny; for the cap it can matter, especially when the test period has a different amount distribution, for example a new high-value merchant. Finally re-check `isna().sum()`, dtypes and row counts after each step so every change is explained."
sources: [{"title": "pandas docs: Working with missing data", "url": "https://pandas.pydata.org/docs/user_guide/missing_data.html", "note": "NaN, NaT and pd.NA, nullable dtypes, fillna, ffill and interpolate."}, {"title": "pandas docs: Working with text data", "url": "https://pandas.pydata.org/docs/user_guide/text.html", "note": "The str accessor, regex extraction and the string dtypes."}, {"title": "pandas docs: Duplicate labels", "url": "https://pandas.pydata.org/docs/user_guide/duplicates.html", "note": "How duplicate index labels behave and how to detect or disallow them."}, {"title": "scikit-learn: Common pitfalls and recommended practices", "url": "https://scikit-learn.org/stable/common_pitfalls.html", "note": "Why preprocessing must be fitted on training data only, with examples of leakage."}]
practice: ["handle-missing-values", "drop-duplicates", "replace-values", "change-data-types", "rename-columns"]
tags: ["missing values", "data cleaning", "leakage"]
tools: ["pandas"]
---

## What you will learn

- How pandas marks missing values, and how to expose fakes like `-1` or `"unknown"`.
- How to convert text to numbers and dates, and count what failed.
- How to normalise strings so that `"Pro"`, `"pro "` and `"PRO"` stop being three categories.
- How to find duplicates by a business key, and why copies on both sides of a split inflate scores.
- How to fill gaps and cap outliers with values learned on training data only.

*Before you start:* loading a CSV and reading dtypes (02.01), boolean masks and `.loc` writes (02.02), and the idea of a train/test split.

## From selecting to cleaning

Lesson 02.02 showed how to pick out rows and columns. This lesson uses those tools for the least glamorous, most important job in applied ML: turning the table you were given into one a model can learn from honestly.

Real data arrives broken in predictable ways: numbers stored as text, disguised gaps, one category spelled five ways, repeated rows, wild values. Each problem either crashes your code (the lucky case) or silently teaches the model something false. The aim is a routine you run on every new table.

## The idea in one picture

Here is a tiny customer table for a churn model, exactly as a colleague exported it:

```python
import io, numpy as np, pandas as pd

csv = '''user_id,age,city,plan,fee,signup,churned
101,34,London,Pro,"1,200",2024-01-15,0
102,-1,london ,pro ,950,2024-02-03,1
103,27,,PRO,n/a,2024-02-30,0
104,999,Paris,Basic,400,2024-03-11,
105,41,unknown,Basic,420,2024-03-19,1
102,-1,london ,pro ,950,2024-02-03,1
106,29,Paris,Pro,"£1,150",2024-04-02,0
107,,Berlin,basic,380,2024-04-20,0
'''
df = pd.read_csv(io.StringIO(csv))
print(df.dtypes.astype(str).to_dict())
# {'user_id': 'int64', 'age': 'float64', 'city': 'str', 'plan': 'str',
#  'fee': 'str', 'signup': 'str', 'churned': 'float64'}
```

Eight rows, and almost every column has a problem. The figure colours each one.

~~~dg-fig 02.03-table.svg
Top: the raw table, with each problem cell coloured by kind. Bottom: the same data after cleaning. Two rows are gone (a duplicate and a row with no label), every column has a proper type, and every gap is an explicit NaN or NaT.
~~~

Reading the top grid in plain words:

- `age` uses `-1` and `999` for "we don't know". These are **sentinels**: ordinary-looking values standing in for missing data.
- `fee` is text (`str`), because two rows contain a comma or a pound sign.
- `city` and `plan` mix capitals and trailing spaces (shown as a dot), so one plan becomes three categories.
- `signup` contains 30 February, a date that does not exist.
- User 102 appears twice, and user 104 has no `churned` label at all.

The bottom grid is the goal: every value is either trustworthy or openly **missing**. Cleaning did *not* invent ages for users 102 and 107; filling gaps is a separate, later decision, and the one that can leak.

## Step 1: make missing values visible

pandas has three markers for "no value", chosen by the column's type:

- `NaN` ("not a number") in float columns, and in the default `str` text columns of pandas 3.
- `NaT` ("not a time") in datetime columns.
- `pd.NA` in the nullable dtypes such as `Int64` (capital I), `boolean` and `string`.

`isna()` (and `notna()`) recognises all three. Never write `x == np.nan`: NaN equals nothing, not even itself.

`read_csv` already turns empty fields and spellings like `NA`, `N/A`, `n/a` and `null` into NaN (so user 103's fee is missing, not `"n/a"`). It cannot know that `-1` or `"unknown"` mean missing, so translate sentinels first, then measure:

```python
df = df.replace({"age": {-1: np.nan, 999: np.nan}, "city": {"unknown": np.nan}})
print(df.isna().mean().round(2).sort_values(ascending=False).head(4).to_dict())
# {'age': 0.5, 'city': 0.25, 'fee': 0.12, 'churned': 0.12}
```

`.mean()` of a True/False table is the fraction of True values, so this line reports each column's **missing rate**. (`read_csv(..., na_values={"age": [-1, 999]})` does the same at load time.)

~~~dg-fig 02.03-missing.svg
Missing rate per column before and after translating sentinels. Before, age looks only 1 in 8 missing; once -1 and 999 are treated as gaps, half of the ages are unknown, which changes what you should do with the column.
~~~

Every later decision (drop the column? fill it? add a flag?) depends on the true missing rate, and sentinels hide it. To find sentinels nobody told you about, look at `describe()` for impossible minimums and maximums and at `value_counts()` for suspiciously popular values such as `0`, `-1` or `9999`.

## Step 2: give every column its real type

A column's **dtype** decides what you can do with it: you can average a `float64` but not text, and one bad cell makes pandas read a whole numeric column as text.

`astype(float)` raises on the first value it cannot parse. `pd.to_numeric(..., errors="coerce")` converts what it can and turns the rest into NaN ("coerce": force, and give up quietly). Quiet failure is dangerous unless you count it, so compare values that *became* missing with those that already were:

```python
raw = df["fee"].str.replace(",", "", regex=False).str.strip()
fee = pd.to_numeric(raw, errors="coerce")
bad = fee.isna() & raw.notna()          # had text, now NaN: the parse failed
print(bad.sum(), raw[bad].tolist())      # 1 ['£1150']
```

The count reveals a pound sign. Fix the rule, not the row, and check again:

```python
raw = df["fee"].str.replace(r"[£,\s]", "", regex=True)   # drop pound signs, commas, spaces
df["fee"] = pd.to_numeric(raw, errors="coerce")
print((df["fee"].isna() & raw.notna()).sum())            # 0
```

For dates, always pass an explicit `format`. Since pandas 2.0 the format is guessed from the first non-missing value and applied to every row, so a file mixing `03/04/2024` with `2024-04-03` raises, or with `errors="coerce"` quietly turns rows into `NaT`:

```python
df["signup"] = pd.to_datetime(df["signup"], format="%Y-%m-%d", errors="coerce")
print(df["signup"].isna().sum())   # 1  -> the 30 February row, now NaT
```

An `int64` column cannot hold NaN, so a gap promotes it to `float64` (why `churned` loaded as `0.0`/`1.0`). The nullable `Int64` keeps integers whole with `<NA>` gaps: `.astype("Int64")`.

## Step 3: normalise strings

To a computer `"Pro"`, `"pro "` and `"PRO"` are three strings, so a group-by or one-hot encoder splits one plan into three thin categories. The `.str` accessor applies a string operation to every value and passes missing values through:

```python
print(df["plan"].unique().tolist())          # ['Pro', 'pro ', 'PRO', 'Basic', 'basic']
for c in ["city", "plan"]:
    df[c] = df[c].str.strip().str.lower()     # trim spaces, then lower-case
print(df["plan"].value_counts().to_dict())   # {'pro': 5, 'basic': 3}
```

Other patterns: `.str.replace(r"\s+", "_", regex=True)` collapses whitespace, `.str.extract(r"^\((\d{3})\)\s*(\d{3}-\d{4})$")` pulls regex groups into new columns, and `df.columns.str.strip().str.lower()` tidies column names.

In pandas 3.0 text columns get the `str` dtype by default instead of the old catch-all `object`, and when pyarrow is installed the `.str` methods run on fast Arrow kernels.

## Step 4: remove duplicates on purpose

There are two kinds of duplicate, and they need different thinking.

A **whole-row duplicate** is an identical copy, usually because a pipeline loaded the same file twice. `df.duplicated().sum()` counts them and `drop_duplicates()` removes all but the first.

A **key duplicate** is subtler: two rows for the same entity, identified by a **business key** such as `(user_id, event_id)`, that differ in some column, for example a payment first recorded as `pending` and later as `paid`. Whole-row equality misses these. You must decide which version is authoritative and keep it explicitly:

```python
events = pd.DataFrame({"user": [1, 1, 2], "event": [7, 7, 8],
                       "updated_at": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"]),
                       "status": ["paid", "pending", "paid"]})
latest = (events.sort_values("updated_at")
                .drop_duplicates(subset=["user", "event"], keep="last"))
print(latest["status"].tolist())   # ['paid', 'paid']  -> the later 'paid' row won
```

Sort first: `keep="last"` means last in the current row order, not most recent.

For ML there is a nastier problem. If copies of one customer land on both sides of a random split, the model memorises the training copy and "predicts" the test copy perfectly; the score measures memory, not generalisation. Measured on a synthetic task with 30% of rows duplicated:

~~~dg-fig 02.03-dupleak.svg
Measured test accuracy on 3,000 synthetic customers with 30% of them copied. With a random row split, copies leak into the test set and the scores rise by 8 to 11 points; splitting by user id gives the honest numbers.
~~~

The deep decision tree, which memorises best, gains the most: 0.756 instead of an honest 0.646. Deduplicate before you split, and when near-duplicates are natural (several sessions of one user, several photos of one product), split by the group key, for example with scikit-learn's `GroupShuffleSplit` or `GroupKFold`.

## Step 5: decide what to do with the gaps

Now every unknown value is openly missing. What to do with each gap is a per-column choice:

- **Drop rows** whose *label* is missing (`dropna(subset=["churned"])`). A supervised model cannot learn from an example with no answer.
- **Drop a column** that is almost empty, after checking that *being* missing does not predict the label.
- **Fill with a constant** when missing has a meaning: no orders yet means 0 orders.
- **Fill with a statistic** such as the median, learned on the training rows: `med = train["age"].median()`.
- **Fill along time within a group**: `df.groupby("station")["temp"].ffill()` carries the last reading forward per station (02.07). `fillna(method="ffill")` was deprecated in pandas 2.1 and fails in pandas 3; use `.ffill()`. Avoid `.bfill()` on time-ordered data: it copies future values into the past.
- **Add an indicator** column before filling, `df["age_missing"] = df["age"].isna().astype("int8")`, so the model still knows the value was absent.

Why bother with the indicator? Because filling hides information. In the next figure, young customers skip the age field more often than older ones. After a median fill, 201 people are suddenly exactly 43 years old.

~~~dg-fig 02.03-impute.svg
Left: the observed ages under-represent young customers, because they leave the field blank more often. Right: filling every gap with the median piles 201 rows onto one value and shrinks the spread from 11.1 to 9.9 years. An indicator column keeps the "was missing" signal that the fill erased.
~~~

The fill makes the column look less varied than it is, and the spike is not a real group of people. Worse, missingness itself was informative ("probably young"), and only the indicator keeps that.

The median is a number **learned from data**, so it is a model parameter like any weight, and it must come from the training rows only. Here is why that matters, by hand.

<div class="ex">
<div class="lab">Worked example</div>

**where the fill value comes from, and a cap learned on train**

Training ages: $(23, 31, \text{NaN}, 45, 38, \text{NaN})$. Test ages: $(\text{NaN}, 62, 70)$.

Honest median: sort the four observed training ages, $(23, 31, 38, 45)$. With an even count the median is the mean of the middle two: $(31 + 38)/2 = 34.5$.

Leaky median, computed before splitting: the observed ages are $(23, 31, 38, 45, 62, 70)$, and the middle two are 38 and 45, so the median is $(38 + 45)/2 = 41.5$.

The two test values moved every filled training age by $41.5 - 34.5 = 7$ years. The model trained on numbers it could only know by peeking at the test set.

The same rule applies to outlier caps. Eleven training amounts: $(12, 15, 18, 20, 22, 25, 27, 30, 34, 40, 900)$. pandas' default quantile places value $q$ at sorted position $q \times (n - 1)$, counting from 0. For the 10th percentile: $0.1 \times 10 = 1$, so the value is 15. For the 90th: $0.9 \times 10 = 9$, so the value is 40.

Clipping to $[15, 40]$ changes 12 to 15 and 900 to 40. The sum drops from $1143$ to $286$, so the mean drops from $1143/11 = 103.9$ to $286/11 = 26.0$. One wild value had quadrupled the average.

Those two cut-offs, 15 and 40, are then applied unchanged to the validation and test rows, whatever their own percentiles are.

</div>

In code the honest version is a matter of order: split first, run `med = train["age"].median()`, then apply `fillna(med)` to every split (or let a `Pipeline` do it, below).

## Step 6: outliers, errors or rare truths?

An **outlier** is a value far from the rest. Before touching one, decide which kind it is:

- An **error** is impossible or clearly wrong: a negative age, a signup in 1970 for an app launched in 2021, a price of $10^9$. Turn it into NaN and treat it as missing (Step 5).
- A **rare but real** value is a genuinely large order or a very long session. Keep the row; production will contain such rows too. Instead, limit its pull on the model.

The usual ways to limit pull are **capping** (also called winsorising: values beyond a cut-off are set to the cut-off) with thresholds learned on train, or a log transform:

```python
lo, hi = train["amount"].quantile([0.01, 0.99])
for part in (train, val, test):
    part["amount"] = part["amount"].clip(lo, hi)
# or, for positive skewed values: part["amount_log"] = np.log1p(part["amount"])
```

~~~dg-fig 02.03-clip.svg
3,000 simulated transaction amounts. Left: a long right tail, with the training 99th percentile (329) marked. Middle: clip() keeps every row but piles the extreme ones at the cap. Right: log1p squeezes the tail smoothly instead, giving a roughly bell-shaped feature.
~~~

Tree models split on thresholds and barely notice how extreme a value is. Linear models and neural networks multiply inputs by weights, so one huge value can dominate a prediction or gradient; they benefit from clipping or `np.log1p` (log of 1 + x, safe at zero). Never drop outlier rows from validation or test: that tests the model on a cleaner world than it will meet.

## Putting it together: stateless and fitted steps

The steps fall into two kinds. **Stateless** steps need no statistics (trimming text, parsing dates, mapping sentinels, dropping exact duplicates) and behave identically on any row. **Fitted** steps learn a number from data (a median, a cap, the known categories); learn those on training data only and replay them unchanged on validation, test and live traffic.

~~~dg-flow Stateless steps run anywhere; fitted steps learn on train only and are replayed everywhere.
Raw table -> Parse types -> Normalise strings -> Deduplicate -> *Fit stats on train -(medians, caps, categories)-> Apply to train, val, test, serving
~~~

## Cleaning in code

Put the stateless part in one function, so the same rules run on training data and in production:

```python
def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Stateless cleaning: safe to run on any split, any day.
    out = df.replace({"age": {-1: np.nan, 999: np.nan}, "city": {"unknown": np.nan}})
    for c in ["city", "plan"]:
        out[c] = out[c].str.strip().str.lower()
    out["fee"] = pd.to_numeric(out["fee"].str.replace(r"[£,\s]", "", regex=True), errors="coerce")
    out["signup"] = pd.to_datetime(out["signup"], format="%Y-%m-%d", errors="coerce")
    out = out.drop_duplicates().dropna(subset=["churned"])
    out["churned"] = out["churned"].astype("Int64")
    return out

df = clean(pd.read_csv(io.StringIO(csv)))
print(df.shape)    # (6, 7)
```

Then let scikit-learn own the fitted part. Inside a `Pipeline`, `fit` learns the medians on the training rows only, and cross-validation refits them in every fold automatically:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median", add_indicator=True), ["age", "fee"]),
    ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                          OneHotEncoder(handle_unknown="ignore")), ["city", "plan"]),
])
model = make_pipeline(pre, LogisticRegression())
X, y = df[["age", "fee", "city", "plan"]], df["churned"].astype(int)
model.fit(X.iloc[:4], y.iloc[:4])                           # first 4 rows play "train"
print(model[0].named_transformers_["num"].statistics_)      # [ 34. 950.]  train medians
print(model[0].get_feature_names_out()[:4].tolist())
# ['num__age', 'num__fee', 'num__missingindicator_age', 'num__missingindicator_fee']
```

`add_indicator=True` adds the "was missing" columns for you, and `handle_unknown="ignore"` stops a city never seen in training from crashing predictions later.

## Checking the result

A cleaning step is finished when you can show what it did. Habits that catch most problems:

- Print `df.isna().sum()` before and after each step; every change should have a reason.
- After any `errors="coerce"`, count the newly missing values. A sudden jump between two data deliveries is a data-quality alert, not noise.
- Check `df.dtypes`: nothing you plan to average should still be `str`.
- Check `value_counts()` of every category column for near-twins such as `"london"` and `"londn"`.
- Record row counts: 8 rows in, 1 duplicate out, 1 unlabelled out, 6 rows left. If 40% of rows vanish, stop and find out why.
- Turn facts you rely on into assertions: `assert df["user_id"].is_unique`.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Sentinels left in place | odd spikes at -1 or 999; a mean age of 150 | map them to NaN, then re-measure missing rates
| `errors="coerce"` without counting | a column is quietly 30% NaN | count `new.isna() & raw.notna()` and inspect the failures
| Date format left to inference | half the dates are `NaT`, or days and months swap | pass an explicit `format=`
| Categories not normalised | three versions of one plan after one-hot encoding | `.str.strip().str.lower()` before grouping or encoding
| Duplicates across the split | test score far above live performance | deduplicate by key before splitting; split by group
| Medians or caps fitted on all data | slightly optimistic test score | split first, fit on train, or use a `Pipeline`
| Backward fill on time series | features that "know" the future | use `ffill` only, per entity, in time order
| Dropping outlier rows from test | metrics better than production | keep test rows; cap or transform instead

The fitted-on-all-data mistake is common because its error is small and invisible. A typical case: a churn model filled missing `tenure_days` with a median computed before splitting; moving the median after the split and adding a `tenure_missing` flag lowered test AUC from 0.842 to 0.836. The lower number was the honest one.

## Going deeper

**Why values go missing matters more than how many.** *Missing completely at random* (MCAR): the gap is unrelated to anything; any sensible fill is harmless. *Missing at random* (MAR): the gap depends on other observed columns, as when young users skip age; fills that use those columns (group medians, `IterativeImputer`, `KNNImputer`) recover much of it. *Missing not at random* (MNAR): the gap depends on the hidden value itself, as when high earners hide income; no fill recovers it, and the indicator is your best tool.

**Why the median, not the mean.** The mean moves with every extreme value (one 900 lifted our mean from 26 to 104); the median does not. Both shrink the column's variance (the impute figure); fine for prediction, but for statistical inference use multiple imputation.

**Some models handle NaN natively.** scikit-learn's `HistGradientBoostingClassifier`, XGBoost and LightGBM learn, at each split, which side missing values should go. With them you can often skip imputation entirely.

**Three-valued logic.** `pd.NA > 1` is `pd.NA`, but `pd.NA | True` is `True` and `pd.NA & False` is `False`. Fill a mask containing `pd.NA` (`mask.fillna(False)`) before selecting with it.

**Near-duplicates** (the same photo resized, the same text with one word changed) escape `drop_duplicates`; at scale they are found with hashing tricks such as perceptual hashes or MinHash.

## Key terms

| Term | Meaning
| --- | --- |
| Missing value | A cell with no known value, marked `NaN`, `NaT` or `pd.NA`; detected by `isna()`
| Sentinel | An ordinary-looking value such as -1, 999 or "unknown" that secretly means missing
| Nullable dtype | A dtype like `Int64` or `boolean` that can hold `pd.NA` without changing type
| Business key | The columns that identify one real-world entity or event, used to define duplicates
| Imputation | Filling missing values, for example with a training-set median
| Missing indicator | A 0/1 column recording that a value was absent before it was filled
| Winsorising / capping | Setting values beyond a cut-off to the cut-off, keeping the row
| Leakage | Information from validation or test data reaching the model during training

<div class="take">
<div class="lab">Key takeaway</div>

make missing values and bad types visible first, keep stateless cleaning separate from fitted cleaning, and fit every statistic (medians, caps, category lists) on the training split only, then replay it unchanged everywhere else.

</div>
