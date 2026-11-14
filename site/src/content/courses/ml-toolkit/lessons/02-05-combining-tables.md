---
id: "02.05"
course: "ml-toolkit"
module: "02"
slug: "combining-tables"
title: "Combining tables"
keyIdeas: ["`merge` is a SQL-style join: `how=\"inner\"`, `\"left\"`, `\"right\"`, `\"outer\"` or `\"cross\"`, on one or more key columns. For attaching features to labelled rows, `how=\"left\"` keeps every labelled row; the default inner join silently drops rows with missing features, often the new users you most need to model.", "Every matching pair of rows becomes an output row, so a key that appears $m$ times on the left and $n$ times on the right produces $m \\times n$ rows. Pass `validate=\"many_to_one\"` so pandas raises instead of quietly multiplying your training set.", "Check every join: assert the row count, use `indicator=True` to count unmatched rows, and make sure key dtypes, string formatting and missing keys are handled on both sides (in pandas, NaN keys match each other).", "`concat` stacks frames by aligning column names (or the index for `axis=1`); mismatched columns become `NaN` and mixed types fall back to `object`. Collect batches in a list and concatenate once, never inside a loop.", "For time-varying features, `merge_asof` joins each row to the latest record at or before its timestamp (sorted inputs, `by=` for the entity, `tolerance=` for staleness), which keeps training data point-in-time correct and free of leakage."]
question: "You train a model to predict whether a support ticket escalates. `tickets` has one row per ticket with `customer_id` and `created_at`. `accounts` is a nightly export with one row per customer per day (`customer_id`, `snapshot_date`, `plan`, `open_tickets`). You write `tickets.merge(accounts, on=\"customer_id\")`. What goes wrong, and what join should you write instead?"
answer: "`accounts` has many rows per customer, so the merge is many-to-many: each ticket is repeated once per snapshot of its customer, which explodes the row count (a customer with 400 daily snapshots and 5 tickets yields 2,000 rows). Even after deduplicating, every ticket would carry feature values from all dates, including snapshots taken after the ticket was created, so `open_tickets` would count the ticket itself and later tickets: target leakage. The default `how=\"inner\"` also drops tickets from customers missing from the export. The correct join is point in time: sort both frames by time and use `pd.merge_asof(tickets.sort_values(\"created_at\"), accounts.sort_values(\"snapshot_date\"), left_on=\"created_at\", right_on=\"snapshot_date\", by=\"customer_id\", direction=\"backward\", allow_exact_matches=False)`, so each ticket gets the latest snapshot strictly before it was created (both time columns as datetimes with the same time zone; a snapshot dated the same day could already reflect the ticket, so excluding exact matches is safer, or compare against the previous day's date). Add a `tolerance` so very stale snapshots become missing, assert that the output has exactly `len(tickets)` rows, and count tickets that found no snapshot. Keep `snapshot_date` in the output and assert it is always before `created_at`. The serving path must use the same rule: the latest snapshot available at prediction time."
sources: [{"title": "pandas docs: Merge, join, concatenate and compare", "url": "https://pandas.pydata.org/docs/user_guide/merging.html", "note": "merge semantics, validate, indicator, concat options and merge_asof."}, {"title": "pandas API: DataFrame.merge", "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html", "note": "Every merge parameter, including validate and suffixes."}, {"title": "pandas API: merge_asof", "url": "https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html", "note": "Sorting requirements, by, direction, tolerance and allow_exact_matches."}]
practice: ["merge-dataframes", "point-in-time-feature-join", "concatenate-dataframes", "setting-index"]
tags: ["merge", "joins", "point-in-time"]
tools: ["pandas"]
---

## What you will learn

- What a join does, row by row, and how the five kinds of `merge` (`inner`, `left`, `right`, `outer`, `cross`) differ on a tiny example you can check by eye.
- Why `how="left"` is the default choice for attaching features to labelled rows, and what an inner join silently throws away.
- How duplicate keys multiply rows ($m \times n$), how to predict the output size before you run a merge, and how `validate=` stops an explosion.
- How to make keys really match (dtypes, string clean-up, missing keys) and how to audit a join with row counts and `indicator=True`.
- How `pd.concat` stacks batches, what happens when their columns differ, and why you never call it inside a loop.
- How `merge_asof` joins each row to the latest record before its timestamp, which keeps time-varying features free of leakage.

*Before you start:* you should be comfortable with DataFrames, selecting columns and dtypes (02.01, 02.02), cleaning strings and missing values (02.03), and computing per-entity aggregates with `groupby` (02.04).

In lesson 02.04 you turned raw events into per-user aggregates: clicks per user, average basket size, days since last visit. Those aggregates live in their own table, one row per user. Your labels live in another table, one row per prediction you want to make. A model needs both in the same row. This lesson is about putting them there correctly.

## The idea in one picture

Here are two small tables. `labels` has one row per impression we want to predict (did the user click?). `feats` has one row per user with a few profile features.

```python
import pandas as pd

labels = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u9"],
                       "day": [3, 3, 4, 4, 5], "clicked": [1, 0, 1, 0, 1]})
feats = pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4"],
                      "age": [34, 27, 41, 22], "plan": ["pro", "free", "pro", "free"]})
```

Two details matter. User `u9` has labels but no features (a brand-new user). User `u4` has features but no labels. The column both tables share, `user_id`, is the **key**: the value used to decide which rows belong together. A **join** (pandas calls it a **merge**) builds a new table in which each output row glues together one row from the left table and one matching row from the right table.

~~~dg-fig 02.05-joins.svg
The same two tables joined three ways. Inner keeps only users present on both sides and loses u9; left keeps every label row and fills u9's missing features with NaN; outer keeps everything, including u4 who has no labels.
~~~

Reading the figure from left to right:

- **Inner join**: only keys found on both sides survive. Four rows; the `u9` impression is gone.
- **Left join**: every row of the left table survives exactly once (as long as the right keys are unique). Five rows; `u9` gets `NaN` for `age` and `plan`.
- **Outer join**: every key from either side survives. Six rows; `u4` appears with `NaN` for `day` and `clicked`.

For building a training set, the left join is the one you want: it keeps the number of labelled examples fixed and makes missing features visible instead of deleting the rows that have them.

## Step 1: choose the kind of join

The `how=` argument decides what happens to rows whose key has no partner on the other side.

| how | keeps | unmatched rows | typical use
| --- | --- | --- | --- |
| `inner` | keys present on both sides | dropped | both tables required
| `left` | every left row | right columns become `NaN` | attach features to labels
| `right` | every right row | left columns become `NaN` | rarely needed; swap and use left
| `outer` | all keys | `NaN` on the missing side | reconciliation, diffing two tables
| `cross` | every pair | n/a | candidate generation, small grids

The default is `how="inner"`, and that default is the most common source of quietly shrinking datasets. An inner join used to attach features drops every labelled example whose features are missing, and those examples are rarely random. They are new users and new items, the cold-start cases the model will certainly meet in production. Use `how="left"` and decide explicitly what a missing feature means.

A **cross join** has no key at all: it pairs every left row with every right row. Our tables give $5 \times 4 = 20$ rows. It is useful for building small grids (every user times every candidate item) and dangerous on anything large.

## Step 2: name the keys and handle clashing columns

The simplest call names one shared column with `on=`. Real tables are messier, so `merge` gives you a few more knobs:

```python
train = labels.merge(feats, on="user_id", how="left")               # same name both sides
train = train.merge(items, left_on="item", right_on="item_id",      # different names
                    how="left", suffixes=("", "_item"))
train = a.merge(b, on=["user_id", "day"], how="left")               # composite key
```

- `on=["user_id", "day"]` is a **composite key**: rows match only if every listed column matches.
- `left_on` / `right_on` handle keys with different names. Both columns are kept in the output; drop the redundant one afterwards.
- If both tables have a non-key column with the same name (say both have `plan`), pandas renames them `plan_x` and `plan_y`. Those names tell a reader nothing, so pass `suffixes=("", "_item")` to keep the left name unchanged and tag the right one.

`DataFrame.join` is a shortcut for joining on the index: `left.join(right, on="key")` matches `left["key"]` against `right.index`, and its default is already `how="left"`. `merge` is the general tool and is explicit about keys, so prefer it in pipelines.

## Step 3: predict the number of output rows

A merge produces **one output row for every matching pair** of rows. That sentence explains everything about row counts. If a key value $k$ appears $m_k$ times on the left and $n_k$ times on the right, it contributes $m_k \times n_k$ rows. For an inner join the total is

$$N_{\text{inner}} = \sum_{k} m_k\, n_k ,$$

where the sum runs over keys present on both sides. A left join adds one row for each left row whose key has no partner; an outer join adds the unmatched right rows too.

In our example every right key is unique ($n_k = 1$), so the inner join has $1 + 2 + 1 = 4$ rows: the left counts of the matched keys. That is the comfortable **many-to-one** case: many label rows may point at the same user, but each user has one feature row. Problems start when the right side is not unique.

<div class="ex">
<div class="lab">Worked example</div>

**count the rows before you merge**

Suppose a profile service wrote a second row for `u2` (age 27 on the old row, 28 on the new one). Now the right side has key counts u1: 1, u2: 2, u3: 1, u4: 1. The left side has u1: 1, u2: 2, u3: 1, u9: 1.

Matched keys: u1 gives $1 \times 1 = 1$ row, u2 gives $2 \times 2 = 4$ rows, u3 gives $1 \times 1 = 1$ row. Inner total: $1 + 4 + 1 = 6$.

Left join: add the unmatched left rows. Only u9 is unmatched, which adds 1 row: $6 + 1 = 7$ rows, from a label table of 5.

Outer join: also add the unmatched right rows. Only u4, so $7 + 1 = 8$ rows.

Running the merges in pandas prints 6, 7 and 8. The two extra rows are copies of u2's impressions, each appearing once with age 27 and once with age 28.

</div>

~~~dg-fig 02.05-explosion.svg
Left: each of u2's two label rows pairs with each of u2's two feature rows, giving four rows. Right: measured output size when 5,000 labels are merged with a table holding d rows per user; the result grows to 30 times the correct size at d = 30.
~~~

With a daily snapshot table (one row per user per day), 5,000 labels become 150,000 rows after a month.

Duplicated rows do more than waste memory:

- They **reweight training** towards users who happen to have more feature rows.
- If the duplicates disagree (old and new age), the model sees the same label with contradictory inputs.
- If the duplicates come from different dates, some of them come from the future relative to the label: **leakage**, the subject of Step 6.

## Step 4: guard every join

Three cheap checks catch almost all join bugs. Make them a habit on every pipeline merge.

```python
n0 = len(labels)
out = labels.merge(feats, on="user_id", how="left",
                   validate="many_to_one", indicator=True)
assert len(out) == n0, f"row count changed: {n0} -> {len(out)}"
print(out["_merge"].value_counts())
# _merge
# both          4
# left_only     1
# right_only    0
out = out.drop(columns="_merge")
```

**`validate=`** states what you believe about key uniqueness and makes pandas check it before joining. It accepts `"one_to_one"`, `"one_to_many"`, `"many_to_one"` and `"many_to_many"` (short forms `"1:1"`, `"1:m"`, `"m:1"`, `"m:m"`). With the duplicated `u2` row from the example it fails immediately:

```python
labels.merge(feats_dup, on="user_id", how="left", validate="many_to_one")
# MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
feats_dup["user_id"].value_counts().head(2)      # find the culprits
# u2    2
# u1    1
```

**Row-count assertion.** For a left join against a table with unique keys, the output must have exactly as many rows as the left table. If it does not, the right keys were not unique.

**`indicator=True`** adds a `_merge` column saying where each row came from: `both`, `left_only` or `right_only`. The `left_only` count is your feature coverage gap. A sudden jump in it between pipeline runs usually means an upstream table broke.

## Step 5: make the keys really match

A join that runs without error can still match far fewer rows than it should. The usual reasons:

- **dtype mismatch.** Merging an `int64` key with a string key raises an error in recent pandas (`ValueError: You are trying to merge on int64 and str columns`). An `int64` key against a `float64` key merges silently; a float key usually means the column had `NaN` somewhere upstream, which deserves a look. Convert both sides to one dtype before joining.
- **String normalisation.** `"ABC-1 "` does not match `"abc-1"`. Normalise both sides with the same function (`.str.strip().str.lower()`, lesson 02.03).
- **Missing keys.** In pandas, rows whose key is `NaN` or `None` *match each other*, unlike SQL where `NULL` never equals `NULL`. Two unrelated rows with a missing user id get glued together. Drop or fill missing keys before joining.

Quick test: `labels["user_id"].isin(feats["user_id"]).mean()` is the share of left keys found on the right (0.8 in our example). If you expected 0.99, stop and look.

## Step 6: point-in-time joins with merge_asof

Many features change over time: a user's plan, their number of open tickets, their 30-day spend. For each training example you need the value **that was known at the example's timestamp**, not the latest value and not all values. A join that respects this is called **point-in-time correct**.

`pd.merge_asof` does exactly this. For each left row it finds the last right row whose time is at or before the left row's time, optionally only among rows with the same entity (`by=`).

```python
labels = labels.sort_values("ts")
snapshots = snapshots.sort_values("ts")
train = pd.merge_asof(
    labels, snapshots, on="ts", by="user_id",
    direction="backward",            # latest snapshot at or before ts
    allow_exact_matches=False,       # strictly before: features computed at ts are excluded
    tolerance=pd.Timedelta("30D"),   # stale snapshots count as missing
)
```

- `on=` is the time column, and both frames must be **sorted** by it (pandas raises otherwise). Its dtypes must match: both `datetime64`, with the same time zone.
- `by=` restricts matches to the same user, like an equality key.
- `direction="backward"` looks into the past. `"forward"` and `"nearest"` exist, but looking forward is almost always leakage in a training set.
- `allow_exact_matches=False` excludes a snapshot stamped at exactly the label time. That is the conservative choice when a snapshot at time $t$ may already include the event being predicted.
- `tolerance=` turns very old matches into `NaN`, so the model is not fed a "current" plan that is a year stale.

Unlike `merge`, `merge_asof` returns exactly one row per left row: it never multiplies rows.

~~~dg-fig 02.05-asof.svg
For user u2, ticket A on 5 March skips the snapshot stamped the same day and takes 1 March; ticket B on 7 March takes 5 March; ticket C on 20 April finds only a 42-day-old snapshot, beyond the 30-day tolerance, and gets NaN.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**trace merge_asof by hand**

Snapshots for u2 (`open_tickets`): 1 Mar is 0, 5 Mar is 2, 9 Mar is 4. Tickets for u2 are created on 5 Mar, 7 Mar and 20 Apr. Settings: backward, exact matches excluded, tolerance 30 days.

Ticket on 5 Mar: snapshots strictly before 5 Mar are {1 Mar}. Latest is 1 Mar, which is 4 days old, within 30. Feature = 0.

Ticket on 7 Mar: snapshots strictly before are {1 Mar, 5 Mar}. Latest is 5 Mar, 2 days old. Feature = 2.

Ticket on 20 Apr: all three snapshots are before it; the latest is 9 Mar. March has 31 days, so the gap is $22 + 20 = 42$ days, more than 30. Feature = NaN.

With the defaults (exact matches allowed, no tolerance) pandas returns 2, 2 and 4 instead. Both the 5 Mar answer and the 20 Apr answer change, which is why these two settings deserve a deliberate choice.

A plain `merge(..., on="user_id")` would instead give $3 \times 3 = 9$ rows for u2, six of which pair a ticket with a snapshot taken after it.

</div>

Serving must use the same rule (the latest snapshot available at prediction time); otherwise features differ between training and production (**training-serving skew**).

## Step 7: stacking tables with concat

`merge` puts tables side by side by matching keys. `pd.concat` stacks them, which is what you do with daily files, shards of a dataset or train and test parts.

```python
jan = pd.DataFrame({"user_id": ["u1", "u2"], "clicks": [3, 5]})
feb = pd.DataFrame({"user_id": ["u3"], "clicks": [2], "device": ["ios"]})

pd.concat([jan, feb], ignore_index=True)
#   user_id  clicks device
# 0      u1       3    NaN
# 1      u2       5    NaN
# 2      u3       2    ios

pd.concat([jan, feb], ignore_index=True, join="inner")   # only shared columns
pd.concat({"jan": jan, "feb": feb}, names=["month"]).reset_index(level="month")
```

~~~dg-fig 02.05-concat.svg
Stacking along rows aligns columns by name. The February batch has an extra device column, so the January rows get NaN there; join="inner" would keep only the shared columns.
~~~

Things to know:

- Along rows (`axis=0`), columns are aligned **by name**; along columns (`axis=1`), rows are aligned **by index**, which behaves like an outer join on the index and can create `NaN` rows if the indexes differ.
- `ignore_index=True` gives a fresh `0..n-1` index. Without it, every batch keeps its own `0, 1, 2...` and the result has duplicate index labels, which later breaks `.loc` lookups.
- Passing a dict labels each piece, so you can keep track of which file a row came from.
- If a column's dtype differs between batches, the result falls back to `object`; check `df.dtypes`.

Never grow a frame by calling `concat` inside a loop. Each call copies every row collected so far, so the total work grows with the square of the number of batches. Collect pieces in a Python list and concatenate once.

~~~dg-fig 02.05-concat-loop.svg
Measured on this machine (timings are machine-dependent): building one frame from up to 800 batches of 1,000 rows. Concat inside the loop grows roughly quadratically; one concat of a list grows linearly and is tens of times faster at 800 batches (the exact factor is printed on the plot).
~~~

## Checking the result

After building a training table, check:

1. **Row count** equals the label table after every left join with a unique right side (assert it).
2. **Coverage**: the fraction of `NaN` in each newly attached column, compared with the previous run.
3. **Uniqueness**: `df.duplicated(subset=key_columns).sum()` is 0 if each label should appear once.
4. **Time ordering**: keep the snapshot time as a column and assert `(train["snapshot_ts"] < train["ts"]).all()`.
5. **Suspiciously good metrics** after adding a join: look for leakage before celebrating.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Training set shrinks after adding features | default `how="inner"` dropped unmatched labels | `how="left"`, handle missing features explicitly
| Training set grows after a "lookup" | duplicate keys on the right side | `validate="many_to_one"`, deduplicate or use `merge_asof`
| Almost nothing matches | dtype mismatch, untrimmed or mixed-case strings | cast and normalise keys on both sides
| Odd rows glued together | `NaN` keys match each other in pandas | drop or fill missing keys first
| Columns called `plan_x`, `plan_y` | same non-key column on both sides | `suffixes=("", "_right")` or rename first
| Great offline score, poor in production | features joined from the future | `merge_asof` with `direction="backward"`
| Pipeline slows down as data grows | `concat` inside a loop | collect a list, concat once
| Numbers became `object` dtype | batches disagree on a column's type | fix dtypes per batch before concat

## Going deeper

**How a merge is computed.** pandas turns key values into integer codes (factorising both sides together), then builds a hash table from one side and probes it with the other. The work is roughly proportional to $n_{\text{left}} + n_{\text{right}} + N_{\text{out}}$. The last term is why an explosion is also a performance problem: the output size, not the input size, sets the cost. Sorting both sides first is not needed; pass `sort=True` only if you want the output ordered by key.

**Why merge_asof is cheap.** Because both sides are sorted by time, pandas can sweep through them with two pointers, like merging two sorted lists, so the cost is about $n + m$ and the output always has exactly $n$ rows. The equivalent with plain `merge` would be a many-to-many join on `by`, a filter on time and a per-row maximum, which builds the full $\sum m_k n_k$ intermediate first.

**Point-in-time correctness at scale.** Feature stores implement the same rule: each feature value carries an "event time" (when it became true) and sometimes a "created time" (when it landed in the store). A strict backward as-of join on the time at which a value was *available*, not when it describes, is what prevents late-arriving data from leaking. Shift snapshot times to when they became available before calling `merge_asof`.

## Key terms

| Term | Meaning
| --- | --- |
| Key | Column (or columns) whose values decide which rows belong together
| Merge / join | Builds rows that combine one left row and one matching right row
| Left join | Keeps every left row; missing right values become NaN
| Many-to-one | Many left rows may share a key, but the right side has each key once
| Row explosion | Output larger than expected because duplicate keys pair up $m \times n$ times
| `validate` | Argument that checks the expected key uniqueness and raises `MergeError` if it is wrong
| `indicator` | Adds `_merge` (`both`, `left_only`, `right_only`) to show where rows came from
| `concat` | Stacks frames along rows or columns, aligning labels
| Point-in-time join | Each row receives feature values known at its own timestamp
| `merge_asof` | pandas function for point-in-time joins on a sorted time column

<div class="take">
<div class="lab">Key takeaway</div>

join features to labels with `how="left"` and `validate`, assert the row count, inspect unmatched keys with `indicator=True`, stack batches with one `concat` call, and use `merge_asof` whenever a feature changes over time.

</div>
