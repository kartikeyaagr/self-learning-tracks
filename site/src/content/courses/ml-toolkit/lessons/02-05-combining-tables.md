---
id: "02.05"
course: "ml-toolkit"
module: "02"
slug: "combining-tables"
title: "Combining tables"
keyIdeas: ["`merge` is a SQL join: `how=\"inner\"`, `\"left\"`, `\"right\"`, `\"outer\"` or `\"cross\"`, on one or more key columns. For attaching features to labelled rows, `how=\"left\"` keeps every labelled row.", "Duplicate keys multiply rows: a key that appears $m$ times on the left and $n$ times on the right produces $m \\times n$ output rows. Pass `validate=\"many_to_one\"` so pandas raises instead of exploding.", "Check every join: compare row counts before and after, use `indicator=True` to count unmatched rows, and make sure key dtypes and string normalisation match on both sides.", "`concat` stacks frames along rows or columns by aligning labels; mismatched columns become `NaN`. For time-dependent features, `merge_asof` joins each row to the latest record at or before its timestamp, which keeps joins point-in-time correct."]
question: "You train a model to predict whether a support ticket escalates. `tickets` has one row per ticket with `customer_id` and `created_at`. `accounts` is a nightly export with one row per customer per day (`customer_id`, `snapshot_date`, `plan`, `open_tickets`). You write `tickets.merge(accounts, on=\"customer_id\")`. What goes wrong, and what join should you write instead?"
answer: "`accounts` has many rows per customer, so the merge is many-to-many: each ticket is repeated once per snapshot of its customer, which explodes the row count (a customer with 400 daily snapshots and 5 tickets yields 2,000 rows). Even after deduplicating, every ticket would carry feature values from all dates, including snapshots taken after the ticket was created, so `open_tickets` would count the ticket itself and later tickets: target leakage. The default `how=\"inner\"` also drops tickets from customers missing from the export. The correct join is point in time: sort both frames by time and use `pd.merge_asof(tickets.sort_values(\"created_at\"), accounts.sort_values(\"snapshot_date\"), left_on=\"created_at\", right_on=\"snapshot_date\", by=\"customer_id\", direction=\"backward\", allow_exact_matches=False)`, so each ticket gets the latest snapshot strictly before it was created (both time columns as datetimes with the same time zone; a snapshot dated the same day could already reflect the ticket, so excluding exact matches is safer, or compare against the previous day's date). Add a `tolerance` so very stale snapshots become missing, assert that the output has exactly `len(tickets)` rows, and count tickets that found no snapshot. The serving path must use the same rule: the latest snapshot available at prediction time."
sources: [{"title": "pandas docs: Merge, join, concatenate and compare", "url": "https://pandas.pydata.org/docs/user_guide/merging.html", "note": "merge semantics, validate, indicator, concat options and merge_asof."}, {"title": "pandas API: DataFrame.merge", "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html", "note": "Every merge parameter, including validate and suffixes."}, {"title": "pandas API: merge_asof", "url": "https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html", "note": "Sorting requirements, by, direction, tolerance and allow_exact_matches."}]
practice: ["merge-dataframes", "point-in-time-feature-join", "concatenate-dataframes", "setting-index"]
tags: ["merge", "joins", "point-in-time"]
tools: ["pandas"]
---

## merge is a join

Feature pipelines are mostly joins: labels joined to user features, item features, and aggregates from lesson 4. pandas `merge` follows SQL semantics (the SQL module covers the same ideas in a database).

```python
train = labels.merge(user_feats, on="user_id", how="left", validate="many_to_one")
train = train.merge(items, left_on="item", right_on="item_id", how="left",
                    suffixes=("", "_item"))
```

| how | keeps | unmatched rows | typical use
| --- | --- | --- | --- |
| `inner` | keys present on both sides | dropped | both tables required
| `left` | every left row | right columns become `NaN` | attach features to labels
| `right` | every right row | left columns become `NaN` | rarely needed; swap and use left
| `outer` | all keys | `NaN` on the missing side | reconciliation, diffing two tables
| `cross` | every pair | n/a | candidate generation, small grids

An inner join used to attach features quietly drops labelled examples whose features are missing, and those are often not random (new users, new items). The model then never trains on the cold-start case it will meet in production. Use `how="left"` and handle the missing features explicitly.

`DataFrame.join` is a convenience for joining on the index (`left.join(right, on="key")` uses the right frame's index); `merge` is the general tool and is clearer about keys.

## Row explosions

A merge produces one output row for every matching pair of rows. If a key value appears $m$ times on the left and $n$ times on the right, it contributes $m \times n$ rows, so the output size is $\sum_k m_k n_k$.

~~~dg-flow Why a "lookup" join can multiply rows.
labels: user u7 x 3 rows -(merge on user_id)-> user_feats: u7 x 2 rows (stale + fresh) -> *u7 x 6 rows in train
~~~

Duplicated rows are not just wasted memory: they reweight training toward users with duplicated features, and if the duplicates differ (an old and a new feature value) the model sees contradictory inputs. Protect every join:

```python
n0 = len(labels)
out = labels.merge(user_feats, on="user_id", how="left",
                   validate="many_to_one", indicator=True)
assert len(out) == n0
print(out["_merge"].value_counts())        # both / left_only / right_only
out = out.drop(columns="_merge")
```

`validate` accepts `"one_to_one"`, `"one_to_many"`, `"many_to_one"` and `"many_to_many"` (or the short forms `"1:1"`, `"1:m"`, `"m:1"`, `"m:m"`) and raises `MergeError` if the key uniqueness assumption is wrong. It costs a uniqueness check and is worth it on every pipeline join. To find the culprits, `user_feats["user_id"].value_counts().head()` shows the duplicated keys.

## Keys must really match

Common reasons a join matches less than expected:

- **dtype mismatch.** Merging an `int64` key with an `object` or string key raises a `ValueError` in recent pandas. `int64` against `float64` will merge, but a float key that came from a column with `NaN` is a hint that something upstream is wrong.
- **String normalisation.** `"ABC-1 "` does not match `"abc-1"`; normalise both sides with the same function (lesson 3).
- **Missing keys.** Rows whose key is `NaN` can match each other in pandas merges, unlike SQL where `NULL` never equals `NULL`. Drop or fill missing keys before joining.
- **Time.** A feature table snapshotted today, joined to labels from last year, gives every training row today's feature values. That is leakage, and it is the subject of the last section.

## concat

`pd.concat` stacks objects. Along rows (`axis=0`) it aligns columns by name; along columns (`axis=1`) it aligns rows by index.

```python
batches = [pd.read_parquet(p) for p in paths]
df = pd.concat(batches, ignore_index=True)          # fresh 0..n-1 index
df = pd.concat({"train": tr, "test": te}, names=["split"]).reset_index(level="split")
```

If one batch has an extra column, the result has that column with `NaN` for the other batches; if a column changes dtype between batches, the result may fall back to `object`. Check `df.dtypes` after concatenating files from different days, or pass `join="inner"` to keep only shared columns. Growing a frame with `concat` inside a loop copies all previous rows every iteration; collect the pieces in a list and concatenate once.

## Point-in-time joins with merge_asof

For each training example, a feature must be the value that was known at the example's timestamp, not the latest value. `merge_asof` does exactly this: for each left row it takes the last right row whose time is at or before the left row's time, optionally within the same entity.

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

Both frames must be sorted by the `on` column, and its dtypes must match (both `datetime64` with the same time zone). `allow_exact_matches=False` is the conservative choice when a snapshot stamped at time *t* may already include the event being predicted.

<div class="ex">
<div class="lab">Worked example</div>

**the join that doubled the dataset**

A label table has 1.2 million rows. After `merge(user_profile, on="user_id")` it has 1.9 million, and offline accuracy improves suspiciously. `user_profile["user_id"].value_counts()` shows 210,000 users with two rows: a profile service wrote a new row on each address change instead of updating.

With `validate="many_to_one"` the merge would have raised immediately. The fix keeps one row per user as of each label's time, using `merge_asof(..., by="user_id")` on the profile's `updated_at`, which removes the duplication and also stops later address changes leaking into earlier labels.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

join features to labels with `how="left"` and `validate`, assert the row count, inspect unmatched keys, and use `merge_asof` whenever a feature changes over time.

</div>
