---
id: "09.02"
course: "machine-learning"
module: "09"
slug: "reproducible-data"
title: "Reproducible data"
keyIdeas: ["A model is reproducible only if you can rebuild its exact training set; that needs a pinned **data version**, deterministic splits and point-in-time-correct features, not just a saved random seed.", "**Hashed splits** assign each entity by a stable hash, e.g. `int(sha256(salt + user_id)) % 100 < 80` for train, so membership never changes as data grows or rows are reordered.", "Split by the entity that must not leak (user, patient, session), not by row, or near-duplicate rows of one entity end up on both sides.", "A **point-in-time join** pairs each label at time $t$ with the latest feature value known at or before $t$ (minus any availability delay); a plain join on the current table leaks the future.", "Version data by immutable snapshots (table time travel, content-addressed files) and record the triple (code commit, data version, config) with every model artifact."]
question: "A fraud team splits 18 months of transactions randomly by row (seed 42) and joins the current customer table to get account age and dispute count. Their test AUC is 0.97; in production it is 0.84. Identify every leak in this setup and the corrected design."
answer: "There are three leaks. First, the row-level split puts transactions from the same card or customer in both train and test, so the model memorises entity-specific patterns; split by a salted hash of customer (or card) ID. Second, the random split ignores time: the model trains on transactions after the test ones, learning fraud patterns from the future; hold out the latest months as test, optionally hashing customers within it. Third, the join against the current customer table leaks post-transaction information: dispute count includes disputes caused by the very fraud being predicted, and account age is computed as of today. Use a point-in-time join so each transaction gets values as of its timestamp minus the delay with which those fields reach production. Also pin a data snapshot so the rebuilt dataset is reproducible, since a random seed alone stops working once rows are added."
sources: [{"title": "Feast: Point-in-time joins", "url": "https://docs.feast.dev/getting-started/concepts/point-in-time-joins", "note": "How a feature store builds leak-free training sets from timestamped feature tables."}, {"title": "Google ML Crash Course: Dividing the original dataset", "url": "https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets", "note": "Train/validation/test splits and why test examples must stay unseen."}, {"title": "Delta Lake: table batch reads and time travel", "url": "https://docs.delta.io/latest/delta-batch.html", "note": "The 'Query an older snapshot of a table' section: versions and timestamps as of."}, {"title": "DVC documentation", "url": "https://dvc.org/doc", "note": "Versioning datasets and models alongside code with content-addressed storage."}]
practice: ["hashed-dataset-split", "point-in-time-feature-join", "row-number", "drop-duplicates"]
tags: ["data versioning", "leakage"]
tools: ["pandas", "SQL"]
---

## Why rerunning the notebook is not enough

Three weeks after launch, someone asks why the model approved a particular customer, or a regression appears and you want to retrain last month's model to compare. If the training table has been updated in place since then, the split depends on row order, and features were joined against current values, you cannot rebuild the dataset, let alone the model. Reproducibility in ML is mostly a data problem, and it has three parts.

~~~dg-layers What has to be pinned to rebuild a training set exactly.
Data version | immutable snapshot ID of every source table or file
Split assignment | deterministic function of entity ID, never of row order or a random draw
*Feature values | as of each label's timestamp, not as of today
Code and config | commit hash, feature definitions, hyperparameters, seed
~~~

## Hashed splits

A random split with a fixed seed is reproducible only while the input rows and their order stay identical. Add one day of data and every row's assignment changes, so some former test users are now in training and your new test score is contaminated. The robust alternative is to derive the split from a hash of a stable ID:

```python
import hashlib
def bucket(entity_id: str, salt: str = "churn-v1", buckets: int = 100) -> int:
    h = hashlib.sha256(f"{salt}:{entity_id}".encode()).hexdigest()
    return int(h[:15], 16) % buckets
split = "test" if bucket(uid) < 10 else "val" if bucket(uid) < 20 else "train"
```

Properties worth knowing:

- **Stable**: a user's split never changes, however much data arrives, so test users stay unseen forever.
- **Entity-level**: hash the ID that must not leak. For a model predicting per-transaction fraud, hashing transaction IDs puts one card's transactions on both sides, and the model learns the card. Hash card or customer IDs.
- **Salted**: change the salt to draw a fresh, independent split for a new project; reuse it to compare models on the same users.
- **Do not use Python's built-in `hash()`** for strings: it is randomised per process (PYTHONHASHSEED), so assignments differ between runs.

For time-dependent problems, combine: hold out the latest period (temporal split, as in ML lesson "Train, validation and test splits") and hash within it if you also need user-disjoint sets.

## Point-in-time feature joins

Most feature tables are keyed by entity and updated over time: `customer_id, ts, n_orders_30d`. A training example is a label event: `customer_id, label_ts, churned`. The naive join on `customer_id` alone picks the current row, which contains information from after the label. That leak is invisible offline (it makes the model look great) and is only discovered when online performance falls short.

The correct join, for each label row, takes the most recent feature row with `feature_ts <= label_ts - delay`, where `delay` is how long the feature takes to become available in production (a nightly batch job that lands at 06:00 makes yesterday's value unavailable until then).

```python
import pandas as pd
labels = labels.sort_values("label_ts")
feats = feats.sort_values("feature_ts")
feats["available_ts"] = feats["feature_ts"] + pd.Timedelta(hours=6)
train = pd.merge_asof(labels, feats, left_on="label_ts", right_on="available_ts",
                      by="customer_id", direction="backward")
```

In SQL the same thing is a window function: join on entity with `feature_ts <= label_ts`, then keep `ROW_NUMBER() OVER (PARTITION BY label_id ORDER BY feature_ts DESC) = 1`. Feature stores (Feast and others) implement this as a built-in "point-in-time join" so each team does not reinvent it.

<div class="ex">
<div class="lab">Worked example</div>

**the leak a plain join creates**

A churn label is "cancelled within 30 days of 1 March". The feature `support_tickets_30d` in the current table counts tickets up to today, including the angry tickets customers filed while cancelling. Joined naively, the feature is strongly predictive (offline AUC 0.93). Joined point-in-time as of 1 March, AUC is 0.78, and 0.78 is the number production will see.

</div>

Late-arriving data needs a decision too: if a feature value with timestamp 09:00 actually landed in the table at 14:00, was it available for a 10:00 prediction? Keep both an event time and an ingestion time and join on the one that matches serving.

## Versioning data and models

- **Immutable snapshots**: table formats such as Delta Lake and Iceberg support time travel (`VERSION AS OF 412`); object stores with content-addressed files (DVC, lakeFS) give the same for files. Never overwrite a training extract in place.
- **Lineage record** saved next to every model artifact: data snapshot IDs, split salt, feature definitions version, code commit, hyperparameters, seed, library versions. A model registry stores it; a small JSON file next to the weights is better than nothing.
- **Determinism**: even with all of that, GPU kernels and parallel data loaders can make training non-bitwise-reproducible. Aim for *statistical* reproducibility: rebuilding gives the same dataset bit for bit and a model within noise of the original.

<div class="take">
<div class="lab">Key takeaway</div>

reproducibility comes from three habits: assign splits by a salted hash of the leaking entity, join features as of each label's time plus availability delay, and record immutable data versions with every model.

</div>
