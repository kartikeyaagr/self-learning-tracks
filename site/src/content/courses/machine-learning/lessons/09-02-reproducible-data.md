---
id: "09.02"
course: "machine-learning"
module: "09"
slug: "reproducible-data"
title: "Reproducible data"
keyIdeas: ["A model is reproducible only if you can rebuild its exact training set; that needs a pinned **data version**, deterministic splits and point-in-time-correct features, not just a saved random seed.", "A seed fixes a sequence of random numbers, not which user goes where: reorder or add rows and most test users move into training. **Hashed splits** such as `int(sha256(salt + user_id)) % 100 < 80` for train keep each entity's split fixed forever.", "Split by the entity that must not leak (user, card, patient, session), not by row, or the model memorises entity fingerprints and the test score overstates production.", "A **point-in-time join** pairs each label at time $t$ with the latest feature value available at or before $t$ (feature time plus availability delay); joining today's table leaks the future and can even train a worse model.", "Version data with immutable snapshots, save a lineage record (data versions, salt, commit, config) next to every model, and check a rebuild by its dataset fingerprint and by landing within the metric's seed noise."]
question: "A fraud team splits 18 months of transactions randomly by row (seed 42) and joins the current customer table to get account age and dispute count. Their test AUC is 0.97; in production it is 0.84. Identify every leak in this setup and the corrected design."
answer: "There are three leaks, plus a reproducibility gap. First, the row-level split puts transactions from the same card or customer in both train and test, so the model memorises entity-specific patterns (device, merchant mix, home location) together with their labels; split by a salted hash of customer (or card) ID, e.g. `int(sha256('fraud-v1:' + customer_id)[:15], 16) % 100`, so no customer is on both sides. Second, the random split ignores time: the model trains on transactions after the test ones, learning fraud patterns from the future; hold out the latest months as test, optionally hashing customers within it so the test set is both later and made of unseen customers. Third, the join against the current customer table leaks post-transaction information: dispute count includes disputes caused by the very fraud being predicted, and account age is computed as of today rather than at the transaction. Use a point-in-time join (pandas `merge_asof` with `direction='backward'`, or a ROW_NUMBER window in SQL) so each transaction gets values as of its timestamp minus the delay with which those fields reach production; keep ingestion time as well as event time for late-arriving disputes. Finally, the seed-42 split is not reproducible: as soon as rows are added or come back in a different order, most former test rows move into training. Read from pinned snapshots of the transaction and customer tables, record their version IDs, the split salt and the code commit in a lineage file next to the model, and verify that a rebuild gives the same dataset fingerprint. After these fixes the offline AUC should drop towards the production level; that drop is the leak being removed, not the model getting worse."
sources: [{"title": "Feast: Point-in-time joins", "url": "https://docs.feast.dev/getting-started/concepts/point-in-time-joins", "note": "How a feature store builds leak-free training sets from timestamped feature tables."}, {"title": "Google ML Crash Course: Dividing the original dataset", "url": "https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets", "note": "Train/validation/test splits and why test examples must stay unseen."}, {"title": "Delta Lake: table batch reads and time travel", "url": "https://docs.delta.io/latest/delta-batch.html", "note": "The 'Query an older snapshot of a table' section: versions and timestamps as of."}, {"title": "DVC documentation", "url": "https://dvc.org/doc", "note": "Versioning datasets and models alongside code with content-addressed storage."}]
practice: ["hashed-dataset-split", "point-in-time-feature-join", "row-number", "drop-duplicates"]
tags: ["data versioning", "leakage"]
tools: ["pandas", "SQL"]
---

## What you will learn

- Why "I saved the random seed" does not make a model reproducible, and what has to be pinned instead.
- How to assign train, validation and test sets with a salted hash of an ID, so membership never changes as data grows.
- Which ID to hash: the entity that must not leak (customer, card, patient), not the row.
- How to join features **as of** each label's time, including the delay before a feature is really available.
- How to version data, record a lineage file next to every model, and check that a rebuild matches.

*Before you start:* know why a test set must stay unseen (01.03 Train, validation and test splits) and what leakage looks like in cross-validation (06.04). Lesson 09.01 turned a model into a running service; this one is about rebuilding that model, and its data, months later.

## Why rerunning the notebook is not enough

Three weeks after launch, someone asks why the model approved a particular customer, or you want to retrain last month's model to compare. You press "Run all" and the numbers come out different. Was it the code, the data, or luck?

If the table was updated in place, the split depended on row order, and features were joined against today's values, you cannot rebuild the dataset, let alone the model. **Reproducibility** here means: given the same inputs you can rebuild the exact training set and get a model whose metrics agree with the original within normal random noise. In ML that is mostly a data problem, and it has three parts.

~~~dg-layers What has to be pinned to rebuild a training set exactly.
Data version | immutable snapshot ID of every source table or file
Split assignment | deterministic function of entity ID, never of row order or a random draw
*Feature values | as of each label's timestamp, not as of today
Code and config | commit hash, feature definitions, hyperparameters, seed
~~~

## The idea in one picture

Take a table of 10,000 users. You split it the usual way, `train_test_split(ids, test_size=0.2, random_state=42)`, and get 2,000 test users. The seed is fixed, so surely the split is fixed too? Now rerun the same call in five situations and count how many of the original 2,000 test users end up in the *training* set:

| Situation | Seeded random split | Hashed split on user ID
| --- | --- | --- |
| Rerun, same rows, same order | 0% | 0%
| Same rows, returned in a different order | 80% | 0%
| One new user appended | 11% | 0%
| One more day of data (+1% rows) | 49% | 0%
| One more month (+30% rows) | 63% | 0%

~~~dg-fig 09.02-reshuffle.svg
Share of the original test users that land in training when you rerun a split. A seeded random split is only stable if the input is byte-for-byte identical; a hashed split never moves anyone.
~~~

In plain words: a seed fixes a *sequence of random numbers*, not *which user goes where*. Those numbers are handed to rows by position, so once the rows change (one extra row, or a database returning rows in another order, which SQL never guarantees without `ORDER BY`) they land on different users. A full reshuffle sends about 80% of old test users into training, so the "unseen" test set is mostly seen and any comparison with the old score is contaminated.

The fix is to make the split a property of the user rather than of the row's position. That is the **hashed split**, and it is the first step below.

## Step 1: split by a salted hash of a stable ID

A **hash function** turns any string into a fixed-length number that looks random but is completely determined by the input. SHA-256 is a standard one that gives the same answer on every machine, in every language, forever. Take the hash of the ID, reduce it to a number from 0 to 99 (a **bucket**), and decide the split from the bucket:

$$\text{bucket}(u) = \text{int}\big(\text{SHA256}(\text{salt} + u)[\text{first 15 hex digits}]\big) \bmod 100.$$

Here $u$ is the entity ID as a string, the **salt** is a fixed string you choose per project (for example `churn-v1`), and $\bmod\ 100$ keeps the remainder after dividing by 100. Fifteen hex digits (60 bits) spread evenly over 100 buckets. Then: buckets 0 to 9 are test, 10 to 19 validation, 20 to 99 train. The ten-line Python version is in the code section below.

<div class="ex">
<div class="lab">Worked example</div>

**assign four users by hand**

Salt `churn-v1`, so we hash the strings `churn-v1:u1001`, `churn-v1:u1002`, and so on.

User u1001: the first 15 hex digits of the SHA-256 are `ec8bd044a9291c1`. As a decimal integer that is 1,065,309,138,001,760,705. Dividing by 100 leaves remainder 05 (for $\bmod\ 100$ just read the last two digits). Bucket 5 is below 10, so u1001 is in **test**.

User u1002: digits `61cd5da708f741d`, integer 440,461,997,743,502,365, bucket 65, so **train**.

User u1003: digits `31d25b7f82c2827`, integer 224,377,028,511,279,143, bucket 43, so **train**.

User u1004: digits `065b91f6c7269f9`, integer 28,632,517,262,338,553, bucket 53, so **train**.

Nothing here depends on other users, row order or table size: add a million users tomorrow and u1001 is still bucket 5, still test.

</div>

Properties worth knowing:

- **Stable**: a user's split never changes, however much data arrives, so test users stay unseen forever.
- **Salted**: change the salt to draw a fresh, independent split for a new project; reuse it to compare models on the same users.
- **Portable**: SHA-256 is available in Python, Spark SQL (`sha2(x, 256)`), BigQuery (`SHA256`) and most other systems, so the warehouse and the training code can compute identical buckets, as long as everyone uses the same hash, salt format and digit-to-bucket rule.
- **Do not use Python's built-in `hash()`** for strings: it is randomised per process (`PYTHONHASHSEED`), so assignments differ between runs.

## Step 2: hash the entity that must not leak

The hashed split is only as good as the ID you feed it. Hash the ID of the thing whose information must not cross from train to test. For a model predicting per-transaction fraud, hashing transaction IDs puts one card's transactions on both sides, and the model learns the card rather than fraud. Hash card or customer IDs. For a medical model, hash the patient, not the scan.

Rows of one customer share near-identical values (device, home location, typical basket). A flexible model memorises those **fingerprints** with the customer's label; that pays off when the customer reappears in test, not in production where customers are new.

~~~dg-fig 09.02-entity.svg
A random forest on simulated churn data. With one row per customer the two splits agree; as customers get more rows, the row-level test score climbs while the honest customer-level score stays flat.
~~~

In this simulation (6,000 rows, a weak genuine signal, a hidden per-customer tendency) both splits give about 0.60 AUC at one row per customer. At 20 rows per customer the row split reports 0.83 while the customer split still says 0.59. The gap is pure memorisation.

For time-dependent problems, combine the two ideas: hold out the latest period as test (a **temporal split**, as in 01.03) and hash entities within it if you also need user-disjoint sets. Then the test set is both "later" and "people the model never saw".

## Step 3: join features as they were at prediction time

Most feature tables are keyed by entity and updated over time: `customer_id, feature_ts, n_tickets_30d`. A training example is a label event: `customer_id, label_ts, churned`. The naive join on `customer_id` alone picks the current row, which contains information from after the label. That leak is invisible offline (it makes the model look great) and is only discovered when online performance falls short.

The correct join, a **point-in-time join**, takes for each label row the most recent feature row that was *available* at the label time:

$$\text{feature}(u, t) = \text{value of the latest row with } \text{feature\_ts} + d \le t.$$

Here $u$ is the entity, $t$ the label (prediction) time, $\text{feature\_ts}$ the time the value describes, and $d$ the **availability delay**: how long the feature takes to reach production. A nightly batch job that computes values at midnight and lands them at 06:00 has $d = 6$ hours; a value computed at midnight on 1 March cannot be used by a prediction at 03:00 that night.

~~~dg-fig 09.02-timeline.svg
One customer's ticket count over time. At the prediction (1 March, 03:00) the honest value is 1. Ignoring the 6-hour delay gives 2; joining today's table gives 3, a number that counts tickets filed while cancelling.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a point-in-time join by hand**

Feature rows for customer c7 (nightly job, lands 6 hours after `feature_ts`): 28 Feb 00:00, value 1, available 28 Feb 06:00; 1 Mar 00:00, value 2, available 1 Mar 06:00; 20 Mar 00:00, value 6, available 20 Mar 06:00.

Label A: c7 at 1 Mar 03:00. Candidate rows with available time $\le$ 03:00 on 1 Mar: only the 28 Feb row (the 1 Mar row lands at 06:00, three hours too late). Feature = **1**.

Label B: c7 at 1 Mar 10:00. Now both the 28 Feb row (06:00 the day before) and the 1 Mar row (06:00 today) are available. Take the latest: feature = **2**.

Naive join on the current table: both labels get the 20 Mar row, value **6**, which counts tickets filed during the cancellation, 19 days after the prediction. That is the future leaking backwards.

</div>

In pandas this is one call to `merge_asof` (shown in full in the code section below), which for each label row finds the last feature row whose availability time is less than or equal to the label time. In SQL it is a window function: join on entity with `available_ts <= label_ts`, then keep `ROW_NUMBER() OVER (PARTITION BY label_id ORDER BY feature_ts DESC) = 1`. Feature stores (Feast and others) implement this as a built-in "point-in-time join" so each team does not reinvent it.

How much does the leak cost? A simulated churn model, trained once with each join:

~~~dg-fig 09.02-join-auc.svg
The naive join looks better offline (0.90 against 0.75) but is worse in production (0.72), because at serving time the leaked tickets do not exist yet. The point-in-time model scores the same offline and online.
~~~

Here the label is "cancelled within 30 days of 1 March" and the current table's `support_tickets_30d` includes the angry tickets filed while cancelling. In production the naive model only ever sees honest, as-of-now counts, much smaller than the ones it learned from. The leak did not just inflate the estimate; it trained a worse model.

Late data needs a decision too: a value stamped 09:00 that landed at 14:00 was not available for a 10:00 prediction. Keep both an **event time** (when it happened) and an **ingestion time** (when it reached the table) and join on the one that matches serving.

## Step 4: version the data and record the lineage

A deterministic split and a correct join still give different results if the source tables change, and they change silently: backfills, deduplication jobs, late labels, privacy deletions.

- **Immutable snapshots**: table formats such as Delta Lake and Iceberg support time travel (`VERSION AS OF 412`); tools for files such as DVC and lakeFS store content-addressed versions so a version ID always means the same bytes. Never overwrite a training extract in place.
- **Lineage record** next to every model: snapshot IDs, split salt, feature definitions version, delays, code commit, hyperparameters, seed, library versions. A model registry stores it; a JSON file beside the weights is better than nothing.
- **Determinism**: GPU kernels and parallel loaders can still make training non-bitwise-reproducible. Aim for *statistical* reproducibility: the same dataset bit for bit, and a model within noise of the original.

~~~dg-flow Every model artifact carries the IDs needed to rebuild it.
Snapshot IDs -(read as of version)-> Training set -(hashed split + point-in-time join)-> *Model -(save next to weights)-> Lineage record
~~~

"Within noise" can be measured. Train the same configuration with 20 different seeds and look at the spread of the metric: that is your **seed noise**. A rebuild that lands inside that spread reproduces the original; one that lands far outside it means something other than luck changed.

~~~dg-fig 09.02-silent.svg
Test AUC for 20 seeds per pipeline. A rebuild from the pinned snapshot is indistinguishable from the original. A rebuild from the live table, where a backfill silently rewrote one source's amounts into cents, sits about 0.017 lower, roughly eight times the seed-to-seed standard deviation.
~~~

Seed noise here is about 0.002 AUC around 0.830; the snapshot rebuild also averages 0.830, the live-table rebuild 0.813 with no code change. Without the snapshot you would hunt a code bug that does not exist; with it, diffing the two versions points straight at the backfill.

## Reproducible data in code

Putting the pieces together on the tiny example above: point-in-time join, hashed split, a fingerprint of the exact rows, and a lineage file.

```python
import hashlib, json
import pandas as pd

def bucket(entity_id, salt="churn-v1", buckets=100):
    h = hashlib.sha256(f"{salt}:{entity_id}".encode()).hexdigest()
    return int(h[:15], 16) % buckets

def split_of(entity_id):
    b = bucket(entity_id)
    return "test" if b < 10 else "val" if b < 20 else "train"

labels = pd.DataFrame({"customer_id": ["c7", "c7", "c9"],
    "label_ts": pd.to_datetime(["2026-03-01 03:00", "2026-03-01 10:00", "2026-03-01 10:00"]),
    "churned": [1, 1, 0]})
feats = pd.DataFrame({"customer_id": ["c7", "c7", "c7", "c9", "c9"],
    "feature_ts": pd.to_datetime(["2026-02-28", "2026-03-01", "2026-03-20", "2026-02-28", "2026-03-01"]),
    "n_tickets_30d": [1, 2, 6, 0, 0]})

feats["available_ts"] = feats["feature_ts"] + pd.Timedelta(hours=6)   # lands at 06:00
train = pd.merge_asof(labels.sort_values("label_ts"), feats.sort_values("available_ts"),
                      left_on="label_ts", right_on="available_ts",
                      by="customer_id", direction="backward")
train["split"] = train["customer_id"].map(split_of)
print(train[["customer_id", "label_ts", "n_tickets_30d", "split"]])
#   customer_id            label_ts  n_tickets_30d  split
# 0          c7 2026-03-01 03:00:00              1  train
# 1          c7 2026-03-01 10:00:00              2  train
# 2          c9 2026-03-01 10:00:00              0    val

# Fingerprint the exact rows, so a rebuild can prove it is identical
fp = hashlib.sha256(pd.util.hash_pandas_object(train, index=False).values.tobytes()).hexdigest()[:12]
print(fp)   # 03f4eb910c9c

lineage = {"data": {"labels": "churn_labels@v412", "features": "cust_feats@v977"},
           "split": {"method": "sha256", "salt": "churn-v1", "test": "<10", "val": "10-19"},
           "feature_delay_hours": 6, "dataset_sha256": fp,
           "code_commit": "3f9c2ab", "seed": 0}
json.dump(lineage, open("lineage.json", "w"), indent=2)
```

Note that both rows of customer c7 got the same split (bucket 87, train): the split follows the customer, not the row. In a real pipeline both DataFrames come from pinned snapshots.

## Checking that it worked

Reproducibility is a claim you can test, so test it.

- **Rebuild and compare the fingerprint.** Rebuild the training set from the lineage record and hash it; a different hash means a source or the code changed. Run this in CI for the production model.
- **Split stability.** Store the test IDs; after every refresh, assert none is now in train.
- **Entity disjointness.** Assert that the sets of customer IDs in train and test do not intersect.
- **No future features.** Assert that every joined row has `available_ts <= label_ts`. A single violation means a join bug.
- **Seed noise band.** Keep the mean and standard deviation of the metric over a handful of seeds in the lineage file. A rebuilt model should land within about two standard deviations.
- **Offline against online.** If offline is much better than early production, suspect leakage before drift.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Random split with a fixed seed on a growing table | Test score jumps after a refresh; old test users appear in training | Hashed split on a stable ID, salted per project
| Splitting by row when entities repeat | Test score far above production; worse the more rows per entity | Hash the customer, card, patient or session ID
| Python `hash()` for bucketing | Different splits on each run or worker | `hashlib.sha256` or the warehouse's stable hash
| Joining today's feature table | Offline AUC suspiciously high; a "too good" feature | Point-in-time join on availability time
| Ignoring availability delay or late data | Small but steady offline-online gap | Join on ingestion time plus the job's delay
| Training extract overwritten in place | Rebuild gives different numbers with unchanged code | Read from immutable snapshots; record the IDs
| No lineage saved with the model | Nobody can say which data a model saw | Write a lineage JSON next to every artifact

## Going deeper

**Why hashing gives a fair split.** A cryptographic hash behaves like a fixed random function: its outputs are spread evenly and unrelated to any pattern in the inputs, so consecutive IDs (`u1001`, `u1002`) land in unrelated buckets. Reducing a 60-bit number modulo 100 introduces a bias of order $100/2^{60}$, far too small to matter. The split is therefore a random sample of entities, chosen once, with the randomness frozen into the function. Because each entity is assigned independently, the test fraction follows a binomial distribution: with $n$ entities and test probability $p$, the standard deviation of the test share is $\sqrt{p(1-p)/n}$, about 0.95 percentage points for $n = 1{,}000$ and $p = 0.1$. Check the counts and class balance per split; you can also hash within strata. Re-salting gives an independent split but breaks comparability with models scored on the old one.

**Point-in-time correctness and time travel are different things.** Time travel (`VERSION AS OF`) gives you the table as it was at some moment, which is great for reproducibility but on its own does not prevent leakage: a snapshot from April still contains values from after a March label. Point-in-time joins use the *row-level* timestamps inside the table. You need both: the snapshot pins the bytes, the as-of join picks the right row for each label.

**Retention and deletion.** Cleanup jobs physically delete old files (Delta's `VACUUM` keeps 7 days by default; Iceberg expires snapshots), so a recorded version ID can stop working: retain training versions or copy the extract. Privacy deletions make exact rebuilds impossible by design; record which were applied, so a rebuild is "the original minus documented deletions".

**Bitwise determinism.** Floating-point addition is not associative, so parallel reductions on a GPU can sum in a different order and give slightly different weights. Frameworks offer deterministic modes (for example `torch.use_deterministic_algorithms(True)` plus fixed seeds for every library and data-loader worker), at some speed cost. For most teams the seed noise band above is the right target; bitwise reproducibility matters mainly for debugging and audits.

## Key terms

| Term | Meaning
| --- | --- |
| Reproducibility | Being able to rebuild the exact training set and a model within seed noise of the original
| Hashed split | Assigning each entity to train, validation or test by a stable hash of its ID
| Salt | A fixed string mixed into the hash so each project gets its own independent split
| Entity-level split | Splitting by the thing that must not leak (customer, card, patient), so no entity is on both sides
| Point-in-time join | Pairing each label with the latest feature value available at or before its time
| Availability delay | The time between when a feature value describes and when production can actually read it
| Immutable snapshot | A data version whose ID always refers to the same bytes (time travel, content addressing)
| Lineage record | The data versions, split salt, code commit and config saved next to a model
| Seed noise | The spread of a metric across training runs that differ only in random seed

<div class="take">
<div class="lab">Key takeaway</div>

reproducibility comes from three habits: assign splits by a salted hash of the leaking entity, join features as of each label's time plus availability delay, and record immutable data versions with every model, then prove a rebuild by matching the dataset fingerprint and landing within seed noise.

</div>
