---
id: "09.01"
course: "machine-learning"
module: "09"
slug: "from-notebook-to-service"
title: "From notebook to service"
keyIdeas: ["A model in production is a small part of a larger system: data collection, feature computation, validation, serving, logging and monitoring are most of the code, and most failures there are **silent** (the model keeps returning plausible numbers).", "**Batch inference** precomputes scores on a schedule and serves them by lookup; **online inference** computes per request. Decide by how fresh the inputs must be, how many entities need scores versus how many are requested, and the latency budget.", "Online latency is a distribution: report p50, p95 and p99 per step. The feature fetch, not the model, usually dominates the tail; batching shares a fixed per-call overhead and can raise throughput by thousands of times.", "**Training/serving skew** is any difference between training inputs and serving inputs: two code paths for one feature, different freshness, or a changed population (drift).", "The strongest defence is to **log features at serving time** and train on those logs, with one feature definition shared by training and serving and the whole pipeline shipped as one versioned artifact.", "Validate inputs at the boundary against a schema (types, ranges, categories, units, null rates): a model fed garbage returns confident garbage without raising an error."]
question: "An e-commerce ranking model was trained on daily snapshots of item features (price, stock, rating). In production it reads the same features from a live cache. Offline AUC is 0.81, but online click-through is below the old model's. Name two specific skew mechanisms that could explain this and how you would confirm each, then say what you would change so the problem cannot recur."
answer: "First, freshness skew with leakage: in training, features came from end-of-day snapshots, so a click that happened at 10:00 is paired with feature values that may reflect events after the click, such as a price drop later that day or a rating updated by that very purchase. The model learned relationships that live features, read before the click, do not have, which inflates offline AUC. Confirm it by rebuilding the training set with point-in-time feature values (the value as of each impression's timestamp) and re-evaluating: if offline AUC drops towards what online performance implies, this was the cause. Second, code or default skew: the live cache may return nulls or defaults on a miss (for example stock = 0 or rating = 0), or encode a field differently (price in pence versus pounds, rating on a 0-1 versus 1-5 scale), at rates never seen in the warehouse snapshot. Confirm it by logging the served features for a day, joining them to offline values for the same request and item, and reporting mismatch rate and null rate per feature; also run a parity test that scores the same rows through the notebook and the service. To prevent recurrence: log the exact served features with each impression and train the next model on those logs; keep one feature definition (a feature store with offline and online materialisations); add a schema check at the service boundary for ranges, units and null rates; ship preprocessing and model as one versioned artifact; and monitor per-feature null and mismatch rates and the hourly click-through rate so a step change raises an alert."
sources: [{"title": "Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)", "url": "https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "The catalogue of ML-specific debt: entanglement, hidden feedback loops, pipeline jungles."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Rules 29-37 on training/serving skew and logging features at serving time."}, {"title": "TensorFlow Data Validation guide", "url": "https://www.tensorflow.org/tfx/guide/tfdv", "note": "Schema inference, anomaly detection and skew/drift comparators between training and serving data."}]
practice: ["training-serving-skew", "record-schema-validation", "point-in-time-feature-join"]
tags: ["serving", "training/serving skew"]
tools: ["pandas"]
---

## What you will learn

- Why a trained model is only a small part of a working ML system, and where production failures actually come from.
- How to choose between **batch** and **online** inference using three questions: how fresh, how many entities, how fast.
- How to read a latency distribution (p50, p95, p99) and why batching changes throughput by orders of magnitude.
- What **training/serving skew** is, its three common forms, and the structural fixes: one feature definition and logging served features.
- How to put a schema check at the model's boundary so that bad inputs raise an alarm instead of producing confident nonsense.

*Before you start:* you should be able to fit and evaluate a scikit-learn model (modules 02 and 06) and know why preprocessing must live inside a pipeline (07.05). Familiarity with train/validation/test splits (01.03) helps. No web-framework or cloud knowledge is needed; the ideas apply whatever stack you use.

## Where this module starts

Every earlier module ended with a notebook, a fitted object and a validation metric. This module is about what happens next, when the model must keep making good predictions for months while the world it learned from moves.

Four habits make that work, one per lesson: serve the model the same inputs it was trained on (this lesson), build training data reproducibly with no peeking into the future (09.02), watch inputs and outputs in production (09.03), and change the model only through controlled experiments (09.04). The Inference Engineering course covers containers, health checks and rollouts for large-model servers; here we focus on the data and statistics side, which applies to every model.

## The idea in one picture

A bank deploys a credit model that uses `income_monthly`, in pounds per month. In training, the middle 99.8% of incomes lie between £809 and £8,432. At 14:00 one day, the team that owns the application form switches to a new upstream API. Nobody notices that the new API reports income **per year**.

| Hour | 11:00 | 12:00 | 13:00 | 14:00 | 15:00 | 16:00
| --- | --- | --- | --- | --- | --- | --- |
| Median income seen by the model (£) | 2,710 | 2,690 | 2,740 | 32,600 | 32,300 | 32,800
| Approval rate | 41% | 42% | 41% | 76% | 78% | 77%

~~~dg-fig 09.01-incident.svg
Left: the approval rate leaps from about 41% to 78% when the new API goes live. Right: the input the model receives jumps twelvefold at the same hour, far above anything seen in training. No error was raised at any point.
~~~

The business sees approvals almost double. The model sees applicants who all look twelve times richer, and does exactly what it learned: approve rich people. Nothing raised an exception. The model was not wrong; it was fed the wrong thing.

That is the central lesson of production ML: most failures are **silent**. Two cheap checks would have caught this within minutes: a range check on each input (a sudden majority above £8,432 is an alarm), and a dashboard of approval rate per hour.

A few words for this lesson:

- **Serving** means using a trained model to answer real requests. The code that does it is the **service**.
- **Inference** is one call of the model on new inputs: features in, prediction out.
- **Training/serving skew** is any difference between what the model saw in training and what it sees when serving. The income bug is an example.

## Step 1: the model is the small box

A production model sits inside a system that must also collect raw data, compute and check features, load the right model version, answer in time, record what happened and raise alarms. Sculley and colleagues at Google drew this as a famous diagram: the ML code is a small box surrounded by much larger boxes for configuration, data collection, feature extraction, verification, serving infrastructure and monitoring.

~~~dg-flow The path of one prediction. The model is one step; every other step can silently change what the model receives.
Raw event -(collect)-> Feature computation -(check against schema)-> Validated features -(forward pass)-> *Model -(score)-> Response -(log features + score + version)-> Training log
~~~

When a production model "gets worse", look at the arrows before the model. In the incident above, the model artifact was identical before and after 14:00; only the first arrow changed.

## Step 2: choose batch or online inference

There are two basic ways to deliver predictions.

**Batch inference** (also called offline inference) runs the model on a schedule, say every night, for every entity you care about, and stores the scores in a table. When someone needs a prediction, the application just looks it up by key. **Online inference** (real-time inference) runs the model when a request arrives, using the request's own context.

| Dimension | Batch (offline) inference | Online (real-time) inference
| --- | --- | --- |
| When it runs | on a schedule, e.g. nightly | per request
| Serving path | key-value lookup of a stored score | feature fetch, then model forward pass
| Latency budget | none for the model; lookup is ~1 ms | typically 10-100 ms end to end
| Freshness | as old as the last run | uses the current request context
| Coverage | only entities known at run time | any entity, including brand-new ones
| Cost | proportional to all entities, every run | proportional to traffic
| Failure mode | stale or missing scores | latency spikes, feature-store outages

Three questions decide it:

1. **How fresh must the inputs be?** If decisive information exists only at request time (the transaction, the query), you need online.
2. **How many entities are scored versus requested?** Batch pays for everyone every run; online pays only for those that show up.
3. **How fast must the answer be?** A lookup is nearly free; online must fit feature fetch and model inside a budget.

~~~dg-fig 09.01-batch-online.svg
Left: a nightly batch score ages from 0 to 24 hours in a sawtooth; an online score is always current. Right: batch cost is flat in traffic while online cost grows with requests. With these illustrative numbers they cross at 250,000 requests per day.
~~~

The left panel shows **staleness**: a batch score served at 02:59 is almost a day old, fine for a weekly churn campaign, useless for fraud. The right panel shows **cost**: scoring 5 million customers nightly costs the same however many visit, while each online prediction pays for its own feature fetch and round trip, so online wins only when few entities are requested.

Batch suits churn scores, weekly demand forecasts, or "recommended for you" rails that may be a day old. Online is needed for fraud scoring a transaction or ranking results for a query. Many systems are **hybrid**: expensive user and item embeddings are computed in batch, and a cheap online step combines them with the request context. **Near-real-time features**, such as "orders in the last 5 minutes" from a stream processor, sit between the two.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a churn score both ways**

A telecom has $N = 5{,}000{,}000$ customers. The retention app shows a churn score when an agent opens a customer, about $300{,}000$ times per day.

Batch: suppose the job scores 200,000 rows per second including reading features. Time per night $= 5{,}000{,}000 / 200{,}000 = 25$ seconds. Storage: each row is an 8-byte id plus a 4-byte score, so $5{,}000{,}000 \times 12 = 60{,}000{,}000$ bytes, about 60 MB. Serving is a key lookup of about 1 ms. Staleness: up to 24 hours.

Online: average traffic is $300{,}000 / 86{,}400 \approx 3.5$ requests per second. If the busiest hour runs at 5 times the average, the peak is about $17$ per second. Each request needs a feature fetch (median 15 ms) plus the model (well under 1 ms).

Decision: the score depends on billing and usage history that changes daily, and 25 seconds a night is trivial, so batch wins: simpler, cheaper, no latency risk. Online would pay off only if the score had to react to something said five minutes ago.

</div>

## Step 3: measure latency as a distribution

For online inference, every request takes a slightly different time, so latency is a distribution, described with **percentiles**: p50 (the median) is the time half of requests beat; p99 is the time 99% beat. A **latency budget** or SLO (service level objective) is usually set on a high percentile, such as "p99 under 100 ms".

The end-to-end latency is a sum of steps:

$$t_{\text{total}} = t_{\text{network}} + t_{\text{features}} + t_{\text{preprocess}} + t_{\text{model}} + t_{\text{log}}.$$

Each $t$ is a random time for one step of one request. Medians roughly add; tails do not, because the slow tail of one step lands on top of ordinary times from the others.

~~~dg-fig 09.01-latency.svg
Left: 200,000 simulated requests have p50 34 ms, p95 56 ms and p99 73 ms, inside a 100 ms budget. Right: the model is a small slice of the median, and the feature fetch dominates the tail.
~~~

Here the medians are 8 ms network, 15 ms feature fetch, 2 ms preprocessing, 6 ms model and 2 ms logging: $8 + 15 + 2 + 6 + 2 = 33$ ms, close to the simulated p50 of 34 ms. The p99 is 73 ms, more than twice the median, mostly from the feature fetch. This is typical: speeding up the model is often the least effective fix; caching, precomputing or parallelising feature fetches buys more.

## Step 4: batch requests to buy throughput

Every model call pays a fixed **overhead** (Python function calls, input conversion, memory allocation, a GPU kernel launch) plus a small cost per row. **Throughput** is rows scored per second. If the overhead dominates, scoring many rows per call is almost free.

~~~dg-fig 09.01-batching.svg
Measured on one CPU with a scikit-learn pipeline: one row takes about half a millisecond, sixteen thousand rows take about one millisecond. Throughput rises by several thousand times because the fixed per-call overhead is shared.
~~~

As a rough model, the time of one call with batch size $B$ is

$$t(B) \approx t_0 + c\,B, \qquad \text{throughput}(B) = \frac{B}{t_0 + c\,B},$$

where $t_0$ is the fixed overhead per call and $c$ the extra time per row. Reading the figure, $t_0 \approx 0.5$ ms and $c \approx 0.00004$ ms: one row gives $1 / 0.0005 = 2{,}000$ rows per second, while $B = 16{,}384$ takes about 1.1 ms, giving $16{,}384 / 0.0011 \approx 15$ million rows per second. Your machine will differ; the shape will not.

This is why batch jobs score millions of rows in seconds, and why online servers for large models use **dynamic batching**: hold incoming requests for a few milliseconds and score them together, trading a little latency per request for much more throughput.

## Step 5: understand training/serving skew

Skew is the gap between what the model learned from and what it is given. It comes in three forms.

- **Code skew**: one feature, two implementations. Training computes `days_since_signup` in SQL; the service recomputes it in Java from another timestamp field in another timezone. The income bug is code skew too: same name, different units.
- **Data (freshness) skew**: same computation, different moment. Training used end-of-day values while serving reads live ones, so "orders so far" means different things. End-of-day values can even include events *after* the label, which is leakage; point-in-time joins (09.02) fix that.
- **Distribution skew**: inputs are correct but the population changed, because the product changed or the model's own actions did (it blocks fraud, so fraudsters adapt). That is **drift**, the subject of 09.03.

Google's Rules of Machine Learning give the bluntest defence: record the exact feature values the service used for each prediction, and build the next training set from those records, so training and serving share one computation by construction. A **feature store** removes most code skew: one definition per feature, materialised in an offline store for training and an online store for serving. Logging removes the rest, at the cost of waiting for enough logged data. To find skew that already exists, join offline features to served features for the same requests and compare them.

<div class="ex">
<div class="lab">Worked example</div>

**a five-row skew report by hand**

Five requests, feature `days_since_signup`. Offline (warehouse SQL): 120, 45, 300, 7, 88. Served (service code): 120, 46, 300, 7, 88.

Compare row by row: only request 2 differs (45 vs 46). Mismatch rate $= 1/5 = 20\%$.

Feature `orders_7d`. Offline: 2, 0, 5, 1, 3. Served: 2, 0, null, 1, 3. Null rate offline $= 0/5 = 0\%$, null rate served $= 1/5 = 20\%$.

Reading: an off-by-one-day mismatch points to a timezone or day-boundary bug; nulls only at serving point to cache misses, which the model never saw in training.

</div>

## Step 6: guard the boundary with a contract

Treat model inputs like an API **contract**: per feature, the type, allowed range or categories, expected null rate and units. Check every request or batch against it, and decide in advance what a violation does: reject, impute a logged default, or score and alert.

Tools such as TensorFlow Data Validation infer a **schema** from training data and flag anomalies in new data: unseen categories, missing columns, out-of-range values, and skew or drift between datasets. Even a quantile range per numeric column catches many incidents.

~~~dg-fig 09.01-schema.svg
Training incomes (filled) sit inside the schema range learned from training. After the bug, about half the served values (outline) form a second hump twelve times higher, almost entirely outside the range, so a range check flags them immediately.
~~~

Also **version the whole path together**: model weights, feature definitions and preprocessing ship as one artifact. A model trained with one tokenizer or bucket boundary and served with another is skew of the most frustrating kind, because each half looks correct alone.

## From notebook to service in code

The smallest honest version of a service: save preprocessing and model as one object, store the schema and a version string next to it, load once at start-up, validate every request, and return the version with the prediction so it can be logged.

```python
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0); n = 5000
X = pd.DataFrame({"income_monthly": rng.lognormal(7.9, 0.4, n),   # pounds per MONTH
                  "age": rng.integers(18, 80, n),
                  "debt_ratio": rng.uniform(0, 1, n)})
y = (0.0012 * X.income_monthly - 3 * X.debt_ratio + rng.normal(0, 1, n) > 1.2).astype(int)

# 1. Training side: one artifact = preprocessing + model + schema + version
pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
SCHEMA = {c: (float(X[c].quantile(0.001)), float(X[c].quantile(0.999))) for c in X}
joblib.dump({"model": pipe, "schema": SCHEMA, "features": list(X.columns),
             "version": "credit-2026-11-14"}, "model.joblib")

# 2. Serving side: load once, validate, predict, return what to log
art = joblib.load("model.joblib")
def predict(request: dict) -> dict:
    row = pd.DataFrame([request])[art["features"]]          # fixed column order
    problems = [f for f, (lo, hi) in art["schema"].items()
                if pd.isna(row.at[0, f]) or not lo <= row.at[0, f] <= hi]
    p = float(art["model"].predict_proba(row)[0, 1])
    return {"p_approve": round(p, 3), "version": art["version"], "out_of_range": problems}

print(predict({"income_monthly": 2800, "age": 41, "debt_ratio": 0.3}))
# {'p_approve': 0.896, 'version': 'credit-2026-11-14', 'out_of_range': []}
print(predict({"income_monthly": 2800 * 12, "age": 41, "debt_ratio": 0.3}))  # yearly by mistake
# {'p_approve': 1.0, 'version': 'credit-2026-11-14', 'out_of_range': ['income_monthly']}
print(SCHEMA["income_monthly"])   # (808.66..., 8432.02...)
```

In the second call the model happily returns 1.0; only the schema check reveals the problem. In a real service `predict` sits behind an HTTP endpoint, and its output plus the input row go to the prediction log.

The skew report on 10,000 synthetic requests, where 8% of signups fall across midnight in the service's timezone and 3% of `orders_7d` lookups miss the cache:

```python
import numpy as np, pandas as pd
# offline: features recomputed in the warehouse; served: features logged by the service
j = offline.merge(served, on="request_id", suffixes=("_off", "_srv"))
for f in ["price", "days_since_signup", "orders_7d"]:
    a, b = j[f + "_off"], j[f + "_srv"]
    both = a.notna() & b.notna()
    mismatch = (~np.isclose(a[both], b[both], rtol=1e-6)).mean()
    print(f"{f:18s} mismatch={mismatch:6.2%}  null_off={a.isna().mean():.2%}  null_srv={b.isna().mean():.2%}")
# price              mismatch= 0.00%  null_off=0.00%  null_srv=0.00%
# days_since_signup  mismatch= 8.23%  null_off=0.00%  null_srv=0.00%
# orders_7d          mismatch= 0.00%  null_off=0.00%  null_srv=2.96%
```

Mismatches and nulls are counted separately so a missing value does not masquerade as a disagreement.

## Checking that the service is healthy

Before and after every release, check:

- **Parity test**: score a fixed few hundred rows in the notebook and through the deployed service; predictions should match to rounding.
- **Skew report**: per-feature mismatch and null rates on a daily sample of logged requests; healthy is near 0%.
- **Schema violations**: the share of requests with any out-of-range input; alert on a sudden rise, not a single bad row.
- **Latency percentiles**: p50, p95 and p99 against the budget, per step if possible.
- **Prediction distribution**: mean score or approval rate per hour; a step change with no product change is the cheapest early warning.

## What goes wrong

| Failure | Symptom | Fix
| --- | --- | --- |
| Units or format changed upstream | prediction rate jumps, no errors | range checks per feature; units in the schema
| Feature computed twice in two languages | offline metric good, online metric poor | one feature definition (feature store) or log served features
| Training features fresher than serving (or the reverse) | strong offline AUC that never materialises live | point-in-time joins (09.02); train on logged features
| Cache miss returns 0 or null | slow, steady degradation, worse at peak load | track served null rate; explicit missing-value handling
| Preprocessing shipped apart from the model | breaks when one part is updated | one versioned pipeline artifact
| p99 too high | timeouts at peak, fine on average | profile per step; cache or precompute features; batch requests

## Going deeper

**Tail latency multiplies with fan-out.** If one online request calls $m$ independent backends in parallel and must wait for all of them, the request is slow whenever *any* backend is slow. If each backend exceeds its own p99 with probability $0.01$, the chance that at least one of $m$ does is

$$P(\text{slow}) = 1 - (1 - 0.01)^m.$$

For $m = 3$: $1 - 0.99^3 = 1 - 0.970299 = 0.0297$, so about 3% of requests hit some backend's p99 tail. For $m = 100$: $1 - 0.99^{100} \approx 0.634$, so most requests do. This is why systems that fan out widely care about each component's p99.9, and use hedged requests (send a duplicate after a short delay and take whichever answers first).

**Batching, quantified.** With $t(B) = t_0 + cB$, throughput approaches $1/c$, and reaches half of it at $B = t_0/c$. Beyond that knee, doubling the batch nearly doubles latency for little gain; dynamic batchers expose maximum batch size and maximum wait time to sit near it.

**Feedback loops.** Once a model acts, its future training data is shaped by its own decisions: rejected applicants never get a repayment label. Logging served features and scores, plus a little randomised exploration, keeps that data usable. Sculley et al. call this a hidden feedback loop.

**Logged features are not free.** A new feature has no logs yet, so it needs a backfill (with the skew risk that implies) or a wait while logs accumulate. Many teams backfill to prototype, then retrain on logs before launch.

## Key terms

| Term | Meaning
| --- | --- |
| Serving / inference | Using a trained model to answer real requests; one model call is one inference
| Batch inference | Scoring all entities on a schedule and serving stored scores by lookup
| Online inference | Computing a prediction per request, inside a latency budget
| p50 / p99 | Latency that 50% / 99% of requests beat; budgets are usually set on a high percentile
| Throughput | Predictions completed per second; raised by batching
| Training/serving skew | Any difference between the inputs used in training and those seen in serving
| Feature store | One definition per feature, materialised offline for training and online for serving
| Schema / data contract | Expected type, range, categories, units and null rate for each input, checked at the boundary

<div class="take">
<div class="lab">Key takeaway</div>

pick batch or online inference by freshness, coverage and latency needs, then remove training/serving skew structurally: one feature definition, served features logged for training, a schema check at every boundary, and model plus preprocessing shipped as one versioned artifact.

</div>
