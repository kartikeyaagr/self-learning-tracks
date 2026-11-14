---
id: "02.07"
course: "ml-toolkit"
module: "02"
slug: "time-series-in-pandas"
title: "Time series in pandas"
keyIdeas: ["Parse timestamps once into `datetime64` (with an explicit `format` and a time zone, UTC for storage), then sort by entity and time. A `DatetimeIndex` unlocks slicing such as `df.loc[\"2024-03\"]` and `resample`.", "`resample` is a groupby on time bins: it turns irregular events into a regular series and makes empty periods explicit (sum 0, mean `NaN`). Use the pandas 2.2+ aliases `\"h\"`, `\"min\"`, `\"ME\"`; the old `\"H\"` and `\"M\"` are rejected in pandas 3.", "`rolling(7)` covers the last 7 *rows*, `rolling(\"7D\")` the last 7 days of *time*, `expanding` everything so far and `ewm` everything with fading weights. Row windows silently stretch when days are missing.", "Windows include the current row by default. A feature for time $t$ must use only data before $t$: `shift(1)` before rolling, or `closed=\"left\"` on time windows, always inside `groupby(entity)`.", "Validate with time-ordered splits (a cutoff or `TimeSeriesSplit`) with a gap at least as long as the longest feature or label window, never with random splits."]
question: "You predict whether a user will make a purchase in the next 7 days, scoring every Monday at 00:00 UTC. For training you create one row per user per Monday with features `purchases_30d = rolling(\"30D\").count()` over the user's purchase events, computed with `groupby(\"user_id\")` on a frame indexed by purchase time, then merged onto the Monday rows by `(user_id, date)`. The label is purchases in `[Monday, Monday + 7 days)`. Identify the leakage and other bugs in this setup and describe a correct construction."
answer: "The rolling count is evaluated at each purchase event, not at the Monday scoring time, so merging on `(user_id, date)` only works for Mondays on which the user purchased, and then the window includes purchases made on Monday itself, i.e. inside the label window: direct leakage. For other Mondays the merge finds no row, so the feature is missing or gets filled with 0, which is also wrong: a user with ten purchases last week would look like a user with none. The correct construction computes features at the scoring times. Build the grid of `(user_id, monday)` rows, then for each one count purchases in `[monday - 30 days, monday)`. One way is `pd.merge_asof` from the Monday grid (sorted by time) onto a per-user cumulative count of events with `by=\"user_id\"` and `allow_exact_matches=False`, then subtracting the cumulative count as of `monday - 30 days` found the same way. Another is resampling purchases to daily counts per user (so days without purchases become 0 rows rather than missing rows), computing `rolling(30)` on that regular daily series, shifting by one day within each user so the window ends on Sunday, and selecting the Mondays; `rolling(\"30D\", closed=\"left\")` evaluated on the Monday rows is equivalent. All timestamps should be in UTC to match the scoring time. The label must be computed from events in `[monday, monday + 7 days)` only. Split train and validation by Monday dates, not randomly, with at least a 7-day gap so the last training labels do not overlap the first validation features. Check the result by recomputing a few rows by hand from the raw events, by truncating the raw data at some date and confirming that earlier feature rows do not change, and by confirming the same code runs at serving time with only data up to Sunday 23:59."
sources: [{"title": "pandas docs: Time series / date functionality", "url": "https://pandas.pydata.org/docs/user_guide/timeseries.html", "note": "DatetimeIndex, time zones, offset aliases (including the 2.2 changes), resample and asfreq."}, {"title": "pandas docs: Windowing operations", "url": "https://pandas.pydata.org/docs/user_guide/window.html", "note": "rolling, expanding and ewm, time-based windows, closed and min_periods."}, {"title": "scikit-learn API: TimeSeriesSplit", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html", "note": "Time-ordered cross-validation folds with an optional gap."}]
practice: ["point-in-time-feature-join", "delayed-label-monitoring", "apply-custom-transforms", "popularity-baseline-leave-last-out"]
tags: ["time series", "rolling windows", "leakage"]
tools: ["pandas"]
---

## What you will learn

- How to turn timestamp strings into real `datetime64` values, with an explicit format and time zone, and why sorting comes next.
- How `resample` turns irregular events (one row per purchase) into a regular series (one row per day), and how to read its bin labels.
- The difference between a window of *rows* (`rolling(7)`), a window of *time* (`rolling("7D")`), a running window (`expanding`) and a fading one (`ewm`).
- How to build lag and window features that use only the past, per entity, so the model never sees the answer it is asked to predict.
- How to split time-ordered data for validation, and why a random split gives scores you will never see again in production.

*Before you start:* you should be comfortable with indexing and `.loc` (02.02), `groupby` and `transform` (02.04) and joins, including `merge_asof` (02.05). The idea of leakage was introduced with target encoding in 02.04; this lesson is its most common real-world form.

Lesson 02.06 reshaped tables between long and wide. Time series data is almost always long, and it adds one thing a plain table lacks: an order that matters. Yesterday can explain today, but today must never explain yesterday.

## The idea in one picture

A small shop records every sale as an event. Here are the 11 sales of its first week of March 2024, as they arrive from the payment system (strings, UTC):

```python
import pandas as pd
raw = pd.DataFrame({
    "ts": ["2024-03-01T09:15:00Z", "2024-03-01T17:40:00Z", "2024-03-02T10:05:00Z",
           "2024-03-04T08:30:00Z", "2024-03-04T13:20:00Z", "2024-03-05T11:00:00Z",
           "2024-03-05T16:45:00Z", "2024-03-06T12:10:00Z", "2024-03-07T09:50:00Z",
           "2024-03-07T15:30:00Z", "2024-03-07T18:05:00Z"],
    "amount": [5, 7, 15, 4, 5, 10, 8, 14, 6, 9, 5]})
raw["ts"] = pd.to_datetime(raw["ts"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
ev = raw.set_index("ts").sort_index()
daily = ev["amount"].resample("D").sum()
print(daily.tolist())          # [12, 15, 0, 9, 18, 14, 20]   (Mar 1 ... Mar 7)
```

~~~dg-fig 02.07-resample.svg
Top: sales arrive at irregular times. Bottom: resample("D").sum() puts them into one bin per day. March 3 had no sales, and instead of vanishing it becomes an explicit 0.
~~~

The raw events have no rhythm: two on Friday, one on Saturday, none on Sunday. A model that predicts "sales tomorrow" needs one row per day, so we bin events by day and add them up: a **resample**. The empty Sunday becomes a row with 0, so "no sales" is recorded as information rather than silently missing.

A few terms that the rest of the lesson uses:

- A **timestamp** is one point in time; pandas stores it as `datetime64`, a count of time units since 1970.
- A **DatetimeIndex** is an index made of timestamps, which unlocks date slicing and resampling.
- A **window** is the set of rows a statistic looks at, such as "the last 7 days".
- A **lag** is a past value moved forward to today's row, such as "yesterday's sales".

## Step 1: parse timestamps once, correctly

Everything else depends on this step. Strings like `"2024-03-01T09:15:00Z"` sort alphabetically, cannot be subtracted, and mean nothing to `resample`. Convert them once, at load time:

```python
raw["ts"] = pd.to_datetime(raw["ts"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
print(raw["ts"].dtype)                      # datetime64[us, UTC]
ev.loc["2024-03-05"]                         # both sales on 5 March (partial-string slicing)
daily.loc["2024-03-02":"2024-03-04"]         # 15, 0, 9: both ends included, as with any .loc slice
ev.index.tz_convert("Europe/Amsterdam").hour # local hour, e.g. 09:15 UTC -> 10
```

Three choices are made here, and each one prevents a class of bugs:

1. **An explicit `format`.** Without it pandas guesses, and a string like `03/04/2024` is 3 April in some countries and 4 March in others. With a format, a malformed row raises an error instead of becoming a wrong date.
2. **A time zone.** `utc=True` makes the column **time-zone-aware**: every value is a definite instant. Store in UTC; convert to local time only when you need a local feature.
3. **Sort.** Windows and lags follow row order, so an unsorted frame gives windows that jump around in time. Sort by entity and time (`df.sort_values(["store_id", "ts"])`) right after parsing.

The dtype says `us`: pandas 3 infers the resolution from the data, where pandas 2 always used nanoseconds.

Mixing aware and naive timestamps raises an error: `ev.index > pd.Timestamp("2024-03-03")` fails with *Invalid comparison between dtype=datetime64[us, UTC] and Timestamp*. That error stops you comparing a local wall-clock time with a UTC instant; make both sides aware, e.g. `pd.Timestamp("2024-03-03", tz="UTC")`.

Compute calendar features (`.dt.dayofweek`, `.dt.hour`) in the time zone where the behaviour happens: an Amsterdam shop opening at 9 local time opens at 8:00 UTC in winter and 7:00 UTC in summer.

## Step 2: resample to a regular frequency

`resample` is a `groupby` whose groups are time bins. You say how wide the bins are and how to combine the rows inside each bin:

```python
ev["amount"].resample("D").sum()     # 12, 15, 0, 9, 18, 14, 20
ev["amount"].resample("D").mean()    # 6.0, 15.0, NaN, 4.5, 9.0, 14.0, 6.67
ev["amount"].resample("ME").sum()    # one row, labelled 2024-03-31: 88
```

Notice the different treatment of the empty day: a sum over no rows is 0, but a mean over no rows is undefined, so it is `NaN`. Both are correct; choose the aggregation that matches the question. With several entities, group first and resample inside each group:

```python
sales = df.set_index("ts").groupby("store_id")["amount"].resample("D").sum()
sales = sales.reset_index()          # columns: store_id, ts, amount
```

**Frequency aliases** are short codes for bin widths. They changed in pandas 2.2: use `"h"`, `"min"`, `"s"` for hours, minutes and seconds, and `"ME"`, `"QE"`, `"YE"` for month, quarter and year ends. The old `"H"`, `"T"` and `"M"` spellings were deprecated in 2.2 and are rejected in pandas 3 (`resample("H")` raises *Invalid frequency: H ... Did you mean h?*).

Each bin is labelled by one of its edges. Daily and hourly bins are labelled by their **left** edge (the bin for 5 March is labelled 5 March 00:00), but month-end bins are labelled by their **right** edge (all of March is labelled 31 March). If you join a monthly feature onto daily labels by date, that difference decides whether the March total lands on 31 March, when the month is over, or appears to be known on 1 March. Check the `label` and `closed` arguments before aligning resampled features with labels.

Two relatives: `asfreq("D")` changes frequency without aggregating (missing timestamps become `NaN` rows), and `ffill()` copies the last known value forward, which is safe for features because it only uses the past.

## Step 3: rolling, expanding and fading windows

Once the series is regular, most features are statistics over windows. pandas offers four kinds:

| window | covers | example use
| --- | --- | --- |
| `rolling(7)` | the last 7 **rows** | 7-day mean if there is exactly one row per day
| `rolling("7D")` | rows within the last 7 days of **time** | robust to missing days and irregular events
| `expanding()` | all rows so far | running mean, cumulative counts
| `ewm(halflife="7D", times=...)` | all rows, exponentially weighted | smooth recency-weighted average

~~~dg-fig 02.07-windows.svg
Ninety days of one store. The 7-day rolling mean follows the recent level; the expanding mean settles near the long-run average and stops reacting; the exponentially weighted mean sits in between.
~~~

A short rolling window reacts quickly but is noisy; an expanding window is stable but slow to notice change. The **exponentially weighted mean** (`ewm`) gives every past row a weight that halves every `halflife` days, so it uses all the history but cares most about recent days.

On the shop's daily series, with `daily = [12, 15, 0, 9, 18, 14, 20]`:

```python
daily.rolling(3).mean().round(2).tolist()
# [nan, nan, 9.0, 8.0, 9.0, 13.67, 17.33]
daily.rolling(3, min_periods=1).mean().round(2).tolist()
# [12.0, 13.5, 9.0, 8.0, 9.0, 13.67, 17.33]
daily.expanding().mean().round(2).tolist()
# [12.0, 13.5, 9.0, 9.0, 10.8, 11.33, 12.57]
```

`min_periods` sets how many observations a window needs before it produces a value. For an integer window the default equals the window size, so the first two rows are `NaN`. Lowering it fills the start of the series with averages over fewer days, which is often fine but should be a conscious choice.

**Rows versus time.** Row windows assume one row per time step. That assumption breaks silently when days are missing, which is exactly what happens if you build the daily table with `groupby(date)` instead of `resample`: days without sales simply have no row.

~~~dg-fig 02.07-rows-vs-time.svg
If 3 March has no row, "the last 3 rows" ending on 5 March reaches back to 2 March and spans four days. A time window of "3D" covers 3 to 5 March whatever rows exist.
~~~

On the gappy version (no Sunday row) the two windows disagree on 5 March: `rolling(3).sum()` gives $15 + 9 + 18 = 42$ (2, 4 and 5 March), while `rolling("3D").sum()` gives $9 + 18 = 27$ (4 and 5 March only). Either make the series regular first (`resample("D")` or `asfreq("D")`) or use a time-based window.

## Step 4: lags and windows without leakage

Here is the rule that matters most in this lesson:

$$\text{feature}(t) \text{ may use only data with timestamp} < t.$$

Here $t$ is the moment the prediction is made, and "data" includes the target itself. By default every pandas window ends at, and *includes*, the current row. If the target is today's sales, a 7-day mean that includes today contains one seventh of the answer. This is **leakage**: the feature carries information that will not exist when the model is used.

~~~dg-flow Building a feature for day t from strictly earlier data.
Sales up to day t-1 -(shift 1 per store)-> *rolling 7 mean -> feature at t -> model predicts sales at t
~~~

The fix is to move the series down by one row before the window, with `shift(1)`, and to do it per entity:

```python
df = df.sort_values(["store_id", "date"])
g = df.groupby("store_id")["sales"]
df["lag_1"] = g.shift(1)                       # yesterday, per store
df["lag_7"] = g.shift(7)                       # same weekday last week
df["mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
df["mean_all"] = g.transform(lambda s: s.shift(1).expanding().mean())

feat = (df.set_index("date").groupby("store_id")["sales"]
          .rolling("7D", closed="left").sum())   # time window [t - 7D, t)
```

`closed="left"` is the time-window equivalent of the shift: it makes the window $[t - 7\text{D}, t)$, open at the right end, so the current timestamp is excluded.

Why the `groupby`? A plain `shift` over the sorted frame lets one store's history spill into the next. With two stores, A and B, four days each:

```python
#  store  date        sales  df["sales"].shift(1)  g.shift(1)  mean_2 (shift then rolling 2)
#  A      2024-03-01     12        NaN               NaN          NaN
#  A      2024-03-02     15       12.0              12.0          NaN
#  A      2024-03-03      0       15.0              15.0         13.5
#  A      2024-03-04      9        0.0               0.0          7.5
#  B      2024-03-01     40        9.0   <- store A!  NaN          NaN
#  B      2024-03-02     44       40.0              40.0          NaN
```

The ungrouped column claims that store B sold 9 the day before 1 March; that 9 is store A's 4 March. The grouped version correctly says "unknown".

<div class="ex">
<div class="lab">Worked example</div>

**the leaky and the honest window, by hand**

Use the shop's daily totals: 1 March 12, 2 March 15, 3 March 0, 4 March 9, 5 March 18. We want a feature for predicting sales on 5 March (the target is 18).

Leaky version, `rolling(3).mean()` on 5 March: the window is 3, 4 and 5 March, so the feature is $(0 + 9 + 18)/3 = 27/3 = 9$. The target 18 is inside it.

Honest version, `shift(1).rolling(3).mean()` on 5 March: after the shift, the row for 5 March holds 4 March's value, so the window covers 2, 3 and 4 March: $(15 + 0 + 9)/3 = 24/3 = 8$. Only the past is used.

Time-window version, `rolling("3D", closed="left").sum()` on 5 March: the window is $[2\ \text{March}, 5\ \text{March})$, which is again 2, 3 and 4 March: $15 + 0 + 9 = 24$. Dividing by 3 days gives the same 8.

Check with pandas: `daily.shift(1).rolling(3).mean()` prints `[nan, nan, nan, 9.0, 8.0, 9.0, 13.67]`, and the fifth value is 8.0. The leaky `daily.rolling(3).mean()` prints 9.0 in the same position.

</div>

How much does this matter? We measured it on a synthetic dataset of 20 stores over one year, with a linear model trained on everything before October and scored on the last two months.

~~~dg-fig 02.07-leak.svg
Left: a plain 7-day rolling mean contains the target day; the shifted version ends the day before. Right: measured on synthetic data, the leaky feature scores an MAE of 15.3 offline, but once only past data exists its error rises to 17.6, the same as the honest model's offline score.
~~~

The leaky model looks about 13% better offline. In production today's sales are unknown at forecast time, so the feature can only use the past and the error jumps to the honest level; the honest offline score (17.5) was the true number all along. On spikier data the gap is far larger.

If the prediction is made some time *before* the target (forecasting next week, or labels that arrive late), shift by that horizon, not by one: features for a forecast made on Monday for Friday may use data only up to Monday. `center=True` windows and `bfill` (backward fill) use future rows; they are fine for smoothing a chart and wrong for features.

## Step 5: split by time

Random train/test splits on time series let the model train on Thursday and test on Wednesday. Neighbouring days are similar, so the offline score measures interpolation between known days, not forecasting. Split by a cutoff instead, and leave a **gap** at least as long as your longest feature window or label window, so the two sides cannot share information:

```python
cutoff = pd.Timestamp("2024-06-01", tz="UTC")
train = df[df["date"] < cutoff - pd.Timedelta("7D")]
test = df[df["date"] >= cutoff]
```

~~~dg-fig 02.07-split.svg
Random days (top) leak neighbouring information into training. A single cutoff with a gap (middle) imitates deployment. TimeSeriesSplit (bottom) repeats that pattern with a training window that grows fold by fold.
~~~

For cross-validation, scikit-learn's `TimeSeriesSplit` produces folds in which training always comes before testing, with an optional `gap`:

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
for tr, te in TimeSeriesSplit(n_splits=3, test_size=2, gap=1).split(np.arange(12)):
    print(tr, te)
# [0 1 2 3 4] [6 7]
# [0 1 2 3 4 5 6] [8 9]
# [0 1 2 3 4 5 6 7 8] [10 11]
```

Each fold skips one row between train and test (row 5, then 7, then 9). The folds index *rows*, so the frame must be sorted by time, and with several stores you usually split on unique dates rather than on raw rows.

## Checking your features

A feature pipeline that runs is not yet a correct one. Before training, run these checks:

- **Recompute a few rows by hand.** Pick one store and one date, filter the raw events to strictly before that date, and compute the feature yourself. It must match the table exactly.
- **Truncate and recompute.** Cut the raw data at some date $T$, rebuild the features, and compare the rows before $T$ with the full build. If any feature value changes, it was using data from after $T$.
- **Distrust a feature that is too good.** Near-perfect offline power is a leak until proven otherwise.
- **Compare offline and production error.** A large drop after launch, like the 6% to 19% MAPE story below, is the classic leakage signature.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Parsing without a format | some days and months swapped, odd spikes on the 1st to 12th of months | pass `format=`; reject rows that fail to parse
| Mixing local and UTC times | features off by one or two hours; jumps at daylight-saving changes | store in UTC, convert with `tz_convert` for local features
| Unsorted frame | lags and windows reach into random rows | `sort_values([entity, time])` right after parsing
| Row window on gappy data | "7-day" features that sometimes cover three weeks | `resample("D")` first, or use `rolling("7D")`
| Window includes current row | excellent offline score, much worse in production | `shift(1)` before rolling, or `closed="left"`
| Shift without `groupby` | first rows of each entity show another entity's values | `groupby(entity).shift(...)`
| `center=True` or `bfill` in features | uses tomorrow's data | trailing windows and `ffill` only
| Random train/test split | validation far better than any later period | cutoff split with a gap, `TimeSeriesSplit`

<div class="ex">
<div class="lab">Worked example</div>

**a sales feature that peeked**

A demand model's feature `rolling_mean_7 = df.groupby("store_id")["sales"].transform(lambda s: s.rolling(7).mean())` gives a validation MAPE of 6%. In production the error is 19%.

Because the window includes the current day, one seventh of each feature is the target itself, and on volatile days that share is what the model leans on. Rewriting the feature with `s.shift(1).rolling(7)` brings offline MAPE to 17%, matching production, and is the version that can actually be computed before the day starts.

</div>

## Going deeper

**Point-in-time joins.** Often the features live in another table with its own timestamps (prices, user profiles, inventory). For each label row you need the latest feature value *as of* the prediction time. `pd.merge_asof(labels, features, on="ts", by="store_id", allow_exact_matches=False)` does exactly this: for each left row it takes the last right row with a strictly earlier timestamp, per store. Both frames must be sorted by `ts`. Setting `allow_exact_matches=False` excludes a feature row stamped at exactly the prediction time, which is usually what you want, since that value was probably written after the prediction was made.

**Daylight saving time.** A local "day" is not always 24 hours. Midnight in Amsterdam on 31 March 2024 is 23:00 UTC on 30 March, but midnight on 1 April is 22:00 UTC on 31 March, because clocks moved forward at 02:00 on the 31st, so that local day lasted only 23 hours. `pd.date_range("2024-03-30", periods=3, freq="D", tz="Europe/Amsterdam")` handles this correctly; adding `pd.Timedelta("24h")` to local times does not. Resampling to local calendar days therefore means converting to the local zone first, then resampling.

**Event time versus arrival time.** A record has the time the event happened and the time it reached your database. A refund for a March sale may arrive in April. A feature "as of 1 April" must only count records that had *arrived* by 1 April, or the training data will look more complete than production data ever is. Keep both timestamps if you can, and filter on arrival time when building features.

**Horizon and label windows.** For a label such as "purchases in the next 7 days", the label at time $t$ covers $[t, t + 7\text{D})$ and every feature must end before $t$. When you split, the last training labels reach 7 days past the training cutoff, so the gap must be at least the label window, or the validation features will overlap the training labels.

## Key terms

| Term | Meaning
| --- | --- |
| `datetime64` / Timestamp | pandas' type for a point in time, optionally with a time zone
| Time-zone-aware | a timestamp tied to a definite instant (e.g. UTC), not a local wall-clock reading
| DatetimeIndex | an index of timestamps; enables `.loc["2024-03"]` slicing and `resample`
| Resample | group rows into regular time bins and aggregate each bin
| Rolling window | a statistic over the last N rows, or the last span of time such as `"7D"`
| Expanding / EWM | a statistic over all rows so far, equally weighted / fading with age
| Lag | a past value moved forward with `shift`, such as yesterday's sales
| Leakage | a feature that uses information not available at prediction time
| Point-in-time join | attaching the latest value known strictly before each row's timestamp (`merge_asof`)

<div class="take">
<div class="lab">Key takeaway</div>

parse and sort timestamps first, compute every window per entity on data strictly before the prediction time (`shift` or `closed="left"`), and split train and test by time with a gap.

</div>
