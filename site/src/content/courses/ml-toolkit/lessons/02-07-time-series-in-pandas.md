---
id: "02.07"
course: "ml-toolkit"
module: "02"
slug: "time-series-in-pandas"
title: "Time series in pandas"
keyIdeas: ["Parse timestamps once into `datetime64` (with an explicit format and time zone), sort, and either set them as a `DatetimeIndex` or keep them as a column used with `on=`. Partial-string slicing such as `df.loc[\"2024-03\"]` then works.", "`resample` changes frequency (events to daily totals, for instance); `rolling` computes statistics over a moving window of rows (`rolling(7)`) or of time (`rolling(\"7D\")`); `expanding` uses everything so far.", "Windows include the current row by default. A feature for predicting time $t$ must use only data before $t$: `shift(1)` before rolling, or `closed=\"left\"` on time-based windows, and always per entity via `groupby`.", "Validate with time-ordered splits, never random ones, and make sure label windows and feature windows do not overlap."]
question: "You predict whether a user will make a purchase in the next 7 days, scoring every Monday at 00:00 UTC. For training you create one row per user per Monday with features `purchases_30d = rolling(\"30D\").count()` over the user's purchase events, computed with `groupby(\"user_id\")` on a frame indexed by purchase time, then merged onto the Monday rows by `(user_id, date)`. The label is purchases in `[Monday, Monday + 7 days)`. Identify the leakage and other bugs in this setup and describe a correct construction."
answer: "The rolling count is evaluated at each purchase event, not at the Monday scoring time, so merging on `(user_id, date)` only works for Mondays on which the user purchased, and then the window includes purchases made on Monday itself, i.e. inside the label window: direct leakage. For other Mondays the merge finds no row, so the feature is missing or gets filled with 0, which is also wrong. The correct construction computes features at the scoring times: build the grid of `(user_id, monday)` rows, then for each one count purchases in `[monday - 30 days, monday)`. One way is `pd.merge_asof` from the Monday grid onto a per-user cumulative count of events with `allow_exact_matches=False`, then subtracting the cumulative count as of `monday - 30 days`; another is resampling purchases to daily counts per user, computing `rolling(30)` on the regular daily series, shifting by one day so the window ends on Sunday, and selecting the Mondays. The label must be computed from events in `[monday, monday + 7 days)` only. Split train and validation by Monday dates, not randomly, with at least a 7-day gap so the last training labels do not overlap the first validation features. Check the result by recomputing a few rows by hand and by confirming the same code runs at serving time with only data up to Sunday 23:59."
sources: [{"title": "pandas docs: Time series / date functionality", "url": "https://pandas.pydata.org/docs/user_guide/timeseries.html", "note": "DatetimeIndex, time zones, offset aliases (including the 2.2 changes), resample and asfreq."}, {"title": "pandas docs: Windowing operations", "url": "https://pandas.pydata.org/docs/user_guide/window.html", "note": "rolling, expanding and ewm, time-based windows, closed and min_periods."}, {"title": "scikit-learn API: TimeSeriesSplit", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html", "note": "Time-ordered cross-validation folds with an optional gap."}]
practice: ["point-in-time-feature-join", "delayed-label-monitoring", "apply-custom-transforms", "popularity-baseline-leave-last-out"]
tags: ["time series", "rolling windows", "leakage"]
tools: ["pandas"]
---

## Getting timestamps right

Time series work starts with a clean datetime column. Parse with an explicit format, decide on a time zone (UTC is the safe default for storage), and sort:

```python
df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
df = df.sort_values(["store_id", "ts"])
daily = df.set_index("ts")
daily.loc["2024-03"]                      # all of March 2024
daily.loc["2024-03-01":"2024-03-07"]      # both ends included, as with any .loc slice
df["ts"].dt.tz_convert("Europe/Amsterdam").dt.hour   # local hour as a feature
```

Mixing time-zone-aware and naive timestamps raises an error in comparisons and joins, which is a feature: it stops you comparing local and UTC times by accident. Calendar features (`.dt.dayofweek`, `.dt.hour`, `.dt.month`) should be computed in the time zone where the behaviour happens.

## Resampling

`resample` is a groupby on time bins. It turns irregular events into a regular series:

```python
sales = df.set_index("ts").groupby("store_id")["amount"].resample("D").sum()
sales = sales.reset_index()    # columns: store_id, ts, amount
```

Days with no events appear with sum 0 (and `NaN` for `mean`), which makes gaps explicit. Frequency aliases changed in pandas 2.2: use `"h"`, `"min"`, `"s"` for hours, minutes and seconds, and `"ME"`, `"QE"`, `"YE"` for month, quarter and year ends; the old `"H"`, `"T"`, `"M"` spellings are deprecated. Each bin is labelled by one edge, which is the left edge for daily and hourly bins and the right edge for month-end bins, so read the `label` and `closed` defaults before aligning resampled features with labels.

## Rolling and expanding windows

| window | covers | example
| --- | --- | --- |
| `rolling(7)` | the last 7 **rows** | 7-day mean if there is exactly one row per day
| `rolling("7D")` | rows within the last 7 days of **time** | robust to missing days and irregular events
| `expanding()` | all rows so far | running mean, cumulative counts
| `ewm(halflife="7D", times=...)` | all rows, exponentially weighted | smooth recency-weighted average

Row-based windows silently become wrong when days are missing: seven rows might span three weeks. Either make the series regular first (`resample("D")` or `asfreq("D")`) or use a time-based window. `min_periods` sets how many observations are required before a value is produced; the default for an integer window is the window size, so the first rows are `NaN`.

## Lags without leakage

By default every window ends at, and includes, the current row. If the target is today's sales, a 7-day mean that includes today contains the target. The rule: a feature for time $t$ uses only information available before $t$.

~~~dg-flow Building a feature for day t from strictly earlier data.
Sales up to day t-1 -(shift 1 per store)-> *rolling 7 mean -> feature at t -> model predicts sales at t
~~~

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

Three details matter. The `groupby` stops one store's history spilling into the next store's first rows, which a plain `shift` over the sorted frame would do. The `shift(1)` happens *inside* each group before the window. And if your prediction is made some time before the target (forecasting next week, or labels that arrive late), shift by that horizon, not by one: features for a forecast made on Monday for Friday may use data only up to Monday.

`center=True` windows and `bfill` use future rows; they are fine for smoothing a plot and wrong for features. Forward fill (`ffill`) is safe, since it only copies the past forward.

## Splitting by time

Random train/test splits on time series let the model train on Thursday and test on Wednesday, so offline scores reflect interpolation between neighbouring days rather than forecasting. Split by a cutoff instead, and leave a gap at least as long as your longest feature window or label window so the two sides cannot share information:

```python
cutoff = pd.Timestamp("2024-06-01", tz="UTC")
train = df[df["date"] < cutoff - pd.Timedelta("7D")]
test = df[df["date"] >= cutoff]
```

scikit-learn's `TimeSeriesSplit` (with its `gap` argument) gives the same structure for cross-validation.

<div class="ex">
<div class="lab">Worked example</div>

**a sales feature that peeked**

A demand model's feature `rolling_mean_7 = df.groupby("store_id")["sales"].transform(lambda s: s.rolling(7).mean())` gives a validation MAPE of 6%. In production the error is 19%.

Because the window includes the current day, one seventh of each feature is the target itself, and on volatile days that share is what the model leans on. Rewriting the feature with `s.shift(1).rolling(7)` brings offline MAPE to 17%, matching production, and is the version that can actually be computed before the day starts.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

parse and sort timestamps first, compute every window per entity on data strictly before the prediction time (`shift` or `closed="left"`), and split train and test by time with a gap.

</div>
