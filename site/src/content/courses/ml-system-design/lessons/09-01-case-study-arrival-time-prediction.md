---
id: "09.01"
course: "ml-system-design"
module: "09"
slug: "case-study-arrival-time-prediction"
title: "Case study: arrival-time prediction"
keyIdeas: ["Arrival-time prediction is a regression on a route, usually built as a correction on top of a routing engine's physics-style estimate rather than from scratch.", "Labels are actual trip durations, which arrive only after the trip ends and are biased by which routes drivers actually took.", "Mean error alone hides the cost of being late; track error by percentile, by trip length and by city, and consider predicting a range.", "Serving needs low latency at very high request volume, because every map view, quote and dispatch decision asks for an ETA."]
question: "Design the system that predicts arrival time for a ride-hailing or delivery app. It must answer about 500,000 ETA requests per second at peak, for pickup and for trip duration, in under 50 ms. Cover labels, features, model, serving and how you would evaluate it."
answer: "Use the routing engine's segment-based estimate as a baseline feature and learn a correction (the residual) with a model that sees route, time, weather, live traffic and city features. Labels are actual durations from completed trips, joined point-in-time to the features known at request time. Evaluate with mean absolute error and with percentiles of error, sliced by city, distance and time of day, and track late-arrival rate separately because lateness costs more than earliness. Serve with cached segment speeds from a streaming pipeline, a light model on CPU, and batching across candidate routes. Monitor error daily by city, feature freshness and drift during events such as storms. Trade-offs: a single global model versus per-city models, and a point estimate versus a range shown to users."
rubric: ["Clarifies which ETA (pickup, trip, delivery), volume and latency budget", "Builds on the routing engine estimate and predicts a correction rather than starting from scratch", "Describes labels from completed trips with point-in-time feature joins and notes selection bias", "Includes live traffic or segment speeds from a streaming pipeline, with freshness handled", "Chooses an asymmetric or quantile view of error, since lateness costs more than earliness", "Evaluates by city, distance band and time of day, not only overall MAE", "Serves within the latency budget with caching and a light model", "Monitors error, feature freshness and drift during unusual events"]
sources: [{"title": "Hu et al., DeeprETA: An ETA Post-processing System at Scale (2022)", "url": "https://arxiv.org/abs/2206.02127", "note": "A learned correction on top of a routing engine's estimate."}, {"title": "Derrow-Pinion et al., ETA Prediction with Graph Neural Networks in Google Maps (2021)", "url": "https://arxiv.org/abs/2108.11482", "note": "Road segments as a graph, with traffic propagating between them."}, {"title": "Wikipedia, Quantile regression", "url": "https://en.wikipedia.org/wiki/Quantile_regression", "note": "Predicting percentiles rather than the mean."}, {"title": "Wikipedia, Map matching", "url": "https://en.wikipedia.org/wiki/Map_matching", "note": "Turning noisy GPS traces into road segments."}]
practice: ["gradient-boosted-regressor", "regression-metrics", "point-in-time-feature-join", "p50-p95-and-p99-inference-latency"]
tags: ["ETA prediction", "regression", "residual model", "quantiles"]
---

## Requirements

Start by asking which ETA is meant. A ride app shows at least two: time for a driver to reach the rider (pickup ETA) and time to the destination (trip ETA). A delivery app adds preparation time at the restaurant. Each has different features and different costs of error. Assume, for this design, both pickup and trip ETA, 500,000 requests per second at peak (every map refresh and every dispatch candidate needs one), and a 50 ms budget at p99.

Also ask how the number is used. Dispatch compares ETAs across candidate drivers, so ranking consistency matters more than absolute accuracy. Pricing and the rider's screen use the absolute value, so calibration matters there.

## Metrics

Offline, report mean absolute error (MAE) and the error percentiles (p50, p90). Users forgive an early arrival more easily than a late one, so track the late-arrival rate (actual exceeds predicted by more than, say, two minutes) separately. Relative error matters too: two minutes is small on a 40-minute trip and large on a 5-minute one, so report error by distance band.

Online, measure how often the shown ETA is met, cancellation rate after the ETA is shown, and complaints. For dispatch, measure total pickup time across the fleet.

## Data and labels

The label is the actual duration of a completed trip. It arrives only after the trip ends, which is fine for training but means evaluation lags by the trip length. Join features as they were at request time (a point-in-time join), not as they were when the trip finished; otherwise the model sees future traffic.

There is selection bias. Trips that were cancelled have no duration, and cancelled trips are more likely to be ones with long waits. Drivers also deviate from the suggested route. Record the route actually taken (via map matching on GPS traces) and decide whether the label refers to the planned route or the trip as driven.

## Features

- The routing engine's estimate: the sum of predicted segment travel times along the route. This is the strongest single feature.
- Live traffic: recent speeds per road segment from GPS pings, aggregated in short windows by a streaming job.
- Time: hour, day of week, holidays, local events.
- Route shape: distance, number of turns, number of traffic lights, highway share.
- Context: weather, city, pickup type (airport, stadium), driver's recent speed.

<div class="ex">
<div class="lab">Worked example</div>

**a correction beats a replacement**

A team trained a model to predict trip duration directly from origin, destination and time, and it lost to the routing engine on short urban trips. Switching to predicting the residual (actual minus engine estimate) let the model focus on what the engine missed, such as airport pickup delays and school-run traffic. MAE fell on every distance band, and the model needed far fewer features to do it.

</div>

## Model

A gradient-boosted tree or a small neural network predicts the residual on top of the routing estimate. Neural models handle many categorical features (city, segment IDs) through embeddings and can be trained with an asymmetric loss. To show a range, train quantile heads for p10 and p90 with the pinball loss. Graph neural networks over road segments can model how congestion spreads, at higher serving cost.

## Serving

~~~dg-flow ETA serving. Segment speeds are precomputed by a streaming job; the request path only looks up and scores.
GPS pings -(windowed aggregation)-> Segment speed store
ETA request -(route from routing engine)-> Feature assembly -(lookup speeds, context)-> *Residual model -(engine ETA + correction)-> Response
~~~

Dispatch asks for ETAs for many candidate drivers at once, so batch those into one call. Cache results for identical origin cell, destination cell and minute.

## Monitoring

Track MAE and late-arrival rate per city daily. Alert on stale segment speeds, because a stuck stream silently turns the model into a time-of-day average. Watch for unusual conditions (storms, road closures, large events), where error can triple; a fallback to the routing estimate is safer than a confident wrong correction.

## Trade-offs

- One global model shares data across cities but may miss local patterns; per-city models are the reverse. A global model with city embeddings is a common compromise.
- A range is more honest but harder to show in a product; many apps show a point estimate and use the upper quantile internally for promises.
- Richer graph models cost more per request; use them where error is largest.

<div class="take">
<div class="lab">Key takeaway</div>

predict a correction to the routing engine's estimate with fresh segment speeds and point-in-time labels, judge it by error percentiles and lateness by city and distance, and serve it cheaply with caching and batching.

</div>
