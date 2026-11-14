---
id: "06.04"
course: "ml-system-design"
module: "06"
slug: "fallbacks-and-model-rollouts"
title: "Fallbacks and model rollouts"
keyIdeas: ["Every ML call needs a defined behaviour when it is slow or failing: timeouts with defaults, cached or batch predictions, a simpler model, or a heuristic.", "Missing features are the most common partial failure. Decide per feature whether to impute, fall back, or skip the model, and make sure training saw the same defaults.", "Deploy models like code: versioned artifacts, shadow and canary stages, automatic rollback on health and guardrail checks.", "Keep the previous model warm and the feature schema backward compatible so rollback takes minutes."]
question: "Your checkout fraud model calls three feature services and a model server, with a 60 ms budget. Design the behaviour when (a) one feature service times out, (b) the model server is down, and (c) a new model version starts blocking twice as many payments as usual."
answer: "(a) Each feature call has its own timeout (for example 15 ms). If the device-reputation service times out, use the default value the model saw in training for \"unknown\" (trained explicitly by randomly masking the feature during training), log the event, and continue. If several critical features are missing, use a rule-based score instead. (b) A circuit breaker detects model-server failures and routes to a fallback: a small model embedded in the checkout service, or rules (amount over a limit from a new device triggers step-up authentication). Payments are not blocked solely because the model is down, unless the risk is very high. (c) Monitor block rate per minute against an expected band per model version; the canary stage receives 2% of traffic, and an automatic rollback fires if its block rate or the share of challenged customers deviates beyond a threshold compared with control. Rollback switches routing to the previous version, which stays loaded, within minutes."
sources: [{"title": "Google SRE Workbook, Canarying Releases", "url": "https://sre.google/workbook/canarying-releases/", "note": "Canary design and automatic rollback."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Rules on heuristics as fallbacks and on launch processes."}, {"title": "Martin Fowler, Circuit Breaker", "url": "https://martinfowler.com/bliki/CircuitBreaker.html", "note": "Stopping calls to failing dependencies."}]
practice: ["canary-mann-whitney", "consecutive-breach-alerts", "population-stability-index", "p50-p95-and-p99-inference-latency"]
tags: ["reliability", "graceful degradation"]
---

## Failure is normal

ML systems depend on more moving parts than ordinary services: feature stores, embedding indexes, model servers, upstream data pipelines. Some part is always slow, stale or broken somewhere. Design decides whether that means a slightly worse result or an outage. The Inference Engineering course covers health checks and draining for large-model servers (ieng 22, 24); this lesson covers fallback design for any model.

## A fallback ladder

~~~dg-flow Each rung is cheaper and less personalised than the one above. Requests fall to the next rung on timeout or error.
Full model + live features -(feature timeout)-> Full model + defaults -(model down)-> Cached or batch prediction -(no cache)-> Simple model or rules -(nothing)-> *Safe default
~~~

| Rung | Example for recommendations | Example for fraud
| --- | --- | --- |
| Full model | personalised ranking | real-time risk score
| Defaults for missing features | ranking without session features | score without device reputation
| Cached or batch prediction | yesterday's top-50 list for this user | last known account risk
| Simple model or rules | popular in region and category | amount and velocity rules
| Safe default | editorial picks | allow with step-up check

Define the ladder per surface. A safe default for recommendations is harmless; for fraud, "allow everything" may not be.

## Missing features

Partial failure is more common than total failure. If a feature is missing at serving time, the model receives whatever default the code supplies. If training never saw that default, predictions can be wild. Two habits make this safe:

- train with the same default, and randomly mask important features during training so the model learns to cope;
- monitor the missing rate per feature, and treat a jump as an incident.

## Timeouts and circuit breakers

Give each dependency a timeout well inside the overall budget, and fail fast. A **circuit breaker** stops calling a dependency after repeated failures and retries it after a cool-down, so a broken service does not slow every request.

## Deploying models safely

Treat a new model like a code release:

- **versioned artifact** from the registry (04.04), with its feature schema;
- **shadow mode**: run alongside production, compare outputs and latency without serving them;
- **canary**: a small slice of traffic, with automatic rollback if errors, latency or ML guardrails break (prediction rate, block rate, score distribution);
- **ramp**: increase in steps while watching dashboards;
- **rollback**: switch traffic back to the previous version, which stays loaded.

ML-specific health checks matter as much as error rates: a model can return 200 OK for every request while blocking every payment.

<div class="ex">
<div class="lab">Worked example</div>

**a silent model failure**

A recommendation model was deployed with a feature-name mismatch: the server sent `user_country` but the model expected `country`. Every request used the default, so all users got the same generic list. Error rates stayed at zero. A canary check comparing the diversity of recommendations across users with the previous version would have caught it in minutes; it was found two days later from an engagement drop.

</div>

## Compatibility

Rollback only works if the old model can still run. Keep feature changes backward compatible: add new features rather than changing the meaning of old ones, and keep old features available until no deployed model needs them. Version feature schemas together with models.

## Testing fallbacks

Fallbacks that are never exercised tend to be broken when needed. Inject failures on purpose in staging, and occasionally in production during quiet hours: time out a feature service, stop a model server, empty a cache. Check that each rung of the ladder engages, that latency stays within budget and that quality drops by the expected amount. Record the quality of each rung so the team knows what a degraded mode costs.

## Monitoring the fallbacks

Count how often each rung is used, per minute and per surface. A rise in "defaults for missing features" or "cached prediction" is often the first visible sign of an upstream incident, appearing before engagement metrics move. Alert on these rates as well as on errors, and include them in the canary checks for new models.

<div class="take">
<div class="lab">Key takeaway</div>

define a fallback ladder per surface, from full model to safe default, handle missing features with trained-in defaults, protect calls with timeouts and circuit breakers, and deploy models through shadow, canary and ramp with ML-aware automatic rollback.

</div>
