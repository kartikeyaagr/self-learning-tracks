---
id: "09.03"
course: "ml-system-design"
module: "09"
slug: "case-study-payment-fraud-detection"
title: "Case study: payment fraud detection"
keyIdeas: ["Payment fraud is a rare-event classification made at authorisation time, within a tight latency budget, where both blocking good customers and approving fraud cost money.", "Labels (chargebacks) arrive weeks or months late, and the model's own blocks remove labels, so evaluation needs a small randomised holdout of uncertain transactions.", "Velocity and graph features (how many cards on this device, how many accounts at this address) usually matter more than the model family.", "Fraudsters adapt within days of a change, so monitoring, fast retraining and rules for emerging patterns are part of the design."]
question: "Design a fraud detection system for card payments at an online payments company processing 5,000 transactions per second. The decision (approve, decline or step-up verification) must be made in under 100 ms. Cover labels, features, the decision policy and how you evaluate when labels arrive late."
answer: "Score each transaction with a gradient-boosted model on transaction, card, merchant, device and velocity features, plus graph features linking cards, devices, emails and addresses. Labels come from chargebacks and confirmed fraud reports, arriving 30 to 90 days later; train on matured windows and treat recent unlabelled data carefully. Map scores to actions with cost-based thresholds: decline when expected fraud loss exceeds the value of the sale, step-up (one-time code) in a middle band, approve otherwise. Keep a small randomised approval holdout among moderately risky transactions to measure fraud the model would have blocked. Serve with precomputed aggregates in a low-latency store updated by streaming. Monitor approval rate, step-up rate, early fraud signals and feature drift. Trade-offs: false declines versus fraud losses, and model freshness versus label maturity."
rubric: ["Clarifies volume, latency, actions and the cost of each error type", "Explains delayed and censored labels, and trains on matured windows", "Proposes velocity and graph features computed by streaming, with point-in-time correctness", "Uses a cost-based decision with a step-up band rather than a single threshold", "Evaluates with precision-recall at operating points and a randomised holdout for counterfactual labels", "Handles adversarial adaptation with monitoring, rules and frequent retraining", "Keeps the request path within latency using precomputed aggregates", "Discusses false declines and customer friction, not only fraud caught"]
sources: [{"title": "Stripe, A primer on machine learning for fraud detection", "url": "https://stripe.com/guides/primer-on-machine-learning-for-fraud-protection", "note": "Features, thresholds and the cost of false positives."}, {"title": "Wikipedia, Chargeback", "url": "https://en.wikipedia.org/wiki/Chargeback", "note": "Where fraud labels come from and why they are late."}, {"title": "imbalanced-learn documentation", "url": "https://imbalanced-learn.org/stable/", "note": "Resampling methods for rare classes."}]
practice: ["precision-recall-curve-and-average-precision", "sigmoid-focal-loss", "delayed-label-monitoring", "isolation-forest"]
tags: ["fraud detection", "class imbalance", "delayed labels", "adversaries"]
---

## Requirements

Clarify the decision and its costs. At authorisation the system can approve, decline, or step up (ask for a one-time code or 3-D Secure). Approving fraud costs the transaction value plus a fee; declining a good customer loses the sale and often the customer. Assume 5,000 transactions per second, a fraud rate near 0.1%, and 100 ms for the whole decision, leaving perhaps 30 ms for the model.

## Metrics

Accuracy is meaningless at 0.1% positives. Use precision-recall curves and report recall at a fixed false-positive rate. The business metrics are fraud loss in money (basis points of volume), false decline rate, and step-up rate. Weight errors by amount: missing a large fraud matters more than a small one.

## Data and labels

Labels come from chargebacks and from customers reporting fraud. They arrive 30 to 90 days after the transaction, and some fraud is never reported. Two consequences follow.

- Train on windows that have matured (at least 60 to 90 days old), and treat recent data as unlabelled rather than as non-fraud.
- Declined transactions never get a label, so the training data is shaped by the previous model. Keep a small randomised holdout: approve a tiny fraction of transactions in the medium-risk band that the model would decline, and use their outcomes to estimate true fraud rates.

<div class="ex">
<div class="lab">Worked example</div>

**the model that grew blind**

A payments team retrained monthly on approved transactions only. Each model blocked a pattern, so the next training set had no examples of it, and the following model stopped recognising it. Fraud of that type returned within two retrains. Adding a 0.5% randomised approval band for medium scores restored the labels and stopped the cycle, at a measured cost far below the losses.

</div>

## Features

- Transaction: amount, currency, merchant category, time, card-not-present flags.
- Velocity: counts and sums for this card, device, IP and email over the last minute, hour and day.
- Mismatch: billing country versus IP country, new device, new shipping address.
- Graph: number of cards seen on this device, accounts sharing a shipping address, links to known fraud.
- History: the customer's typical amounts and merchants.

Velocity and graph features are computed by a streaming job and read from a low-latency store; they must be point-in-time correct in training.

## Model

Gradient-boosted trees are the standard choice: strong on tabular features, fast and interpretable enough for dispute teams. Train with class weights rather than heavy resampling, and calibrate scores, because the decision layer uses probabilities. An unsupervised anomaly score (an isolation forest on recent behaviour) can be added as a feature to catch new patterns. Rules remain for known patterns and immediate responses.

## Serving and decision

~~~dg-flow Authorisation path. Aggregates are precomputed; the request only looks up, scores and decides.
Payment events -(streaming aggregates)-> Feature store
Authorisation request -(lookup)-> Feature store -(features)-> *Fraud model -(probability)-> Decision policy -(approve, step-up, decline)-> Issuer
~~~

Decline when expected loss (probability times amount plus fees) exceeds the expected margin; step up in a middle band; approve otherwise. Thresholds can vary by merchant risk tolerance.

## Monitoring

Watch approval, decline and step-up rates hourly, score distributions, and early signals such as customer disputes and step-up failure rates, which arrive long before chargebacks. Alert on sudden shifts by merchant or region. Retrain weekly or faster and add rules within hours when a new attack appears.

## Trade-offs

- False declines versus fraud losses: set by money, not by a fixed recall target.
- Fresh training data versus mature labels: recent data reflects current attacks but is incompletely labelled.
- Step-up reduces fraud but adds friction and abandonment.

## Likely follow-up questions

- "A merchant says too many good customers are declined. What do you check?" False declines estimated from the randomised band and from customers who retry and succeed with step-up, broken down by that merchant's features.
- "How do you handle a brand-new merchant with no history?" Fall back to category-level and network-wide features, use tighter step-up bands at first, and relax them as history builds.
- "Would a deep sequence model over a card's transactions help?" Often a little, as an extra feature; the latency budget and the need for explanations usually keep trees as the final scorer.
- "How do you explain a decline?" Use reason codes from the top feature contributions, which dispute teams and merchants need.

<div class="take">
<div class="lab">Key takeaway</div>

treat fraud as a cost-weighted, rare-event decision with late and censored labels; rely on streaming velocity and graph features, map calibrated scores to approve, step-up and decline, and keep a randomised holdout so the model can still see what it blocks.

</div>
