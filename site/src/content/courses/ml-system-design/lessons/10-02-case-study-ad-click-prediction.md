---
id: "10.02"
course: "ml-system-design"
module: "10"
slug: "case-study-ad-click-prediction"
title: "Case study: ad click prediction"
keyIdeas: ["Ad click prediction feeds an auction, so the model must be calibrated: the predicted probability multiplies the bid to decide which ad wins and what it pays.", "Data is huge and sparse (ad, user, context IDs and their crosses), and positives are rare, so negative downsampling with a calibration correction is standard.", "Ads and campaigns change constantly, so freshness matters; models are retrained or updated online every few hours or continuously.", "Evaluation uses normalised log loss and calibration offline, and revenue, advertiser value and user experience online."]
question: "Design the click-through-rate prediction system for an ads platform that runs 200,000 auctions per second, each with about 1,000 eligible ads, and must return a winner within 50 ms. Explain the labels, model, calibration, how predictions enter the auction, and how you keep the model fresh."
answer: "Retrieve a few hundred eligible ads by targeting and a light model, then score them with a CTR model. Labels are clicks on impressions, joined after a short attribution window; downsample negatives (for example keep 5%) and correct predicted probabilities back to the true base rate. Use a deep model with ID embeddings and explicit crosses (DCN or similar) over user, ad, advertiser, query and context features, plus historical CTR counters. Calibrate per segment, because expected value equals bid times pCTR; check reliability curves. Rank by bid times pCTR plus a quality term, with the price set by the auction rules. Train continuously or every few hours on fresh logs, warm-starting from the previous model. Evaluate with normalised cross-entropy and calibration offline, then online revenue, advertiser return and user metrics such as ad hides. Monitor calibration drift and feature delays."
rubric: ["Explains that predictions enter an auction, so calibration matters as much as ranking", "Describes a retrieval or filtering stage before the heavy CTR model", "Uses negative downsampling with a correct calibration adjustment", "Proposes sparse ID embeddings and feature crosses with counters for history", "Keeps the model fresh with frequent or online training and handles new ads", "Evaluates with log loss or normalised entropy and calibration, not only AUC", "Considers advertiser value and user experience alongside platform revenue", "Monitors calibration and feature pipeline delays in production"]
sources: [{"title": "He et al., Practical Lessons from Predicting Clicks on Ads at Facebook (2014)", "url": "https://quinonero.net/Publications/predicting-clicks-facebook.pdf", "note": "Negative downsampling, freshness and normalised entropy."}, {"title": "McMahan et al., Ad Click Prediction: a View from the Trenches (2013)", "url": "https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/", "note": "Online learning for click prediction at scale."}, {"title": "Wang et al., Deep and Cross Network for Ad Click Predictions (2017)", "url": "https://arxiv.org/abs/1708.05123", "note": "Learning feature crosses explicitly."}, {"title": "Wikipedia, Real-time bidding", "url": "https://en.wikipedia.org/wiki/Real-time_bidding", "note": "How auctions use predicted click and conversion rates."}]
practice: ["log-loss-binary-cross-entropy", "expected-calibration-error", "auc-roc", "uniform-negative-sampling"]
tags: ["CTR prediction", "calibration", "auctions", "online learning"]
---

## Requirements

Clarify the auction. In a typical design, each ad's expected value is bid times predicted click probability (for cost-per-click ads) or bid times click probability times conversion probability (for cost-per-conversion ads). The ranking and the price both depend on the probability, so it must be calibrated, not just well ordered. Assume 200,000 auctions per second, 1,000 eligible ads per auction after targeting, and 50 ms for the whole auction.

## Metrics

Offline: log loss, often normalised by the log loss of always predicting the average CTR (normalised entropy), and calibration (predicted versus observed CTR in buckets). AUC is useful but ignores calibration. Online: revenue per thousand impressions, advertiser return on spend, click-through, and user-side guardrails such as ad hides and session length.

## Data and labels

Each impression is an example; a click within a short window is a positive. CTR is often around 1%, so data is very imbalanced and huge. Keep all positives and a random fraction $w$ of negatives. The model then predicts $p'$ on the downsampled data; recover the true probability with
$p = p' / (p' + (1 - p')/w)$.
Forgetting this correction inflates every bid.

Conversions arrive later than clicks (hours to days). For conversion models, handle delayed labels by waiting, or by modelling the delay.

<div class="ex">
<div class="lab">Worked example</div>

**a calibration bug that cost money**

After a pipeline change, the negative sampling rate for one region changed from 10% to 5%, but the correction still used 10%. The model's predicted CTR for that region rose by roughly 80%. Ranking order within the region barely changed, so AUC stayed flat, but advertisers there were charged more per click than the clicks were worth. A daily calibration check by region would have caught it in hours.

</div>

## Features

- IDs: user, ad, campaign, advertiser, query, page, with learned embeddings.
- Historical counters: CTR of this ad, this advertiser, this user on this category, over several windows.
- Context: device, time, position, page type.
- Crosses: user interest by ad category, query by ad text.

## Model

Logistic regression on hashed crosses, trained online, was the classic baseline and remains a sensible fallback. Modern systems use deep models with embeddings and explicit crossing layers (DCN, DCN-v2) or attention over the user's recent behaviour. Multi-task heads can predict click and conversion together.

## Serving

~~~dg-flow Ad auction. Targeting and a light model cut candidates; the heavy model scores a few hundred.
Ad request -(targeting, budget, eligibility)-> ~1,000 ads -(light model)-> ~200 ads -(feature lookup)-> *CTR model -(calibrated pCTR)-> Auction -(bid x pCTR + quality)-> Winner and price
~~~

## Freshness

New ads appear constantly and old ones change. Train every few hours, or continuously, warm-starting from the previous weights. For new ads with no history, rely on advertiser and content features, and allow a little exploration so they gather data.

## Monitoring

Track calibration by segment hourly, normalised entropy on fresh data, feature freshness (a stale CTR counter quietly harms results), and revenue. Alert on sudden calibration shifts after deployments.

## Likely follow-up questions

- "Why not optimise AUC?" Because the auction uses the probability itself; a perfectly ordered but miscalibrated model sets wrong prices.
- "How do you handle position bias?" Include position as a feature in training and set it to a fixed value at serving, or model examination separately.
- "How do you protect users from too many low-quality ads?" Add a quality or user-cost term to the auction score and cap ad load per session.

## Trade-offs

- Model size versus latency: the heavy model can only score a few hundred ads.
- Freshness versus stability: frequent updates follow trends but risk bad pushes; keep automatic validation gates.
- Platform revenue versus advertiser return and user experience: optimising only revenue erodes the other two.

## A numeric sanity check

At 200,000 auctions per second and 200 ads scored per auction, the heavy model makes 40 million predictions per second. If one prediction costs about 1 million floating-point operations, that is $4 \times 10^{13}$ FLOP/s, which a few dozen modern accelerators can supply in theory, but memory bandwidth for embedding lookups is usually the real limit. This is why embedding tables are sharded across machines, user features are computed once per request rather than once per ad, and the light model exists at all.

<div class="take">
<div class="lab">Key takeaway</div>

ad CTR models feed an auction, so calibration is essential; downsample negatives and correct for it, learn sparse IDs and crosses, keep the model fresh, and monitor calibration by segment.

</div>
