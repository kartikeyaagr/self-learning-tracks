---
id: "09.02"
course: "ml-system-design"
module: "09"
slug: "case-study-dynamic-pricing"
title: "Case study: dynamic pricing"
keyIdeas: ["Dynamic pricing combines a demand model (how many people buy or ride at each price) with a decision rule that picks a price to meet a business goal under constraints.", "Observational data confounds price and demand, because prices were raised exactly when demand was high; experiments or randomised price variation are needed to estimate elasticity.", "The objective is rarely pure revenue: fairness, regulation, trust and marketplace balance (riders and drivers) constrain the price.", "Prices change behaviour on both sides of a marketplace, so evaluation needs experiments by region or time rather than by user."]
question: "Design a dynamic pricing system for a ride-hailing marketplace. Prices should respond to supply and demand in each area every few minutes. Explain how you estimate the effect of price, what you optimise, and how you test changes safely."
answer: "Split into forecasting and decision. Forecast requests and available drivers per area for the next 5 to 30 minutes from recent counts, time, weather and events. Estimate how conversion falls with price, using randomised small price variations because historical data confounds price with demand. The decision layer chooses a multiplier per area that keeps expected wait times under a target while respecting caps, smoothness between neighbouring areas and fairness rules. Evaluate with switchback experiments (alternating policies by area and time window), since user-level splits leak through shared supply. Monitor conversion, wait times, driver earnings, complaints and cap hits. Trade-offs: responsiveness versus price stability, short-term revenue versus long-term trust, and exploration cost versus better elasticity estimates."
rubric: ["Separates demand and supply forecasting from the pricing decision", "Explains why historical price data is confounded and proposes randomised variation to estimate elasticity", "States an objective beyond revenue, such as wait time or completed trips, with explicit constraints", "Includes caps, smoothing over time and space, and fairness or regulatory rules", "Uses switchback or region-level experiments because of marketplace interference", "Monitors both sides of the marketplace and user trust signals", "Discusses exploration cost and the stability versus responsiveness trade-off"]
sources: [{"title": "Wikipedia, Dynamic pricing", "url": "https://en.wikipedia.org/wiki/Dynamic_pricing", "note": "Background and examples across industries."}, {"title": "Wikipedia, Price elasticity of demand", "url": "https://en.wikipedia.org/wiki/Price_elasticity_of_demand", "note": "The quantity a pricing model must estimate."}, {"title": "Wikipedia, Multi-armed bandit", "url": "https://en.wikipedia.org/wiki/Multi-armed_bandit", "note": "Learning while choosing, for price exploration."}]
practice: ["linear-regression", "thompson-sampling-beta-bernoulli", "linucb-contextual-bandit", "b-test-significance"]
tags: ["dynamic pricing", "demand forecasting", "price elasticity", "bandits"]
---

## Requirements

Clarify the product. Ride-hailing pricing reacts within minutes to local imbalances; airline and hotel pricing plans over weeks against fixed inventory; retail pricing changes daily per product. Assume ride-hailing: a city is divided into small areas (hexagons a few hundred metres across), the price multiplier is updated every two minutes, and a quote must be returned in under 100 ms.

Ask about constraints early. Many cities cap surge multipliers, some regulators prohibit raising prices during emergencies, and prices must not vary by protected attributes. These shape the design more than the model does.

## Metrics

Choose one primary objective. Revenue alone would raise prices until volume collapses. Common choices are completed trips, or gross bookings subject to a wait-time target. Guardrails: rider conversion, median and p90 pickup wait, driver earnings per hour, cancellations, complaints, and the share of time at the cap.

## Data and labels

Two quantities are needed.

- Demand and supply forecasts: counts of app opens, requests and available drivers per area per minute. Labels are simply the future counts.
- Price response: the probability that a rider who sees a quote accepts it, as a function of price and context. This is the hard part.

Historical data is confounded. Prices were high exactly when demand was high, so a naive regression can show demand rising with price. To estimate elasticity, add small randomised variation to prices (for example, plus or minus 5% within allowed bounds) and fit conversion on that variation. This is an exploration cost worth paying.

<div class="ex">
<div class="lab">Worked example</div>

**a backwards elasticity**

A team fit acceptance rate against surge multiplier on six months of logs and found acceptance rising with price. The reason was that high multipliers occurred at concert endings and rainstorms, when riders had no alternative. After adding small random price offsets for two weeks and fitting only on that variation, acceptance fell by about 3% per 10% price rise, which matched intuition and made the pricing rule usable.

</div>

## Model

- Forecasting: gradient-boosted trees or a small sequence model on recent per-area counts, time features, weather and event calendars, predicting the next 5, 15 and 30 minutes.
- Conversion: a logistic model of acceptance given price, ETA, trip distance and context, fitted on randomised variation. Monotonic constraints keep acceptance from rising with price.
- Decision: for each area, choose the multiplier that brings expected demand at that price close to expected supply, so that pickup waits stay under target. A contextual bandit can keep exploring around the chosen price to refine elasticity.

## Serving

~~~dg-flow Pricing loop. Forecasts and multipliers update every two minutes; quotes read the latest multiplier.
Request and driver events -(per-area counts)-> Forecasts -(demand, supply)-> *Pricing optimiser -(capped, smoothed multipliers)-> Multiplier store
Quote request -(area lookup)-> Multiplier store -(price)-> Rider app
~~~

Smooth multipliers over time (limit the change per update) and across neighbouring areas, so that walking one street does not halve the price.

## Evaluation and experiments

User-level A/B tests are biased in a marketplace: if half the riders see lower prices, they take drivers from the other half. Use switchback experiments, alternating policies across areas and time windows, and compare averages over many windows. Measure marketplace outcomes, not just conversion.

## Monitoring

Track forecast error per area, multiplier distributions, time at cap, conversion and wait times. Alert when prices hit caps for long periods (supply shortage, not a pricing problem) or when conversion drops sharply after a model update.

## Trade-offs

- Fast response improves balance; stable prices build trust. Smoothing trades one for the other.
- Exploration improves elasticity estimates but costs revenue and can look unfair; keep it small and within policy.
- Higher prices can attract drivers in the medium term, which a short-horizon model ignores.

## Likely follow-up questions

- "How do you stop a feedback loop where high prices suppress demand, which then looks like low demand?" Forecast requests from app opens and searches, which happen before the price is seen, rather than from completed requests.
- "What if drivers move towards surge areas and the surge disappears?" That is the intended effect; model supply response over 10 to 20 minutes and avoid oscillation by smoothing.
- "How would you explain a price to a rider?" Show the multiplier and the reason (high demand nearby), and offer to notify them when it drops. Transparency reduces complaints more than small price changes do.
- "How do you protect against emergency pricing?" Hard caps and a manual override per area, enforced outside the model.

<div class="take">
<div class="lab">Key takeaway</div>

forecast demand and supply, estimate price response from randomised variation rather than confounded logs, choose prices under explicit caps and smoothing, and test with switchback experiments that respect marketplace interference.

</div>
