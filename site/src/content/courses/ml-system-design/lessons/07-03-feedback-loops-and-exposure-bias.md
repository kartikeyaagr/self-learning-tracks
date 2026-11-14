---
id: "07.03"
course: "ml-system-design"
module: "07"
slug: "feedback-loops-and-exposure-bias"
title: "Feedback loops and exposure bias"
keyIdeas: ["Deployed models change the data they will be trained on: recommenders decide what gets seen, fraud models decide which transactions complete, ranking decides which results get clicked.", "**Exposure bias** means feedback only exists for what the model chose to show; **position bias** means feedback depends on where it was shown. Both make the current model look better than alternatives.", "Feedback loops can narrow content (popularity concentration), hide errors (blocked cases never get labels) or amplify unfairness.", "Countermeasures: logged propensities and inverse propensity weighting, a small amount of randomised exploration, holdout groups, and diversity and fairness constraints."]
question: "A job-recommendation platform notices that jobs posted by large companies get most applications, and small employers complain about getting no candidates. Explain how a feedback loop could cause this and propose design changes, including how you would evaluate them."
answer: "The recommender is trained on applications to jobs it previously showed. Large-company jobs got early exposure (strong brand, more historical data), collected more applications, so the next model scored them higher and showed them more. Small employers' jobs rarely got shown, so they never gathered the data to prove relevance: a rich-get-richer exposure loop, plus position bias that inflates clicks on top slots. Changes: log the probability each job was shown at each position and train with inverse propensity weighting or a position-bias correction; reserve an exploration budget (for example 5% of slots via Thompson sampling) for jobs with few impressions, using content features (title, skills, location match) for priors; add an exposure-fairness constraint in re-ranking so relevant small-employer jobs get a minimum share. Evaluate with an A/B test measuring application quality and hires for candidates (primary), plus guardrails on employer-side coverage (share of jobs getting at least five qualified applicants) and candidate satisfaction, and use counterfactual estimates on the randomised slice to compare policies offline."
sources: [{"title": "Chaney, Stewart and Engelhardt, How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility (2018)", "url": "https://arxiv.org/abs/1710.11214", "note": "Simulation of recommender feedback loops."}, {"title": "Joachims, Swaminathan and Schnabel, Unbiased Learning-to-Rank with Biased Feedback (2016)", "url": "https://arxiv.org/abs/1608.04468", "note": "Propensity-weighted learning from clicks."}, {"title": "Bottou et al., Counterfactual Reasoning and Learning Systems (2013)", "url": "https://arxiv.org/abs/1209.2355", "note": "Off-policy estimation in a production ad system."}]
practice: ["epsilon-greedy-action-selection", "thompson-sampling-beta-bernoulli", "catalog-coverage-and-novelty", "softmax-boltzmann-exploration"]
tags: ["feedback loops", "exposure bias", "counterfactual evaluation"]
---

## The model shapes its own data

Supervised learning assumes training data is drawn independently of the model. In deployed systems that is false: what the model decides changes what gets logged. A recommender only learns about items it shows. A fraud model never sees outcomes for transactions it blocks. A search ranker sees more clicks on whatever it puts first. Over retraining cycles these effects compound into **feedback loops**.

~~~dg-flow A feedback loop. Exposure decides feedback, feedback decides the next model, the model decides exposure.
Model -(chooses what to show)-> Exposure -(users react only to shown items)-> Logged feedback -(retrain)-> Next model -(favours items with feedback)-> Model
~~~

## Common loops

| Loop | Mechanism | Symptom
| --- | --- | --- |
| Popularity concentration | popular items get shown, collect clicks, get shown more | coverage falls; new items starve
| Filter bubble | users shown narrow topics engage with them, model narrows further | diversity falls; long-term engagement drops
| Hidden errors | blocked or rejected cases never produce labels | model under-learns what it already blocks
| Position bias | top slots get clicks regardless of relevance | offline metrics favour the current ranking
| Self-fulfilling predictions | low-scored applicants get fewer opportunities, confirming low scores | fairness harms, stuck segments

## Exposure and position bias in data

Logged clicks answer "what did users click among what we showed, where we showed it?", not "what would users like?". Treating them as ground truth rewards the current policy and penalises alternatives that would have shown other items.

Corrections:

- **Log propensities**: record the probability that each item was shown at each position under the serving policy, including randomisation.
- **Inverse propensity weighting (IPW)**: weight each logged example by $1/p(\text{shown})$ when training or evaluating, so rarely shown items count more. Clip large weights to control variance.
- **Position-bias models**: estimate how likely each position is to be examined, from a small randomised experiment, and divide it out.

## Exploration

Without exploration, the system cannot learn about options it never tries. A small amount of controlled randomness, such as epsilon-greedy, Thompson sampling or randomly swapping adjacent positions, generates unbiased data at small cost. It also enables **off-policy evaluation**: estimating how a new policy would perform from data logged under the old one, using the logged propensities.

The exploration budget is a product decision. Even 1% of traffic, logged carefully, is often enough to debias evaluation and keep new items alive. Explore where the cost of a poor choice is small, such as lower positions on a page or users who browse widely, and never in places where one bad result does real harm.

<div class="ex">
<div class="lab">Worked example</div>

**an exploration slice that paid for itself**

A podcast app ranked episodes by predicted listens. New shows rarely surfaced, and the catalogue's long tail shrank each quarter. The team randomised the order of positions 3 to 6 for 2% of sessions and logged propensities. The data let them train a position-debiased model and evaluate new rankers offline with IPW. The debiased ranker raised listening from new shows by 18% with no loss in total listening time.

</div>

## Holdouts

A long-lived holdout group that never receives the model (or receives an older one) measures the cumulative effect of the system and of feedback loops, which short A/B tests miss. For fraud and risk models, a holdout of borderline cases that are approved with extra checks provides labels the model would otherwise hide (02.05).

## Fairness and long-term effects

Feedback loops can systematically disadvantage groups: sellers, creators or applicants who start with less exposure. Monitor exposure and outcomes by group, set constraints in re-ranking where appropriate, and evaluate on long horizons. Short-term engagement metrics rarely reveal these effects.

## Designing for it from the start

In an interview, mention feedback loops whenever the model decides what users see or which cases are acted on. A short paragraph covering propensity logging, a small exploration budget, a holdout and coverage guardrails shows you understand that the system and its data are intertwined.

<div class="take">
<div class="lab">Key takeaway</div>

deployed models shape their own training data through exposure, position and decisions; log propensities, debias with inverse propensity weighting, keep a small randomised exploration slice and long-lived holdouts, and watch coverage and fairness over long horizons.

</div>
