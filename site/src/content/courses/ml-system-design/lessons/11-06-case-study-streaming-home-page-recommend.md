---
id: "11.06"
course: "ml-system-design"
module: "11"
slug: "case-study-streaming-home-page-recommendations"
title: "Case study: streaming home-page recommendations"
keyIdeas: ["A streaming service's home page is a two-dimensional layout: rows (themes) of titles, so the design ranks titles within rows and rows on the page.", "The catalogue is small (thousands to tens of thousands of titles) but each viewing decision matters; long-term retention is the true objective.", "Presentation matters: which artwork or trailer is shown for a title can change play rates, and can be personalised with bandits.", "Evaluation relies on experiments with retention and viewing hours, plus offline replay with care for presentation bias."]
question: "Design the home page for a subscription video streaming service with 200 million members and 20,000 titles. The page has rows such as \"Continue watching\", \"Because you watched X\", genres and \"Trending now\". Explain how you choose titles within rows, which rows to show and in what order, artwork selection, and how you evaluate against retention."
answer: "Generate candidate rows from templates: continue watching, because you watched (seeded from recent titles), personalised genres, trending, new releases, top picks. Fill each row by ranking eligible titles with a model predicting the probability of playing and of watching substantially (for example, completing an episode), using member history, title features and context such as device and time. Rank rows with a page-level model that predicts the value of each row in each position, with constraints: deduplicate titles across rows, limit similar rows, and keep continue watching near the top. Personalise artwork per member with a contextual bandit over a few candidate images per title, rewarded by quality plays rather than clicks. Evaluate offline with replay on logged pages, correcting for presentation bias, and online with experiments on viewing hours, retention and the share of members who find something to watch. Monitor diversity, catalogue coverage, new-release reach and cold-start member engagement."
rubric: ["Treats the page as rows of titles and ranks both titles within rows and rows on the page", "Generates rows from templates with personalised seeds", "Ranks titles with a model of meaningful viewing, not just clicks", "Deduplicates titles across rows and enforces row diversity constraints", "Personalises artwork with a bandit rewarded by quality plays", "Evaluates with experiments on retention and viewing, with long-term holdouts", "Handles presentation bias in offline evaluation", "Addresses new members and new titles"]
sources: [{"title": "Wikipedia, Recommender system", "url": "https://en.wikipedia.org/wiki/Recommender_system", "note": "Overview of methods and evaluation."}, {"title": "Wikipedia, Multi-armed bandit", "url": "https://en.wikipedia.org/wiki/Multi-armed_bandit", "note": "For artwork and exploration choices."}, {"title": "Naumov et al., Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM, 2019)", "url": "https://arxiv.org/abs/1906.00091", "note": "A reference architecture for ranking with sparse features."}, {"title": "Covington, Adams and Sargin, Deep Neural Networks for YouTube Recommendations (2016)", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", "note": "Candidate generation and ranking for video."}]
practice: ["mmr-diversity-rerank", "linucb-contextual-bandit", "ndcg-at-k", "catalog-coverage-and-novelty"]
tags: ["page construction", "rows", "artwork", "contextual bandits"]
---

## Requirements

Clarify the setting. A subscription service earns from members staying, not from each click; the goal is that members find something worth watching quickly and keep subscribing. Assume 200 million members, a catalogue of 20,000 titles that varies by region and licence, a home page of about 40 rows with 20 to 40 titles each, rendered within 300 ms, on TVs, phones and browsers.

## Metrics

Online: retention (cancellations over months), viewing hours, the share of sessions that start a play within a few minutes, and the share of plays that last beyond a few minutes ("quality plays"). Guardrails: catalogue coverage and diversity. Offline: per-row ranking metrics such as NDCG on logged plays, and page-level replay metrics.

## Data and labels

Signals: plays, play duration and completion, ratings, adding to a list, browsing without playing. Labels for the title ranker are "played and watched at least N minutes" after an impression. Log every impression with its row and position, because presentation strongly affects what gets played.

## Rows

Candidate rows come from templates:

- Continue watching (ordered by likelihood of resuming).
- Because you watched X, seeded from recent titles the member finished.
- Personalised genres or themes ("Dark crime dramas").
- Trending in your country, new releases, top picks.

## Ranking titles within rows

A ranker scores eligible titles for each row with member features (history embeddings, preferred genres, viewing times), title features (genre, cast, popularity, freshness, embeddings) and context (device, time of day, country). The objective is a quality play, not a click; a misleading trailer produces clicks and early abandonment.

## Ranking rows on the page

A page-level model scores each row in each position: the expected value of a member engaging with that row there. Then a greedy page builder fills positions under constraints: no title repeated across rows near the top, no two very similar rows in a row, continue watching near the top on most devices, and a minimum share of discovery rows. Device matters: a TV shows fewer titles per row before scrolling.

<div class="ex">
<div class="lab">Worked example</div>

**the page that repeated itself**

A team's first page builder chose rows independently, and members with a strong interest in crime dramas saw the same five titles in four of the first six rows. Adding a cross-row deduplication rule and a penalty for row similarity increased distinct titles in the first screen from 21 to 38, and raised the share of sessions starting a play within two minutes.

</div>

## Artwork personalisation

Each title has several candidate images. A contextual bandit chooses the image per member using their history (a member who watches comedies may respond to an image showing a comedic actor). Reward quality plays, not clicks, so that misleading images are not favoured. Keep a small exploration rate.

## Serving

~~~dg-flow Home page construction. Title scores per row are computed, then the page builder picks rows under constraints.
Member history -(features)-> Row templates -(candidate rows)-> *Title ranker per row -(scored rows)-> Page builder -(dedup, diversity, device rules)-> Artwork bandit -(images)-> Home page
~~~

Precompute most of the page offline per member and refresh key rows (continue watching, trending) at request time.

## Evaluation

Offline replay on logged pages is biased: members could only play what was shown, in the positions shown. Use inverse propensity weighting where impressions were randomised, or treat offline metrics as a filter before experiments. Online experiments should run for weeks, with retention measured on a longer holdout.

## Likely follow-up questions

- "How do you handle a new member?" Ask for a few titles they like at sign-up, then use popular and region-trending titles with fast adaptation from the first plays.
- "How do you promote a new original series?" Give it guaranteed exploration slots and measure quality plays, not just starts.

## Trade-offs

- Exploitation versus discovery of the catalogue.
- Row personalisation versus a stable, learnable page layout.
- Artwork that attracts clicks versus artwork that represents the title honestly.

## A numeric sanity check

With 20,000 titles and 200 million members, scoring every title for every member is $4 \times 10^{12}$ scores, too many to do on each visit but manageable in a daily batch spread over a large cluster, especially as many titles are not available in a given region. At request time only a few rows are re-scored, and the page builder works on about 40 candidate rows, which is cheap. Precomputed pages per member take a few kilobytes each, so storing all of them is under a terabyte.

<div class="take">
<div class="lab">Key takeaway</div>

build the home page as rows of ranked titles, choose and order rows with a page-level model under deduplication and diversity rules, personalise artwork with bandits rewarded by quality plays, and judge everything by retention.

</div>
