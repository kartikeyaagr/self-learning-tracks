---
id: "10.04"
course: "ml-system-design"
module: "10"
slug: "case-study-product-recommendations-at-scale"
title: "Case study: product recommendations at scale"
keyIdeas: ["E-commerce recommendation covers several surfaces (home page, product page, cart, email) with different goals: discovery, substitutes, complements and reminders.", "Signals are implicit (views, add-to-cart, purchases) and sparse, with a long tail of products that have little data.", "A standard design has multiple candidate sources (item-to-item co-occurrence, two-tower embeddings, popularity by segment) and a ranker predicting purchase or revenue.", "Business constraints (availability, margins, repeated purchases, diversity) belong in re-ranking, and evaluation must use time-based splits."]
question: "Design product recommendations for a large online store with 100 million products and 200 million monthly users. Cover the home page and the \"customers also bought\" unit on product pages. Explain candidates, the ranker, cold start, business constraints and evaluation."
answer: "Treat the two surfaces differently. Product page: item-to-item candidates from co-purchase and co-view counts (normalised to avoid popularity bias) plus embedding neighbours, ranked for complements, excluding the same product from other sellers. Home page: candidates from a two-tower model on recent behaviour, recently viewed items, co-purchase neighbours of recent purchases, and trending items in the user's segments. A ranker predicts click, add-to-cart and purchase, combined into expected value with margin as an optional weight. Cold start: content embeddings from titles, images and categories for new products; popular-by-segment and onboarding signals for new users. Re-rank for availability, diversity, and to avoid recommending what was just bought (unless consumable). Evaluate with time-based splits (recall at k, NDCG), then experiments on revenue per visitor and conversion. Monitor coverage, popularity concentration and stock-outs."
rubric: ["Distinguishes surfaces and their goals (complements on product pages, discovery on the home page)", "Uses multiple candidate sources including item-to-item co-occurrence and two-tower retrieval", "Corrects for popularity bias in co-occurrence or retrieval training", "Ranks with a model of purchase or value, not only clicks", "Handles cold start for new products and new users with content features and popularity", "Applies business rules in re-ranking: availability, already purchased, diversity", "Evaluates with time-based splits and online revenue or conversion experiments", "Monitors catalogue coverage and concentration on popular items"]
sources: [{"title": "Wikipedia, Collaborative filtering", "url": "https://en.wikipedia.org/wiki/Collaborative_filtering", "note": "Item-based and user-based methods."}, {"title": "Wang et al., Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (2018)", "url": "https://arxiv.org/abs/1803.02349", "note": "Item embeddings from behaviour graphs with side information for cold start."}, {"title": "Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (2019)", "url": "https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/", "note": "Two-tower retrieval with logQ correction."}, {"title": "Naumov et al., Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM, 2019)", "url": "https://arxiv.org/abs/1906.00091", "note": "A reference ranking architecture with sparse embeddings."}]
practice: ["item-item-cf-scores", "implicit-als-user-factors", "in-batch-softmax-logq", "catalog-coverage-and-novelty"]
tags: ["e-commerce recommendation", "collaborative filtering", "two-tower", "cold start"]
---

## Requirements

Clarify the surfaces and their purpose:

- Product page "customers also bought": complements (a case for a phone) and close alternatives.
- Home page: discovery and reminders of items the user is considering.
- Cart and email: complements and replenishment.

Assume 100 million products, 200 million monthly users, page latency budget of 100 ms for recommendations, and stock and prices that change hourly.

## Metrics

Online: conversion rate and revenue per visitor attributed to the units, plus add-to-cart rate. Guardrails: diversity, share of recommendations for out-of-stock items, and complaints. Offline: recall at k and NDCG on the next purchases, with time-based splits (train on weeks 1 to 8, test on week 9) so that future data never leaks into training.

## Data

Implicit signals of increasing strength: views, add-to-cart, purchase. Purchases are sparse (most users buy rarely) and views are noisy. Weight signals by strength, and keep session order, since what someone viewed in the last ten minutes says much about their current intent.

## Candidates

- Item-to-item co-occurrence: products often bought or viewed together, normalised (for example with a lift or cosine measure) so that bestsellers do not appear for everything.
- Two-tower retrieval: a user tower over recent behaviour and profile, an item tower over item ID and content features, trained with in-batch negatives and a logQ correction for popularity.
- Recently viewed and abandoned-cart items.
- Popular and trending items within the user's segments (region, category interests).

<div class="ex">
<div class="lab">Worked example</div>

**everything recommends the same charger**

A store's first co-purchase recommender suggested the same cheap phone charger on 30% of product pages, because it appeared in so many baskets. Replacing raw co-occurrence counts with a lift score (how much more often two items co-occur than chance) and capping any single item's share of slots spread recommendations across many more products and raised add-to-cart from the unit by 9%.

</div>

## Ranking

A ranker (DLRM-style or gradient-boosted) scores a few hundred candidates with user, item, cross and context features. Multi-task heads predict click, add-to-cart and purchase; combine them as expected value, optionally weighted by margin. Include price relative to the user's usual spend, and whether the item is in stock.

## Cold start

- New products: content embeddings from title, description, images and category put them into the retrieval space on day one. Reserve a small share of slots for exploration.
- New users: popular items by region and entry point, then quick adaptation from session behaviour.

## Re-ranking

Remove out-of-stock items and items just purchased (unless consumable), deduplicate variants (sizes, colours, sellers), ensure category diversity, and apply sponsored-slot rules.

## Serving

~~~dg-flow Product page recommendations. Item neighbours are precomputed; session context re-ranks them live.
Purchase and view logs -(daily batch)-> Item neighbours and embeddings
Product page request -(fetch neighbours + user context)-> Candidates -(features)-> *Ranker -(stock, dedup, diversity)-> Top 12 items
~~~

## Monitoring

Track catalogue coverage (share of products ever recommended), concentration on top items, out-of-stock recommendations, latency, and per-surface conversion. Rebuild item neighbours daily and refresh embeddings weekly or faster.

## Likely follow-up questions

- "How do you avoid recommending a fridge to someone who just bought one?" Track purchase cycles per category; durable goods are suppressed for months, consumables are promoted near their repurchase time.
- "Should margin be in the objective?" It can be, as a weight, but test it carefully; optimising margin alone can reduce trust and conversion.
- "How would you evaluate the effect on long-tail sellers?" Track their share of impressions and sales by experiment arm.

## Trade-offs

- Relevance versus diversity and discovery.
- Short-term conversion versus long-term trust (recommending only expensive or sponsored items).
- Precomputed neighbours (cheap, stale) versus live retrieval (fresh, costly).

## A numeric sanity check

Item-to-item neighbours for 100 million products at 50 neighbours each are 5 billion pairs; stored as 8-byte IDs plus a 4-byte score, that is about 60 GB, which fits comfortably in a distributed key-value store and is cheap to read per page. Two-tower item embeddings at 128 dimensions in 8-bit form take about 13 GB, small enough to replicate the nearest-neighbour index widely. The ranker sees a few hundred candidates per request, so its cost is set by request volume, not catalogue size.

<div class="take">
<div class="lab">Key takeaway</div>

design per surface, combine normalised co-occurrence, embeddings and popularity as candidate sources, rank by expected purchase value, solve cold start with content features, and apply stock and repetition rules in re-ranking.

</div>
