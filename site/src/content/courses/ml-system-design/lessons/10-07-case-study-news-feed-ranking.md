---
id: "10.07"
course: "ml-system-design"
module: "10"
slug: "case-study-news-feed-ranking"
title: "Case study: news feed ranking"
keyIdeas: ["Feed ranking orders posts from friends, groups, pages and recommendations by a value model that combines many predicted actions with tuned weights.", "Candidate inventory is mostly given (posts from connections since the last visit) plus recommended content, so ranking quality and objectives matter more than retrieval.", "Integrity (misinformation, harmful content, engagement bait) is handled with demotions and filters inside the ranking stack.", "Weights in the value model are policy choices; they are tuned by long experiments on retention and user-reported satisfaction."]
question: "Design the ranking system for a social network's home feed. Each user has about 1,500 eligible posts since their last visit from friends, groups and pages, plus recommended posts. Explain the objective, the model, integrity handling, freshness and how you evaluate changes."
answer: "Gather inventory: posts from connections since last visit, unseen older high-value posts, and recommended posts from a retrieval model. A light ranker trims to about 500; a heavy multi-task model predicts like, comment, share, long view, hide and report, using attention over the user's recent interactions. A value model combines these as a weighted sum, with weights favouring meaningful interactions (comments, shares between friends) and penalising negative feedback. Integrity classifiers demote or remove misinformation, borderline content and engagement bait. Re-rank for diversity of authors and content types, and freshness. Evaluate with experiments on retention, meaningful interactions and survey-based satisfaction, with long-running holdouts. Monitor integrity prevalence, creator distribution and training-serving skew. Trade-offs: engagement versus wellbeing and integrity, and friends' content versus recommendations."
rubric: ["Describes inventory sources including connections and recommended content", "Uses a multi-stage ranker with a multi-task heavy model", "Defines a value model with explicit weights, including negative signals", "Integrates integrity classifiers as demotions or filters inside ranking", "Re-ranks for diversity, freshness and author variety", "Evaluates with long-running experiments and satisfaction surveys, not only engagement", "Addresses feature freshness and training-serving skew", "Discusses trade-offs between engagement, wellbeing and integrity"]
sources: [{"title": "Zhou et al., Deep Interest Network for Click-Through Rate Prediction (2017)", "url": "https://arxiv.org/abs/1706.06978", "note": "Attending over a user's recent behaviour when scoring an item."}, {"title": "Zhao et al., Recommending What Video to Watch Next: A Multitask Ranking System (2019)", "url": "https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/", "note": "Multi-task ranking with competing objectives."}, {"title": "Wikipedia, Recommender system", "url": "https://en.wikipedia.org/wiki/Recommender_system", "note": "Overview of approaches and evaluation."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Objectives, launches and monitoring."}]
practice: ["ndcg-at-k", "mmr-diversity-rerank", "training-serving-skew", "point-in-time-feature-join"]
tags: ["feed ranking", "multi-objective", "integrity", "value model"]
---

## Requirements

Clarify inventory and constraints. The feed shows friends' posts, group and page posts, and recommended content from outside the user's network. Assume 1,500 eligible connected posts per visit plus a few hundred recommended, a few visits a day, and 300 ms to return the first screen.

## Metrics

Online: daily and weekly active users, meaningful interactions (comments and shares between friends), time spent, hides and reports, and survey answers to "was this worth your time?". Integrity: prevalence of violating content among views, measured by sampling. Offline: per-task AUC and calibration, and NDCG of the value score against logged engagement.

## Inventory

- Connected posts since the last visit.
- Older connected posts not yet seen that had strong engagement ("you may have missed").
- Recommended posts from a two-tower retrieval model over the user's interests.

## Model

- Light ranker: a small model on precomputed features cuts 2,000 to about 500.
- Heavy ranker: a multi-task network with heads for like, comment, share, click, long view, hide and report. Attention over the user's recent interactions (DIN-style) captures current interests.
- Value model: $V = \sum_i w_i \hat{p}_i$, with negative weights for hide and report. Choosing the weights is a product decision: a high weight on comments promotes discussion but also promotes provocative posts.

<div class="ex">
<div class="lab">Worked example</div>

**weights that rewarded outrage**

After a team raised the weight on comments to promote conversation, comments rose 8%. Sampling showed that much of the gain came from heated, divisive threads, and survey satisfaction fell. Adding a predicted-report penalty, down-weighting comments on posts from non-friends, and adding a satisfaction-survey head to the value model kept most of the conversation gain without the drop in satisfaction.

</div>

## Integrity in ranking

Integrity classifiers score each candidate for misinformation (often linked to fact-checker ratings), borderline content, engagement bait ("comment yes if you agree") and clickbait. Violating content is removed upstream; borderline content is demoted by a multiplier in the value score. Integrity models must be fast enough to run on every candidate, or their scores cached per post.

## Re-ranking

Avoid long runs of posts from one author or of one type (all videos), mix connected and recommended content within limits, boost freshness for time-sensitive posts, and respect user controls such as "see first" and snoozes.

## Serving

~~~dg-flow Feed ranking. Post features are cached; user features come from a feature store; the first screen is ranked first.
Post events -(streaming counters)-> Post feature cache
Feed request -(inventory: connected + recommended)-> ~2,000 posts -(light ranker)-> ~500 -(heavy multi-task model)-> *Value model + integrity demotions -(re-rank: diversity, freshness)-> Feed
~~~

Rank the first screen with priority and the rest asynchronously as the user scrolls.

## Monitoring

Watch training-serving skew (features computed differently online and offline), counter freshness, per-head calibration, integrity prevalence, and the distribution of impressions across authors. Changes to weights should go through the same experiment process as model changes.

## Likely follow-up questions

- "How do you rank posts from a new friend with no interaction history?" Use the friendship's recency and mutual friends as features, and give new connections a temporary boost.
- "How do you avoid a feed full of recommendations?" Cap the recommended share and measure its effect on retention separately for light and heavy users.
- "How do you evaluate a change that may only matter over months?" Keep a long-running holdout and track retention and satisfaction surveys there.

## Trade-offs

- Engagement versus wellbeing and integrity: explicit negative weights and survey heads are the levers.
- Connected content versus recommendations: recommendations add engagement but can crowd out friends.
- Model size versus latency, solved by staged ranking and caching.

## A numeric sanity check

If 1 billion daily users open the feed three times and each request scores 2,000 candidates with the light model and 500 with the heavy one, that is 6 trillion light and 1.5 trillion heavy predictions a day, about 17 million heavy predictions per second on average. Post features are shared across all users who see a post, so caching them per post rather than per request is the single largest saving. Computing user features once per request, not once per candidate, is the second.

<div class="take">
<div class="lab">Key takeaway</div>

rank feed inventory with a multi-stage, multi-task model combined into a value score whose weights are product choices, build integrity demotions into ranking, re-rank for diversity, and judge changes by long-term retention and satisfaction.

</div>
