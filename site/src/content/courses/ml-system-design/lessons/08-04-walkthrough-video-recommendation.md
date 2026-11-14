---
id: "08.04"
course: "ml-system-design"
module: "08"
slug: "walkthrough-video-recommendation"
title: "Walkthrough: video recommendation"
keyIdeas: ["Video recommendation is the canonical multi-stage ranking problem: huge corpus, tight latency, many engagement signals and strong feedback loops.", "Frame the objective around satisfaction, not clicks: expected watch time, completions, likes and surveys combined, with guardrails against clickbait and harmful content.", "Use multi-source candidate generation, a multi-task ranker and a re-ranking layer for diversity, freshness and policy.", "Evaluate offline with debiased metrics and online with long-running A/B tests on retention; plan for cold start and exploration."]
question: "In an interview you are asked to design the home-feed recommender for a video platform. The interviewer wants you to go deep on the ranking objective. What do you propose and how would you justify it?"
answer: "Propose a multi-task ranker with heads for click, watch time (or completion fraction), like, share, subscribe and negative feedback (dislike, \"not interested\", report), trained on logged impressions with position as a training-only feature or a separate shallow position tower to absorb position bias. Combine heads into a value score such as expected watch time plus weighted positive actions minus weighted negative ones, for example score = E[watch seconds] + a·P(like) + b·P(share) − c·P(not interested). Justify: clicks alone reward clickbait; watch time aligns better with satisfaction but favours long videos, so normalise by video length or add completion; explicit signals capture quality; negative signals protect trust. Choose weights by online experiments measuring long-term retention and user surveys, and keep guardrails on report rates and creator diversity. Mention calibration of each head, since they are combined linearly."
rubric: ["Clarifies surfaces, scale, latency and the satisfaction goal", "Uses multi-source candidate generation with high recall and explains why", "Proposes a multi-task ranker and a justified value function, not clicks alone", "Handles position and exposure bias in training and evaluation", "Adds re-ranking for diversity, freshness and policy filtering", "Evaluates with retrieval recall, debiased offline metrics and long A/B tests on retention", "Addresses cold start for users and videos with exploration", "Discusses latency budget, caching and fallbacks"]
sources: [{"title": "Covington, Adams and Sargin, Deep Neural Networks for YouTube Recommendations (2016)", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", "note": "Candidate generation and ranking with watch-time weighting."}, {"title": "Zhao et al., Recommending What Video to Watch Next: A Multitask Ranking System (2019)", "url": "https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/", "note": "Multi-task ranking and position-bias correction."}, {"title": "Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (2019)", "url": "https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/", "note": "Two-tower retrieval."}]
practice: ["in-batch-softmax-logq", "ndcg-at-k", "mmr-diversity-rerank", "catalog-coverage-and-novelty", "popularity-baseline-leave-last-out"]
tags: ["recommendation", "interview walkthrough"]
---

## The prompt

"Design the recommendation system for a video platform's home feed." Every interviewer has seen this question, so a structured, numeric answer stands out. Module 10 includes the full case study; here is how to present it in 45 minutes.

## 1. Clarify

- Surface: home feed in the app, 20 videos per page, infinite scroll.
- Scale: 200M daily users, 5 home loads a day each, so about 11,600 per second on average, 30,000 at peak.
- Corpus: 500M videos, 500,000 new each day.
- Latency: 200 ms for the page, with about 150 ms for recommendation.
- Goal: long-term satisfaction and retention, not just clicks.

## 2. Frame

Rank videos for a user by expected value. Online metrics: daily active users, watch time per user, 28-day retention; guardrails: report rate, creator diversity, latency. Offline metrics: retrieval recall@500, ranking NDCG@20 with watch-time gains, calibration per head.

## 3. Data and labels

Impression logs with position, watch events (seconds, completion), likes, shares, subscribes, "not interested" and reports. Labels per impression per head. Time-based splits. Sample negatives from shown-but-not-watched impressions.

## 4. Candidates and ranking

~~~dg-flow Home-feed pipeline with budgets. The ranker's objective is the deep-dive topic.
Request -(parallel sources, 40 ms)-> 2,000 candidates -(filters)-> Light ranker -(15 ms)-> 400 -(multi-task ranker, 60 ms)-> *50 -(re-rank, 5 ms)-> 20 shown
~~~

Candidate sources: two-tower retrieval (user tower from watch history and context, video tower from content and engagement), subscriptions, co-watch neighbours of recent videos, trending by region and language, and an exploration pool of new videos.

Ranker: multi-task network over user features, video features, context and cross features, with heads for click, watch time, like, share, subscribe and negative feedback. Value score combines them with weights tuned by experiment.

## 5. Bias and feedback loops

Position bias: include position as a feature during training (set to a constant at serving) or use a shallow position tower. Exposure bias: log propensities, reserve a small exploration slice, evaluate with debiased metrics (07.03). Popularity concentration: diversity re-ranking and coverage guardrails.

## 6. Serving

Video embeddings and candidate lists precomputed; user embedding computed per request from recent history (streaming features). Ranker on GPU or optimised CPU with batching. Cache candidates for a few minutes between page loads. Fallbacks: if the ranker times out, use the light ranker's order; if retrieval fails, trending in region.

## 7. Evaluation and rollout

Offline: retrieval recall, NDCG with watch-time gains on debiased data, slices for new users and new videos. Online: A/B tests run for at least two weeks with retention as a key metric, and a long-term holdback. Interleaving can screen ranker changes quickly.

## 8. Monitoring and iteration

Score distributions per head, calibration, coverage, share of new videos, report rate, latency. Retrain the ranker daily and refresh embeddings for new videos hourly.

<div class="ex">
<div class="lab">Worked example</div>

**a strong deep-dive answer**

"Clicks alone would favour thumbnails over substance, so I would rank by expected watch time plus weighted likes and shares, minus weighted 'not interested' signals, with weights tuned on 28-day retention in experiments. Watch time favours long videos, so I'd use completion fraction alongside seconds watched."

</div>

## Likely follow-up questions

Expect probes on four points. "How do you handle a brand-new video?": content embeddings from title, thumbnail and audio place it in the retrieval space, and an exploration slot gives it a small guaranteed audience whose early engagement feeds back within hours. "How do you handle a new user?": start from popular and locally trending items, ask for a few interests at sign-up, and lean on session signals, which become informative after a handful of plays. "Why not optimise clicks?": clicks reward misleading thumbnails; completion, satisfaction surveys and return visits track what users value. "How would you know the system is making things worse over months?": keep a small long-running holdout that gets a simpler policy and compare retention and diversity of consumption against it.

## A numeric sanity check

Suppose 100 million daily users each open the home page three times and each page ranks 500 candidates. That is $1.5 \times 10^{11}$ ranker scores a day, or about 1.7 million per second on average and several times that at peak. This is why the heavy ranker sees only hundreds of items per request, why features are cached, and why retrieval must be an approximate nearest-neighbour lookup rather than a scan.

<div class="take">
<div class="lab">Key takeaway</div>

present video recommendation as a multi-stage pipeline with multi-source candidates, a multi-task ranker on a satisfaction-based value score, re-ranking for diversity and freshness, explicit handling of position and exposure bias, and evaluation tied to long-term retention.

</div>
