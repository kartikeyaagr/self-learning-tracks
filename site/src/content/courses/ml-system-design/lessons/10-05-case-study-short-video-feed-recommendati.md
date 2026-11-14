---
id: "10.05"
course: "ml-system-design"
module: "10"
slug: "case-study-short-video-feed-recommendation"
title: "Case study: short-video feed recommendation"
keyIdeas: ["A short-video feed shows one item at a time and gets a strong signal after every swipe, so the system can adapt within a single session.", "Watch signals need normalising: completion of a 10-second clip means less than completion of a 60-second one, and replays and shares are stronger than views.", "New creators and new videos need deliberate exploration, usually as staged distribution to growing audiences based on early engagement.", "The objective must look beyond time spent in one session, towards satisfaction and return visits, with diversity and safety in re-ranking."]
question: "Design the recommendation system for a full-screen short-video feed with 500 million daily users, 50 million new videos a day, and an average session of 40 videos. Explain the signals, candidate generation, ranking, exploration for new videos, and how you avoid optimising only for time spent."
answer: "Treat every swipe as feedback. Normalise watch time by video length (completion rate, rewatches), and add explicit signals: likes, shares, follows, comments, and negative ones such as quick skips and \"not interested\". Candidates come from a two-tower model on the user's long-term and in-session history, followed creators, trending items by region, and a fresh-content pool. A multi-task ranker predicts completion, like, share, follow and skip, combined into a value score tuned by experiments on retention. Re-rank for diversity of creators and topics, safety and fatigue. Fetch the next few videos in small batches so the session state updates after each swipe. Exploration: give each new video a small initial audience; promote to larger pools if engagement beats its peers. Evaluate with experiments on day-7 retention and satisfaction surveys, with long-running holdouts. Monitor creator concentration, new-video reach and harmful-content prevalence."
rubric: ["Explains the differences of a single-item swipe feed (dense feedback, in-session adaptation)", "Normalises watch signals by video length and includes explicit and negative signals", "Uses several candidate sources including in-session behaviour and fresh content", "Ranks with multi-task heads combined into a value score tuned for retention or satisfaction", "Designs staged exploration for new videos and creators", "Re-ranks for diversity, safety and fatigue", "Evaluates beyond time spent, with retention, surveys and long-term holdouts", "Handles serving with small batches and fast session-state updates"]
sources: [{"title": "Covington, Adams and Sargin, Deep Neural Networks for YouTube Recommendations (2016)", "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/", "note": "Candidate generation and ranking for video."}, {"title": "Zhao et al., Recommending What Video to Watch Next: A Multitask Ranking System (2019)", "url": "https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/", "note": "Multi-task ranking and position bias."}, {"title": "Wikipedia, Multi-armed bandit", "url": "https://en.wikipedia.org/wiki/Multi-armed_bandit", "note": "Balancing exploration and exploitation."}]
practice: ["in-batch-softmax-logq", "ndcg-at-k", "mmr-diversity-rerank", "thompson-sampling-beta-bernoulli"]
tags: ["short-video feed", "sequential recommendation", "exploration", "watch signals"]
---

## Requirements

This differs from a long-form home page (see 08.04). The feed shows one video at a time, full screen, and the user swipes within seconds. Each swipe is feedback. Assume 500 million daily users, 40 videos per session, 50 million new videos a day, and a budget of 100 ms to return the next batch of videos.

At 500 million users times a few sessions times 40 videos, the system serves tens of billions of views a day; the ranker must be efficient, and most work must be cached.

## Signals and labels

- Watch time, normalised: completion rate, and whether the video was replayed. A 15-second video watched twice is a stronger signal than a 3-minute video abandoned halfway.
- Explicit positives: like, share, comment, follow the creator, save.
- Negatives: skip within one or two seconds, "not interested", report.

Combine these in a multi-task model rather than one blended label, so that weights can be tuned later without retraining.

## Candidates

- Two-tower retrieval with a user tower over long-term interests and the current session's last 20 videos.
- Followed creators' new videos.
- Trending in the user's region and language.
- A fresh pool of recently uploaded videos for exploration.
- Similar videos to ones just liked (item-to-item embeddings).

## Ranking

A multi-task ranker predicts completion, like, share, follow, skip and report for a few hundred candidates. The value score is a weighted sum, e.g. $V = w_1 \hat{p}_{\text{complete}} + w_2 \hat{p}_{\text{share}} + w_3 \hat{p}_{\text{follow}} - w_4 \hat{p}_{\text{skip}}$, with weights set by experiments on retention rather than by guesswork.

<div class="ex">
<div class="lab">Worked example</div>

**a feed that narrowed too fast**

A model update raised average session length by 6%, but a month later users in that arm reported the feed felt repetitive and day-30 retention was slightly lower. Sessions had narrowed to two or three topics. Adding a topic-diversity constraint in re-ranking and a small exploration slot every eight videos recovered the retention loss while keeping most of the session gain.

</div>

## Exploration for new videos

Staged distribution: show each new video to a small, relevant audience (a few hundred viewers). If its completion and share rates beat peers in its category, promote it to a larger pool, then larger still. This gives every creator a chance, limits exposure for poor or harmful videos, and produces labels quickly. Bandit methods (Thompson sampling on early engagement) formalise the promotion decision.

## Re-ranking

Limit consecutive videos from one creator or topic, insert exploration slots, apply safety and integrity filters, and respect user fatigue (do not repeat a video unless rewatch is likely).

## Serving

~~~dg-flow Short-video feed. The client fetches small batches so each swipe can change what comes next.
Swipe events -(streaming)-> Session state
Feed request -(user + session state)-> Candidate sources -(~1,000)-> *Multi-task ranker -(value score)-> Re-rank -(diversity, safety, exploration)-> Next 4-8 videos
~~~

Prefetch video files for the next few items on the client so playback starts instantly.

## Evaluation

Offline: per-head AUC and calibration, NDCG on sessions. Online: day-1 and day-7 retention, satisfaction survey responses, shares, and time spent as one metric among several. Keep a long-running holdout to watch for slow effects.

## Monitoring

Track creator and topic concentration, new-video reach, prevalence of harmful content in sampled views, and per-head calibration. Alert when one topic or creator takes an unusual share of impressions.

## Likely follow-up questions

- "How do you handle a new user?" Show diverse popular videos by region and language, learn fast from the first 10 to 20 swipes, and ask optional interest questions.
- "How do you stop clickbait thumbnails or first seconds?" Weight completion and post-view actions more than the first seconds, and penalise high early-skip videos.
- "How do you protect teenagers?" Stricter content filters, limits on sensitive topics, and session-time reminders, enforced outside the ranker.

## Trade-offs

- Time spent versus satisfaction and long-term retention.
- Exploration cost versus fresh content supply and creator fairness.
- In-session adaptation versus stability; too much adaptation chases noise from a single swipe.

<div class="take">
<div class="lab">Key takeaway</div>

use every swipe as normalised feedback, combine multi-task predictions into a value score tuned on retention, explore new videos with staged audiences, and re-rank for diversity and safety rather than optimising time spent alone.

</div>
