---
id: "09.04"
course: "ml-system-design"
module: "09"
slug: "case-study-trending-topics-and-bot-detection"
title: "Case study: trending topics and bot detection"
keyIdeas: ["Trending detection compares a topic's current rate with its own expected rate, not with other topics, so seasonal and always-popular topics do not dominate.", "Coordinated inauthentic accounts can manufacture trends, so bot and coordination signals must filter what counts before ranking trends.", "Bot detection mixes per-account classifiers with behaviour across accounts, because coordinated groups look normal one at a time.", "Both systems run on streams with seconds to minutes of delay, which favours approximate counting and simple, robust statistics."]
question: "Design a system that surfaces trending topics on a social platform with 100 million posts a day, refreshed every few minutes, and that prevents coordinated bot networks from pushing topics into the trending list."
answer: "Extract candidate topics (hashtags, named entities, clustered phrases) from the post stream and count them in sliding windows with approximate structures. Score each topic by how far its current rate exceeds its own baseline for that hour and day, using a z-score or ratio with smoothing, and require growth across distinct, credible accounts. Before counting, weight each post by its author's authenticity score from a bot classifier (account age, posting cadence, content similarity, network features) and by coordination signals such as many accounts posting near-identical text within seconds. Rank trends by the weighted surprise, apply safety and policy filters, and personalise by region. Evaluate with human review of sampled trends and with known coordinated campaigns. Monitor the share of trends later removed and the bot-score distribution. Trade-offs: speed versus false trends, and aggressive filtering versus suppressing genuine grassroots topics."
rubric: ["Defines trending as surprise against the topic's own baseline rather than raw volume", "Uses windowed or approximate streaming counts with a clear refresh interval", "Requires breadth of distinct, credible authors, not only post counts", "Designs a bot classifier with account, behaviour and network features", "Detects coordination across accounts (near-duplicate content, synchronised timing)", "Evaluates with human review and known campaigns, given scarce labels", "Monitors removed trends and adapts to adversaries", "Discusses the trade-off between speed and false trends, and risks of suppressing genuine topics"]
sources: [{"title": "Varol et al., Online Human-Bot Interactions: Detection, Estimation, and Characterization (2017)", "url": "https://arxiv.org/abs/1703.03107", "note": "Account-level features for bot classification."}, {"title": "Wikipedia, Count-min sketch", "url": "https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch", "note": "Approximate counts over a stream with bounded memory."}, {"title": "Wikipedia, Social bot", "url": "https://en.wikipedia.org/wiki/Social_bot", "note": "Background on automated accounts and their uses."}, {"title": "Wikipedia, Streaming algorithm", "url": "https://en.wikipedia.org/wiki/Streaming_algorithm", "note": "Algorithms that process data in one pass with limited memory."}]
practice: ["page-hinkley-drift", "consecutive-breach-alerts", "isolation-forest", "z-test-one-sample"]
tags: ["trending detection", "bot detection", "streaming", "anomaly detection"]
---

## Requirements

Clarify what "trending" means for the product: new and rising topics, not what is always popular. Assume 100 million posts a day (about 1,200 per second, several times that during major events), a global list plus per-country lists, and refresh every two to five minutes. Trends must be safe to show (no harassment campaigns) and resistant to manipulation.

## Metrics

Online: click-through on trends, time to first appearance of a major event, and the share of trends later removed by reviewers. Offline: precision of trends against human judgement on a sample, and recall against a list of known events. For bots: precision and recall against a labelled set of accounts, and the share of trending volume coming from accounts later suspended.

## Data and labels

Trending has almost no labels; it is mostly a statistics problem. Bot detection has labels from accounts suspended for automation, reviewer decisions and known campaigns, all incomplete and biased towards obvious bots. Use them, but evaluate on fresh human review too.

## Detecting trends

1. Extract topics from each post: hashtags, named entities, and clusters of similar phrases.
2. Count each topic in short windows (one minute) with a count-min sketch for the long tail and exact counts for the top few thousand.
3. Compare the current rate with the topic's expected rate for this hour and weekday, smoothed over past weeks. A score such as $z = (c - \mu) / \sqrt{\mu + k}$ (with a constant $k$ to stop tiny topics from scoring high on a few posts) captures surprise.
4. Require breadth: the number of distinct authors, weighted by authenticity, must rise too.

<div class="ex">
<div class="lab">Worked example</div>

**a trend from 400 accounts**

A hashtag reached the top of a country's trending list with 60,000 posts in an hour. Grouping by author showed that 400 accounts, created in the same week and posting within seconds of each other, wrote over 80% of them. Weighting posts by author authenticity and capping each account's contribution per window dropped the topic below the threshold, while a genuine local news story with 15,000 posts from 12,000 authors rose into its place.

</div>

## Detecting bots

- Account features: age, profile completeness, follower and following ratios, verification.
- Behaviour: posting cadence and regularity, activity at all hours, share of reposts, client used.
- Content: similarity to other accounts' posts (MinHash on text), links to the same domains.
- Network: dense clusters that follow and amplify each other; synchronised actions across accounts.

A gradient-boosted classifier gives a per-account score. A separate coordination detector finds groups posting near-identical content within short windows; groups matter because each account alone can look human.

## Serving

~~~dg-flow Trending pipeline. Authenticity weights are applied before counting so manipulation never reaches the ranking.
Post stream -(topic extraction)-> Weight by author score -(windowed counts)-> *Surprise scoring -(breadth, safety filters)-> Trend lists by region
Account events -(batch and streaming features)-> Bot classifier -(authenticity scores)-> Weight by author score
~~~

## Monitoring

Track trends removed after appearing, the share of trending volume from low-authenticity accounts, and the time between a real event and its trend. Adversaries adapt by aging accounts and adding variation to text, so retrain the bot classifier often and add new coordination signals.

## Trade-offs

- Faster refresh catches breaking news sooner but admits more noise and more manipulation.
- Strong filtering reduces false trends but risks suppressing genuine grassroots topics from small or new communities; review a sample of filtered topics.
- Explaining why a topic trends builds trust but helps attackers learn the rules.

## Likely follow-up questions

- "How do you avoid trending a topic that is popular every Monday morning?" The baseline is per topic and per hour of the week, so a regular peak produces no surprise.
- "How do you handle a topic spelt many ways?" Normalise case and spelling, and cluster hashtags and phrases by co-occurrence and embedding similarity so that variants count together.
- "How would you evaluate a new bot classifier without good labels?" Compare its flags with later suspensions, run human review on a stratified sample of high, medium and low scores, and measure how much trending volume it removes from known campaigns.
- "Should a verified account be trusted?" More, but not fully; hijacked accounts are a common attack.

<div class="take">
<div class="lab">Key takeaway</div>

score trends by surprise against each topic's own baseline and by breadth of credible authors, weight every post by an authenticity score before counting, and detect coordination across accounts because bots that act together each look normal alone.

</div>
