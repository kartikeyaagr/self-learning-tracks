---
id: "10.03"
course: "ml-system-design"
module: "10"
slug: "case-study-notification-ranking"
title: "Case study: notification ranking"
keyIdeas: ["Notification ranking decides which notifications to send, when, and how many, trading short-term engagement against annoyance and disabled notifications.", "The right target is the causal effect of sending (uplift), not the probability of opening, because some users would have visited anyway.", "Volume control is a constrained optimisation per user: a daily budget, minimum value thresholds and quiet hours.", "Long-term effects (users turning notifications off or uninstalling) appear slowly, so experiments need long-running holdouts."]
question: "Design the system that decides which push notifications to send to each user of a social app with 300 million daily users. Many events (likes, comments, new posts, friend activity) could trigger a notification. Explain what you optimise, how you control volume and timing, and how you measure long-term effects."
answer: "Collect candidate notifications from event streams per user, deduplicate and aggregate them (for example \"5 people liked your photo\"). Score each candidate with models for the probability of open and, ideally, the uplift in sessions or meaningful actions compared with not sending, plus the probability of a negative response (dismiss, disable, uninstall). Value = uplift benefit minus annoyance cost. A per-user scheduler sends candidates whose value exceeds a threshold, within a daily budget, outside quiet hours, at times when the user is likely to engage; urgent types such as direct messages bypass the ranker. Labels come from sends with outcomes and from randomised holdouts that estimate uplift. Evaluate with long-running holdout groups that receive fewer notifications to measure retention and opt-out. Monitor opt-out rates by type and volume per user."
rubric: ["Defines the objective as incremental value (uplift) rather than open rate", "Includes a negative-outcome model (dismissals, opt-outs, uninstalls) in the value", "Aggregates and deduplicates candidate notifications before ranking", "Designs per-user volume control with budgets, thresholds and quiet hours", "Handles timing (when to send) and urgent types that bypass ranking", "Uses randomised holdouts to estimate uplift and long-term effects", "Monitors opt-outs and engagement per notification type", "Discusses the short-term versus long-term trade-off explicitly"]
sources: [{"title": "Wikipedia, Uplift modelling", "url": "https://en.wikipedia.org/wiki/Uplift_modelling", "note": "Predicting the effect of an action rather than an outcome."}, {"title": "Wikipedia, Push technology", "url": "https://en.wikipedia.org/wiki/Push_technology", "note": "Background on push notifications."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Choosing objectives and watching long-term effects."}]
practice: ["log-loss-binary-cross-entropy", "b-test-significance", "linucb-contextual-bandit", "thompson-sampling-beta-bernoulli"]
tags: ["notifications", "uplift", "volume control", "long-term effects"]
---

## Requirements

Clarify what types exist and which are mandatory. Direct messages and security alerts must always be delivered promptly; likes, friend posts and recommendations are optional. Assume 300 million daily users, tens of candidate events per user per day, a target of a few optional notifications per user per day, and a decision within seconds of an event (or batched for digest types).

## Metrics

The tempting metric, open rate, is misleading: a notification about something the user would have seen anyway is opened but adds nothing. Better metrics:

- Incremental daily active users or sessions caused by notifications (measured against a holdout).
- Opt-out rate (notifications disabled for the app or for a type) and uninstalls.
- Meaningful actions after opening (replying, posting), not just opening.

## Data and labels

Each sent notification has outcomes: opened, dismissed, led to actions, followed by an opt-out. To estimate uplift, some candidates that would have been sent must be randomly withheld; the difference in outcomes between sent and withheld groups, conditioned on features, is the effect. Keep a small, permanent holdout of users who get a reduced volume to measure long-term effects.

<div class="ex">
<div class="lab">Worked example</div>

**open rate up, users down**

A team trained a ranker on open probability and raised sends for users who opened often. Open rate rose 12%, but a three-month holdout showed that heavy users who received more notifications disabled them at twice the previous rate, and daily users in that group fell slightly. Switching the target to uplift in sessions and adding a predicted opt-out cost reduced volume by 20% with no loss in daily users.

</div>

## Features

- User: notification history, recent opens and dismissals, time since last visit, typical active hours, settings.
- Notification: type, actor (close friend or stranger), content popularity, freshness.
- Context: how many notifications already sent today, time of day, device.

## Model

Train a multi-task model with heads for open, meaningful action and negative outcome. For uplift, use a two-model approach (one on treated, one on control data) or a model with treatment as an input, trained on randomised data. Value per candidate is roughly
$V = \Delta P(\text{session}) \cdot b - P(\text{negative}) \cdot c$,
with business weights $b$ and $c$ tuned by experiment.

## Volume control and timing

A per-user scheduler holds candidates and decides:

- Threshold: send only when $V$ exceeds a minimum.
- Budget: at most $k$ optional notifications per day, with $k$ adapted per user.
- Timing: delay non-urgent candidates to the user's likely active time; never during quiet hours.
- Aggregation: combine related events into one notification.

## Serving

~~~dg-flow Notification pipeline. Urgent types bypass ranking; others wait in a per-user scheduler.
Event streams -(candidate generation, deduplication, aggregation)-> Candidates per user -(features)-> *Value model -(uplift minus annoyance)-> Per-user scheduler -(budget, timing, quiet hours)-> Push service
Direct messages and security alerts -(bypass)-> Push service
~~~

## Monitoring

Track opt-out rates by type, volume per user distribution, uplift estimates from holdouts, and delivery delays. Alert when a new notification type causes a spike in opt-outs.

## Likely follow-up questions

- "How do you launch a new notification type?" Start with a small randomised rollout, measure uplift and opt-outs, and only then let the ranker trade it against existing types.
- "What about users who never open notifications?" Reduce volume to a minimum and rely on in-app surfaces; sending more rarely helps them.
- "Could email and push be ranked together?" Yes, with channel as a feature and per-channel budgets.

## Trade-offs

- Short-term sessions versus long-term opt-outs: the core tension; long holdouts are the only reliable measure.
- Personalised volume versus fairness to content creators who rely on notifications for reach.
- Fast sending versus better timing.

## A numeric sanity check

With 300 million daily users and an average of three optional notifications each, the system sends about 900 million optional pushes a day, around 10,000 per second, with sharp peaks in the evening. If each user has 30 candidates, the value model scores about 9 billion candidates a day; a small model on CPU handles this, but the per-user scheduler needs durable state, because a lost queue means lost or duplicated notifications. Holdouts of 1% of users still contain 3 million people, enough to detect small changes in retention.

<div class="take">
<div class="lab">Key takeaway</div>

rank notifications by their causal value (uplift minus annoyance) rather than open rate, control volume and timing per user, let urgent types bypass ranking, and measure long-term effects with permanent holdouts.

</div>
