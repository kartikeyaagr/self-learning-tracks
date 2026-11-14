---
id: "08.01"
course: "ml-system-design"
module: "08"
slug: "approaching-the-interview"
title: "Approaching the interview"
keyIdeas: ["An ML system design interview tests judgement: can you turn a vague product problem into a working, measurable, maintainable ML system and explain the trade-offs?", "Interviewers look for structure, clarifying questions, sensible defaults, numbers, awareness of data and labels, evaluation and monitoring, and depth where it matters.", "Drive the conversation: propose a plan, check it with the interviewer, and adjust when they steer. Silence and long monologues both hurt.", "Prepare by practising whole designs out loud against a timer, and by knowing a few standard architectures well enough to adapt them."]
question: "You have 45 minutes and are asked to \"design a system that detects fake reviews on a shopping site\". Write the outline you would put on the whiteboard in the first three minutes, before any detail."
answer: "A short outline: (1) Clarify: where reviews appear, volume (for example 500,000 new reviews a day), what action we take (remove, down-rank, send to review), latency (at submission or within hours), what labels exist (moderator decisions, user reports), cost of false removals versus missed fakes. (2) Framing: binary classifier for fake probability per review, plus account-level and network-level signals; thresholds for auto-remove and review queue. (3) Data and labels: moderator decisions, reports, known fraud rings; delayed and biased labels. (4) Features: text, reviewer history, purchase verification, timing bursts, graph links between accounts and products. (5) Model: baseline rules, then gradient-boosted trees plus a text encoder; graph features for rings. (6) Serving: score at submission and rescore in batch as graph signals accumulate. (7) Evaluation: precision at the auto-remove threshold, recall, reviewer load. (8) Monitoring and adversaries: drift, new attack patterns, appeals. Then ask the interviewer which area they want to go deeper on."
sources: [{"title": "Chip Huyen, Machine Learning Systems Design (free notes)", "url": "https://huyenchip.com/machine-learning-systems-design/toc.html", "note": "Interview-oriented notes and exercises."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "A compact source of defensible defaults."}, {"title": "Chip Huyen, Designing Machine Learning Systems (book page)", "url": "https://huyenchip.com/books/", "note": "The lifecycle these interviews probe."}]
tags: ["interviews", "communication"]
---

## What the interview is testing

An ML system design interview gives you a product problem and about 45 minutes. There is no single right answer. The interviewer wants to see how you think: whether you clarify the goal, choose a sensible framing, reason about data and labels, pick models for good reasons, plan evaluation and serving, and anticipate what goes wrong in production. A candidate who designs a simple, well-measured system and explains trade-offs scores better than one who names many advanced models without connecting them.

## What interviewers usually assess

| Area | Strong signal
| --- | --- |
| Problem understanding | asks about users, decisions, scale, latency, error costs
| Framing | turns the goal into a clear ML task with a label and a metric
| Data and labels | knows where labels come from, their delays and biases
| Features | proposes concrete features, aware of freshness and leakage
| Modelling | starts with a baseline, justifies the model and loss
| Evaluation | offline metrics tied to online ones, A/B testing, slices
| Serving | batch or online, latency budget, multi-stage if needed
| Operations | monitoring, retraining, fallbacks, feedback loops
| Communication | structured, concise, checks in with the interviewer

## Before the interview

- **Practise out loud**, with a timer, on a whiteboard or document. Designing in your head is not the same as explaining under time pressure.
- **Know a few reference architectures** well: a two-tower retrieval plus ranking pipeline, a real-time risk scorer, a forecasting pipeline, a RAG system. Most questions are variations on these. The case studies in modules 09 to 11 cover them.
- **Memorise a handful of numbers**: seconds in a day (86,400), typical latency budgets, memory per embedding, rough model inference costs (01.06).
- **Prepare questions** you always ask at the start (01.04).

~~~dg-flow How to spend preparation time. Breadth first, then depth on the common patterns.
Learn the framework -(practise 10+ prompts out loud)-> Reference architectures -(go deep)-> *Ranking, risk, forecasting, RAG -(mock interviews)-> Feedback and gaps
~~~

## During the interview

- **Restate the problem** in one sentence and confirm it.
- **Ask clarifying questions** for two to five minutes, then state assumptions with numbers.
- **Outline the plan** on the board and say which areas you will cover.
- **Go breadth first**, sketching the whole system, then go deep where the interviewer shows interest or where the problem is hardest.
- **Think aloud** about trade-offs, and say which option you choose and why.
- **Check in** every few minutes: "Shall I go deeper on features or move to serving?"
- **Manage time**: leave five minutes for monitoring and iteration, which candidates often skip.

<div class="ex">
<div class="lab">Worked example</div>

**two candidates, one prompt**

Asked to design a restaurant recommender, one candidate immediately proposed a graph neural network with attention over user histories and spent 25 minutes on its architecture. Another asked about surfaces and scale, proposed popularity and location-based candidates as a baseline, then a two-tower retrieval and a gradient-boosted ranker with concrete features, explained the metrics and an A/B test, and finished with cold start, monitoring and a feedback-loop caveat. The second candidate got the offer; the first never discussed data, labels or evaluation.

</div>

## When you are stuck

Say what you know and propose a simple approach: "I'm not sure of the best architecture for this; a reasonable baseline would be X, and I'd measure Y to decide whether to invest more." Interviewers value calibrated honesty over confident guessing.

## After a practice session

Review what you skipped. The most common gaps are labels, evaluation, and monitoring. Keep a checklist and add to it after each mock.

## A note on time

Most candidates run out of time rather than ideas. Glance at the clock at three points: after clarifying (about 8 minutes in), after the high-level design (about 20 minutes in), and when 10 minutes remain. If you are behind at the second check, say so and propose where to go deep, rather than trying to cover every component at the same level. Interviewers would rather see two parts done well and the rest named than eight parts sketched. Keep the last five minutes for monitoring, risks and questions, because those are where strong candidates separate themselves and where rushed candidates say nothing.

<div class="take">
<div class="lab">Key takeaway</div>

ML system design interviews test structured judgement; practise whole designs out loud, know a few reference architectures, clarify and state numeric assumptions, go breadth first, check in often, and leave time for evaluation, monitoring and trade-offs.

</div>
