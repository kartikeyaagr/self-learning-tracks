---
id: "01.04"
course: "ml-system-design"
module: "01"
slug: "clarifying-requirements"
title: "Clarifying requirements"
keyIdeas: ["Requirements come in two kinds: **functional** (what the system decides, for whom, with what inputs and outputs) and **non-functional** (latency, throughput, freshness, cost, privacy, fairness).", "Ask about the decision, the users, the scale, the latency budget, the data available and the cost of each kind of error before proposing anything.", "Write assumptions down with numbers. A stated assumption the interviewer can correct is better than a hidden one.", "Requirements decide the architecture: a 20 ms budget rules out some models, and a weekly freshness need rules out a streaming pipeline."]
question: "You are asked to \"design a system that recommends restaurants\". List the eight questions you would ask before designing anything, and explain how two different answers to one of them would change the design."
answer: "Questions: (1) Where is the recommendation shown and what action follows (home feed, search, notification)? (2) Who are the users and how many (daily active users, requests per second at peak)? (3) What is the success metric (orders, retention, rating)? (4) What is the latency budget for a page load? (5) How fresh must it be (does today's opening hours or current wait time matter)? (6) What data exists (orders, views, ratings, location, menus)? (7) What are the constraints (delivery radius, dietary filters, fairness to small restaurants)? (8) What is the cost of a bad recommendation? For example, if recommendations appear in a push notification sent once a day, batch scoring overnight is fine and the model can be heavy; if they power a live home screen that reacts to the current location and time, you need online retrieval and ranking under about 100 ms with location-aware candidate generation."
sources: [{"title": "Google, Problem framing course", "url": "https://developers.google.com/machine-learning/problem-framing", "note": "Defining the goal, output and success metrics."}, {"title": "Chip Huyen, Machine Learning Systems Design (free notes)", "url": "https://huyenchip.com/machine-learning-systems-design/toc.html", "note": "Project setup, objectives and constraints."}]
practice: ["p50-p95-and-p99-inference-latency", "optimize-throughput-and-cost-under-latency-slo"]
tags: ["requirements", "constraints"]
---

## Why ask first

A design is an answer to a specific question. "Recommend restaurants" could mean a nightly email, a live home screen or a search ranking. Each needs a different system. Spending the first few minutes on requirements avoids designing the wrong thing well, and in an interview it shows you know that constraints drive architecture.

## Functional requirements

These describe what the system does:

- **The decision**: what does the prediction change? Ranking a list, blocking a payment, setting a price, sending a notification.
- **Inputs and outputs**: what is known at prediction time, and what form the output takes (a score, a class, a list of ten items, a price).
- **The users**: who sees the result, and are there distinct groups (new versus returning, buyers versus sellers)?
- **Surfaces**: where it appears (web, app, email) and how often.

## Non-functional requirements

These describe how well it must do it:

| Requirement | Question to ask | Example answer
| --- | --- | --- |
| Scale | how many predictions per second at peak? | 5,000 QPS, 50M users
| Latency | what is the budget for this call? | p99 under 50 ms
| Freshness | how quickly must new data matter? | within 1 hour
| Cost | what can we spend per prediction or per month? | under $0.0001 per request
| Availability | what happens if the model is down? | fall back to popularity
| Privacy and policy | which data may we use? | no precise location
| Fairness | which groups must be treated consistently? | sellers of all sizes get exposure

Latency budgets should be stated as percentiles, usually p99, because users notice the slow tail. The ML part usually gets only a fraction of the page budget; the rest goes to network, other services and rendering.

~~~dg-flow From vague ask to design input. Each answer narrows the space of designs.
Vague ask -(decision + users)-> Functional spec -(scale, latency, freshness, cost)-> Non-functional spec -(constraints + data)-> *Design space
~~~

## The cost of errors

Ask what a false positive and a false negative each cost. Blocking a legitimate £50 payment costs the sale and a support call; missing a fraudulent one costs £50 plus a chargeback fee. Showing an irrelevant restaurant costs a little attention; recommending one that is closed costs trust. These costs choose the metric and the operating threshold later (05.01).

## State assumptions with numbers

When the interviewer does not know or does not say, assume and write it down: "Assume 20M daily users, 10 sessions a day each, so about 200M ranking calls a day, or roughly 2,300 per second on average and maybe 10,000 at peak." Numbers make trade-offs concrete and let the interviewer redirect you early.

<div class="ex">
<div class="lab">Worked example</div>

**one answer that changes everything**

"How fresh must fraud scores be?" If the answer is "a transaction must be scored before authorisation", the design needs online features (spend in the last hour, new device) and a model under about 50 ms. If the answer is "we review flagged accounts the next day", a nightly batch job with heavy features and a large model is simpler and cheaper.

</div>

## A short checklist

Decision and action; users and surfaces; success metric and guardrails; scale; latency; freshness; available data and labels; cost of each error; constraints on privacy and fairness; fallback behaviour. Ten items, two or three minutes, and the rest of the design follows from them.

## Requirements that conflict

Requirements often pull against each other, and naming the conflict is part of the design. Higher recall in fraud detection means more false alarms for analysts to review. Fresher recommendations mean a streaming pipeline and higher cost. Strict privacy rules may remove the most predictive feature. When two requirements conflict, ask which one wins, or propose a compromise with numbers: "we can hit p99 under 50 ms if we drop the external credit-score call, which costs about 2% recall; is that acceptable?" Interviewers look for exactly this kind of explicit trade-off.

<div class="take">
<div class="lab">Key takeaway</div>

before designing, pin down the decision, inputs and outputs, scale, latency, freshness, cost, error costs and constraints, and write assumptions down with numbers, because each one rules designs in or out.

</div>
