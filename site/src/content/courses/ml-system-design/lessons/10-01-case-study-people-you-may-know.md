---
id: "10.01"
course: "ml-system-design"
module: "10"
slug: "case-study-people-you-may-know"
title: "Case study: people you may know"
keyIdeas: ["People-you-may-know is link prediction on a social graph: score pairs of users who are not yet connected by how likely a connection is, and how valuable it would be.", "Candidates come mainly from the graph itself (friends of friends, shared groups, schools, workplaces, contact uploads), because the space of all pairs is far too large.", "A good suggestion needs both people to act: one sends a request, the other accepts. The label and objective should reflect both steps.", "Privacy constraints shape what signals may be used and what a suggestion may reveal."]
question: "Design the \"people you may know\" feature for a social network with 1 billion users and an average of 300 connections each. Suggestions appear on the home page and in a dedicated tab. Cover candidates, labels, the ranking model, serving and privacy."
answer: "Generate candidates from the graph: friends of friends (scored by mutual count and their closeness), shared groups, schools and workplaces, and uploaded contacts where permitted, plus nearest neighbours in a graph embedding space. Precompute a few thousand candidates per user in a daily batch job with a streaming update when a new connection forms. Rank with a model predicting the probability that a request is sent and accepted, using mutual-connection features, embedding similarity, profile overlap and activity; weight by value such as later interaction. Labels are impressions followed by request and acceptance, with non-acceptances as negatives. Filter blocked users and apply privacy rules. Evaluate offline with recall at k on future connections; online with accepted connections per user and downstream engagement. Monitor graph growth and fairness to new users. Trade-offs: graph growth versus spam requests, and broad reach versus privacy."
rubric: ["Frames the task as link prediction with a two-step outcome (request, then acceptance)", "Generates candidates from graph structure, especially friends of friends, plus other sources", "Explains why scoring all pairs is impossible and how candidate counts are bounded", "Uses mutual-connection and embedding features, with point-in-time correctness", "Precomputes candidates in batch and refreshes on new connections", "Evaluates with recall on future connections and online accepted connections, not just clicks", "Addresses privacy, blocking and sensitive inferences", "Discusses new users with few connections and avoiding spammy request behaviour"]
sources: [{"title": "Wikipedia, Link prediction", "url": "https://en.wikipedia.org/wiki/Link_prediction", "note": "Problem definition and classic graph heuristics."}, {"title": "Wikipedia, Triadic closure", "url": "https://en.wikipedia.org/wiki/Triadic_closure", "note": "Why friends of friends are the strongest candidate source."}, {"title": "Grover and Leskovec, node2vec: Scalable Feature Learning for Networks (2016)", "url": "https://arxiv.org/abs/1607.00653", "note": "Learning node embeddings from random walks."}, {"title": "Ying et al., Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage, 2018)", "url": "https://arxiv.org/abs/1806.01973", "note": "Graph neural networks at the scale of billions of edges."}]
practice: ["precision-recall-at-k", "hit-rate-and-mrr", "word2vec-training-pairs", "embedding-similarity"]
tags: ["friend recommendation", "link prediction", "graph features", "two-sided acceptance"]
---

## Requirements

Clarify the goal. More connections help a social network because people with friends on it return more often; but unwanted suggestions irritate users and can expose sensitive relationships. Assume 1 billion users, 300 connections on average, suggestions refreshed daily with immediate updates after a new connection, and under 100 ms to render the list.

## Metrics

The primary online metric is accepted connections per user per week, ideally weighted by whether the pair later interacts. Guardrails: request rejection or ignore rate, blocks and reports after a suggestion, and hides of the suggestion unit. Offline: recall at 20 against connections formed in the following week, and calibration of the acceptance probability.

## Why candidate generation dominates

There are about $10^{18}$ possible pairs; nothing can score them. Friends of friends alone give $300 \times 300 = 90{,}000$ two-hop paths per user before deduplication, which is already large. Triadic closure (two people with a common friend tend to connect) makes this the richest source. Rank two-hop candidates by the number of mutual connections and how strong those mutual ties are, keeping the top 1,000 or so.

Other sources add people the graph does not reach yet:

- Shared groups, schools and employers within overlapping years.
- Contact uploads, with consent and within regional rules.
- Nearest neighbours in a graph embedding (node2vec-style or a graph neural network).
- Recent profile views in either direction, if policy allows.

## Data and labels

Each impression of a suggestion is a training example. The outcome has two steps: did the viewer send a request, and did the other person accept? Model both, and rank by their product, perhaps weighted by the predicted value of the connection. Suggestions that were never shown have no label, so log impressions with their positions.

<div class="ex">
<div class="lab">Worked example</div>

**optimising the wrong step**

A team optimised for requests sent. Request volume rose by 15%, but acceptance fell from 40% to 28% because the model learned to suggest popular strangers whom many people request. Ranking by the probability of an accepted request, and adding the receiver's past acceptance rate as a feature, restored acceptance while keeping most of the gain in accepted connections.

</div>

## Features

- Graph: number of mutual connections, their strength, overlap of groups, two-hop path counts, embedding similarity.
- Profile: same school, employer, city, language.
- Activity: how active each person is, how many pending requests the receiver already has.
- Context: how the candidate was generated, and how many times it has been shown before (fatigue).

## Model

A gradient-boosted tree or a two-tower plus cross-feature network predicts the two probabilities. Two-tower embeddings also serve retrieval. Graph neural networks over the friendship graph give strong embeddings but are costly to train at this scale; train on sampled neighbourhoods.

## Serving

~~~dg-flow People-you-may-know. Candidates are built offline and patched when the graph changes.
Friendship graph -(daily batch: two-hop, groups, embeddings)-> Candidate store per user
New connection event -(streaming two-hop update)-> Candidate store per user
Page request -(fetch ~1,000 candidates)-> *Ranker -(filters: blocks, privacy, already pending)-> Top 20 suggestions
~~~

## Privacy

Do not suggest people in ways that reveal sensitive facts (for example, two people who saw the same doctor because both uploaded the doctor's number). Allow opt-out from contact-based suggestions, exclude blocked users in both directions, and avoid explanations that disclose private information.

## Monitoring

Track accepted connections, reject and block rates, and the share of suggestions from each source. Watch new users separately: they have no graph yet and depend on contacts and profile signals.

## Likely follow-up questions

- "How do you help a new user with zero connections?" Use contact upload (with consent), school and workplace, and location-based groups, then refresh immediately after the first few connections.
- "Should popular accounts be suggested to everyone?" Cap how often one person is suggested per day, because receivers with thousands of pending requests rarely accept.
- "How would you test a new candidate source?" Measure its incremental recall offline, then run an experiment and check accepted connections, not just impressions.

## Trade-offs

- Growth versus annoyance: more aggressive suggestions add connections but raise reject and block rates.
- Signals versus privacy: contact uploads and profile views are strong but sensitive.
- Batch freshness versus cost: daily batch plus event-driven updates is a common compromise.

<div class="take">
<div class="lab">Key takeaway</div>

generate candidates from the graph (mostly friends of friends), rank by the probability that a request is both sent and accepted, refresh on new connections, and design privacy rules in from the start.

</div>
