---
id: "03.04"
course: "ml-system-design"
module: "03"
slug: "new-users-and-new-items"
title: "New users and new items"
keyIdeas: ["**Cold start** is predicting for users or items with little or no history. Collaborative signals are missing exactly when the first impression matters most.", "For new users, use context (location, device, referral source), onboarding questions, and popular or trending items within a segment, then adapt quickly as signals arrive.", "For new items, rely on **content features** (text, image, category, price, creator) that map into the same space as established items.", "New items need **exploration**: a reserved share of exposure, boosted by uncertainty, so the system can learn about them. Otherwise popular items stay popular forever."]
question: "A marketplace adds 200,000 new listings a day. Your recommender is a matrix factorisation model retrained nightly, so new listings get no exposure until the next day and many never get any. Propose changes so new listings are shown, learned about and judged fairly."
answer: "Give new listings a content-based representation from day one: an item tower that embeds title, photos, category, price and seller history into the same space as the collaborative embeddings, trained so that for established items its output matches their learned vectors. Use it to retrieve new listings alongside established ones. Reserve a small exposure budget (for example 5% of slots) for listings with few impressions, chosen by a bandit such as Thompson sampling on click or conversion rate with a prior from the content model, so uncertain items get tried and good ones rise quickly. Update listing statistics in near real time from a stream so a listing that performs well in its first hours gets more exposure the same day. Evaluate separately for new and established listings, and track the share of new listings that receive at least N impressions within 24 hours, along with buyer conversion so exploration does not hurt overall quality."
sources: [{"title": "Wikipedia, Cold start (recommender systems)", "url": "https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)", "note": "User, item and system cold start, with common mitigations."}, {"title": "Li et al., A Contextual-Bandit Approach to Personalized News Article Recommendation (2010)", "url": "https://arxiv.org/abs/1003.0146", "note": "LinUCB for exploration with fresh items."}, {"title": "Volkovs, Yu and Poutanen, DropoutNet: Addressing Cold Start in Recommender Systems (NeurIPS 2017)", "url": "https://papers.nips.cc/paper_files/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html", "note": "Training models to work when collaborative inputs are missing."}]
practice: ["popularity-baseline-leave-last-out", "epsilon-greedy-action-selection", "thompson-sampling-beta-bernoulli", "linucb-contextual-bandit", "catalog-coverage-and-novelty"]
tags: ["cold start", "exploration"]
---

## Where collaborative signals run out

Recommenders and rankers learn a lot from interaction history: users who liked A also liked B. That signal does not exist for a user who signed up a minute ago or an item listed this morning. Yet these are often the moments that matter most: a new user decides whether to stay, and a new item either gets discovered or sinks. Cold start is a design problem, not just a modelling one, because the fixes involve data collection, exposure policy and evaluation.

## New users

Start from what you do know:

- **context**: country, language, device, time, referral source or campaign, the first search query;
- **onboarding**: a few quick choices ("pick three topics") that seed preferences;
- **segment popularity**: what similar new users in the same context engage with;
- **fast adaptation**: update the user's representation within the session from their first clicks, rather than waiting for the nightly retrain.

Session-based models, which take the sequence of recent actions as input, handle new users naturally because they never needed a long history.

## New items

For items, content is the bridge:

- text (title, description), images, category, price, brand, creator and their track record;
- a **content tower** that maps these into the same embedding space as items with history, trained so its output matches the learned embedding of established items;
- metadata-based similarity to established items ("similar to listings that sold well").

When the item collects interactions, blend toward its collaborative representation.

~~~dg-flow A new item's path to a learned representation.
New item -(content features)-> Content embedding -(exploration exposure)-> First interactions -(near-real-time stats)-> *Blended embedding -(enough history)-> Collaborative embedding
~~~

## Exploration

A model that ranks only by predicted engagement will rarely show items it is unsure about, so it never learns about them. Popular items get more exposure, more data and higher scores, which is a rich-get-richer loop. Breaking it needs deliberate exploration:

| Strategy | How it works | Trade-off
| --- | --- | --- |
| Reserved slots | a fixed share of positions for new or low-exposure items | simple, but blind to quality
| Epsilon-greedy | show a random candidate with probability $\epsilon$ | wastes impressions on poor items
| Upper confidence bound | boost items with high uncertainty | needs uncertainty estimates
| Thompson sampling | sample a plausible score from each item's posterior | explores in proportion to the chance of being best
| Contextual bandits (LinUCB) | uncertainty bonus based on features | personalised exploration

The exposure budget is a business decision: more exploration helps new sellers and long-term catalogue health, but costs short-term engagement. State it explicitly, for example "5% of impressions go to items with fewer than 100 impressions".

<div class="ex">
<div class="lab">Worked example</div>

**a content model that unlocked a catalogue**

A music service found that 60% of newly released tracks were never recommended, because their collaborative embeddings were random until the weekly retrain. Adding an audio-and-metadata tower that predicted the collaborative embedding from content, plus a 3% exploration budget for tracks under a week old, raised new-track exposure fourfold while overall listening time stayed flat.

</div>

## Evaluating cold start

Average metrics hide cold-start performance because established users and items dominate the data. Report metrics separately for new users (first session, first week), new items (first day, first week), and established ones. Track coverage: what share of new items get at least some impressions within a day. Offline evaluation with leave-last-out splits overstates quality for new users; simulate cold start by hiding a user's history and scoring their first actions.

## The system around it

Cold start touches several components: onboarding UI, a content embedding pipeline that runs when an item is created, a streaming job that updates item statistics within minutes, and an exposure policy in the ranker. A good design names all four.

## Avoiding a bad first impression

Exploration should not show a new user poor content. Explore mostly with established users who have shown tolerance, cap how many exploratory items appear on one page, and filter candidates by quality signals from the content model before they enter the exploration pool. New users get the safest personalisation available: popular items in their segment, adjusted by their first few actions.

<div class="take">
<div class="lab">Key takeaway</div>

for new users lean on context, onboarding and fast in-session adaptation; for new items use content features mapped into the same space as learned embeddings, give them a deliberate exploration budget, and measure cold-start performance separately.

</div>
