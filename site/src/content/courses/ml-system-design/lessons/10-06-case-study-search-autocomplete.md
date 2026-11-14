---
id: "10.06"
course: "ml-system-design"
module: "10"
slug: "case-study-search-autocomplete"
title: "Case study: search autocomplete"
keyIdeas: ["Autocomplete predicts the query a user is typing from a prefix, under a strict latency budget (tens of milliseconds) per keystroke.", "A strong design retrieves candidates from popular past queries with a prefix index (trie), then ranks them with personal, contextual and freshness features.", "Typos, rare prefixes and new events need fallbacks: fuzzy matching, and a generative model for completions not seen before.", "Suggestions are shown to everyone, so filtering offensive, unsafe and personal content is a core requirement."]
question: "Design query autocomplete for a search engine with 50,000 queries per second at peak (each typed with about 8 keystrokes). Suggestions must appear within 30 ms of each keystroke. Explain candidate generation, ranking, handling of typos and new events, and safety filtering."
answer: "Build a prefix index (trie or sorted table) over popular queries from logs, storing the top 20 completions per prefix precomputed by frequency with time decay; this answers most requests from memory in a few milliseconds. Add candidates from the user's own recent searches, fuzzy matching (edit distance 1 to 2) for typos, and a small language model for prefixes with no matches. A light ranker scores candidates on popularity, freshness (trending queries in the last hour), the user's history, location and language. Filter offensive, sexual, violent and personal-information suggestions with blocklists and a classifier, and require a minimum number of distinct users for any query to appear. Evaluate offline with mean reciprocal rank and keystrokes saved; online with suggestion acceptance and time to search. Serve from in-memory replicas close to users; update the trending layer every few minutes."
rubric: ["Clarifies keystroke volume and the strict per-keystroke latency budget", "Uses a prefix index with precomputed top completions per prefix", "Ranks candidates with popularity, freshness, personalisation and context", "Handles typos with fuzzy matching and rare prefixes with a generative fallback", "Adds a fast-updating trending layer for new events", "Filters unsafe and personal content, with a minimum distinct-user threshold", "Evaluates with MRR or keystrokes saved and online acceptance", "Serves from in-memory replicas near users with caching"]
sources: [{"title": "Wikipedia, Autocomplete", "url": "https://en.wikipedia.org/wiki/Autocomplete", "note": "Background and common approaches."}, {"title": "Wikipedia, Trie", "url": "https://en.wikipedia.org/wiki/Trie", "note": "The prefix tree used for fast lookup of completions."}, {"title": "Wikipedia, Learning to rank", "url": "https://en.wikipedia.org/wiki/Learning_to_rank", "note": "Ranking candidate completions."}]
practice: ["n-gram-language-model", "hit-rate-and-mrr", "edit-distance", "spell-correction"]
tags: ["query autocomplete", "prefix matching", "language models", "latency"]
---

## Requirements

Assume 50,000 searches per second at peak, about 8 keystrokes each (so 400,000 autocomplete requests per second, fewer with client-side debouncing), a 30 ms budget per keystroke including network, and up to 10 suggestions. Ask about languages and whether personalisation is allowed.

## Metrics

Offline: mean reciprocal rank (MRR) of the query the user eventually submitted, and keystrokes saved. Online: suggestion acceptance rate, time to complete the search, and searches abandoned. Safety: rate of reported or offensive suggestions, which must be near zero.

## Data

Query logs give the candidates and their popularity. Normalise case, whitespace and punctuation, and count by distinct users rather than raw volume, so that one script repeating a query cannot make it popular. Require a minimum number of distinct users before a query is eligible, which also protects privacy: a query typed by only one person may contain personal data.

## Candidates

- Prefix index: a trie or sorted list mapping each prefix to its top completions, precomputed with a score of decayed frequency (recent weeks weigh more). Precomputing top-k per prefix makes lookup a single read.
- Personal history: the user's own recent queries that match the prefix.
- Trending layer: queries rising sharply in the last hour, updated every few minutes from the stream.
- Fuzzy matching: prefixes within edit distance 1 or 2, for typos.
- Generative fallback: a small language model that completes rare prefixes, filtered and constrained to well-formed queries.

<div class="ex">
<div class="lab">Worked example</div>

**a news event the index missed**

When a major storm was named, users typed its name for hours before the daily index rebuild included it, and autocomplete offered unrelated older queries. Adding a trending layer that counted query prefixes in five-minute windows and merged rising queries into suggestions brought the storm's name into the list within ten minutes of the first spike.

</div>

## Ranking

A light model (gradient-boosted trees or a small network) re-ranks 50 to 100 candidates using: global popularity, trend score, match type (exact prefix versus fuzzy), the user's history, location, language and device. It must run in a few milliseconds.

## Safety filtering

Suggestions put words in front of users who did not type them. Filter with blocklists, a classifier for offensive, sexual, violent and hateful content, and rules for names of private individuals. Be careful with queries that pair a person's name with a negative claim. Filtering happens when building the index and again at serving time for the trending and generative layers.

## Serving

~~~dg-flow Autocomplete request. Most answers come from an in-memory prefix index; other layers are merged in.
Query logs -(daily: normalise, count, filter)-> Prefix index with top-k
Query stream -(5-minute windows)-> Trending layer
Keystroke -(prefix)-> Prefix index + trending + personal history -(candidates)-> *Light ranker -(safety filter)-> Top 10 suggestions
~~~

Serve from in-memory replicas in each region; cache popular prefixes at the edge. The client debounces keystrokes and drops stale responses.

## Monitoring

Track latency percentiles, acceptance, MRR on fresh logs, and reports of offensive suggestions with an urgent removal path. Watch for coordinated attempts to push a query into suggestions.

## Likely follow-up questions

- "How large is the prefix index?" With 100 million eligible queries and top-10 per prefix, storing query IDs rather than strings keeps it to tens of gigabytes, which fits in memory per region.
- "How do you personalise without leaking?" Use only the user's own history on their own device or account, never show other users' rare queries.
- "Should the generative model be used for every keystroke?" No; it is a fallback for rare prefixes, because it is slower and harder to make safe.

## Trade-offs

- Popularity versus personalisation and freshness.
- Coverage of rare prefixes versus safety: generated suggestions are riskier.
- Latency versus ranking quality: every millisecond spent ranking is taken from the network budget.

## A numeric sanity check

At 400,000 requests per second and a 30 ms budget, the network round trip usually takes 10 to 20 ms, leaving roughly 5 to 10 ms for lookup, ranking and filtering. A precomputed top-k lookup takes well under a millisecond; ranking 100 candidates with a small tree model takes one or two. If each server handles 20,000 requests per second, about 20 servers per region suffice before replication for failures. Debouncing on the client, sending a request only after 50 to 100 ms without a new keystroke, can cut traffic substantially.

<div class="take">
<div class="lab">Key takeaway</div>

answer most keystrokes from a precomputed prefix index, merge personal, trending and fuzzy candidates, rank with a light model, and treat safety filtering and distinct-user thresholds as core requirements.

</div>
