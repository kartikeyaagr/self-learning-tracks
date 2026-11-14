---
id: "09.16"
course: "system-design-101"
module: "09"
slug: "related-searches"
title: "Related searches"
keyIdeas: ["Related searches suggest queries similar to the current one.", "Mine query logs: queries issued in the same session (people often refine \"python list\" → \"python list comprehension\"), and queries that lead to clicks on the same results (co-click).", "Score pairs by co-occurrence, normalized so popular queries don't dominate (e.g. pointwise mutual information).", "Embeddings of queries can find semantically similar ones for rare queries.", "Precompute offline and serve from a key-value store keyed by normalized query."]
question: "How would you mine \"related searches\" from query logs?"
answer: "Split the logs into sessions and count pairs where query B follows query A, giving extra weight to reformulations where A got no clicks and B did. Add pairs of queries that lead to clicks on the same documents, using the query-document click graph. Normalise by overall popularity (lift or PMI) so generic queries like \"weather\" don't dominate, require a minimum count and number of distinct users so rare or private queries never surface, drop near-duplicates, diversify across meanings, filter unsafe suggestions, and keep only queries that perform well. Store the top few per query in a key-value table rebuilt daily in a batch job, with content- or embedding-based fallbacks for rare queries."
sources: [{"title": "Query Understanding: Query expansion and suggestions", "url": "https://queryunderstanding.com/", "note": "Using query logs."}]
tags: ["Design"]
---

## What related searches are

**Related searches** are the suggestions shown at the bottom or side of a results page ("People also search for", "Related searches"): other queries the user might want to run next. They help in three situations:

- The first query was **too broad**: "jaguar" could mean the car, the animal or the American football team. Suggestions like "jaguar f-type price" and "jaguar habitat" let the user say which.
- The first query was **too narrow** or phrased oddly, and a more common phrasing works better: "inexpensive airfare" to "cheap flights".
- The user wants to **explore**: after "tents", "camping stoves", "sleeping bags".

Related searches are close cousins of **query expansion**, which silently changes the query rather than offering alternatives. They use the same raw material, so this lesson covers expansion first, then how to mine related searches from logs, then how to score, clean and serve them.

## Query expansion

**Expansion** adds terms so a query matches documents that use different words for the same thing. For example, "vp marketing" becomes `(vp OR "vice president") marketing`, so job ads that spell out the title are found.

Sources of expansions:

- **Synonyms** ("couch" for "sofa", "trainers" for "sneakers") and **abbreviations** ("vp", "nyc", "tv").
- **Stemming done as expansion**: instead of stemming the index, expand "shoe" to "shoe OR shoes" at query time. This keeps the index exact (so phrase and exact searches still work) and lets you control which variants are added.
- **Mined equivalences** from logs: queries that lead to the same clicks (below) can be treated as near-synonyms.
- **Embedding neighbours**: terms whose vectors are close (lesson 09.13).

Expansion is a **precision-versus-recall trade-off** (lesson 09.09). It increases recall (more documents match) but risks precision when the added term changes the meaning. "apple" expanded to "fruit" is great on a grocery site and terrible on an electronics site. "Java" expanded to "coffee" ruins a programming search.

How much should an expanded term count? Three options:

- **Treat it equally** with the original terms. Simple, but expansions that are only roughly equivalent pull in noise.
- **Discount** it: give expanded terms a lower weight (for example 0.5), so documents matching the user's exact words still rank first.
- **Feed it to the ranking model** as features ("matched an original term", "matched only an expansion") and let a learned re-ranker decide (lesson 09.08).

Related searches avoid much of this risk by **showing** the alternatives and letting the user choose instead of silently changing the query.

~~~dg-compare Expansion silently broadens the query; related searches offer alternatives and leave the choice to the user.
[Query expansion]
"vp marketing" -(add synonyms)-> One broader query -> More results
+ Recall improves invisibly
- Wrong synonyms hurt precision
- User can't see or undo it
[*Related searches]
"vp marketing" -(look up mined list)-> Suggestions shown -> User picks
+ User stays in control
+ Helps disambiguate broad queries
- Needs a click to help
~~~

## Mining related searches from query logs

The strongest signal is **what other users did next**. Query logs record, for each search, the user (or anonymous session ID), the time, the query, which results were shown and which were clicked. Three common sources of related pairs come from them.

### 1. Session co-occurrence

Group each user's searches into **sessions**: runs of searches with no gap longer than, say, 30 minutes. Queries in the same session are likely related, because people search in bursts around one task. For every pair (A, B) where B follows A in a session, increment a count. Over millions of sessions, strong pairs emerge: "tent" is often followed by "sleeping bag".

### 2. Reformulations

A **reformulation** is a special, stronger case: a user searches A, clicks nothing (or clicks and comes straight back), and within a few seconds searches B. That suggests A didn't work and B was the user's better attempt. Reformulations that **lead to clicks or purchases** are especially valuable: they're a log of users fixing their own queries. "inexpensive airfare" → no clicks → "cheap flights" → click is a strong signal to suggest "cheap flights" for "inexpensive airfare". Many reformulations are small edits (adding a word, fixing a typo), which also feed spelling correction (lesson 09.15).

### 3. The click graph

Build a **bipartite graph**: one set of nodes for queries, another for documents (or URLs, or products), and an edge from a query to each document clicked for it, weighted by click count. Two queries that lead to clicks on the **same documents** are related even if they share no words. "cheap flights" and "budget airfare" both lead to clicks on the same flight comparison sites. Similarity can be measured by the overlap of their clicked-document sets (cosine or Jaccard over click vectors), or by **random walks** on the graph: start at a query, step to a clicked document, step back to another query that clicked it, and count where you land.

~~~dg-tree The click graph connects queries that share no words but lead to clicks on the same documents.
Clicked documents
  *flight comparison site A
    "cheap flights" | 9,000 clicks
    "budget airfare" | 1,200 clicks
  flight comparison site B
    "cheap flights" | 6,000 clicks
    "budget airfare" | 800 clicks
    "last minute flights" | 2,500 clicks
~~~

## Scoring and cleaning

Raw co-occurrence counts are dominated by queries that are popular everywhere. "facebook", "weather" and "youtube" follow almost every query in some session, simply because people search for them constantly. Suggesting "weather" after "jaguar" helps nobody.

### Normalise by popularity

Compare how often B follows A with how often you'd expect B by chance:

`lift(A → B) = P(B follows A) / P(B)`

or the closely related **pointwise mutual information**, PMI = log of that ratio. B is suggested for A only if it's **unusually** likely after A. A lift of 1 means no association; a lift of 50 means B is 50 times more likely after A than in general.

<div class="ex">
<div class="lab">Worked example</div>

**lift for "jaguar"**

Suppose 1% of all sessions contain "weather" and 0.01% contain "jaguar f-type". Among 100,000 sessions containing "jaguar", "weather" follows in 2,000 (2%) and "jaguar f-type" in 1,200 (1.2%).

Lift for weather = 2% / 1% = 2: barely above chance. Lift for jaguar f-type = 1.2% / 0.01% = 120: very strongly associated. Raw counts would rank "weather" first; lift correctly drops it and keeps "jaguar f-type".

</div>

Lift has the opposite failure for very rare B: a query seen twice, both times after A, gets an enormous lift from tiny counts. So require a **minimum count** for each pair, or use a smoothed score that shrinks small-count estimates towards no association.

~~~dg-bars Raw follow counts versus lift for queries after "jaguar". Lift removes queries that are popular everywhere.
weather (raw) | 2000 | 2,000 follows, lift 2
jaguar f-type (raw) | 1200 | 1,200 follows, lift 120
*jaguar f-type (lift) | 120 | lift 120, kept
weather (lift) | 2 | lift 2, dropped
~~~

### Further cleaning

- **Remove near-duplicates** of the original and of each other ("jaguar cars" versus "jaguar car", "jaguar f type" versus "jaguar f-type"), so the suggestions are actually different. Normalise and compare analysed forms.
- **Diversify**: for an ambiguous query, make sure the final list covers several meanings (car, animal, team), not eight variants of one.
- **Apply query performance**: only suggest queries that themselves lead to successful searches (clicks, purchases), as with autocomplete (lesson 09.14).
- **Safety**: filter offensive or sensitive suggestions with a block list and a classifier. A suggestion on your page reads as your platform's recommendation.
- **Privacy**: require that each pair was seen from a minimum number of **distinct users** (say 20 or more), so one person's unusual searches can never be shown to others. Don't let a user's own private sessions shape suggestions shown to others in identifiable ways.
- **Fallback for rare queries**: many queries are too rare to have log data. Fall back to content: queries whose top results overlap with this query's results, queries sharing rare terms, or the embedding nearest neighbours among known good queries (lesson 09.13).

## Building and serving

~~~dg-flow Mining related searches: sessions and the click graph give candidate pairs; scoring and filtering turn them into a lookup table.
[Mine] Query logs -(split into sessions)-> Follow pairs and reformulations -(lift, min users)-> *Candidate pairs
[Graph] Query logs -(query to clicked doc)-> Click graph -(shared clicks)-> *Candidate pairs
[Serve] *Candidate pairs -(dedupe, diversify, filter)-> Key-value table -(cache)-> Results page
~~~

The pairs are computed **offline** in a batch job (daily is typical) over the last few weeks of logs:

1. Sessionise the logs and extract follow pairs and reformulations.
2. Build the click graph and compute query-query similarity for queries above a minimum traffic.
3. Merge the candidate sources with weights, compute scores, and apply the cleaning steps.
4. Keep the top 8 or so per query and write them to a **key-value table**: normalised query → list of related queries.

At query time, the search service normalises the query, looks it up in the table (an in-memory cache or Redis in front of a key-value store), and renders the list. It's one lookup, well under a millisecond, and can run in parallel with the main search so it adds no latency.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the table**

Related searches are stored for the 20 million most common normalised queries. Each entry holds 8 related queries of about 25 bytes each, plus the key: roughly 250 bytes per query, so about 5 GB in total. That fits in a replicated in-memory store. The daily batch job processes 30 days of logs (say 3 billion searches); with sessionisation and pair counting in Spark, it runs in an hour or two on a modest cluster. Queries outside the 20 million fall back to content-based suggestions or show none.

</div>

## Measuring success

Track the **click-through rate on related searches**, the **success rate** of searches started from a suggestion (do they lead to clicks or purchases?), and whether sessions that use them end successfully more often. A/B test changes to the mining or scoring.

## Common mistakes

- Using raw co-occurrence counts, so generic queries dominate every list.
- Using lift without a minimum count, so one-off queries with tiny counts appear.
- Showing near-duplicates of the original query.
- No privacy threshold, so a rare personal query from one user appears for others.
- Expanding queries silently with risky synonyms when showing them as suggestions would be safer.

## In an interview

Say: "Mine it offline from logs: session co-occurrence, reformulations that led to clicks, and the click graph for queries sharing clicked results. Score with lift or PMI so globally popular queries don't dominate, require minimum counts and distinct users for privacy, dedupe and diversify, filter for safety and performance, and store the top 8 per query in a key-value table rebuilt daily, with a content-based fallback for rare queries."

<div class="take">
<div class="lab">Key takeaway</div>

related searches show alternative queries rather than silently rewriting. They are mined offline from query logs (what users searched next in a session, reformulations that led to clicks, and queries sharing clicked documents), normalised with lift so globally popular queries don't dominate, filtered for quality, diversity and privacy, and served from a key-value lookup.

</div>
