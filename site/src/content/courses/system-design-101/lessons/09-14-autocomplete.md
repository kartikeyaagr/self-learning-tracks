---
id: "09.14"
course: "system-design-101"
module: "09"
slug: "autocomplete"
title: "Autocomplete"
keyIdeas: ["Autocomplete suggests completions for each keystroke, so it must answer in under ~100 ms.", "Data structure: a trie of popular queries. Cache the top-k completions at each node so lookup is O(prefix length), with no traversal of the subtree.", "Build the trie offline from query logs (aggregate counts, weekly or daily), and serve it from memory, sharded by prefix.", "Client side: debounce keystrokes and cache results in the browser.", "Filter offensive or unsafe suggestions, and blend in trending queries with a faster-updating pipeline."]
question: "Why cache the top 5 completions at each trie node?"
answer: "Without the cache, answering a prefix means traversing the whole subtree below its node and sorting every completion by frequency, which for a short prefix like \"s\" means visiting millions of nodes on every keystroke, far too slow for the few milliseconds available while a user types. Caching the top 5 completions at each node, computed when the trie is built, makes a lookup just a walk down the prefix plus one read, effectively constant time. The cost is extra memory (k entries per node, kept small by limiting the trie to popular queries) and that a count change affects every ancestor's list, so the trie is rebuilt periodically offline instead of updated per search."
sources: [{"title": "ByteByteGo: Design a Search Autocomplete System", "url": "https://bytebytego.com/courses/system-design-interview/design-a-search-autocomplete-system", "note": "Trie, top-k caching, data pipeline."}]
tags: ["Design"]
---

## What autocomplete is for

**Autocomplete** (also called typeahead or query suggestion) shows a short list of likely queries under the search box as the user types. It isn't just about saving keystrokes. Its bigger job is to **guide searchers to queries that work**: correctly spelled, phrased the way the catalogue phrases things, and specific enough to give good results. Users now expect it everywhere, and on mobile, where typing is slow, the suggestion list is often the whole search experience: a large share of mobile searches are completed by tapping a suggestion.

It's also a demanding system. A suggestion request fires on (almost) every keystroke, so request rates are several times the search rate, and the results must come back faster than the user types the next character: the end-to-end budget is about 100 ms, leaving the server perhaps 10 ms.

This lesson covers both halves: **which** completions to offer (a modelling question) and **how** to serve them fast (a data-structure and system question).

## Modelling: which completions to offer

Treat autocomplete as **conditional probability**: given the typed prefix, offer the full queries that are most probable. Estimate the probabilities from a query log.

- If 1,000 of 1,000,000 logged queries are "pants", then P(pants) = 0.001.
- If 50,000 logged queries start with "pa", then P(pants | pa) = 1,000 / 50,000 = 0.02.
- P(pants | pat) would be 0 by pure prefix matching, unless "pat" is a typo for "pant". That's why autocomplete must work together with spelling correction (lesson 09.15): it should tolerate an edit or two in the prefix.

### Complications

- **Recency and seasonality.** Raw frequency over a year suggests "halloween costume" in April. Weight recent counts more, for example with exponential decay (a count from 7 days ago weighs half a count from today), and boost seasonal queries in season.
- **Context.** Location ("pizza" suggests nearby cities), the current page or category, and the session can improve estimates, but too much context fragments the counts until the estimates become noisy (overfitting).
- **Non-prefix matches.** The typed text may not be a prefix of the best suggestion: "pa" should be able to suggest "mens pants" (infix match on a word boundary). This needs an index of word starts, not just query starts.
- **The feedback loop.** If you count queries that were produced by clicking a suggestion, suggestions become popular *because* they were suggested. Exclude or down-weight suggestion-clicked queries when counting.

### Popularity is not enough: query performance

A suggestion is only good if the search it leads to succeeds. So choose suggestions by the probability of a **successful** search (a click or purchase), roughly:

`value(q | prefix) = P(q | prefix) × P(success | q)`

<div class="ex">
<div class="lab">Worked example</div>

**popularity versus performance**

For prefix "pa": "pants" has P = 0.02 and a click-through rate of 0.1, so value = 0.002. "pant cuffs" is ten times rarer (P = 0.002) but has click-through 0.5, value = 0.001. "pa" itself as a query has P = 0.01 but click-through 0.01, value 0.0001.

Ranking purely by success probability would put "pant cuffs" first, but that's a niche product most users don't want, so the combined value still favours the popular query, correctly. What performance data does well is **remove** bad suggestions: set a minimum performance threshold so queries that usually lead to zero results or no clicks ("pa") are never suggested.

</div>

## The data structure: a trie with cached top-k

The standard serving structure stores queries in a **trie** (pronounced "try", from re*trie*val), also called a **prefix tree**. Each node represents a prefix; each edge adds one character; a path from the root to a marked node spells a complete query. All queries sharing a prefix share the path for it, so "tree", "trie" and "try" share the nodes for "t" and "tr".

~~~dg-tree A trie for "tree", "trie", "try" and "toy". Each node caches its top completions, so a lookup is one walk down the prefix.
root
  t | top: tree, try, trie, toy
    *r | top: tree, try, trie
      e | top: tree
        e | tree (end)
      i | top: trie
        e | trie (end)
      y | try (end)
    o | top: toy
      y | toy (end)
~~~

### The naive lookup and why it's too slow

To answer "top 5 completions for prefix p" without help:

1. Walk from the root to p's node: O(length of p) steps.
2. Visit **every** node below it, collecting each complete query and its count.
3. Sort the collected queries by count and take the top 5.

Step 2 is the problem. For a short prefix like "s", the subtree holds every logged query starting with "s", possibly millions of nodes. Doing that on every keystroke, for thousands of users a second, is impossible within a few milliseconds.

### The fix: cache top k at every node

Store at **every node** the list of its top k completions (with scores), computed once when the trie is built. A lookup becomes:

1. Walk down the prefix: a handful of steps, since people type short prefixes before choosing.
2. Read the stored list.

That's effectively **constant time** for the bounded prefix lengths people type.

~~~dg-compare Answering a prefix with and without cached top-k lists at each node.
[Naive trie]
Prefix "s" -(walk 1 node)-> Visit whole subtree -(sort millions)-> Top 5
- Millions of nodes per keystroke
- Latency grows with popularity
[*Cached top-k trie]
Prefix "s" -(walk 1 node)-> Read stored list -> Top 5
+ Constant-time lookup
- k entries stored at every node
- Updates touch every ancestor
~~~

The costs:

- **Memory**: every node stores k entries. That's acceptable because the trie only holds queries above a popularity threshold (rare one-off queries are dropped), k is small (5 to 10), and entries can be pointers or IDs rather than full strings.
- **Updates are expensive**: when one query's count changes, the top-k lists of every ancestor node may change. Doing that per search is too costly, so the trie is usually **rebuilt offline** (hourly or daily) from aggregated logs, and the new version replaces the old atomically.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the trie**

Keep the 10 million most frequent queries, average 20 characters. A trie shares prefixes, so suppose about 50 million nodes. Each node stores a character, child pointers and a top-5 list of 4-byte query IDs plus 4-byte scores: about 40 bytes for the list and maybe 40 more for structure, so about 80 bytes per node, or 4 GB in total. That fits in memory on one server, so the service can be replicated for throughput without sharding. At 20,000 suggestion requests a second at peak and about 0.2 ms of CPU per lookup, four replicas give ample headroom.

</div>

## The surrounding system

~~~dg-flow Autocomplete splits into a slow offline path that builds the trie and a fast read path that serves it.
[Offline] Query logs -(aggregate daily counts)-> Filter and score -> *Trie builder -(snapshot)-> Trie store
[Online] Keystroke -(debounced)-> CDN cache -(miss)-> Suggestion service -(in-memory trie)-> Suggestions
~~~

### Data collection and building

- **Log** every submitted query with timestamp, whether it came from a suggestion, and its outcome (clicks, purchases).
- **Aggregate** counts per query per day in a batch job (Spark, or a stream processor for near-real-time).
- **Filter**: drop queries below a minimum count (also a privacy protection: a query typed by only one person must never be suggested to others), and drop offensive, unsafe or legally sensitive completions using a block list and a classifier. Autocomplete suggestions are seen as the platform speaking, so this step matters for trust.
- **Score** each query (decayed frequency × performance), then **build** the trie with top-k lists bottom-up: each node's list is the best k among its own query and its children's lists, so building is one pass.

### Serving

- The trie lives **in memory** in a suggestion service. If too large for one machine, **shard by prefix** (the first one or two characters), being careful that common first letters ("s", "c") get more shards.
- Replicate for read throughput and availability.
- Swap in new tries by loading the new snapshot alongside and switching a pointer, so there's no downtime.

### Client and edge

- **Debounce** keystrokes: send a request only after a short pause (say 50 to 100 ms) so a fast typist doesn't send a request per character.
- **Cache** recent responses in the browser, so backspacing is instant.
- **Cache popular prefixes at the CDN**: "a", "b", "ip", "sh" are requested constantly and are the same for everyone (in a region). One- and two-character prefixes cached at the edge remove a large share of origin traffic.
- **Cancel** in-flight requests when a newer keystroke arrives, and discard responses that come back out of order.

~~~dg-seq A user types "be". Debouncing skips the request for "b", and the response for "be" comes from the in-memory trie.
User > Browser : types "b"
Browser > Browser : debounce timer starts
User > Browser : types "e" within 80 ms
Browser > Browser : timer resets, then fires
Browser > CDN : GET /suggest?q=be
CDN > Suggestion service : cache miss
Suggestion service > Suggestion service : walk b, e; read top 5
Suggestion service -> CDN : 5 suggestions, cache 5 min
CDN -> Browser : best buy, bed bath, ...
~~~

### Freshness

A daily rebuild misses breaking news and trending topics. Add a small **real-time layer**: a stream job counts queries over the last hour, detects ones rising fast, and keeps them in a small separate structure. The service merges its results with the main trie's at query time.

<div class="ex">
<div class="lab">Worked example</div>

**prefix "be"**

The node for "be" stores [best buy, bed bath, bestbuy hours, beats headphones, bee movie] with scores. Typing "e" sends `GET /suggest?q=be`; the CDN misses (it's late at night in this region), and the service walks b → e and returns the five cached entries in under a millisecond of server time. The full subtree below "be" (perhaps 400,000 queries) is never visited at query time; that work was done when the trie was built. The response is cached at the CDN for 5 minutes, so the next thousand users typing "be" never reach the service.

</div>

## Alternatives

- **Elasticsearch completion suggester**: an in-memory FST (a compressed relative of the trie) with weights, supporting fuzzy prefixes. Good when you already run Elasticsearch.
- **Edge n-grams in a normal index**: index "p", "pa", "pan", "pant" for each query and search them like terms. Flexible, slower, larger.
- **Redis sorted sets** per prefix: simple for small vocabularies.

## Common mistakes

- Computing completions at request time by scanning the subtree or querying the main search index with a prefix wildcard.
- Suggesting by raw frequency only, including queries that lead nowhere.
- Counting suggestion-clicked queries, creating a self-reinforcing loop.
- No safety filter or minimum count, exposing offensive or private queries.
- Firing a request on every keystroke with no debounce or caching.

## In an interview

State the model (most probable successful completions for the prefix, from logs, decayed by time, filtered for safety and privacy), then the structure (trie with top k cached per node), then the system (offline aggregation and rebuild, in-memory replicated service, debounce, browser and CDN caching, real-time layer for trends). Give the memory arithmetic.

<div class="take">
<div class="lab">Key takeaway</div>

autocomplete predicts likely, successful full queries from a prefix, using decayed log counts weighted by how well each query performs and filtered for safety and privacy. A trie with the top k completions cached at each node turns lookups into a short walk, at the cost of memory and offline rebuilds, with debouncing and CDN caching absorbing the keystroke traffic.

</div>
