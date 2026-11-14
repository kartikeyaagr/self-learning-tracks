---
id: "20.06"
course: "inference-eng-course"
module: "20"
slug: "cache-admission-eviction-locality"
title: "Cache Admission, Eviction & Locality"
keyIdeas: ["Admission: decide what is worth caching at all.", "Eviction: decide what to drop; LRU is a baseline, not always best.", "Value of an entry ~ probability of reuse x recompute cost saved / size.", "Locality: route related requests to the same place so caches stay small and hot.", "Measure hit rate, bytes written vs bytes reused, and TTFT saved."]
question: "Why might \"cache everything, evict LRU\" be a poor policy for a lower KV tier, and what would you use instead?"
answer: "Most prefixes are never reused, so caching everything spends bandwidth writing one-off KV and pushes out valuable entries (cache pollution). Better: admit only entries likely to be reused (seen more than once, long prefixes, active conversations), weigh eviction by size and recompute cost rather than recency alone, and route requests so each replica sees a stable subset of prefixes."
sources: [{"title": "Mooncake", "url": "https://arxiv.org/abs/2407.00079", "note": "Production KV caching policies."}, {"title": "Wikipedia: Cache replacement policies", "url": "https://en.wikipedia.org/wiki/Cache_replacement_policies", "note": "LRU, LFU and variants."}]
tags: ["caching"]
---

## Admission

Not every prefix deserves cache space. Admission rules decide what gets in.

| Rule | Rationale |
| --- | --- |
| Admit on the second occurrence | one-off prompts never pollute the cache |
| Admit only prefixes longer than N tokens | short prefixes are cheap to recompute |
| Admit active conversations | the next turn is likely to come soon |

## Eviction

Score each entry by the expected savings per byte:

```
value = p_reuse × recompute_cost / size
```

- **LRU** approximates p_reuse with recency. It works for chat, where recent conversations get the next turn.
- **LFU** or frequency-plus-recency variants protect popular system prompts and documents from being flushed by a burst of one-off traffic.

<div class="ex">
<div class="lab">Worked example</div>

**why LRU alone can hurt**

A 20k-token system prompt is used by 40% of requests. A batch job then sends 10,000 unique 4k-token prompts in five minutes.

Under pure LRU, the batch job fills the cache and pushes out the system prompt. The next interactive request pays a full 20k-token prefill.

A frequency-aware policy, or a rule that only admits on second occurrence, keeps the system prompt and lets the one-off prompts pass through.

</div>

## Locality

Without affinity routing, every replica's cache must hold every popular prefix. With prefix-aware routing, each replica holds only its share, so the same total memory covers much more of the working set.

~~~dg-flow Prefix-aware routing.
Request -> Hash the prefix -> *Pick the replica that holds it -> Fall back to least loaded if overloaded
~~~

## Measuring cache economics

| Metric | Good sign |
| --- | --- |
| Hit rate | rises as capacity grows |
| Bytes reused ÷ bytes written | well above 1 |
| Prefill tokens saved per second | large and stable |

## Deep dive: hit rate against cache size

If prefix popularity follows a Zipf law (the ⟦m984⟧-th most popular is used in proportion to ⟦m985⟧) over ⟦m986⟧ prefixes, a cache holding the top ⟦m987⟧ gets a hit rate of about

⟦M983⟧

~~~dg-raw
%%chart Hit rate against cache size, Zipf popularity over 100,000 distinct prefixes.
x = 10, 100, 1000, 10000, 100000
logx = 1
xlab = Prefixes the cache can hold
ylab = Hit rate (%)
Hit rate: 24, 43, 62, 81, 100
%%
~~~

Every tenfold increase in capacity adds about the same number of points. The first tiers (GPU memory) are small but catch the head of the distribution; a large CPU or disk tier is needed for the long tail, and its value depends on how much of your traffic lives there.

<div class="take">
<div class="lab">Key takeaway</div>

cache what is likely to be reused, evict by value per byte, and route for locality. A bigger cache without these policies mostly stores junk.

</div>
