---
id: "23.03"
course: "inference-eng-course"
module: "23"
slug: "prefix-cache-aware-routing"
title: "Prefix- & Cache-Aware Routing"
keyIdeas: ["Prefix caches are per replica; routing decides whether requests hit them.", "Send requests with the same prefix (session, system prompt, document) to the same replica.", "Pure affinity creates hot spots; cap load per replica.", "Approaches: consistent hashing with bounded loads, approximate prefix trees in the router.", "Measure hit rate per replica and load imbalance together."]
question: "How do you balance cache affinity against load balance when routing?"
answer: "Score each candidate replica by expected cache hit (length of prompt prefix it likely holds) and by its load. Send the request to the replica with the best hit if its load is under a threshold; otherwise fall back to the least-loaded replica (and let the cache build there). Consistent hashing on a prefix key (like session id or system-prompt hash) gives affinity with bounded imbalance."
sources: [{"title": "SGLang: Router", "url": "https://docs.sglang.ai/", "note": "Cache-aware routing across workers."}, {"title": "Consistent Hashing with Bounded Loads", "url": "https://arxiv.org/abs/1608.01350", "note": "Affinity with load limits."}]
tags: ["routing", "prefix"]
---

## Approaches

| Approach | How |
| --- | --- |
| Session affinity | hash the conversation id to a replica |
| Prefix hash | hash the first N tokens, usually the system prompt |
| Router-side prefix tree | the router remembers which prefixes went where |
| Engine-reported cache state | replicas publish the hashes of their cached blocks |

## Balancing affinity and load

```
best = replica with the longest expected prefix match
if load(best) < threshold:
    route to best
else:
    route to least loaded
```

Pure affinity creates hot spots. If one popular system prompt maps to one replica, that replica gets most of the traffic. Bounded-load affinity keeps the cache benefit while capping how overloaded any replica can get.

## Effect

| Routing | Hit rate | Load balance |
| --- | --- | --- |
| Random | low | good |
| Pure affinity | high | hot spots |
| Affinity with bounded load | high | good |

<div class="ex">
<div class="lab">Worked example</div>

**how much a hit saves**

A RAG service sends a 6,000-token system prompt plus documents, then a 300-token question. 8 replicas, 8B model.

Random routing: each replica sees every prefix, but its cache can hold only part of the working set, so the hit rate is about 30%. Average prefill per request is about 4,400 tokens.

Affinity with bounded load: each document set mostly lands on one replica, and the hit rate rises to about 80%. Average prefill falls to about 1,500 tokens.

Prefill work drops by about two thirds, so TTFT p50 drops by a similar factor and the fleet frees capacity for more traffic.

</div>

## Multi-turn chat

Session affinity alone often captures most of the benefit. Each conversation's history stays on one replica, so every new turn only prefills the new message.

## Deep dive: when to follow the cache and when to follow load

Routing a request to the replica that holds its prefix saves prefill time ⟦m1051⟧. It is worth it while the extra queueing there is smaller:

⟦M1049⟧

For a 30,000-token cached document (prefill about 1.3 s on an 8B model), waiting even half a second longer on the cached replica is a good trade. For a 500-token prefix (15 ms), almost any queue difference wins.

Consistent hashing keeps most prefixes on the same replica when the pool changes. Adding one replica to ⟦m1052⟧ moves only

⟦M1050⟧

of the keys, against nearly all of them with plain modulo hashing. Bounded-load variants cap any replica at ⟦m1053⟧ times the average to stop popular prefixes from overloading one replica.

<div class="take">
<div class="lab">Key takeaway</div>

cache-aware routing turns per-replica caches into a fleet-wide win. Prefer the replica that holds the cache, unless it is overloaded.

</div>
