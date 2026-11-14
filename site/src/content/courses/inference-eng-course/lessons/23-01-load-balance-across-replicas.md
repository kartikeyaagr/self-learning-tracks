---
id: "23.01"
course: "inference-eng-course"
module: "23"
slug: "load-balance-across-replicas"
title: "Load Balance Across Replicas"
keyIdeas: ["LLM requests vary in cost by 100x or more; request counts are a bad proxy for load.", "Round robin and random spread requests, not work.", "Least outstanding requests is a big step up.", "Token- or KV-aware balancing is better still.", "Balancing must also consider cache locality (next lessons)."]
question: "Why is round-robin load balancing a poor fit for LLM replicas?"
answer: "Requests vary hugely in cost (prompt and output length), so equal request counts produce very unequal load: one replica can be stuck with several long generations while another is idle. Round robin also ignores KV cache state and prefix locality. Least-outstanding-requests or token-aware balancing does better."
sources: [{"title": "Google SRE Book: Load Balancing in the Datacenter", "url": "https://sre.google/sre-book/load-balancing-datacenter/", "note": "Balancing policies."}, {"title": "vLLM: Production stack router", "url": "https://github.com/vllm-project/production-stack", "note": "Routing options for vLLM fleets."}]
tags: ["routing"]
---

## Policies

| Policy | Uses | Weakness |
| --- | --- | --- |
| Round robin | nothing | ignores how big each request is |
| Random | nothing | same, plus random clumping |
| Least connections or outstanding requests | in-flight count per replica | treats a tiny and a huge request the same |
| Least tokens or KV usage | engine metrics | needs fresh metrics |
| Power of two choices | sample 2 replicas, pick the less loaded | works well even with stale information |

## Why cost varies so much

A 50-token answer to a short prompt and a 4,000-token generation after a 30k-token prompt both count as "one request". The second uses hundreds of times more GPU time and KV memory.

<div class="ex">
<div class="lab">Worked example</div>

**two requests, very different costs**

Request A: 200 prompt tokens, 50 output tokens. Prefill takes a few ms, decode about 0.5 s, and KV peaks at about 32 MB on an 8B model.

Request B: 30,000 prompt tokens, 4,000 output tokens. Prefill takes about 1.5 s, decode about 60 s, and KV peaks at about 4.3 GB.

Round robin sends them to replicas in turn as if they were equal. A replica that happens to get three requests like B in a row has filled 13 GB of KV and is far slower than its neighbors.

</div>

~~~dg-raw
%%widget lb Raise the size spread and compare the p99 latency of each policy. Then push the load toward 95%.
r = 0.75
s = 1.2
%%
~~~

## Practical setup

1. **Start with least outstanding requests.** It needs no engine integration and already beats round robin when request sizes vary.
2. **Add engine signals** such as waiting-queue length and KV usage, when the engine exposes them.
3. **Use power of two choices** to avoid herding when metrics are a few seconds old: many routers all picking the single "best" replica at once.

## Deep dive: why round robin balances requests but not work

Round robin gives each of ⟦m1039⟧ replicas the same number of requests, ⟦m1040⟧. If request cost has mean ⟦m1041⟧ and coefficient of variation ⟦m1042⟧, the work on one replica has

⟦M1037⟧

LLM request costs are heavy-tailed: with prompts from 100 to 30,000 tokens, ⟦m1043⟧ can reach 3 or more. With 16 requests in flight per replica, the CV of per-replica work is ⟦m1044⟧. Two replicas with the same request count can easily carry a factor of two different load. A router that estimates cost per request, for example

⟦M1038⟧

with prompt length ⟦m1045⟧ and expected output ⟦m1046⟧, balances work instead of counts.

<div class="take">
<div class="lab">Key takeaway</div>

balance work, not requests. Least outstanding requests is the minimum; routing on queue and KV signals is better.

</div>
