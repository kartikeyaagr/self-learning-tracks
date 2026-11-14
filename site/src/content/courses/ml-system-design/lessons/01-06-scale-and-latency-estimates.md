---
id: "01.06"
course: "ml-system-design"
module: "01"
slug: "scale-and-latency-estimates"
title: "Scale and latency estimates"
keyIdeas: ["Rough numbers for traffic, data size, model size and latency decide the architecture. Do them early and out loud.", "Convert daily volumes to **queries per second**: divide by 86,400, then multiply by a peak factor of 2 to 10.", "Split the end-to-end **latency budget** across stages (feature fetch, candidate retrieval, ranking, network) and check each against what the hardware can do.", "Estimate storage and compute for training data, features and models; these often decide between batch and online, or between a big and a small model."]
question: "A shopping app ranks 1,000 candidate products per home-screen load. It has 30M daily active users, each loading the home screen 8 times a day. Estimate peak ranking QPS and items scored per second, and decide whether a model costing 20 microseconds per item on one CPU core is affordable."
answer: "Loads per day = 30M × 8 = 240M, or about 2,800 per second on average. With a peak factor of 3, peak is about 8,300 loads per second. Each load scores 1,000 items, so about 8.3M items per second at peak. At 20 µs per item, one core scores 50,000 items per second, so you need about 166 cores busy at peak, or roughly 250 to 350 cores with headroom. That is affordable (a few dozen servers) but significant. Latency per request is 1,000 × 20 µs = 20 ms on one core, so you should parallelise scoring across cores or batch items to hit a ranking budget of around 30 ms. If cost matters, add a cheaper first-stage ranker that cuts 1,000 candidates to 200 before the heavy model."
sources: [{"title": "Latency numbers every programmer should know", "url": "https://gist.github.com/jboner/2841832", "note": "Orders of magnitude for memory, disk and network."}, {"title": "Colin Scott, Interactive latency numbers by year", "url": "https://colin-scott.github.io/personal_website/research/interactive_latency.html", "note": "How the reference numbers have changed over time."}]
practice: ["estimate-model-memory-and-minimum-gpu-count", "p50-p95-and-p99-inference-latency", "optimize-throughput-and-cost-under-latency-slo", "kv-cache-memory-for-mha-mqa-gqa-and-mla"]
tags: ["back-of-the-envelope", "capacity planning"]
---

## Why estimate

Numbers turn opinions into decisions. "We could score every item online" is a feeling; "that is 8 million model evaluations a second" is a design constraint. A few minutes of arithmetic tells you whether a design is feasible, roughly what it costs and where the bottleneck is.

## Traffic

Start from users and actions:
$$\text{average QPS} = \frac{\text{users} \times \text{actions per user per day}}{86{,}400}$$
Then multiply by a peak factor, typically 2 to 3 for consumer traffic with daily cycles and up to 10 for event spikes. 100M actions a day is about 1,160 per second on average.

For ranking systems, multiply by the number of items scored per request. This is usually the number that matters for compute.

## Latency budgets

State the end-to-end target as a percentile, then split it:

| Stage | Budget (p99) | Notes
| --- | --- | --- |
| Network and gateway | 20 ms | outside your control
| Feature fetch | 10 ms | key-value lookups, batched
| Candidate retrieval | 15 ms | approximate nearest neighbour
| Ranking | 30 ms | the ML model
| Business rules and re-ranking | 5 ms | filters, diversity
| Total | 80 ms | leaves headroom under 100 ms

Stages run in sequence add up; stages run in parallel cost the slowest. Remember that the p99 of a request that fans out to many backends is worse than any single backend's p99, because the slowest of many calls dominates.

~~~dg-flow A ranking request's latency budget. The model gets only part of it.
Request -(20 ms)-> Gateway -(10 ms)-> Feature fetch -(15 ms)-> Retrieval -(30 ms)-> *Ranking -(5 ms)-> Response
~~~

## Useful reference numbers

Keep orders of magnitude in mind: a main-memory read is about 100 ns, a read from a key-value store in the same data centre about 0.5 to 1 ms, a round trip across regions 50 to 150 ms. A small gradient-boosted model scores in microseconds per item; a mid-sized neural ranker tens of microseconds per item on CPU; a transformer forward pass on a GPU milliseconds per batch.

## Storage and training data

Estimate the training set: 200M impressions a day × 30 days × 500 bytes of features is 3 TB. That fits on one large machine for gradient-boosted trees with sampling, or calls for distributed training for a neural model. Embedding tables are often the largest part of a recommender: 100M users × 64 dimensions × 4 bytes is 25.6 GB.

For large models, memory decides hardware. A model with $N$ parameters in 16-bit precision needs about $2N$ bytes for weights alone; a 7B-parameter model needs about 14 GB, plus memory for activations or a KV cache.

<div class="ex">
<div class="lab">Worked example</div>

**an estimate that changed the design**

A team planned to compute personalised recommendations online for 50M users. Estimating 50M users × 5 visits × 2,000 candidates gave 500 billion item scores a day. They switched to nightly batch generation of a top-200 list per user (10 billion scores, once a day on spare capacity), with a light online re-ranker for context such as time of day.

</div>

## Present it clearly

In an interview, write the three or four numbers that matter: peak QPS, items scored per second, end-to-end latency budget, and storage for the biggest table. Round aggressively. The point is the order of magnitude and what it implies, not the third digit. Then say what the numbers imply: "8 million item scores a second means we need a cheap first-stage ranker" is the sentence the interviewer is waiting for.

## Cost per prediction

Estimate cost alongside capacity. If a CPU core costs about $0.04 an hour and scores 50,000 items a second, a billion item scores cost under a dollar. A GPU at a few dollars an hour that serves 500 large-model requests a second costs around half a cent per thousand requests. Multiplying these by daily volume shows quickly whether a design fits the budget, and which stage is worth optimising first. Usually one stage dominates, and that is where a cheaper model, caching or batching pays off most.

<div class="take">
<div class="lab">Key takeaway</div>

convert users and actions into peak QPS and items per second, split the p99 latency budget across stages, and estimate data and model sizes; these few numbers decide between batch and online, and between one model and a cascade.

</div>
