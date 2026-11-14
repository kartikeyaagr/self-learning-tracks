---
id: "19.04"
course: "inference-eng-course"
module: "19"
slug: "load-imbalance-hot-experts"
title: "Load Imbalance & Hot Experts"
keyIdeas: ["Step time per layer is set by the most loaded GPU.", "Expert usage is skewed and varies with traffic.", "Replicating hot experts spreads their load.", "Dynamic rebalancing uses recent statistics to move or replicate experts.", "Capacity limits (dropping overflow tokens) trade quality for balance; avoid in serving if possible."]
question: "Why does one hot expert slow down the whole MoE layer under expert parallelism, and name two mitigations."
answer: "All GPUs must finish their experts before the combine all-to-all completes and the next layer starts, so the GPU with the most tokens sets the layer time; others wait idle. Mitigations: replicate hot experts on more GPUs and split their tokens (as in redundant-expert schemes), and rebalance placement periodically from observed statistics; capacity limits that drop or reroute overflow tokens also exist but can affect quality."
sources: [{"title": "DeepSeek-V3 Technical Report", "url": "https://arxiv.org/abs/2412.19437", "note": "Redundant experts for load balancing at inference."}, {"title": "Switch Transformers", "url": "https://arxiv.org/abs/2101.03961", "note": "Capacity factor and token dropping."}]
tags: ["moe"]
---

## Why imbalance hurts

Every GPU must finish its experts before the combine step. The layer is therefore as slow as its busiest GPU.

~~~dg-flow Imbalanced expert-parallel layer.
GPU 1: 120 tokens -> *GPU 2: 900 tokens (hot) -> GPU 3: 140 tokens -> All wait for GPU 2
~~~

If the busiest GPU has three times the mean load, the other GPUs spend about two thirds of the layer idle. Adding GPUs does not help, because the hot expert still lives on one of them.

~~~dg-raw
%%widget moe Change batch size, top-k and routing skew, and watch the busiest GPU. Then turn on replication of the hottest experts.
b = 7
k = 2
s = 0.6
%%
~~~

Things to try in the widget:

- At skew 0 and 16 tokens, imbalance comes from chance alone. Raise the batch to 4096 and it nearly disappears.
- At skew 1.2, a large batch no longer helps: the skew is in the traffic, not in sampling noise.
- Turn on replication at skew 1.2. The max-to-mean ratio drops sharply at the cost of 8 extra expert copies.

## Measuring

- **Tokens per GPU per MoE layer per step**, and its max ÷ mean. Engines can log this cheaply from the router output.
- **Idle time inside MoE layers** in a profiler trace: GPUs waiting in the combine all-to-all while one GPU is still computing.
- **Per-layer patterns:** some layers are far more skewed than others.

## Mitigations

| Method | How it works | Cost |
| --- | --- | --- |
| Redundant experts | keep extra copies of the hottest experts on spare slots and split their tokens | extra memory per copy |
| Periodic rebalancing | move experts between GPUs based on recent load | migration traffic, stats collection |
| Larger batches | sampling noise averages out | more KV memory, higher latency |
| Capacity factor or token dropping | cap tokens per expert, drop or reroute the rest | quality loss; avoid in serving |

<div class="ex">
<div class="lab">Worked example</div>

**sizing redundancy**

A deployment runs 256 experts per layer on 32 GPUs (8 per GPU). Statistics show the top 16 experts take 30% of tokens.

Giving each GPU one extra slot adds 32 slots, enough for two copies of each of the top 16 experts. Their load is now split three ways.

Memory cost: one more expert per GPU per layer, about 12% more expert memory. The max-to-mean ratio in this kind of setup typically falls from about 2.5 to about 1.3.

</div>

## In practice

Large production MoE deployments collect expert load continuously, and every few minutes to hours they recompute placement, replicate the hottest experts and move weights in the background.

## Deep dive: imbalance from random routing alone

Even with a perfectly balanced router, the number of tokens per expert varies. With mean ⟦m958⟧, the busiest of ⟦m959⟧ experts gets about

⟦M957⟧

~~~dg-raw
%%chart Expert-layer efficiency from random imbalance alone, against mean tokens per expert.
x = 8, 32, 128, 512
logx = 1
xlab = Mean tokens per expert μ
ylab = Efficiency (%)
8 experts: 58, 74, 85, 92
256 experts: 46, 63, 77, 87
%%
~~~

Real routing is skewed on top of this. Redundant copies of the hottest experts, placed on less loaded GPUs, attack the skew. Larger batches attack the random part: efficiency rises with ⟦m960⟧, slowly.

<div class="take">
<div class="lab">Key takeaway</div>

in expert parallelism the busiest GPU sets the pace. Measure load per GPU, then replicate or rebalance hot experts. Bigger batches only fix noise, not skew.

</div>
