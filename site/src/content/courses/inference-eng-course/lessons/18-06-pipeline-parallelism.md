---
id: "18.06"
course: "inference-eng-course"
module: "18"
slug: "pipeline-parallelism"
title: "Pipeline Parallelism"
keyIdeas: ["Pipeline parallelism splits the model by layers into stages on different GPUs.", "Communication: activations between stages once per step, small and point to point.", "Good across nodes where links are slow.", "Bubbles: stages idle unless multiple micro-batches are in flight.", "Does not reduce per-token latency; mainly a way to fit large models."]
question: "Why does pipeline parallelism add little communication but can hurt latency, and how do engines keep all stages busy?"
answer: "Each stage holds a contiguous set of layers and only sends the activations of the tokens in flight to the next stage once per step, which is small. But a single request must pass through all stages sequentially, so per-token latency does not improve and adds transfer hops. Engines keep stages busy by having several micro-batches in flight, so while stage 2 processes batch A, stage 1 processes batch B."
sources: [{"title": "GPipe", "url": "https://arxiv.org/abs/1811.06965", "note": "Pipeline parallelism with micro-batches."}, {"title": "vLLM: Parallelism and scaling", "url": "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", "note": "pipeline-parallel-size."}]
tags: ["pipeline-parallel"]
---

## How pipeline parallelism works

Pipeline parallelism (PP) splits the model by layers. Each GPU, or group of GPUs, holds a contiguous range of layers, called a stage. Tokens flow through the stages in order.

~~~dg-flow A two-stage pipeline for an 80-layer model.
Tokens -> *Stage 1: layers 1 to 40 on GPU 0 -(activations)-> Stage 2: layers 41 to 80 on GPU 1 -> Logits
~~~

## Communication is light

Only the hidden states of the tokens in flight cross from one stage to the next, once per stage boundary per step. For 64 decode tokens with hidden size 8,192 in BF16, that is 1 MB per boundary per step, compared with tensor parallelism's 160 all-reduces per step. PP's light communication makes it suitable across servers connected by InfiniBand, where TP would be too slow.

## Bubbles

With a single batch, the stages take turns: stage 2 waits while stage 1 works, then stage 1 waits while stage 2 works. Each GPU is idle about half the time with two stages, and (P − 1) ÷ P of the time with P stages. These idle periods are called **bubbles**.

The fix is to keep several micro-batches in flight. While stage 2 processes micro-batch 1, stage 1 processes micro-batch 2. With at least as many micro-batches as stages, every stage always has work.

| Micro-batches in flight | Stage utilization |
| --- | --- |
| 1 | about 1/P |
| P | high |
| more than P | high, with more scheduling overhead |

Engines implement this by running several independent batches through the pipeline at once, sometimes called virtual engines. In decode, each sequence still advances only one token per full trip through the pipeline.

## Latency does not improve

Each token must pass through every stage in turn, so a single request's TPOT is at least the sum of all stages' times, plus the transfer time between them. PP does not reduce per-request latency the way TP does. It adds throughput capacity and memory, not speed for one user.

<div class="ex">
<div class="lab">Worked example</div>

**TP within servers, PP across them**

A dense model of about 400B parameters in FP8 needs about 400 GB for weights, more than one 8-GPU H100 server (640 GB) can hold with room for KV.

Layout: TP 8 inside each of two servers, PP 2 across them. Each server holds half the layers, split 8 ways.

Between servers, one activation transfer per step per micro-batch crosses InfiniBand. At batch 64 that is about 2 MB, about 40 µs at 50 GB/s plus latency.

Inside each server, TP's all-reduces stay on NVLink. TP across the two servers instead would put 2 all-reduces per layer on InfiniBand, which would be far slower.

</div>

## Load balance between stages

A pipeline runs at the speed of its slowest stage. If stage 1 has the embedding layer plus 40 layers and stage 2 has 40 layers plus the large LM head, stage 2 is slower, and stage 1 waits. Engines balance stages by layer count and sometimes by measured cost. With uneven layer counts (such as 61 layers over 2 stages), one stage gets an extra layer, and the imbalance is small but real.

## When to use it

- The model is too large for one server, even with TP across all its GPUs.
- The links between servers are too slow for TP.
- You need more memory for weights and KV across servers, and per-request latency is acceptable.

## Downsides

- No improvement in per-request latency.
- Bubbles at low load, when there are not enough requests to fill the pipeline.
- More complex scheduling, and failures in any stage stop the whole pipeline.

## Deep dive: the pipeline bubble

With ⟦m915⟧ stages and ⟦m916⟧ micro-batches in flight, the pipeline spends ⟦m917⟧ slots filling and draining:

⟦M914⟧

~~~dg-raw
%%timeline A 4-stage pipeline with 4 micro-batches (time slots). Grey slots are the bubble.
unit = slots
tick = 1
Stage 1 | 0-1 m1 c1 ; 1-2 m2 c2 ; 2-3 m3 c3 ; 3-4 m4 wine ; 4-7 idle line
Stage 2 | 0-1 idle line ; 1-2 m1 c1 ; 2-3 m2 c2 ; 3-4 m3 c3 ; 4-5 m4 wine ; 5-7 idle line
Stage 3 | 0-2 idle line ; 2-3 m1 c1 ; 3-4 m2 c2 ; 4-5 m3 c3 ; 5-6 m4 wine ; 6-7 idle line
Stage 4 | 0-3 idle line ; 3-4 m1 c1 ; 4-5 m2 c2 ; 5-6 m3 c3 ; 6-7 m4 wine
%%
~~~

Here ⟦m918⟧ of slots are idle. In serving, continuous batching keeps many micro-batches in flight, so ⟦m919⟧ is large and the bubble small. But one token's latency still passes through all ⟦m920⟧ stages in sequence, plus ⟦m921⟧ transfers, so PP never lowers TPOT.

~~~dg-raw
%%chart Bubble share against micro-batches in flight.
x = 1, 2, 4, 8, 16, 32
logx = 1
xlab = Micro-batches in flight M
ylab = Idle share (%)
2 stages: 50, 33, 20, 11, 5.9, 3
4 stages: 75, 60, 43, 27, 16, 8.6
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

pipeline parallelism splits layers into stages with very little communication, which makes it the usual way to span servers. Keep stages balanced and fed with several micro-batches, and do not expect it to lower per-request latency.

</div>
