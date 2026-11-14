---
id: "26.07"
course: "inference-eng-course"
module: "26"
slug: "rl-rollout-inference"
title: "RL Rollout Inference"
keyIdeas: ["RL training alternates rollout generation and model updates.", "Rollouts: many samples per prompt, high temperature, long outputs.", "Weights update every step: fast weight sync into the engine is essential.", "Long-tail generation lengths stall synchronous training steps.", "Techniques: parallel sampling sharing prompt KV, async or partial rollouts, colocated or separate rollout GPUs."]
question: "What makes rollout generation in RL training different from production serving?"
answer: "Rollouts generate many samples per prompt (for example 8 to 64) with high temperature and often long outputs; weights change every training step, so the inference engine must reload or sync weights quickly; the generation phase blocks training, so the slowest rollout (long tail of lengths) stalls the whole step; and throughput per training step matters, not per-user latency."
sources: [{"title": "verl", "url": "https://github.com/volcengine/verl", "note": "RL training framework using vLLM/SGLang for rollouts."}, {"title": "OpenRLHF", "url": "https://github.com/OpenRLHF/OpenRLHF", "note": "RLHF framework with vLLM rollouts."}]
tags: ["rl"]
---

## The loop

~~~dg-flow One synchronous RL step.
Prompts -> *Rollout engine generates n samples each -> Reward and scoring -> Training update -> Sync weights to engine
~~~

Reinforcement learning for language models alternates between generating samples (rollouts) with the current policy and updating the policy from their rewards. Rollouts are often more than half of the wall-clock time of each step.

## Differences from serving

| Aspect | Serving | RL rollouts |
| --- | --- | --- |
| Weights | fixed | change every step |
| Samples per prompt | 1 | many, often 8 to 64 |
| Objective | latency SLO | step throughput |
| Tail behavior | the p99 user | the slowest sample blocks the whole step |

<div class="ex">
<div class="lab">Worked example</div>

**the long tail**

A step runs 512 prompts × 16 samples. Most samples finish in 2,000 tokens. A few run to the 16,000-token limit.

At 25 ms per token, most samples are done in under a minute, but the longest takes almost 7 minutes. For most of that time, the GPUs are nearly idle, waiting on a handful of sequences.

Capping length, or letting long samples continue into the next step (asynchronous or partial rollouts), recovers most of the idle time.

</div>

## Techniques

- **Parallel sampling** shares the prompt's KV across all samples of one prompt.
- **Fast weight sync:** update the engine's weights in place over NCCL from the trainer, instead of saving and reloading from disk.
- **Tail handling:** cap lengths, or use asynchronous or partial rollouts.
- **Placement:** colocate the trainer and the engine on the same GPUs, switching memory between them, or use separate rollout GPUs.

## Deep dive: the tail of a rollout batch

RL rollouts start together and end when the longest finishes. If output lengths are roughly exponential with mean ⟦m1125⟧, the share of sequences still running at time ⟦m1126⟧ and the expected longest are

⟦M1124⟧

~~~dg-raw
%%chart Share of a rollout batch still generating, against time in units of the mean length.
x = 0, 1, 2, 3, 4, 5, 6
xlab = Time (multiples of the mean output length)
ylab = Sequences still running (%)
Running: 100, 37, 13.5, 5, 1.8, 0.7, 0.25
%%
~~~

For 512 rollouts with a mean of 2,000 tokens, the longest is about 13,600. For most of that time, fewer than 5% of the batch is running and the GPU is nearly idle. Fixes: refill finished slots with new prompts (continuous batching across rollout steps), cap lengths, or move stragglers to a separate pool.

<div class="take">
<div class="lab">Key takeaway</div>

RL rollouts are throughput jobs with changing weights and many samples per prompt. Weight-sync speed and length tails decide efficiency.

</div>
