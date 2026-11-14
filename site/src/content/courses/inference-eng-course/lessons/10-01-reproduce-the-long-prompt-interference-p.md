---
id: "10.01"
course: "inference-eng-course"
module: "10"
slug: "reproduce-the-long-prompt-interference-problem"
title: "Reproduce the Long-Prompt Interference Problem"
keyIdeas: ["A long prefill in a mixed step makes that step slow for every decoding user.", "The effect shows as ITL p99 spikes, not in averages.", "Reproduce it: steady short requests plus periodic long prompts.", "Plot step time over time to see the stalls directly.", "This motivates chunked prefill and, at larger scale, disaggregation."]
question: "Describe an experiment that shows long-prompt interference and the metric that reveals it."
answer: "Run a steady stream of short chat requests and measure ITL. Then inject occasional 16k-token prompts while keeping the chat load constant. Without chunked prefill, each long prompt's prefill runs in a single step lasting hundreds of milliseconds, so ITL p99 for the chat users jumps sharply while their p50 barely moves. The ITL tail (and per-step time trace) reveals it."
sources: [{"title": "Sarathi-Serve: Taming Throughput-Latency Tradeoff", "url": "https://arxiv.org/abs/2403.02310", "note": "Describes generation stalls from prefills."}, {"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Prefill-decode interference measurements."}]
tags: ["prefill", "interference"]
---

## The problem

With continuous batching, a new request's prefill runs in the same forward pass as everyone else's decode. Decode steps are short, perhaps 15 ms. A long prefill is not. A 16,000-token prompt on an 8B model needs about 16,000 × 16 GFLOP = 256 TFLOP plus attention, which takes several hundred milliseconds even on an H100. If the engine runs it as one step, no decoding user receives a token during that time.

~~~dg-flow One long prompt stalls everyone. Press Play to watch the gap appear.
Decode steps, about 15 ms -> *One long prefill step, about 400 ms -> Decode steps resume
~~~

Users see the stream freeze mid-sentence, then resume. In a voice product, the freeze becomes a pause in speech. In an agent, it adds directly to the time per tool call.

## Why averages hide it

Suppose one 400 ms stall happens every 10 seconds for a user decoding at 15 ms per token. In those 10 seconds the user receives about 640 tokens. One gap of 400 ms among 640 gaps barely moves the average: TPOT rises from 15 ms to about 15.6 ms. But the p99 of ITL captures the slowest 1% of gaps, about 6 per 10 seconds, so it jumps.

That is why you must look at ITL tails, not TPOT averages, to see interference.

## Reproduce it

1. **Background load.** 20 concurrent chat requests, each 300 tokens in and 300 out, in a closed loop.
2. **Disruptors.** Every 10 seconds, send one 16,000-token prompt with 50 output tokens.
3. **Engine setting.** Disable chunked prefill, or set the token budget above 16,000 so the whole prompt runs in one step.
4. **Recording.** Record every token's arrival time for the chat users. Run for at least 5 minutes after warm-up.
5. **Control run.** Repeat without the disruptors.

## What you see

| Metric for the chat users | Without long prompts | With long prompts |
| --- | --- | --- |
| ITL p50 | 15 ms | 16 ms |
| ITL p99 | 25 ms | 380 ms |
| TPOT p95 | 17 ms | 30 ms |
| Maximum ITL | 40 ms | 450 ms |

The median ITL barely moves. The p99 rises 15 times. Plot ITL against time for one chat user and you will see a regular pattern of spikes, one every 10 seconds, lined up with the long prompts' arrivals.

~~~dg-raw
%%widget pct The same median with a fatter tail. Interference changes the tail, not the median.
m = 16
s = 0.9
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**how many users notice**

A product serves 2,000 concurrent chat users across 100 GPUs. Long-document requests arrive at 10 per second in total, about one every 10 seconds per GPU.

Every chat user on that GPU sees a 400 ms freeze every 10 seconds. That is every user, every answer longer than about 10 seconds.

The dashboards show TPOT p50 of 16 ms and TPOT p95 of 30 ms, both "within SLO". Only the ITL p99 and user complaints reveal the problem.

</div>

## The general lesson

Any step that takes much longer than a decode step hurts every running sequence. Long prefills are the most common cause, but there are others:

- Very large batches of new short requests arriving at once.
- Recomputation of a preempted long sequence.
- Host-side pauses, such as Python garbage collection or a slow scheduler step.

The tools to see them are the same: per-token timestamps, ITL percentiles and a time plot.

## Deep dive: how many responses see a stall

Long prompts arrive at rate ⟦m600⟧. A response of ⟦m601⟧ tokens is on screen for about ⟦m602⟧ seconds, so the number of stalls it sees is roughly Poisson with mean ⟦m603⟧:

⟦M599⟧

~~~dg-raw
%%chart Share of 500-token responses that see at least one stall (12 ms steps, no chunking).
x = 0.05, 0.1, 0.25, 0.5, 1
logx = 1
xlab = Long prompts arriving per second
ylab = Responses with a stall (%)
Responses with a stall: 26, 45, 78, 95, 99.8
%%
~~~

At one long prompt every two seconds, 95% of answers contain a visible hitch, while median ITL has not moved at all. This is why interference is a tail problem that most users nonetheless feel.

<div class="take">
<div class="lab">Key takeaway</div>

long prefills stall every decoding user on the same GPU. Averages hide it, so measure ITL p99 and maximum ITL with mixed traffic, and plot ITL over time to see the spikes.

</div>
