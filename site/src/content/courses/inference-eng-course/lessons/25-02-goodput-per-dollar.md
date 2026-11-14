---
id: "25.02"
course: "inference-eng-course"
module: "25"
slug: "goodput-per-dollar"
title: "Goodput per Dollar"
keyIdeas: ["Goodput per dollar = SLO-meeting tokens (or requests) per second / cost per second.", "Compare hardware, models and configs on this single number.", "It changes with the SLO: tighter SLOs favor faster hardware or smaller models.", "Also report the SLO and workload, or the number means nothing.", "Use it to decide between more cheap GPUs and fewer fast ones."]
question: "Why compare configurations by goodput per dollar instead of throughput per dollar?"
answer: "Throughput per dollar rewards overloading the GPU, which raises raw tokens/s but makes many requests miss the SLO. Those tokens have less or no value to the product. Goodput per dollar counts only tokens from requests that met the SLO, so it measures what you are actually buying."
sources: [{"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Goodput as the serving objective."}, {"title": "Google SRE Book: Service Level Objectives", "url": "https://sre.google/sre-book/service-level-objectives/", "note": "SLO basics."}]
tags: ["cost", "goodput"]
---

## The formula

```
goodput_per_dollar = goodput_tokens_per_s / (gpu_cost_per_hour / 3600)
```

Goodput counts only tokens from requests that met the SLO. Throughput counts all tokens, including those from requests that were too slow to be useful.

## Example

| Config | Throughput (tokens/s) | SLO met | Goodput (tokens/s) | \$/hour | Goodput tokens per \$ |
| --- | --- | --- | --- | --- | --- |
| A: overloaded | 6,000 | 60% | 3,600 | 2.5 | 5.2M |
| B: tuned | 4,800 | 99% | 4,750 | 2.5 | 6.8M |
| C: bigger GPU | 9,000 | 99% | 8,900 | 5.0 | 6.4M |

- **A** looks fastest by throughput, but 40% of its tokens are too slow to count.
- **B** runs a smaller batch so that nearly every request meets the SLO. It wins on goodput per dollar.
- **C** is very fast, but it costs twice as much per hour.

## SLO sensitivity

The ranking depends on the SLO. Tighten the TPOT SLO and A and B may both fail, leaving C as the only option. Loosen it and A may meet it, becoming the cheapest. Always state the SLO next to a goodput-per-dollar number.

<div class="ex">
<div class="lab">Worked example</div>

**the same hardware under two SLOs**

An 8B model on an L40S reaches 3,000 tokens/s at batch 64 with TPOT 35 ms, and 1,800 tokens/s at batch 24 with TPOT 18 ms.

With a 40 ms TPOT SLO, run batch 64: about 9M goodput tokens per dollar at \$1.20 per hour.

With a 20 ms TPOT SLO, batch 64 fails. Batch 24 gives about 5.4M per dollar. An H100 might now be cheaper per useful token, because it can hold 20 ms at a much larger batch.

</div>

## Deep dive: cost as a function of the TPOT SLO

A TPOT target caps the batch size, and the batch size sets cost per token. Using the step model from 05.02:

⟦M1092⟧

~~~dg-raw
%%chart Cost per million output tokens against the TPOT SLO (Llama 3 8B, H100, 1k context, $2.50/h).
x = 7.1, 8.3, 9.8, 12.9, 19
xlab = TPOT SLO (ms)
ylab = $ per million output tokens
Cost: 0.62, 0.18, 0.11, 0.070, 0.052
%%
~~~

Going from a 7 ms target (batch 8) to 13 ms (batch 128) cuts cost almost 9 times. A user cannot see the difference between 140 and 78 tokens per second when reading, so a tight TPOT SLO is often the most expensive line in a product spec.

<div class="take">
<div class="lab">Key takeaway</div>

buy goodput, not throughput. Goodput per dollar under a stated SLO is the number to compare configurations on.

</div>
