---
id: "00.04"
course: "inference-eng-course"
module: "00"
slug: "probability-sampling-benchmark-statistics"
title: "Probability, Sampling & Benchmark Statistics"
keyIdeas: ["An LLM outputs a probability distribution over the vocabulary at each step. Sampling picks a token from it.", "Temperature reshapes the distribution; top-k and top-p cut off unlikely tokens.", "Benchmarks are samples too. Report percentiles (p50, p95, p99), not just means.", "Tail percentiles need many samples. p99 from 200 requests rests on 2 data points.", "Repeat runs and report spread. A 5% \"speedup\" inside run-to-run noise is not a result."]
question: "You run a benchmark twice and get p99 latency of 820 ms and 1,140 ms. What does this tell you, and what would you do before reporting a number?"
answer: "The p99 is unstable, likely because too few requests were measured or the system was not at steady state. Run longer (thousands of requests), discard warm-up, repeat several times, and report the median of runs with a spread or confidence interval rather than one number."
sources: [{"title": "The Curious Case of Neural Text Degeneration (nucleus sampling)", "url": "https://arxiv.org/abs/1904.09751", "note": "Why top-p sampling exists."}, {"title": "The Tail at Scale", "url": "https://research.google/pubs/the-tail-at-scale/", "note": "Why tail latency matters for services."}]
tags: ["statistics", "sampling"]
---

## Two kinds of randomness

Probability shows up in inference in two very different places:

1. **Sampling**: the model outputs a probability distribution over the vocabulary, and you choose the next token from it.
2. **Measurement**: latency varies from request to request, so a benchmark gives you a distribution, not a number.

Mixing up these two is common. Seeds control the first. Neither seeds nor wishes control the second.

## From logits to probabilities

The model's last layer outputs one score, a **logit**, per vocabulary entry (128,256 entries for Llama 3). Softmax turns logits into probabilities. Then a sampler picks a token.

- **Greedy**: always take the highest-probability token. Deterministic, can loop or sound flat.
- **Temperature T**: divide logits by T before softmax. T < 1 sharpens the distribution, T > 1 flattens it. T → 0 is greedy.
- **Top-k**: keep only the k most likely tokens, renormalize.
- **Top-p (nucleus)**: keep the smallest set of tokens whose probabilities sum to at least p, renormalize.

~~~dg-raw
%%widget sample Change temperature, top-k and top-p and watch the distribution reshape. Draw samples to see the effect.
%%
~~~

## Sampling and performance

Sampling looks trivial but is not free at scale. The logit vector for one token is 128,256 numbers. With batch 256 that is 33 million logits per step, and a naive top-p needs a sort. Engines use fused sampling kernels that avoid a full sort. Also note that greedy decoding is not perfectly reproducible across batch sizes: floating-point sums happen in different orders when the batch changes, which can flip a near-tie.

## Benchmark statistics

Latency is a distribution with a long right tail: most requests are fast, a few are much slower because they queued behind a long prompt, hit a garbage-collection pause or a preemption. So:

- The **mean** is pulled up by the tail and describes nobody's experience.
- The **median (p50)** is the typical request.
- **p95 and p99** describe your unhappiest users and are what SLOs are written against.

~~~dg-raw
%%widget pct Raise the spread and watch p99 run away from the median.
m = 400
s = 0.6
%%
~~~

## How many samples do you need?

To estimate p99 you need enough requests that the top 1% is not just a handful. With 10,000 requests, p99 rests on 100 samples. With 300 requests, it rests on 3, and two runs can differ by 2×.

<div class="ex">
<div class="lab">Worked example</div>

**comparing two configs**

Config A: p95 TTFT 820 ms. Config B: 760 ms. Each from 200 requests.

Re-run each three times. A gives 820, 910, 770. B gives 760, 840, 900. The ranges overlap completely, so you cannot claim B is faster.

Run 2,000 requests per config, or report a bootstrap confidence interval, before deciding.

</div>

## Variance you can control

- Warm up the server first: the first requests pay for CUDA graph capture, compilation and memory allocation.
- Fix prompts and output lengths, or fix the distribution they are drawn from and its seed.
- Pin GPU clocks if you can, and watch for thermal throttling on long runs.
- Run each configuration several times and report the spread, not only the best run.

## Deep dive: temperature as a change of distribution

With logits ⟦m109⟧ and temperature ⟦m110⟧:

⟦M107⟧

As ⟦m111⟧ the largest logit dominates and sampling becomes greedy. As ⟦m112⟧ all tokens become equally likely. Entropy, ⟦m113⟧, measures how spread out the choice is; it rises steadily with ⟦m114⟧.

~~~dg-raw
%%chart Probability of the top three tokens as temperature changes (logits 4, 3, 2, then five tokens at 1).
x = 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2
xlab = Temperature
ylab = Probability
Top token: 0.98, 0.86, 0.7, 0.57, 0.48, 0.41, 0.32
Second: 0.02, 0.12, 0.19, 0.21, 0.21, 0.21, 0.2
Third: 0, 0.02, 0.05, 0.08, 0.1, 0.11, 0.12
%%
~~~

Top-p (nucleus) sampling keeps the smallest set ⟦m115⟧ of tokens, taken in order of probability, such that ⟦m116⟧, then renormalizes within ⟦m117⟧. The set grows when the model is unsure and shrinks when it is confident, which is why top-p usually behaves better than a fixed top-k.

## Deep dive: how many samples a percentile needs

The ⟦m118⟧-th percentile of ⟦m119⟧ samples is the value at rank ⟦m120⟧. The number of samples that fall below the true percentile is binomial, with standard deviation ⟦m121⟧. An approximate 95% confidence interval for the percentile is the range of sorted values between ranks

⟦M108⟧

<div class="ex">
<div class="lab">Worked example</div>

**is my p99 trustworthy?**

With ⟦m122⟧ requests: rank ⟦m123⟧. The true p99 lies somewhere between your 984th and 996th fastest request.

If those two values are 1.8 s and 2.6 s, your "p99 = 2.1 s" could really be anywhere in that range. Comparing two configs whose p99s differ by 0.2 s tells you nothing.

With ⟦m124⟧: rank ⟦m125⟧, a much tighter band. Rule of thumb: you want at least 100 samples in the tail beyond the percentile, so ⟦m126⟧, which is 10,000 for p99.

</div>

## Deep dive: is a difference real?

For means of two configs measured over ⟦m127⟧ repeated runs each, with run-to-run standard deviation ⟦m128⟧, the standard error of the difference is ⟦m129⟧. Treat a difference smaller than about twice that as noise.

<div class="ex">
<div class="lab">Worked example</div>

**Example**

Throughput over 5 runs: config A averages 4,120 tokens/s, B averages 4,260, and the run-to-run standard deviation is 90.

Standard error of the difference: ⟦m130⟧. The observed gap is 140, about 2.5 standard errors. That is probably real, but a sixth and seventh run would make it convincing.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

sampling controls what the model says; statistics control whether your benchmark means anything. Report percentiles, use enough samples for the tail, and repeat runs.

</div>
