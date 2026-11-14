---
id: "05.02"
course: "inference-eng-course"
module: "05"
slug: "sweep-concurrency-request-rate"
title: "Sweep Concurrency & Request Rate"
keyIdeas: ["Sweep concurrency in powers of two (1, 2, 4 ... 256) with a fixed workload.", "Plot throughput vs concurrency and latency vs concurrency side by side.", "Throughput rises then plateaus; latency stays flat then rises.", "Sweep open-loop request rate too, to find where queues start growing.", "The usable point is where latency still meets the SLO, usually before the plateau."]
question: "Describe the curves you expect for throughput and TPOT as concurrency rises from 1 to 256 on a real engine, and where the knee usually comes from."
answer: "Throughput rises almost linearly at first, then bends and flattens; TPOT stays near baseline at low concurrency, then climbs. The knee appears when decode stops being purely bandwidth bound (compute starts to matter), when the KV cache fills and the engine queues or preempts, or when batched prefills start interfering with decode."
sources: [{"title": "Anyscale: How continuous batching enables 23x throughput", "url": "https://www.anyscale.com/blog/continuous-batching-llm-inference", "note": "Throughput vs latency curves under batching."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Automatic rate sweeps."}]
tags: ["load-testing"]
---

## The sweep

Hold the workload fixed, for example 1,000 tokens in and 256 out with `ignore_eos`, and change only the load. Run each level for at least two to three minutes after warm-up:

- **Closed loop**: concurrency 1, 2, 4, 8, 16, 32, 64, 128 and 256.
- **Open loop**: request rates from well below to just above the expected capacity.

At each point record throughput, p50 and p95 TTFT, p50 and p95 TPOT, and the engine's running, waiting and KV usage metrics.

## The expected shape

| Concurrency | Output tok/s | TPOT (ms) | p95 TTFT (ms) |
| --- | --- | --- | --- |
| 1 | 140 | 7 | 50 |
| 8 | 1,050 | 7.6 | 70 |
| 32 | 3,600 | 9 | 140 |
| 128 | 8,500 | 15 | 600 |
| 256 | 9,200 | 28 | 3,000 |

These are illustrative numbers for an 8B model on an H100. The pattern is what to look for:

- From 1 to about 32, throughput rises almost linearly and TPOT barely moves. The weight read is shared, so extra sequences are nearly free.
- From 32 to 128, throughput still rises but TPOT starts to grow. KV reads and compute are catching up with the weight read.
- From 128 to 256, throughput gains only 8% while TPOT nearly doubles and TTFT jumps five times. This is past the knee.

~~~dg-raw
%%widget batch Raise the batch size and look for the point where per-user speed drops faster than total throughput rises.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
ctx = 10
b = 5
%%
~~~

## What causes the knee

Three mechanisms bend the curve, and the engine's metrics tell you which one is active.

- **Compute.** At a few hundred sequences, decode matmuls are no longer memory bound. Step time starts to grow in proportion to batch size. Signal: TPOT rises steadily while KV usage is still low.
- **KV capacity.** The cache fills, and new requests wait in the queue until running ones finish. TTFT jumps while TPOT stays steady. Signal: KV usage near 100% and a growing waiting queue.
- **Prefill interference.** At higher load, more new requests arrive per second, so more prefill work is mixed into each step. Decodes slow down, and ITL tails grow. Signal: p99 ITL much higher than p50.

## Closed loop versus open loop results

The two sweeps answer different questions.

- The closed-loop sweep shows the throughput ceiling and how per-user latency trades against it. It can never show an unbounded queue, because load stops growing when the server slows.
- The open-loop sweep shows what happens with real arrivals. Below capacity, TTFT stays near its baseline plus a little queueing. Just above capacity, the queue grows without limit.

A long open-loop run makes overload obvious. Plot TTFT against each request's start time. Below capacity, the points form a flat band. Above capacity, they form a rising line, because each request waits behind more requests than the last.

<div class="ex">
<div class="lab">Worked example</div>

**spotting overload in a time plot**

At 10 requests per second, TTFT stays between 80 and 200 ms for the whole 10-minute run.

At 12 requests per second, TTFT starts at 150 ms and climbs steadily to 40 s by the end of the run.

The capacity is between 10 and 12 requests per second. The average TTFT over the 12 per second run (about 20 s) is meaningless, because it depends on how long you ran the test.

</div>

## Choosing the operating point

The best operating point sits before the knee, at the highest load where every SLO still holds with some margin. In the table above, with an SLO of p95 TTFT under 500 ms and p95 TPOT under 30 ms, that is somewhere between 32 and 128 concurrent requests. A finer sweep in that range finds it. Leave headroom of 20 to 30% for bursts, since the knee is steep.

## Deep dive: the batch sweep, predicted

With the step model from 03.03, per-user speed and total throughput are

⟦M426⟧

As ⟦m428⟧ grows, ⟦m429⟧ approaches ⟦m430⟧, the rate at which the GPU can stream KV. It reaches half of that at

⟦M427⟧

For Llama 3 8B at 1k context, 2.8 TB/s and 1 ms overhead: the ceiling is about 20,900 tokens/s, and ⟦m431⟧.

~~~dg-raw
%%chart Predicted total throughput against batch size (Llama 3 8B, H100, 1k context).
x = 1, 8, 32, 64, 128, 256
logx = 1
xlab = Batch size
ylab = Output tokens per second
Total throughput: 148, 1127, 3879, 6544, 9961, 13474
%%
~~~

~~~dg-raw
%%chart Predicted per-user speed against batch size (same setup).
x = 1, 8, 32, 64, 128, 256
logx = 1
xlab = Batch size
ylab = Tokens per second per user
Per-user speed: 148, 141, 121, 102, 78, 53
%%
~~~

From 1 to 32, throughput rises 26 times while each user loses 18%. From 128 to 256, throughput rises 35% while each user loses 32%. That change of slope is the knee. Your measured curve should track these within 10 to 20%; if it falls well short, the gap is overhead or a bad attention kernel.

<div class="take">
<div class="lab">Key takeaway</div>

sweep load while holding the workload fixed, and plot throughput and latency together. The useful operating point sits before the knee, and the engine's metrics tell you whether compute, KV capacity or prefill interference caused it.

</div>
