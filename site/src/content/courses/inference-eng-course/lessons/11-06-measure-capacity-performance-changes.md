---
id: "11.06"
course: "inference-eng-course"
module: "11"
slug: "measure-capacity-performance-changes"
title: "Measure Capacity & Performance Changes"
keyIdeas: ["Measure quantization at three levels: kernel, single-request latency, SLO-safe capacity.", "Gains at batch 1 rarely equal gains at production load.", "Freed memory helps only if KV was the constraint.", "Report capacity and goodput, not just tokens per second at one point.", "Keep the same workload and SLO as the baseline."]
question: "After switching to FP8 weights, throughput at batch 1 improved 1.7 times but SLO-safe capacity improved only 1.2 times. Give plausible explanations."
answer: "At the operating load the batch is large and closer to compute bound, where FP8 gains are smaller than at batch 1; capacity may be limited by TTFT (prefill) or the scheduler rather than decode speed; and freed memory may not translate into more concurrency if KV was not the binding constraint. Overheads outside GEMMs (attention, sampling, CPU) also do not get faster."
sources: [{"title": "vLLM: Quantization", "url": "https://docs.vllm.ai/en/latest/features/quantization/index.html", "note": "Supported formats and hardware."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Sweep tool."}]
tags: ["quantization", "benchmarking"]
---

## Three levels of measurement

A quantization change can be measured at three levels, and they give very different numbers.

| Level | Question | Tool |
| --- | --- | --- |
| Kernel | is the matmul faster? | micro-benchmark of one matmul shape |
| Request | did TTFT and TPOT improve for one request? | single-request runs at batch 1 |
| System | did SLO-safe capacity improve on the real workload? | open-loop rate sweep with the harness |

Only the system level tells you whether the change saves money. Kernel and batch-1 speedups are upper bounds.

## Why gains shrink from level to level

- **Matmuls are only part of the step.** Attention, norms, sampling, scheduling and CPU work are unchanged. If matmuls are 60% of a step and become 2 times faster, the step becomes 1 ÷ (0.4 + 0.3) ≈ 1.43 times faster, by Amdahl's law.
- **Production batch sizes are larger.** Weight-only gains measured at batch 1 shrink at batch 64 or 256.
- **Capacity may be bound by something else.** If capacity is limited by TTFT for long prompts, a faster decode kernel barely moves it.
- **More memory only helps if memory was the limit.** Freeing 8 GB for KV cache helps only if KV usage was near full.

<div class="ex">
<div class="lab">Worked example</div>

**the same change at three levels**

FP8 W8A8 on an 8B model, H100.

Kernel level: the MLP matmul at 256 tokens is 1.8 times faster.

Request level: batch-1 TPOT falls from 7.0 ms to 4.3 ms, 1.6 times faster.

System level: SLO-safe capacity rises from 9.0 to 11.0 requests per second, 1.22 times. The gain is real but much smaller than the kernel number, because attention, small ops and TTFT for long prompts were unchanged.

</div>

## How to run the comparison

1. Use the same engine version, workload, SLO and harness for both configurations.
2. Record the weight memory and KV capacity from both startup logs.
3. Run batch-1 TTFT and TPOT for a sanity check against the bandwidth math.
4. Run a fine rate sweep near the knee for both.
5. Record engine metrics at the SLO-safe rate: KV usage, waiting queue, preemptions.
6. Run the quality evaluation (next lesson).

## A good report

| Metric | BF16 | FP8 | Change |
| --- | --- | --- | --- |
| Weight memory | 16 GB | 8.5 GB | −47% |
| KV capacity | 330k tokens | 390k tokens | +18% |
| TPOT at batch 1 | 7.0 ms | 4.3 ms | −39% |
| p95 TTFT at expected load | 520 ms | 410 ms | −21% |
| SLO-safe capacity | 9.0 req/s | 11.0 req/s | +22% |
| Product eval | 82.4 | 82.1 | −0.3 points |

Every row answers a different question. Together they show where the gain came from: faster matmuls and more KV room, with no meaningful quality loss.

## Explaining unexpected results

- **Capacity improved less than TPOT.** Check which SLO binds at the new knee. Often TTFT or KV becomes the limit.
- **Capacity got worse.** Check the kernel names in a profile for a slow fallback, and check whether activation quantization runs as a separate unfused kernel.
- **KV capacity rose less than expected.** The engine may keep some layers in BF16, or the activation peak may have changed.

## Deep dive: from kernel speed-up to system speed-up

If a kernel speed-up of ⟦m664⟧ applies to a share ⟦m665⟧ of request time, Amdahl's law gives the request-level speed-up:

⟦M663⟧

From the example: a 1.8 times kernel speed-up and a 1.6 times request speed-up imply ⟦m666⟧. The other 16% (attention, norms, sampling, launch gaps) did not get faster.

~~~dg-raw
%%chart Request-level speed-up against the share of time the faster kernels cover.
x = 0.5, 0.6, 0.7, 0.8, 0.9, 1
xlab = Share of time covered f
ylab = Speed-up
Kernel 1.8× faster: 1.29, 1.36, 1.45, 1.55, 1.67, 1.8
Kernel 4× faster: 1.6, 1.82, 2.11, 2.5, 3.08, 4
%%
~~~

At system level, the gain shrinks again because TTFT, queueing and network time also count, and the SLO may be set by a different phase.

<div class="take">
<div class="lab">Key takeaway</div>

measure quantization where it pays, as SLO-safe capacity on your workload. Kernel and batch-1 speedups are upper bounds that shrink by Amdahl's law, so report all three levels and explain the gaps.

</div>
