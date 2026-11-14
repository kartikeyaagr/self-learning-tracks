---
id: "04.04"
course: "inference-eng-course"
module: "04"
slug: "warm-up-steady-state-reproducible-runs"
title: "Warm-Up, Steady State & Reproducible Runs"
keyIdeas: ["First requests are slow: compilation, CUDA graph capture, allocator growth, cold caches.", "Warm up before measuring and discard that window.", "Measure at steady state: after the ramp and before the drain at the end.", "Fix seeds, workload files, versions and flags. Save them with results.", "Repeat runs and report spread."]
question: "List four things you must hold constant or record to make two benchmark runs comparable."
answer: "The exact model revision, dtype and engine version and flags; the hardware and driver; the workload file (prompts, lengths, seed, arrival pattern); and the warm-up procedure and measurement window. Also record GPU clocks or power limits, and whether prefix caching was cold or warm."
sources: [{"title": "NVIDIA: LLM inference benchmarking fundamentals", "url": "https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/", "note": "Methodology guidance."}, {"title": "MLPerf Inference", "url": "https://mlcommons.org/benchmarks/inference-datacenter/", "note": "Example of strict, reproducible benchmark rules."}]
tags: ["methodology"]
---

## Why the first minute lies

The first requests a fresh server handles pay costs that later requests do not:

- **Compilation and capture.** `torch.compile`, CUDA graph capture and Triton autotuning may happen at startup or lazily at first use. A lazy compile can add seconds to the first request of each new shape.
- **Memory pools.** The allocator grows its pools the first time large tensors are needed.
- **Cold caches.** The prefix cache is empty, so the first requests that share a system prompt all pay full prefill.
- **Clocks and power.** GPUs ramp clocks up under load, and thermal state changes over the first minutes of sustained work.
- **Connection setup.** New TCP and TLS connections, and connection pools in the client, add latency at first.

If you include these in your results, the averages look worse than steady state. If you exclude them carelessly, you hide cold-start costs that real users see after every deploy or scale-up. Report them separately.

## The phases of a run

~~~dg-flow The phases of a benchmark run. Only the middle one is measured.
Warm-up -(discard)-> Ramp-up -> *Steady state -> Drain -(discard)-> Report
~~~

- **Warm-up.** Send a few dozen requests of representative shapes and throw the results away.
- **Ramp-up.** In an open-loop test, the number of requests in flight starts at zero and takes some time to reach its steady level, roughly one average latency.
- **Steady state.** The part you measure. It should be long enough to collect thousands of requests and to see slow effects such as thermal throttling.
- **Drain.** After the last request is sent, the remaining ones finish with no new traffic behind them. Their latency is lower than in steady state, so exclude them or end the measurement window before the drain.

<div class="ex">
<div class="lab">Worked example</div>

**how much warm-up skews results**

A team ran 500 requests on a freshly started engine and got p95 TTFT of 1.9 s.

Rerunning on the warm engine gave 420 ms. The difference came from lazy CUDA graph capture for new batch sizes (about 10 s spread across the first minute) and from an empty prefix cache.

The fix in the benchmark: 60 seconds of warm-up. The fix in production: capture graphs at startup and send warm-up traffic before marking the replica ready.

</div>

## The reproducibility checklist

Record all of these automatically, next to every result:

- **Model**: repository, commit hash, dtype, quantization method and its settings.
- **Engine**: version or commit, every non-default flag, and environment variables that change behaviour.
- **Hardware**: GPU type and count, driver and CUDA versions, power limit, clock settings, interconnect.
- **Workload**: the file, its seed, the arrival pattern, the rate or concurrency, the duration.
- **Cache state**: cold or warm prefix cache.
- **Client**: the machine, its location relative to the server, the tool and its version.

A benchmark result without these is an anecdote.

## Variance and repetition

Run each configuration at least three times, ideally on a freshly started engine each time. Then check how much the runs differ.

- Within 2 to 3%: normal. Differences smaller than this between configurations are noise.
- 5 to 10%: something is unstable. Common causes are thermal throttling, other jobs on the host, network noise, or a workload too short to reach steady state.
- More than 10%: do not compare configurations until you find the cause.

Report the median of the runs and their spread. When you claim "configuration B is 8% faster", you should be able to show that the runs of A and B do not overlap.

## Controlling the environment

- **Lock clocks or power** for very precise comparisons (`nvidia-smi -lgc` sets a locked clock range). Unlocked clocks are more realistic but noisier.
- **Pin the client to separate CPU cores**, or run it on a different machine, so it does not steal CPU from the engine's scheduler and tokenizer.
- **Watch the client.** A single Python process can run out of CPU above a few hundred streaming requests per second. If the client is saturated, the result measures the client.

## Deep dive: how much warm-up skews a mean

If the first ⟦m392⟧ seconds of a ⟦m393⟧-second run are slower by a factor ⟦m394⟧, a mean taken over the whole run is inflated by

⟦M390⟧

A 5 s warm-up at 3 times the latency, inside a 30 s run, inflates the mean by ⟦m395⟧. Percentiles are hit harder: those slow requests are all in the tail.

~~~dg-raw
%%chart TPOT over time in one run (illustrative). Capture, allocator growth and cold caches settle in the first seconds.
x = 0, 2, 4, 6, 8, 10, 15, 20, 30
xlab = Seconds since the server started taking traffic
ylab = TPOT (ms)
Measured: 31, 22, 14, 10, 8.4, 8.1, 8.0, 8.0, 8.1
Steady state*: 8, 8, 8, 8, 8, 8, 8, 8, 8
%%
~~~

## Deep dive: how many runs to detect a change

To tell whether a change of ⟦m396⟧ (as a fraction of the mean) is real, with per-run coefficient of variation ⟦m397⟧, 5% false alarms and 80% power, each side needs about

⟦M391⟧

runs.

| Per-run CV | Detect 5% | Detect 2% | Detect 1% |
| --- | --- | --- | --- |
| 1% | 1 | 4 | 16 |
| 3% | 6 | 36 | 142 |
| 5% | 16 | 99 | 393 |

Lowering noise is cheaper than adding runs. Halving the CV cuts the runs you need by four.

<div class="take">
<div class="lab">Key takeaway</div>

warm up, measure only the steady state, record everything automatically, and repeat each run. Treat differences smaller than run-to-run variance as noise.

</div>
