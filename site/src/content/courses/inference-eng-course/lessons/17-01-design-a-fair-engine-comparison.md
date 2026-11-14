---
id: "17.01"
course: "inference-eng-course"
module: "17"
slug: "design-a-fair-engine-comparison"
title: "Design a Fair Engine Comparison"
keyIdeas: ["Fix model weights, precision, hardware and workload across engines.", "Use one external benchmark client for all engines.", "Tune each engine reasonably (equal effort), or compare defaults explicitly labeled.", "Report goodput under the same SLO, not one throughput number.", "Publish configs so others can reproduce."]
question: "Name four ways an engine comparison is commonly unfair."
answer: "Different defaults (one has prefix caching or chunked prefill on, the other off); different precision or quantization; tuning one engine and not the other; different workloads or output lengths (EOS not ignored); different versions or hardware; and measuring with each engine's own benchmark tool, which may define metrics differently."
sources: [{"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Engine-agnostic benchmark client."}, {"title": "MLPerf Inference", "url": "https://mlcommons.org/benchmarks/inference-datacenter/", "note": "Example of strict comparison rules."}]
tags: ["benchmarking", "engines"]
---

## Why engine comparisons go wrong

Engine benchmarks circulate constantly, and many of them disagree. Usually the engines are not the reason. The comparisons differ in model precision, settings, workload, hardware or client, and those differences swamp the engine's own effect. A fair comparison changes exactly one thing, the engine, and holds everything else fixed.

## What to hold constant

- **Model**: the same weights and revision, and the same precision. If one engine runs FP8, the other must run the same FP8 recipe, not BF16.
- **Hardware**: the same GPU type and count, driver and CUDA version, power limits, and the same parallelism layout.
- **Workload**: the same file, seed, arrival pattern, rate or concurrency, and output length control.
- **Context limits**: the same maximum model length and, where possible, the same memory fraction.
- **Measurement**: the same SLO, warm-up, measurement window and number of repeats.
- **Client**: the same benchmark tool and version, run from the same machine.

## Two valid styles

| Style | When to use it | The rule |
| --- | --- | --- |
| Out of the box | choosing defaults, or judging ease of use | state clearly that defaults differ, and list them |
| Tuned | choosing an engine for production | give each engine equal tuning effort, and publish every flag |

Both are legitimate. Mixing them is not: "engine A tuned against engine B defaults" tells you about the tuning, not the engines.

## What to report

- **Goodput against request rate under the SLO** for each engine, the curve that matters most.
- **TTFT and TPOT percentiles at the expected load.**
- **SLO-safe capacity**, the single number for capacity planning.
- **Startup time and memory**: weights, KV capacity and graph memory from each startup log.
- **Features that changed the result**, such as prefix caching or speculation, with a run with them off if possible.
- **Exact versions and commands**, so anyone can rerun it.

<div class="ex">
<div class="lab">Worked example</div>

**a comparison that was really about settings**

A published comparison showed engine A at 2.1 times the throughput of engine B on an 8B model.

A rerun found three differences besides the engine: A ran with FP8 KV cache and B with BF16; A had max sequences at 512 and B at 256; and the workload used one shared system prompt, with prefix caching on in A and off in B.

With matched settings, A led by 8% at high load and trailed by 3% at low load. A useful result, but very different from 2.1 times.

</div>

## Red flags in other people's comparisons

- No description of the workload: input and output lengths, prefix sharing, arrival pattern.
- Throughput only, with no latency or SLO.
- One engine on newer hardware, a newer version or a different GPU count.
- Different quantization or KV precision.
- A single run with no indication of variance.
- Closed-loop concurrency only, which can hide overload differences.

## Versions move

Engines release often, and performance changes by double-digit percentages between versions. A comparison is a snapshot. Record the date and versions, and plan to rerun it when you consider upgrading or switching.

## Deep dive: is engine A really faster?

Throughput comparisons are ratios, so work with logarithms. With ⟦m874⟧ runs per engine and per-run coefficients of variation ⟦m875⟧ and ⟦m876⟧:

⟦M873⟧

<div class="ex">
<div class="lab">Worked example</div>

**a 4% win that is not one**

Engine A: 5,200 tokens/s, engine B: 5,000. Three runs each, CV 4% for both.

SE of the log ratio ⟦m877⟧. Interval: ⟦m878⟧, from 0.98 to 1.11.

The interval includes 1. Report "no significant difference at this sample size", or run more: at 12 runs each, the interval is 1.01 to 1.07.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a fair comparison fixes everything except the engine, measures goodput under an SLO with one client, and publishes every setting. Otherwise you are comparing configurations, not engines.

</div>
