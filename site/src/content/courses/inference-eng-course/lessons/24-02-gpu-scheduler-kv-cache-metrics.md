---
id: "24.02"
course: "inference-eng-course"
module: "24"
slug: "gpu-scheduler-kv-cache-metrics"
title: "GPU, Scheduler, KV & Cache Metrics"
keyIdeas: ["GPU: memory used, SM activity, DRAM activity, power, temperature, clocks, ECC, Xid errors.", "Scheduler: running, waiting, step time, batch size, token budget usage.", "KV: usage, preemptions, swaps, blocks free.", "Prefix cache: hit rate, evictions.", "Correlate layers: a TTFT spike plus KV at 100% plus preemptions tells a clear story."]
question: "Which GPU-level metrics can warn about hardware problems before a crash?"
answer: "Xid errors in driver logs, ECC error counts (correctable rising, uncorrectable any), retired or remapped memory pages, NVLink error counters, thermal throttling and clock drops, and power anomalies. DCGM exposes most of these as metrics; rising correctable errors or row remapping often precede failures."
sources: [{"title": "NVIDIA DCGM Exporter", "url": "https://github.com/NVIDIA/dcgm-exporter", "note": "GPU metrics for Prometheus."}, {"title": "NVIDIA Xid errors", "url": "https://docs.nvidia.com/deploy/xid-errors/index.html", "note": "Meaning of driver error codes."}]
tags: ["observability", "gpu"]
---

## GPU health

| Metric | Warning sign |
| --- | --- |
| ECC correctable errors | a rising trend on one GPU |
| ECC uncorrectable errors, Xid 48, 63, 64, 79 | act now: drain and replace |
| Clocks below base under load | thermal or power throttling |
| Temperature | close to the limit |
| NVLink errors | interconnect trouble |

Xid codes are error codes the NVIDIA driver writes to the kernel log. A few mean "this GPU has failed": Xid 79 means the GPU has fallen off the bus. Others, such as Xid 13 or 31, are usually caused by software and point to a kernel or engine bug.

## Engine internals

| Metric | What it tells you |
| --- | --- |
| Step time and batch size | why throughput changed |
| Waiting requests | queueing |
| KV usage and preemptions | memory pressure |
| Prefix hit rate and evictions | whether the cache is working |
| Speculation acceptance rate | whether speculative decoding pays |

## Correlate on one panel

Build a panel per replica showing TTFT p95, KV usage and preemptions together. Most incidents become obvious when these three share one chart.

<div class="ex">
<div class="lab">Worked example</div>

**reading the three lines together**

At 14:05, TTFT p95 on replica 3 climbs from 0.6 s to 4 s. Traffic to replica 3 has not changed.

On the same panel, KV usage on replica 3 has been at 98% since 14:02, and preemptions jumped from 0 to 40 per minute.

A few very long requests are holding most of the KV. New requests wait or get preempted. The fix is a per-request token cap or KV-aware routing, not more GPUs.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

watch GPU health counters for early hardware warnings, and engine internals to explain latency changes. Put related metrics on one chart.

</div>
