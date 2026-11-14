---
id: "04.08"
course: "inference-eng-course"
module: "04"
slug: "project-build-your-inference-benchmark-harness"
title: "Project: Build Your Inference Benchmark Harness"
keyIdeas: ["Build a harness you will reuse for every later lab.", "Inputs: endpoint, workload file, load pattern (rate or concurrency), duration, SLO.", "Record raw per-request, per-token timestamps; compute summaries afterwards.", "Output JSON plus charts: latency percentiles and goodput vs load.", "Embed env and config in every results file."]
question: "What should the output of your harness contain so that someone else can reproduce and trust a result?"
answer: "A results file with the environment report (hardware, driver, versions), full engine config, workload definition and seed, load pattern, raw per-request timing records (send, first token, every token, end, token counts, errors), and computed summaries: TTFT, TPOT, ITL and E2E percentiles, throughput, goodput against a stated SLO, plus charts over load."
sources: [{"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Reference design for a sweep tool."}, {"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "Reference timing code."}]
tags: ["project"]
tools: ["Project"]
---

## Goal

Build your own benchmark harness. It will be the measuring instrument for every later module, so it should be correct, repeatable and easy to extend. Established tools are good for comparisons; your own harness lets you add custom workloads, metrics and experiments.

## Requirements

1. **Workload.** Load prompts and target output lengths from a file, with a seed. Support synthetic workloads (random lengths from a distribution) and replayed ones.
2. **Load modes.** Closed loop with concurrency C, and open loop with a Poisson rate R. In open loop, schedule each send from the clock, never from completions.
3. **Timing.** For each request, record the scheduled send time, actual send time, every chunk's arrival time, end time, input and output token counts, HTTP status and any error. Use a monotonic clock.
4. **Warm-up.** A configurable warm-up period whose results are excluded.
5. **Sweep.** Run a list of rates or concurrencies in one command, with a pause between points.
6. **Summary.** p50, p95 and p99 of TTFT, TPOT, ITL and E2E. Output tokens per second, requests per second, error rate, SLO attainment and goodput.
7. **Provenance.** The environment report, engine configuration and exact command line, saved with each result.

## Structure

~~~dg-flow The harness pipeline.
Workload file -> Load generator -> *Async SSE client -> Raw records -> Summaries and charts
~~~

- **Load generator**: produces (scheduled time, request) pairs.
- **Async client**: one process with `asyncio` and `httpx` or `aiohttp` can hold hundreds of open streams. Parse SSE lines yourself and stamp each chunk with `time.monotonic()` the moment it is read.
- **Raw records**: write one JSON line per request. Keep them. Every summary can be recomputed from raw records, and new metrics can be added later.
- **Summaries**: computed offline from raw records, so a bug in a summary never forces you to rerun a test.

## Sketch of the client

```
async def run_one(client, req, t_sched):
    await asyncio.sleep(max(0, t_sched - time.monotonic()))
    rec = {"sched": t_sched, "sent": time.monotonic(), "chunks": []}
    async with client.stream("POST", url, json=req) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                rec["chunks"].append(time.monotonic())
    rec["end"] = time.monotonic()
    return rec
```

Compute TTFT from `sched` in open loop, so that client-side delays count as latency, as they would for a real user.

## Validation

A harness you have not validated is a source of confident wrong numbers. Check three things:

- **Against a known model.** Run it against your naive server from Module 01. Throughput should stay flat as concurrency rises, and TTFT should follow the queueing pattern you predicted.
- **Against an established tool.** Run the same workload with vLLM's benchmark. Throughput and median latencies should agree within a few percent.
- **Against Little's law.** Average requests in flight should equal the completed rate times the average E2E latency.

<div class="ex">
<div class="lab">Worked example</div>

**a bug the validation catches**

A first version of a harness reported TTFT 40% lower than vLLM's tool on the same workload.

The cause: it started the timer after the HTTP connection was established, so connection setup and queueing in the client's pool were excluded.

Moving the timestamp to the scheduled send time brought the two tools within 3%.

</div>

## Deliverable

A repository containing:

- The harness, with a README that explains each metric's definition.
- One results folder with a sweep against a real engine: raw records, summary table and charts of p95 TTFT, p95 TPOT and throughput against rate.
- A short note on validation: what you compared against and how close it was.

## Deep dive: invariants the harness should assert

Every record the harness writes can be checked against identities that must hold:

⟦M419⟧

- The first holds to the clock's resolution. If it fails, two timestamps came from different clocks.
- The second fails when the server batches several tokens into one SSE chunk. Count tokens from the usage field, not from chunks, or TPOT will be wrong by the chunking factor.
- The third is Little's law. A gap above about 5% means time is being lost before or after the measured interval.

~~~dg-flow The harness's timing path. Every timestamp comes from one monotonic clock.
Schedule s_k = k/λ -> Send, record lateness -> Stream, record t_1 … t_n -> *Finish: usage tokens, check invariants -> Report histograms
~~~

<div class="ex">
<div class="lab">Worked example</div>

**chunking skews TPOT**

A server sends 4 tokens per SSE chunk. A harness that counts chunks sees 125 "tokens" for a 500-token answer over 4.0 s of decode.

It reports TPOT ⟦m420⟧ ms. The true TPOT is ⟦m421⟧ ms. The harness was off by a factor of 4, in the direction that makes the engine look bad.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

your harness is the instrument for the rest of the course. Schedule sends from the clock, keep raw records with full provenance, and validate against a known tool before you trust it.

</div>
