---
id: "IN.03"
course: "inference-eng-course"
module: "IN"
slug: "levels-of-inference-engineering"
title: "Levels of Inference Engineering"
keyIdeas: ["Level 1: run a model locally and get tokens out. Know what a GPU, a tokenizer and a model checkpoint are.", "Level 2: operate a real engine. Benchmark correctly, tune flags, size hardware against an SLO.", "Level 3: optimize. Quantize, cache prefixes, profile, choose parallelism, run a fleet with routing and autoscaling.", "Level 4: build. Write kernels, change schedulers, read and modify engine code, reproduce papers.", "The course follows these levels. Pick a learner path based on which level you need."]
question: "Describe what distinguishes a level-2 (engine user) from a level-4 (engine builder) inference engineer, with one concrete task each."
answer: "A level-2 engineer runs an existing engine like vLLM, tunes flags such as max batched tokens and GPU memory fraction, and benchmarks against an SLO, for example picking the cheapest GPU that meets p95 TTFT. A level-4 engineer changes the engine itself, for example writing a new attention kernel or modifying the scheduler, and proves the gain with profiles and ablations."
sources: [{"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Where level-4 work often happens."}, {"title": "SGLang GitHub repository", "url": "https://github.com/sgl-project/sglang", "note": "A second engine worth reading."}]
tags: ["career"]
---

## A ladder, not a checklist

Inference work ranges from "get a model answering" to "write a faster attention kernel". It helps to see it as four levels, each building on the one below. You can do valuable work at every level.

| Level | You can | Typical task | Typical evidence of success |
| --- | --- | --- | --- |
| 1. Run | Load and query a model | Serve a 7B model on one GPU over HTTP | It answers, and you know its memory use |
| 2. Operate | Run and tune an engine | Find max concurrency under a p95 TTFT target | A benchmark table with SLO pass or fail |
| 3. Optimize | Change how it runs | FP8 weights, prefix caching, pick tensor-parallel size | Before and after numbers, quality checked |
| 4. Build | Change the engine | Write a fused kernel, modify the scheduler | Profiles, ablations, upstreamable code |

## What each level looks like in practice

**Level 1, run.** You can download a checkpoint, load it with Hugging Face Transformers, generate text and serve it behind a small HTTP endpoint. You can explain why a 7B model in BF16 needs about 14 GB just for weights.

**Level 2, operate.** You run a real engine such as vLLM or SGLang. You know which flags matter (maximum batched tokens, GPU memory fraction, maximum model length) and you benchmark properly: warm-up, realistic prompt lengths, percentiles instead of averages. You can answer "how many users can this GPU handle at p95 TTFT under 1 second?"

**Level 3, optimize.** You change how the model runs and prove the change helped without hurting quality. Quantization, prefix caching, chunked prefill, speculative decoding, choosing parallelism, routing and autoscaling a fleet. You read profiles to find bottlenecks instead of guessing.

**Level 4, build.** You change the engine itself: kernels in Triton or CUDA, scheduler policies, memory managers. You read papers, reproduce them and know where they stop working.

<div class="ex">
<div class="lab">Worked example</div>

**the same problem at three levels**

Problem: chat latency is too high at peak traffic.

Level 2 answer: lower the max batch size so each step is shorter, and add a replica. Verify p95 TTFT drops under the SLO at the expected load.

Level 3 answer: turn on chunked prefill so long prompts stop stalling decodes, plus prefix caching for the shared system prompt. Measure the p95 change and GPU-hours saved.

Level 4 answer: profile shows attention for mixed prefill and decode batches is the hot spot. Write a kernel that handles both in one launch and show the speedup with ablations.

</div>

## How the course maps to the levels

- Parts 1 to 3 (foundations, how inference works, measurement) get you to level 2.
- Parts 4 and 5 (engines, single-GPU optimization) are level 3 on one GPU.
- Part 6 (kernels, compilation, speculation) and Part 7 (multi-GPU) are the deep end of level 3 and the start of level 4.
- Part 8 (production) is level 3 at fleet scale.
- Parts 9 and 10 (specialized workloads, engine reading, mini engine, research) are level 4.

## Four learner paths

- **Applied inference engineer**: the core path through production. Most people stop here, and most production wins come from here.
- **Runtime or performance engineer**: the core path plus kernels, attention, compilation, speculation and engine internals.
- **Distributed inference engineer**: the core path plus multi-GPU, MoE, long context, disaggregation and routing.
- **Research or engine engineer**: everything, including the mini engine and paper reproduction.

## How to use the course

Each lesson has a question at the end. Write your answer before you look at the model answer; the act of retrieving is what makes it stick. Completed lessons enter the review deck and come back at growing intervals. Labs and projects are where the knowledge becomes skill, so do them on a real GPU if you can. A rented cloud GPU for a few hours is enough for most of them.

## Deep dive: what "expert" looks like in numbers

Each level of the ladder can be checked by what you can predict before you measure. A useful test is the gap between your prediction and the measurement:

⟦M23⟧

- **Level 1 to 2:** you can run a benchmark but cannot predict its outcome. Errors above 100% are normal.
- **Level 3:** you can estimate TPOT and KV capacity from first principles to within about 30%.
- **Level 4:** you can predict how a change (FP8, chunked prefill, TP 2) will move goodput to within about 20%, and you can explain the rest with a profile.
- **Level 5:** you can predict where a technique stops working, before running the sweep.

~~~dg-raw
%%grid Which modules build which skills (darker = more central).
heat = 1
cols = L1, L2, L3, L4, L5
Setup (00 to 01) | 3 2 0 0 0
How inference works (02 to 03) | 2 3 2 0 0
Measurement (04 to 05) | 1 3 3 1 0
Engines (06 to 09) | 0 2 3 2 0
Single-GPU speed (10 to 17) | 0 1 3 3 1
Scale-out (18 to 21) | 0 0 2 3 2
Production (22 to 25) | 0 1 2 3 2
Research and capstone (26 to 30) | 0 0 1 3 3
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**checking your own level on one question**

Question: "Llama 3 8B, H100, batch 32, 2k context, BF16. What TPOT?"

Level 3 answer: weights 16 GB plus KV ⟦m24⟧, 24 GB per step, at about 2.7 TB/s effective, about 9 ms. Measured values are typically 10 to 12 ms.

Level 4 adds: "the extra is kernel launch overhead and attention efficiency at short context; CUDA graphs close most of it."

</div>

<div class="take">
<div class="lab">Key takeaway</div>

you do not need level 4 to be useful. Most production wins come from level 2 and 3 work done carefully with good measurements.

</div>
