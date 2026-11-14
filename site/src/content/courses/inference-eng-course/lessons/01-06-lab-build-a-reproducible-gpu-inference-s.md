---
id: "01.06"
course: "inference-eng-course"
module: "01"
slug: "lab-build-a-reproducible-gpu-inference-server"
title: "Lab: Build a Reproducible GPU Inference Server"
keyIdeas: ["Build a small HTTP server that streams tokens from a 1 to 8B model.", "Pin everything: model revision, dtype, library versions, sampling settings.", "Measure TTFT, time per output token and peak memory for one request.", "Send 1, 2, 4 and 8 concurrent requests and plot latency vs concurrency.", "Write down what you observe. You will compare against vLLM in Module 06."]
question: "Describe the three measurements your lab report must include and why each matters."
answer: "TTFT and per-token latency for a single request (the best-case baseline), latency as concurrency rises from 1 to 8 (shows queueing in a naive server), and peak GPU memory with the model loaded (shows headroom for KV cache). Together they show the baseline and why a real engine is needed."
sources: [{"title": "Hugging Face: Generation with LLMs", "url": "https://huggingface.co/docs/transformers/main/en/llm_tutorial", "note": "Reference for the generate loop."}, {"title": "FastAPI documentation", "url": "https://fastapi.tiangolo.com/", "note": "Server framework used in the lab."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Build a small inference server you can rebuild anywhere, and record a baseline you will beat in later modules. The point is not speed. The point is a pinned, measurable reference.

## What "reproducible" means here

Someone else, or you in three months, should get the same model, the same software and roughly the same numbers. That requires pinning four things:

- **Model**: repository name plus commit hash, and the exact dtype.
- **Software**: Python version, and exact versions of PyTorch, Transformers, CUDA and the driver. A lock file or a container image digest is best.
- **Hardware**: GPU model, memory size, power limit and clocks.
- **Workload**: the prompts, their lengths, `max_tokens`, sampling settings and seed.

Write an environment report at startup and save it with every result:

```
import torch, transformers, platform, subprocess
print(platform.python_version(), torch.__version__, transformers.__version__)
print(torch.version.cuda, torch.cuda.get_device_name())
print(subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,power.limit",
                               "--format=csv,noheader"]).decode())
```

## Steps

1. Choose a 7B or 8B instruct model. Pin its revision in a config file.
2. Load it in BF16 inside `torch.inference_mode()` and apply the chat template.
3. Expose `POST /v1/chat/completions` with SSE streaming, in the OpenAI format.
4. Write a load client in Python with `httpx` or `aiohttp`. For each request, record the send time, first-token time, every token's arrival time, the end time and the number of output tokens.
5. Use a fixed prompt set, for example 50 prompts of 200 to 1,000 tokens, with `max_tokens` set to 256 and `ignore_eos` behaviour where your server supports it, so output lengths stay comparable.
6. Run with concurrency 1, 2, 4 and 8. Each run should send at least 50 requests after 5 warm-up requests.
7. Record peak GPU memory, average power draw and the environment report.

## What to compute

For each concurrency level:

- TTFT at p50 and p95.
- TPOT at p50 and p95 (the gaps between tokens, excluding the first).
- Output throughput: total output tokens divided by wall-clock time.
- Requests per second completed.

## Expected results

Because the server handles one request at a time, adding concurrency only adds queueing.

| Concurrency | p95 TTFT | TPOT | Output tokens/s |
| --- | --- | --- | --- |
| 1 | about 0.1 s | 20 to 30 ms | about 40 |
| 2 | about 1 × generation time | same | about 40 |
| 4 | about 3 × generation time | same | about 40 |
| 8 | about 7 × generation time | same | about 40 |

If your throughput grows with concurrency, something is batching that you did not expect. If TPOT grows, the server may be interleaving requests on the CPU side. Either way, find out why before you move on.

<div class="ex">
<div class="lab">Worked example</div>

**reading the baseline**

At concurrency 8, a team measured p95 TTFT of 44 s, TPOT of 24 ms and 41 output tokens per second.

Each generation takes about 256 × 0.024 ≈ 6.1 s. The eighth request in a burst waits for seven others, so its TTFT is about 7 × 6.1 ≈ 43 s. The measurement matches the model.

When a measurement matches a simple model like this, you can trust both. When it does not, one of them is wrong.

</div>

## Report

Write one page:

- Setup: the pinned model, software and hardware.
- Method: the prompt set, the concurrency levels and the number of requests.
- Results: a table and a chart of p95 TTFT and throughput against concurrency.
- Resources: peak memory and average power.
- Two sentences on why this design cannot scale.

Keep the client script. Module 04 turns it into a proper benchmark harness, and every later module compares against this baseline.

## Deep dive: how many runs make a baseline

Run-to-run noise comes from clocks, thermal state and background load. If one run's throughput has standard deviation ⟦m212⟧, the mean of ⟦m213⟧ runs has standard error

⟦M211⟧

and a 95% interval of about ⟦m214⟧. Report the coefficient of variation ⟦m215⟧ next to every baseline number; under 2% is a quiet machine.

<div class="ex">
<div class="lab">Worked example</div>

**Example**

Five runs of TPOT at batch 1: 6.41, 6.38, 6.52, 6.44, 6.40 ms. Mean 6.43 ms, ⟦m216⟧, CV 0.9%.

SE ⟦m217⟧ ms. The baseline is ⟦m218⟧ ms. A change that lands at 6.35 ms is not yet distinguishable from noise; one that lands at 6.1 ms is.

</div>

~~~dg-raw
%%chart How the 95% interval shrinks with more runs (CV of 3% per run).
x = 1, 2, 3, 5, 8, 10, 20
xlab = Runs averaged
ylab = ± % of the mean
95% interval: 5.9, 4.2, 3.4, 2.6, 2.1, 1.9, 1.3
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

your first server is a baseline, not a product. Pin everything, measure TTFT, TPOT and throughput across concurrency, and expect throughput to stay flat. That flat line is the problem the rest of the course solves.

</div>
