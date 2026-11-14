---
id: "02.06"
course: "inference-eng-course"
module: "02"
slug: "visual-lab-trace-one-token-end-to-end"
title: "Visual Lab: Trace One Token End-to-End"
keyIdeas: ["Follow one token across CPU and GPU: request, tokenize, schedule, forward, sample, detokenize, send.", "Capture a profiler trace of one decode step and open it in Perfetto.", "Identify gaps on the GPU timeline: they are CPU overhead.", "Label each kernel with the layer and op it belongs to.", "Write a one-paragraph explanation of where the time went."]
question: "For one generated token, list where it spends time from HTTP arrival to the client receiving it, and which parts are on the GPU."
answer: "Network and HTTP parsing (CPU), tokenization of the prompt (CPU), queueing in the scheduler (CPU), prefill or decode forward pass (GPU: matmuls, attention, norms), sampling (GPU), detokenization (CPU), and SSE write and network back to the client (CPU). Only the forward pass and sampling run on the GPU."
sources: [{"title": "PyTorch Profiler recipe", "url": "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html", "note": "Capture op-level traces."}, {"title": "Perfetto UI", "url": "https://ui.perfetto.dev/", "note": "View Chrome-format traces from the profiler."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Follow one request from the client to the GPU and back, and measure where its time goes. You will produce a timeline and a short bottleneck report. This habit of tracing first and optimizing second carries through the whole course.

## The path of one token

~~~dg-seq One token's journey from client to GPU and back.
Client > API : HTTP request
API > Engine : token ids
Engine > GPU : forward pass
GPU -> Engine : logits
Engine > Engine : sample, detokenize
Engine -> API : text piece
API -> Client : SSE chunk
~~~

Each hop costs time:

- **Network and HTTP parsing**: often under 1 ms inside a datacenter, tens of ms over the internet.
- **Tokenization**: about 1 ms for a few thousand tokens with a fast Rust tokenizer, but it can be much slower with a Python one.
- **Queueing**: zero at low load, and the dominant term at high load.
- **Prefill**: tens to hundreds of ms, depending on the prompt length.
- **Each decode step**: a few to tens of ms.
- **Sampling and detokenization**: under 1 ms each when done well.

## Instrument the server side

Add timestamps at each boundary in your server from lesson 01.06, and log them per request:

```
t = {"recv": now()}
ids = tok(prompt).input_ids;            t["tokenized"] = now()
logits = prefill(ids);                  t["prefill_done"] = now()
first = sample(logits);                 t["first_token"] = now()
```

Also record the client's send and first-receive times. The gap between the client's first-receive time and the server's first_token stamp is network plus streaming overhead.

## Capture the GPU side

Use the PyTorch profiler around a few decode steps:

```
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for _ in range(5):
        model(next_ids, past_key_values=cache, use_cache=True)
prof.export_chrome_trace("decode.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

Open `decode.json` in Perfetto (ui.perfetto.dev). You will see a CPU row where Python and PyTorch launch kernels and a GPU row where the kernels run.

## What to look for

- **Gaps on the GPU row.** The GPU is waiting for the CPU to launch the next kernel. At batch 1 with eager PyTorch, these gaps can be 20 to 40% of the step.
- **The big blocks.** Matmuls for QKV, the output projection, the MLP and the LM head. Their total should be close to weight bytes divided by bandwidth.
- **Attention kernels.** Small at short context. Repeat the trace at 8k tokens of context and watch them grow.
- **Many tiny kernels.** Norms, RoPE, residual adds, type casts. Each is a few microseconds, but there are hundreds per step.

<div class="ex">
<div class="lab">Worked example</div>

**a typical batch-1 trace**

An 8B model in eager PyTorch on an H100 takes 24 ms per decode step.

The matmuls take about 7 ms in total, close to 16 GB ÷ 3.35 TB/s ≈ 4.8 ms plus some inefficiency.

About 9 ms is GPU idle time between kernels, while Python launches about 700 kernels.

The rest is small kernels and attention. The cheapest win is obvious: remove launch gaps with CUDA graphs. That alone can bring the step to about 10 ms.

</div>

## Write the report

Produce three things:

1. A table of the top ten GPU kernels by total time, with their share of the step and the layer they belong to.
2. The total GPU idle time inside one step, as a percentage.
3. A breakdown of TTFT for one request into tokenize, queue, prefill and first sample, and the measured network overhead.

End with one sentence naming the biggest bottleneck and the layer of the stack where it lives.

## Deep dive: reading a batch-1 trace with numbers

A single decode step for Llama 3 8B launches roughly this sequence per layer: norm, QKV matmul, RoPE, attention, output matmul, add and norm, gate-up matmul, SiLU-multiply, down matmul, add. Expected times at batch 1 come from bytes:

| Kernel | Bytes read | Expected time at 2.8 TB/s |
| --- | --- | --- |
| QKV matmul | 50 MB | 18 µs |
| Output matmul | 34 MB | 12 µs |
| Gate-up matmul | 235 MB | 84 µs |
| Down matmul | 117 MB | 42 µs |
| Attention (1k context) | 4.2 MB of KV | 2 to 5 µs (launch bound) |
| Norms, RoPE, adds | about 0.05 MB | 2 to 4 µs each |

~~~dg-raw
%%timeline One layer of a batch-1 decode step on the GPU (µs, idealized, no gaps).
unit = µs
GPU | 0-3 norm c4 ; 3-21 QKV c1 ; 21-24 rope c4 ; 24-29 attn flame ; 29-41 O c1 ; 41-44 norm c4 ; 44-128 gate-up c3 ; 128-131 act c4 ; 131-173 down c3 ; 173-176 add c4
%%
~~~

The four matmuls should be about 90% of the layer. If your trace shows the small kernels taking as long as the matmuls, launch overhead dominates and CUDA graphs or fusion will help.

<div class="take">
<div class="lab">Key takeaway</div>

a token spends time on the network, the CPU and the GPU. A trace shows exactly where, and gaps on the GPU timeline are often the cheapest wins.

</div>
