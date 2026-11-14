---
id: "17.03"
course: "inference-eng-course"
module: "17"
slug: "tensorrt-llm-hardware-specific-optimization"
title: "TensorRT-LLM & Hardware-Specific Optimization"
keyIdeas: ["TensorRT-LLM is NVIDIA's inference library with tuned kernels for its GPUs.", "Strengths: kernel quality, FP8/FP4, fused attention, in-flight batching, NVIDIA-specific features.", "Costs: NVIDIA lock-in, engine build or conversion steps, more configuration.", "Often served behind NVIDIA Triton Inference Server or its own server.", "Worth benchmarking for stable, high-volume NVIDIA deployments."]
question: "What does TensorRT-LLM trade for its performance, compared with vLLM or SGLang?"
answer: "It ties you more closely to NVIDIA hardware and its toolchain, can require building or converting engines for a specific model, precision and GPU (with longer build steps and less flexibility to change shapes), and has a steeper setup. In return it offers highly tuned kernels, strong FP8 and FP4 support and in-flight batching on NVIDIA GPUs."
sources: [{"title": "TensorRT-LLM GitHub repository", "url": "https://github.com/NVIDIA/TensorRT-LLM", "note": "Source and docs."}, {"title": "NVIDIA Triton Inference Server", "url": "https://github.com/triton-inference-server/server", "note": "Common serving layer for TensorRT-LLM."}]
tags: ["tensorrt-llm"]
---

## What TensorRT-LLM is

TensorRT-LLM is NVIDIA's library and runtime for LLM inference. It implements the same core techniques as other engines, including in-flight batching (NVIDIA's name for continuous batching), paged KV cache, chunked prefill, quantization, speculative decoding and multi-GPU parallelism, with kernels tuned specifically for each NVIDIA GPU architecture. It is often served behind Triton Inference Server or NVIDIA's Dynamo framework.

Historically, using it meant **building an engine**: compiling the model for a specific GPU, precision and set of maximum shapes ahead of time. Recent versions add a PyTorch-based path that loads models more like other engines, which reduces the setup cost.

## Where it can win

- **New hardware features early.** Support for FP4 on Blackwell, Hopper-specific kernels and new interconnect features often arrives first in NVIDIA's own stack.
- **Highly tuned kernels.** Matmul, attention and MoE kernels tuned per architecture, often with fused operations other engines lack.
- **Integrated quantization tooling.** NVIDIA's Model Optimizer produces FP8 and FP4 checkpoints with calibration recipes designed for these kernels.
- **Large multi-node deployments** on NVIDIA systems, with NVIDIA's networking and orchestration tools.

## What you pay

| Aspect | TensorRT-LLM | vLLM and SGLang |
| --- | --- | --- |
| Hardware | NVIDIA only | NVIDIA, AMD and others, varying by engine |
| Setup | heavier: engine build or conversion for some paths | load weights directly |
| Iteration speed | slower to change shapes or settings in the build path | fast |
| Model coverage | strong for popular models, newer architectures may lag | very broad |
| Debugging | more layers between you and the kernels | more Python, easier to inspect |

## When it is worth considering

- A stable model that will run for months at high volume.
- An all-NVIDIA fleet, especially on the newest GPUs.
- A team with time to tune the build and maintain it across releases.
- A cost structure where a few percent of efficiency is worth engineering weeks.

<div class="ex">
<div class="lab">Worked example</div>

**is 10% worth it?**

A service spends \$400,000 a month on GPUs for one model. A fair benchmark shows TensorRT-LLM with 12% higher SLO-safe capacity than the current engine on its workload.

Fewer GPUs needed: about 11%, so about \$43,000 a month.

Migration and ongoing maintenance: estimated at one engineer for two months, then a quarter of an engineer. The savings clearly pay for it.

The same 12% on a \$20,000-a-month service saves \$2,100 a month, which does not justify the work.

</div>

## How to evaluate it

Use the fair comparison method from lesson 17.01 with equal tuning effort. Include startup and rebuild time in the comparison: if your models or shapes change often, a slow build process has a real operational cost. Check that every feature you rely on, such as a specific quantization format, structured output or LoRA serving, works in the version you test.

## Deep dive: when an engine switch pays for itself

A switch that makes the fleet faster by a factor ⟦m884⟧ frees a share ⟦m885⟧ of its GPUs. With ⟦m886⟧ GPUs at price ⟦m887⟧ per GPU-hour:

⟦M883⟧

~~~dg-raw
%%chart Annual saving from a 10% faster engine, by fleet size ($2.50 per GPU-hour).
x = 8, 32, 64, 128, 512
logx = 1
logy = 1
xlab = GPUs in the fleet
ylab = Thousand dollars per year
Saving: 16, 64, 128, 255, 1019
%%
~~~

A 10% speed-up frees 9.1% of the fleet. If the switch costs three engineer-months (say \$100,000 including risk), it breaks even at about 50 GPUs. Below that, the faster engine is not worth adopting for speed alone.

<div class="take">
<div class="lab">Key takeaway</div>

TensorRT-LLM trades flexibility and portability for NVIDIA-specific performance and early hardware support. Benchmark it fairly when volume is high enough that a few percent of efficiency pays for the extra setup and maintenance.

</div>
