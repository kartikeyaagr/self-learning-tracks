---
id: "17.04"
course: "inference-eng-course"
module: "17"
slug: "llama-cpp-edge-oriented-runtimes"
title: "llama.cpp & Edge-Oriented Runtimes"
keyIdeas: ["llama.cpp: C/C++ runtime for CPUs, Apple Silicon and consumer GPUs.", "GGUF format with many quantization levels (2 to 8 bits).", "Optimized for single-user, low-memory, dependency-light deployment.", "Has a server mode with basic batching, but not built for large-scale concurrency.", "Edge constraints: memory capacity and bandwidth, power, no datacenter interconnect."]
question: "Why would you choose llama.cpp over vLLM, and when would you not?"
answer: "For local or edge deployment: laptops, phones, CPUs, Apple Silicon or small GPUs, single-user use, minimal dependencies, and aggressive quantization formats (GGUF k-quants). Not for high-concurrency datacenter serving, where vLLM or SGLang's batching, paged KV and multi-GPU support give far higher throughput."
sources: [{"title": "llama.cpp GitHub repository", "url": "https://github.com/ggml-org/llama.cpp", "note": "Source, GGUF format and server."}, {"title": "MLX", "url": "https://github.com/ml-explore/mlx", "note": "Apple Silicon array framework with LLM examples."}]
tags: ["llama.cpp", "edge"]
---

## The edge setting

Not all inference runs in a datacenter. Laptops, phones, workstations and small servers run models locally for privacy, offline use, low latency or zero per-token cost. The constraints are very different.

| Constraint | Datacenter | Edge |
| --- | --- | --- |
| Memory | 80 GB or more of HBM per GPU | 8 to 128 GB, often shared with the CPU |
| Bandwidth | 3 TB/s or more | 50 to 800 GB/s |
| Users | thousands | one or a few |
| Main goal | cost per token at an SLO | latency, privacy, offline operation |
| Power | hundreds of watts per GPU | a few to tens of watts |

## llama.cpp

llama.cpp is the most widely used runtime for local inference.

- **Pure C and C++**, with no Python at runtime, and backends for CPUs with SIMD instructions, Apple's Metal, CUDA, Vulkan, and others.
- **GGUF model files**: a single-file format holding weights, tokenizer and metadata, with many quantization types (from 8-bit down to about 1.5 bits per weight with mixed schemes). A 7B model runs in about 4 to 5 GB at 4-bit.
- **Memory-mapped weights**: the file is mapped into memory, so startup is fast and the operating system pages weights in as needed.
- **Layer offloading**: some layers on the GPU, the rest on the CPU, for models larger than GPU memory.
- **A server** with an OpenAI-compatible API, so local tools can talk to it like any other endpoint.

## The same physics, smaller numbers

Decode is still bandwidth bound. The ceiling is still bandwidth divided by weight bytes.

| Device | Bandwidth | 8B model at about 4.5 GB (4-bit) | Ceiling |
| --- | --- | --- | --- |
| Laptop, dual-channel DDR5 | about 80 GB/s | 4.5 GB per token | about 18 tokens/s |
| Apple M-series Pro | about 200 GB/s | 4.5 GB | about 44 tokens/s |
| Apple M-series Max or Ultra | 400 to 800 GB/s | 4.5 GB | about 90 to 180 tokens/s |
| RTX 4090 | about 1 TB/s | 4.5 GB | about 220 tokens/s |

<div class="ex">
<div class="lab">Worked example</div>

**why quantization is not optional at the edge**

An 8B model in BF16 needs 16 GB. On a laptop with 16 GB of RAM, it does not fit alongside the operating system and apps.

At 4-bit it needs about 4.5 GB, fits easily, and decodes at up to about 18 tokens per second on 80 GB/s of bandwidth, a comfortable reading speed.

At the edge, quantization is how models fit and run at all, not a cost optimization.

</div>

## Prefill on the edge

Prefill is compute bound, and edge devices have much less compute than datacenter GPUs. A 4,000-token prompt that prefills in 0.1 s on an H100 can take 5 to 30 s on a laptop CPU. Long prompts, RAG and large system prompts hurt far more at the edge. Prompt caching across turns, which llama.cpp supports, matters a lot.

## Other edge runtimes

- **MLX**: Apple's array framework for Apple silicon, with an LLM library that makes good use of unified memory.
- **ONNX Runtime**: cross-platform, with execution providers for many accelerators, including mobile NPUs.
- **Mobile runtimes**: frameworks such as ExecuTorch and vendor SDKs that target phone NPUs and GPUs.
- **Ollama and LM Studio**: user-friendly applications built on top of llama.cpp and similar runtimes.

The right choice depends on the device and the accelerator it has.

## Deep dive: decode speed on edge devices, from bandwidth alone

The same bound as on a data-center GPU applies: each token reads all weights once.

⟦M888⟧

~~~dg-raw
%%bars Upper bound on batch-1 decode speed for an 8B model in 4-bit (about 4.9 GB), from memory bandwidth (tokens/s).
unit = tok/s
Desktop GPU, about 1,000 GB/s | 204 | c1
High-end laptop, about 400 GB/s | 82 | c2
Mid-range laptop, about 200 GB/s | 41 | c3
Phone, about 60 GB/s | 12 | c4
%%
~~~

Real speeds land at 60 to 80% of these bounds. Two consequences follow. A 4-bit model is not a luxury on the edge: in BF16 the same model would read 16 GB per token and run at a quarter of the speed, if it fit at all. And a phone can run an 8B model at reading speed, but only just; a 3B model at 4 bits (about 1.9 GB) is the comfortable size there.

<div class="take">
<div class="lab">Key takeaway</div>

llama.cpp is the main tool for local and edge inference, where memory is small and users are few. The bandwidth math is the same as in the datacenter, so quantization decides whether a model fits and how fast it decodes, and long prompts hurt much more.

</div>
