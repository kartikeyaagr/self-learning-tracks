---
id: "14.05"
course: "inference-eng-course"
module: "14"
slug: "paged-attention-flashinfer-serving-backends"
title: "Paged Attention, FlashInfer & Serving Backends"
keyIdeas: ["Serving attention must handle paged KV, variable lengths and mixed prefill/decode batches.", "Libraries: FlashAttention (with paged support), FlashInfer, Triton kernels, vendor kernels.", "FlashInfer offers block-sparse and paged layouts, cascade attention for shared prefixes, and JIT-specialized kernels.", "Backend choice depends on GPU, model architecture and features needed.", "Benchmark backends at your workload's shapes; winners differ by context and batch."]
question: "What does an engine's \"attention backend\" setting choose, and what should you consider when switching backends?"
answer: "It chooses which kernel library implements attention for prefill, decode and mixed batches with the engine's KV layout, for example FlashAttention, FlashInfer, or Triton-based kernels. When switching, check hardware support, dtype support (FP8 KV), features (paged KV, sliding window, MLA, cascade attention), correctness on your model, and performance at your shapes (batch, context, GQA ratio)."
sources: [{"title": "FlashInfer", "url": "https://github.com/flashinfer-ai/flashinfer", "note": "Attention kernel library for serving."}, {"title": "FlashInfer paper", "url": "https://arxiv.org/abs/2501.01005", "note": "Design of a customizable attention engine."}, {"title": "vLLM: Attention backends", "url": "https://docs.vllm.ai/en/latest/design/attention_backends.html", "note": "Supported backends and selection."}]
tags: ["attention", "backends"]
---

## What a serving attention backend must handle

An attention kernel in a research paper takes dense, padded tensors. A serving engine needs much more:

- **Paged KV** with block tables, in the engine's block size.
- **Variable sequence lengths** in one batch, without padding.
- **Mixed batches** of prefill chunks, cached prefixes and decode tokens.
- **Head layouts**: MHA, GQA, MQA and MLA, which stores a compressed latent instead of per-head K and V.
- **Masks and variants**: causal, sliding window, attention sinks, soft-capping of logits, ALiBi on older models.
- **KV precision**: BF16, FP16, FP8, sometimes lower.
- **CUDA graph compatibility**: fixed launch shapes and no host-side work that varies per step.
- **Your hardware**: compute capability, including features such as TMA on Hopper.

No single backend is best at all of these on every GPU, which is why engines support several.

## Common options

| Backend | Strengths |
| --- | --- |
| FlashAttention 2 and 3 | fast prefill, paged and varlen support, widely used |
| FlashInfer | built for serving: paged and ragged layouts, cascade attention, many variants, JIT compilation |
| Triton kernels | portable across NVIDIA and AMD, easy to modify |
| Vendor kernels (TensorRT-LLM, cuDNN, AMD's libraries) | tuned for specific hardware |
| MLA-specific kernels (such as FlashMLA) | DeepSeek-style latent attention |

In vLLM and SGLang, a flag or environment variable selects the backend, and the startup log reports which one is in use.

## Cascade attention for shared prefixes

When many requests in a batch share a long prefix, such as a 4,000-token system prompt, a normal decode kernel reads that prefix's KV once per request. With 100 requests, the same 512 MB is read 100 times per step.

**Cascade attention** splits the computation:

1. Compute attention of all the batch's queries against the shared prefix together. The prefix's K and V are read once, and the queries form a matrix, so the tensor cores are used efficiently.
2. Compute each request's attention against its own unique suffix.
3. Merge the two partial results per request with the log-sum-exp rescaling from online softmax.

<div class="ex">
<div class="lab">Worked example</div>

**the saving from cascade attention**

128 requests share a 4,000-token prefix and each has about 500 unique tokens. Llama 3 8B.

Standard decode attention reads 128 × 4,500 × 128 KB ≈ 74 GB per step.

Cascade attention reads 4,000 × 128 KB once (0.5 GB) plus 128 × 500 × 128 KB (8.2 GB): about 8.7 GB, 8.5 times less.

At 3 TB/s, attention drops from about 25 ms to about 3 ms per step. FlashInfer and others implement this.

</div>

## Choosing a backend

1. **List the features your model needs**: MLA, sliding window, soft-capping, FP8 KV.
2. **Filter** the backends that support all of them on your GPU and with your engine version.
3. **Benchmark** the remaining options on your workload, with your context lengths and batch sizes, at the system level.
4. **Check correctness**: compare outputs with a reference backend on a set of prompts.
5. **Record the choice** in your config with the reason, since defaults change between releases.

## Things to watch

- **Silent fallbacks.** If the chosen backend does not support a feature, some engines fall back to a slower path with only a log warning.
- **Version coupling.** Backends are often compiled against specific PyTorch and CUDA versions. Mismatches cause import failures or force slow paths.
- **Numerical differences.** Backends sum in different orders, so greedy outputs can differ after near-ties. Compare on quality metrics, not exact strings.

## Deep dive: what cascade attention saves, in general

For ⟦m795⟧ sequences sharing a prefix of ⟦m796⟧ tokens, each with ⟦m797⟧ private tokens, KV bytes read per decode step are

⟦M794⟧

The prefix and suffix are attended separately and merged with the same ⟦m798⟧ rule as split-KV (08.03).

~~~dg-raw
%%chart KV read reduction from cascade attention (4,000-token shared prefix, 500 private tokens each).
x = 1, 4, 16, 64, 128, 256
logx = 1
xlab = Sequences sharing the prefix
ylab = Times fewer bytes
Reduction: 1, 3, 6, 8, 8.5, 8.7
%%
~~~

The limit is 9 times, set by ⟦m799⟧. Long shared prefixes and short private parts, typical of agents with large tool definitions, gain most.

<div class="take">
<div class="lab">Key takeaway</div>

the attention backend is a real performance lever. Pick by required features first, then by measured speed on your workload, check correctness, and watch the startup log for silent fallbacks.

</div>
