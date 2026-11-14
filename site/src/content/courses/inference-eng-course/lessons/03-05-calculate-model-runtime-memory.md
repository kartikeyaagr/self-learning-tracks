---
id: "03.05"
course: "inference-eng-course"
module: "03"
slug: "calculate-model-runtime-memory"
title: "Calculate Model & Runtime Memory"
keyIdeas: ["GPU memory = weights + KV cache + activations + workspace + CUDA context + fragmentation.", "Weights are fixed: params x bytes per param.", "Activations scale with tokens in a step (max batched tokens), not with total context.", "Engines reserve a fixed fraction of GPU memory and give what is left after weights and activations to the KV cache.", "More free memory means more KV blocks, which means more concurrent requests."]
question: "Break down the memory on an 80 GB GPU running an 8B BF16 model in vLLM with gpu_memory_utilization 0.9."
answer: "vLLM targets 72 GB in total. Weights take about 16 GB. Activations and temporary buffers for the largest batch take a few GB (measured by a profiling run), CUDA graphs and the CUDA context take around 1 to 2 GB. Whatever remains, roughly 50 GB, becomes the KV cache pool. The other 8 GB is left unused as a safety margin."
sources: [{"title": "vLLM: Engine arguments", "url": "https://docs.vllm.ai/en/latest/configuration/engine_args.html", "note": "gpu-memory-utilization and related flags."}, {"title": "PyTorch: CUDA memory management", "url": "https://pytorch.org/docs/stable/notes/cuda.html#memory-management", "note": "Caching allocator behavior."}]
tags: ["memory"]
---

## Where every gigabyte goes

Before you can say how many users a GPU can serve, you need a memory budget. Engines build one at startup, and reading it is one of the most useful things you can do with an engine's log.

~~~dg-layers Memory on one GPU, from fixed costs to what is left for users.
CUDA context | about 0.5 to 1 GB per process
*Weights | parameters × bytes per parameter
Activations and workspace | peak temporary tensors for the largest step
CUDA graphs | memory pools captured with the graphs
*KV cache pool | everything left inside the configured fraction
Headroom | what the configured fraction leaves free
~~~

## Weights

Weights are parameters times bytes per parameter. Llama 3 8B in BF16 takes 16 GB. In FP8 it takes 8 GB. Quantized formats such as INT4 add a little for scales and zero points, usually 3 to 10% on top of the raw bits.

Weights are fixed once loaded. If you use tensor parallelism, each GPU holds roughly its share, plus any layers that are replicated, such as embeddings in some setups.

## Activations and workspace

Activations are the temporary tensors of one forward pass. They scale with the number of tokens processed in a single step, the hidden size and the MLP width. For an 8B model:

- The largest temporaries are the MLP's gate and up outputs: 2 × tokens × 14,336 × 2 bytes.
- At 8,192 tokens in one step, that is about 470 MB, plus several smaller buffers.
- The logits for sampled positions add tokens × vocab × 4 bytes if kept in FP32. For 256 sequences and a 128k vocabulary, that is 131 MB.

Engines bound activations with a setting such as `max_num_batched_tokens`, then run a dummy forward pass at that size during startup to measure the peak. Libraries such as cuBLAS also allocate workspace buffers.

## CUDA graphs

Engines that use CUDA graphs capture one graph per batch size they expect, often dozens of them. Each graph has its own memory pool for intermediate tensors. This typically costs a few hundred MB to a couple of GB, depending on how many sizes are captured.

## The KV cache pool

After everything else is measured, the engine takes a configured fraction of total memory and gives the remainder to the KV cache. In vLLM:

KV pool = gpu_memory_utilization × total memory − weights − activation peak − other allocations

The pool is divided into fixed-size blocks, for example 16 tokens each. The startup log reports how many blocks fit, or the equivalent number of tokens. That figure sets your maximum concurrency.

<div class="ex">
<div class="lab">Worked example</div>

**an 8B model on an 80 GB GPU**

Target: 0.9 × 80 GB = 72 GB.

Weights: 16 GB. Activation peak at 8,192 batched tokens: about 2 GB. CUDA graphs: about 1 GB. CUDA context and libraries: about 1 GB.

KV pool: 72 − 16 − 2 − 1 − 1 = 52 GB.

At 128 KB per token, that is about 425,000 tokens, or about 26,600 blocks of 16 tokens. The log line you would see says roughly "maximum concurrency for 8,192 tokens per request: 52×".

</div>

~~~dg-raw
%%widget kv Change the model and GPU and watch how much room is left for the KV cache once the weights are loaded.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
t = 12
b = 32
%%
~~~

## Common budget surprises

- **Two processes on one GPU.** Each has its own CUDA context, and each engine assumes it owns the configured fraction. Two engines both set to 0.9 will crash.
- **A long max model length with chunked prefill off.** The activation peak is sized for the largest possible single prefill, which can be several GB.
- **Many captured graph sizes.** Capturing every batch size up to 512 costs more memory than capturing a few dozen.
- **Fragmentation in the PyTorch allocator.** Long-running servers can lose some hundreds of MB to fragmentation. Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps in many cases.

## The knobs you have

| Knob | Effect | Trade-off |
| --- | --- | --- |
| Raise gpu_memory_utilization (0.90 to 0.95) | More KV blocks | Less headroom for spikes and other processes |
| Lower max_num_batched_tokens | Smaller activation peak | Slower prefill for long prompts |
| Quantize weights (BF16 to FP8) | Frees half the weight memory for KV | Possible small quality loss |
| FP8 KV cache | Doubles tokens per GB of pool | Small quality risk, needs kernel support |
| Lower max_model_len | Smaller worst-case reservations | Rejects longer requests |
| Fewer CUDA graph sizes | Frees graph pool memory | More eager steps at uncaptured sizes |

## Deep dive: the memory equation an engine solves at startup

Engines like vLLM size the KV pool by profiling. They load the weights, run one forward pass at the largest batch they will allow, measure peak memory, and give whatever is left (up to the utilization fraction) to KV:

⟦M325⟧

⟦M326⟧

~~~dg-raw
%%bars An 8B model on an 80 GB H100 at 0.90 utilization (GB).
unit = GB
CUDA context and libraries | 0.8 | c4
Weights (BF16) | 16.1 | c3
Peak activations (8k-token prefill) | 2.4 | flame
CUDA graphs | 0.6 | c2
KV pool | 52.1 | c1
Held back (10%) | 8.0 | line
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why raising max-num-batched-tokens shrinks the KV pool**

Activation memory during prefill scales with tokens processed in one step. The MLP's intermediate alone is ⟦m327⟧ bytes, 235 MB at ⟦m328⟧. With other buffers, peak activations are about 2.4 GB.

Raising the step budget to 32,768 tokens raises peak activations to roughly 9 GB, and the KV pool shrinks by about 6.6 GB, about 51,000 tokens of KV. Bigger prefill chunks cost concurrency.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

weights are fixed, activations scale with tokens per step, graphs and context take a little, and the KV cache gets whatever remains. That remainder decides how many users fit, so read it in the startup log.

</div>
