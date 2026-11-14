---
id: "06.02"
course: "inference-eng-course"
module: "06"
slug: "start-a-vllm-engine"
title: "Start a vLLM Engine"
keyIdeas: ["vllm serve <model> starts an OpenAI-compatible server.", "Key flags: max-model-len, gpu-memory-utilization, max-num-seqs, max-num-batched-tokens, tensor-parallel-size.", "Read the startup log: weight memory, KV cache size in tokens, CUDA graph capture.", "max-model-len too high wastes nothing per se but may fail if one max-length sequence cannot fit.", "Pin the version: defaults change between releases."]
question: "Name five vLLM startup flags you would set deliberately and what each controls."
answer: "--model (weights), --dtype or --quantization (precision), --max-model-len (longest sequence, which caps KV per request), --gpu-memory-utilization (fraction of GPU memory vLLM may use, which sizes the KV pool), and --max-num-seqs or --max-num-batched-tokens (scheduler limits on batch size and tokens per step). Also --tensor-parallel-size for multi-GPU."
sources: [{"title": "vLLM: Quickstart", "url": "https://docs.vllm.ai/en/latest/getting_started/quickstart.html", "note": "Starting the server."}, {"title": "vLLM: Engine arguments", "url": "https://docs.vllm.ai/en/latest/configuration/engine_args.html", "note": "Every flag."}]
tags: ["vllm"]
---

## Starting the engine

vLLM installs with pip and serves an OpenAI-compatible API with one command:

```
pip install vllm==<pinned-version>
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256
```

This loads the model, profiles memory, allocates the KV pool, captures CUDA graphs and starts a server on port 8000. Pin the version. Engine defaults change between releases, sometimes in ways that move your benchmarks by 20%.

## The flags that matter

| Flag | What it controls | Trade-off |
| --- | --- | --- |
| --max-model-len | longest prompt plus output for one request | larger values need more KV space for one request |
| --gpu-memory-utilization | fraction of GPU memory vLLM may use | higher means a bigger KV pool and less safety margin |
| --max-num-seqs | maximum sequences in one step | higher means more throughput and slower per-user decode |
| --max-num-batched-tokens | tokens processed in one step | bounds prefill chunk size and activation memory |
| --tensor-parallel-size | GPUs per model replica | fits bigger models, adds communication |
| --enable-prefix-caching | reuse of shared prompt prefixes | on by default in recent versions |
| --quantization, --kv-cache-dtype | precision of weights and of KV | memory against quality |
| --enforce-eager | disables CUDA graphs | easier debugging, slower small batches |

## How the flags interact

- **max-num-seqs and the latency target.** This caps the batch. If your TPOT SLO is tight, a lower cap keeps each step short. If throughput matters most, raise it until KV memory or compute becomes the limit.
- **max-num-batched-tokens and TTFT.** This is the token budget per step. A larger budget lets long prompts prefill in fewer steps (lower TTFT) but makes each of those steps longer (higher ITL for everyone else). Values of 2,048 to 8,192 are common for chat.
- **max-model-len and capacity.** The engine checks at startup that one request of this length fits in the pool. Setting it to 128k on a small GPU can make startup fail, even if your real requests are short.

## Reading the startup log

The startup log answers most capacity questions before you send a single request. Look for:

- **Weight memory**: for example "Loading model weights took 15.2 GB". It should match parameters × bytes.
- **KV cache size**: for example "GPU KV cache size: 412,000 tokens" and "Maximum concurrency for 8,192 tokens per request: 50.3x".
- **CUDA graph capture**: the batch sizes captured and the time taken, typically tens of seconds.
- **Attention backend**: which kernel library it chose, such as FlashAttention or FlashInfer.
- **Warnings**: fallbacks to slower kernels, unsupported features or a dtype that differs from what you asked for.

<div class="ex">
<div class="lab">Worked example</div>

**reading capacity from the log**

The log says "GPU KV cache size: 412,000 tokens" for Llama 3 8B on one H100 at 0.9 memory utilization.

At 2,000 tokens per conversation, about 206 conversations fit. With max-num-seqs at 256, the KV pool, not the sequence cap, will be the limit.

If the product needs 300 concurrent conversations, the options are FP8 KV cache (about 824,000 tokens), FP8 weights (about 480,000 tokens) or a second replica.

</div>

## A smoke test

```
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct",
       "messages": [{"role": "user", "content": "Say hi in five words."}],
       "max_tokens": 32}'
```

Check three things in the response: the text is sensible (the chat template works), the `usage` field reports token counts, and `finish_reason` is "stop" rather than "length" (the stop tokens work).

## Starting it like production

- Run it in a container with pinned versions of vLLM, PyTorch and CUDA.
- Keep the weights on local NVMe or in the image, not on network storage.
- Expose `/health` to your orchestrator and wait for it before sending traffic.
- Send a few warm-up requests of realistic shapes before marking the replica ready.

## Deep dive: reading the startup numbers back into memory

The engine's log prints its KV pool in tokens. Multiply by ⟦m469⟧ to get bytes and check it against your own budget:

⟦M466⟧

412,000 tokens × 128 KB is 54 GB. At max-model-len 8,192, the engine reports a maximum concurrency of about 50: the number of maximum-length requests that fit at once.

Each step of gpu-memory-utilization moves a fixed amount:

⟦M467⟧

⟦M468⟧

Raising utilization from 0.90 to 0.95 adds about 30,500 tokens of KV (7%), and leaves 4 GB instead of 8 GB for anything else that allocates on the GPU.

<div class="ex">
<div class="lab">Worked example</div>

**a configuration that will not start**

Llama 3 8B on a 24 GB GPU at 0.90 utilization: ⟦m470⟧ GB for KV, about 31,000 tokens.

max-model-len 131,072 needs 16 GB for one sequence. The engine refuses to start, because not even one request of the maximum length fits.

Setting max-model-len to 16,384 (2.1 GB per full-length sequence) makes it start, with room for one or two full-length requests or many short ones.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a handful of flags set memory and batching behaviour. Set them deliberately, read the startup log for weights, KV capacity and kernel choices, and pin the engine version.

</div>
