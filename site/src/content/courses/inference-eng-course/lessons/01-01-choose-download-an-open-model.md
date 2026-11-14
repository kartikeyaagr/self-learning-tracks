---
id: "01.01"
course: "inference-eng-course"
module: "01"
slug: "choose-download-an-open-model"
title: "Choose & Download an Open Model"
keyIdeas: ["Pick by task quality, license, size and context length. Instruct or chat variants are tuned to follow prompts.", "Weight memory = params x bytes per param. Leave 20 to 40% of GPU memory for KV cache and overhead.", "Models ship as safetensors shards plus config.json, tokenizer files and a generation config.", "Read the model card: chat template, context length, license, recommended sampling settings.", "Download once with huggingface-cli or snapshot_download and reuse the cache."]
question: "You have one 24 GB GPU. Pick a model size and precision for a chat assistant and justify it with memory math."
answer: "An 8B instruct model in BF16 needs about 16 GB for weights, leaving roughly 6 to 7 GB for KV cache and activations, which is enough for modest concurrency at a few thousand tokens of context. A 14B model would need 28 GB in BF16 and only fits if quantized to 8-bit or 4-bit. So an 8B BF16 model, or a larger model in FP8/INT4 if quality needs it."
sources: [{"title": "Hugging Face Hub: Models", "url": "https://huggingface.co/docs/hub/models", "note": "Model cards, files and licenses."}, {"title": "Safetensors documentation", "url": "https://huggingface.co/docs/safetensors/index", "note": "The safe, fast weight format most models ship in."}]
tags: ["models"]
---

## Start from the job, not the leaderboard

Model choice is the decision that sets every number after it: memory, speed, cost and quality. Engineers often start with a leaderboard and work backwards. It works better the other way round. Write down the task, the hardware and the latency target, then find the smallest model that does the task well enough.

Ask four questions in this order:

1. **What does the model have to do?** A support chatbot, a code assistant, a classifier and a summarizer have very different quality bars.
2. **What hardware will it run on?** One 24 GB card, one 80 GB card or a node of eight GPUs gives very different options.
3. **How long are the inputs?** A model trained for 8k tokens will not handle 100k-token documents, however good it is.
4. **Can you legally ship it?** Licenses differ, and some weights are gated behind an agreement.

## Base, instruct and chat variants

Most model families ship several variants of each size:

- **Base** models are trained only to predict the next token on raw text. They continue text well but do not follow instructions reliably. Use them for fine-tuning or for pure completion tasks.
- **Instruct** or **chat** models have extra training on conversations and instructions, usually followed by preference tuning. Use them for assistants.
- **Specialized** variants (code, math, long context, vision) trade some general quality for one strength.

For serving, the variant matters in one more way: instruct models expect a specific chat template, covered in the next lesson.

## The bytes math that decides fit

Weight memory is parameters times bytes per parameter. This is the first calculation you should do for any model.

| Model | BF16 weights | FP8 | INT4 (with scales) |
| --- | --- | --- | --- |
| 1.5B | 3 GB | 1.5 GB | about 1 GB |
| 8B | 16 GB | 8 GB | about 4.5 GB |
| 14B | 28 GB | 14 GB | about 8 GB |
| 32B | 64 GB | 32 GB | about 18 GB |
| 70B | 140 GB | 70 GB | about 38 GB |

Weights are not the whole story. Three other things need GPU memory:

- **KV cache**: the attention keys and values for every active token. It grows with users and context length, and it is the thing that sets how many people you can serve at once.
- **Activations**: temporary tensors during the forward pass. Small for decode, larger for long prefill chunks.
- **Runtime overhead**: the CUDA context (several hundred MB), CUDA graphs, the allocator's reserve and library workspaces. Budget 1 to 3 GB.

A useful rule: weights should take at most 60 to 80% of GPU memory. Above that, there is little room left for users.

<div class="ex">
<div class="lab">Worked example</div>

**what fits on a 24 GB card?**

Llama 3 8B in BF16 takes 16 GB of weights. Subtract about 1.5 GB of overhead and you have about 6.5 GB for KV cache.

Llama 3 8B uses 32 layers, 8 KV heads and head size 128, so each token's KV cache is 2 × 32 × 8 × 128 × 2 bytes = 128 KB.

6.5 GB ÷ 128 KB ≈ 50,000 tokens of cache in total. That is about 12 users with 4k-token conversations, or 1 user with a very long document.

A 14B model in BF16 (28 GB) does not fit at all. In FP8 (14 GB) it fits with about 8 GB of cache. In INT4 (8 GB) it fits with about 14 GB of cache.

</div>

~~~dg-raw
%%widget kv Pick a model and GPU, then move context length and batch to see when weights plus KV cache stop fitting.
model = Llama 3 8B
gpu = RTX 4090 (24 GB)
t = 12
b = 4
%%
~~~

## Dense and mixture-of-experts models

In a **dense** model every parameter is used for every token. In a **mixture-of-experts (MoE)** model, each layer has many expert MLPs and each token uses only a few of them. Mixtral 8x7B has about 47B parameters in total but uses about 13B per token.

This splits the two numbers you care about:

- **Memory** follows the total parameter count, because every expert must be loaded.
- **Compute and decode bandwidth** follow the active parameter count, at least at small batch sizes.

So an MoE model can be fast for its quality but still needs a lot of memory. Module 19 covers MoE serving.

## Attention variants change memory too

Two models with the same parameter count can have very different KV cache sizes:

- **Multi-head attention (MHA)**: one key and value head per query head. Llama 2 7B uses 32 of each, which gives 512 KB of KV cache per token.
- **Grouped-query attention (GQA)**: several query heads share one KV head. Llama 3 8B uses 8 KV heads for 32 query heads, which gives 128 KB per token, four times less.
- **Multi-head latent attention (MLA)**: used in DeepSeek models. It stores a compressed latent instead of full keys and values and is smaller still.

When you compare candidates, compare KV bytes per token, not only parameter counts.

## What is inside a model repository

A Hugging Face model repository usually contains:

- `config.json`: the architecture's hyperparameters, such as layers, hidden size, attention heads, KV heads, vocabulary size and maximum positions.
- `model-0000X-of-0000N.safetensors`: the weights, split into shards of a few GB each. Safetensors is a simple format that can be memory-mapped and cannot run code when loaded, unlike old pickle-based `.bin` files.
- `model.safetensors.index.json`: which tensor lives in which shard.
- `tokenizer.json` and `tokenizer_config.json`: the vocabulary, merge rules and the chat template.
- `generation_config.json`: default sampling settings and stop tokens.
- `README.md`: the model card, with the license, intended use, context length and recommended settings.

Read the model card before you serve the model. It often says which chat template to use, which sampling settings were evaluated and whether the long-context support needs special settings such as RoPE scaling.

## Downloading well

```
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --revision <commit-sha> --local-dir /data/models/qwen2.5-7b
```

- **Pin the revision.** Repositories change. A pinned commit hash makes your server reproducible.
- **Keep weights on fast local disk.** Loading 16 GB from NVMe takes a few seconds. Loading it from network storage can take minutes and dominates cold start.
- **Download once.** In a fleet, bake the weights into an image or a shared cache instead of having every replica pull from the internet.

## Deep dive: the full memory equation for fitting a model

A model fits on a GPU when

⟦M177⟧

where ⟦m181⟧ is the fraction the engine is allowed to use (often 0.9), ⟦m182⟧ covers the CUDA context, activations and workspace (typically 1 to 4 GB), and the KV term grows with batch ⟦m183⟧ and context ⟦m184⟧:

⟦M178⟧

⟦m185⟧ and ⟦m186⟧ are bytes per weight and per KV value. Rearranging gives the number of tokens of KV you can afford:

⟦M179⟧

<div class="ex">
<div class="lab">Worked example</div>

**three candidates for one 48 GB L40S**

Budget: ⟦m187⟧ GB, minus 2 GB runtime = 41.2 GB.

Llama 3 8B, BF16: weights 16.1 GB, leaves 25.1 GB. At 128 KB per token, ⟦m188⟧ tokens of KV, enough for 46 users at 4k context.

Qwen2.5 32B, BF16: weights 65.6 GB. Does not fit.

Qwen2.5 32B, 4-bit weights (about 0.56 bytes per parameter with scales): 18.4 GB, leaves 22.8 GB. KV per token is ⟦m189⟧ KB, so about 87,000 tokens: 21 users at 4k context.

</div>

~~~dg-raw
%%bars How the L40S budget splits for the three candidates (GB).
unit = GB
stack = Weights, Runtime, KV pool available
Llama 3 8B BF16 | 16.1+2+25.1
Qwen2.5 32B 4-bit | 18.4+2+22.8
Qwen2.5 32B BF16 | 65.6+2+0 | | does not fit: 67.6 GB needed before any KV
%%
~~~

## Deep dive: MoE memory versus compute

For a mixture-of-experts model, memory follows total parameters and per-token compute follows active parameters. The ratio tells you how much "free" quality you buy with memory:

⟦M180⟧

Mixtral 8x7B has about 47B total and 13B active, a ratio of 3.6. You pay 47B worth of memory to run at roughly 13B speed at batch 1.

<div class="take">
<div class="lab">Key takeaway</div>

start from the task and the hardware, then do the bytes math. Weights at more than about 80% of GPU memory leave no room for users, and KV bytes per token matter as much as parameter count.

</div>
