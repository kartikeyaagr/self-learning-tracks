---
id: "06.06"
course: "ai-engineering-course"
module: "06"
slug: "moe-for-app-engineers"
title: "MoE for app engineers"
keyIdeas: ["A mixture-of-experts layer replaces one MLP with many expert MLPs and a router that sends each token to a few of them; memory scales with total parameters, compute per token with active parameters.", "Read a model card for total and active parameters, experts routed per token, shared experts, attention type, checkpoint precision and the hardware it is sized for.", "At batch 1 an MoE decodes like its active size; as the batch grows the tokens touch most experts, so each step reads nearly all expert weights.", "MoE favours hosted endpoints that batch many customers; it hurts small self-hosted deployments that pay for the memory of the whole model, and multi-GPU setups that pay for all-to-all traffic.", "Precision is part of the architecture now: gpt-oss ships MXFP4 experts, DeepSeek-V3.2 ships FP8 weights, so a parameter count alone does not tell you the memory."]
question: "A team wants to self-host gpt-oss-120b (117B total, 5.1B active, MoE weights in MXFP4) for an internal tool that gets a few requests a minute, arguing it 'costs like a 5B model'. Using the model card, explain what it really costs in memory and speed, how that changes with batch size, and what you would recommend."
answer: "Compute per token is like a 5B model, but memory is not: every expert must be resident because any token can route to any expert. The 114.7B expert parameters at 4.25 bits are about 61 GB, plus about 4 GB of BF16 attention, embedding and router weights, so roughly 65 GB, which fills an 80 GB H100 and leaves about 12 GB for KV cache. At batch 1 each layer reads only 4 of its 128 experts, so decode is fast. As concurrency grows, tokens spread over experts: with uniform routing 32 concurrent tokens touch about 82 experts per layer, so each step reads most of the weights. For a few requests a minute the GPU sits idle most of the time, so paying a hosted provider per token is cheaper unless data terms require self-hosting; if they do, gpt-oss-20b (about 14 GB) may meet the quality bar on one smaller GPU, which you check on your own eval."
rubric: ["Separates memory (total parameters, all experts resident) from compute (active parameters)", "Works out memory from the checkpoint precision, not from parameter count times 2 bytes", "Explains that larger batches touch more experts so per-step weight reads grow", "Compares utilisation and cost of self-hosting at low volume with paying per token", "Suggests a smaller model or hosted endpoint, validated on an eval"]
sources: [{"title": "OpenAI: gpt-oss-120b and gpt-oss-20b Model Card (2025)", "url": "https://arxiv.org/abs/2508.10925", "note": "117B/5.1B and 21B/3.6B MoE, top-4 of 128 or 32 experts, MXFP4 expert weights."}, {"title": "Qwen Team: Qwen3 Technical Report (2025)", "url": "https://arxiv.org/abs/2505.09388", "note": "Qwen3-235B-A22B and Qwen3-30B-A3B: 128 experts, 8 active, no shared experts."}, {"title": "DeepSeek-AI: DeepSeek-V3.2 (2025)", "url": "https://arxiv.org/abs/2512.02556", "note": "671B-total MoE with MLA and DeepSeek Sparse Attention."}, {"title": "Hugging Face: Mixture of Experts Explained", "url": "https://huggingface.co/blog/moe", "note": "Routers, load balancing, and why MoE is memory-heavy at inference."}]
practice: ["moe-total-vs-active-params", "top-k-moe-router-with-load-statistics", "expert-parallel-routing-across-gpus", "moe-token-dispatch-and-expert-aggregation", "mxfp4-dequantize"]
tools: ["gpt-oss", "Qwen3", "DeepSeek-V3.2", "vLLM", "llama.cpp"]
tags: ["Inference", "MoE", "Model choice"]
---

## Total and active parameters

Most strong open-weights models released since 2025 are mixtures of experts: gpt-oss, Qwen3's large models, DeepSeek-V3.x, Llama 4, Kimi K2. In each transformer block the attention part is unchanged, but the MLP is replaced by $E$ expert MLPs and a small router. For each token the router scores every expert, keeps the top $k$, runs only those and mixes their outputs by the router weights.

~~~dg-flow One MoE layer, as in gpt-oss-120b: 4 of 128 experts per token.
Token vector -(router scores 128 experts)-> *Top 4 -(run only these)-> Expert 7, 31, 64, 90 -(weighted sum)-> Layer output
~~~

Two numbers follow. **Total parameters** set memory: any token may route to any expert, so every expert must be loaded. **Active parameters** set compute: a forward pass costs about $2 \times N_{\text{active}}$ FLOPs per token. A model card that says "117B with 5.1B active" is telling you it needs the memory of a large model and does the arithmetic of a small one. Routers, load-balancing losses and capacity limits are training concerns; [MoE inference](lesson:ieng:19.01) covers how engines run them.

## Reading an MoE model card

The figures below come from each model's card and Hugging Face config.

| Model | Total / active | Experts per layer | Attention | Shipped precision | Weights on disk |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-120b | 117B / 5.1B | 128, top-4 | GQA (8 KV heads), alternating full and 128-token sliding window | MXFP4 experts, BF16 rest | about 65 GB |
| gpt-oss-20b | 21B / 3.6B | 32, top-4 | same pattern | MXFP4 experts | about 14 GB |
| Qwen3-235B-A22B | 235B / 22B | 128, top-8, no shared expert | GQA (4 KV heads) | BF16 (official FP8 variant) | 470 GB (239 GB FP8) |
| Qwen3-30B-A3B | 30.5B / 3.3B | 128, top-8 | GQA (4 KV heads) | BF16 | 61 GB |
| Qwen3-Next-80B-A3B | 80B / 3B | 512, top-10 + 1 shared | hybrid: linear attention, full attention every 4th layer | BF16 | about 160 GB |
| DeepSeek-V3.2 | 671B / 37B | 256, top-8 + 1 shared | MLA plus DeepSeek Sparse Attention | FP8 (E4M3, 128×128 blocks) | about 690 GB |

What each column tells you:

- **Total / active.** Divide to get the sparsity: gpt-oss-120b uses 4% of its parameters per token, DeepSeek-V3.2 about 5.5%. Price per token from hosted providers tends to track active size; the hardware you would rent tracks total size.
- **Experts and top-k.** More, smaller experts (512 in Qwen3-Next) mean finer routing and more experts touched per batch. A shared expert runs for every token and acts like a small dense core.
- **Attention.** This sets the KV cache, which is separate from the weights. MLA stores a compressed latent: for DeepSeek-V3.2 that is 576 values per layer, about 70 KB per token in BF16 across 61 layers, so a 128k-token context needs about 9 GB. DeepSeek Sparse Attention lets each query attend to 2,048 selected tokens, which cuts long-context compute, but the cache still holds every token.
- **Shipped precision.** gpt-oss was post-trained with its experts in MXFP4 (4.25 bits per value), and its published evals use those weights, so running it in MXFP4 is the reference, not a compromise. DeepSeek-V3.2 ships FP8. Re-quantizing either further is your own risk ([what your provider optimizes](lesson:aieng:06.05)).
- **Licence and context.** gpt-oss and Qwen3 are Apache 2.0, DeepSeek-V3.2 is MIT. Qwen3's base models are 32k tokens natively and reach 128k with YaRN scaling, which the card recommends enabling only when you need it.

## Why cheap per token is not cheap to host

<div class="ex">
<div class="lab">Worked example</div>

**gpt-oss-120b on one 80 GB GPU.** Expert weights: 128 experts × 3 × 2,880² parameters × 36 layers = 114.7B parameters; at 4.25 bits that is 60.9 GB. The remaining 2.2B parameters (attention, embeddings, router) in BF16 add 4.3 GB. Total 65.3 GB, matching the card's claim that it fits on one H100 with about 12 GB left for KV cache and activations. Only half the layers use full attention, so the long-context cache costs about 36 KB per token: about 320k cached tokens in that headroom.

**How many experts one decode step touches.** If routing were uniform, a batch of $B$ tokens choosing $k$ of $E$ experts touches on average

$$
E \left(1 - \left(1 - \tfrac{k}{E}\right)^{B}\right)
$$

experts per layer. For gpt-oss-120b ($E = 128$, $k = 4$):

| Concurrent tokens | Experts touched per layer | Share of expert weights read per step |
| --- | --- | --- |
| 1 | 4 | 3% (1.9 GB) |
| 8 | 28.7 | 22% |
| 32 | 81.7 | 64% (39 GB) |
| 128 | 125.8 | 98% |

At batch 1 the model streams like a small model because each step reads about 1.9 GB of expert weights. At batch 32 each step reads about 39 GB, four times what a dense 5B model in BF16 would read, so per-step latency drifts toward that of a big dense model. The FLOPs per token still match the active size, which is why the economics win only at large batch, where a server is compute-bound and spreads those 39 GB across many users. Real routing is skewed rather than uniform: fewer experts are touched, but the popular ones become hot spots ([load imbalance](lesson:ieng:19.04)).

</div>

## When MoE helps you and when it hurts

~~~dg-layers Who pays for what in MoE serving.
*Hosted endpoint | batches thousands of users, fills experts, charges near active-size prices
Self-hosted, high traffic | full batches amortise all-expert reads; expert parallelism across GPUs
Self-hosted, low traffic | pays for total-size memory that sits idle; dense models of the same quality may be easier
Laptop or single consumer GPU | small MoE (Qwen3-30B-A3B, gpt-oss-20b) with experts offloaded to CPU RAM
~~~

**It helps** when you call it through an API: providers batch many customers onto the same experts, so you get large-model quality at a per-token price closer to the active size. It also helps on a single machine with lots of system RAM. llama.cpp's `--cpu-moe` and `--n-cpu-moe N` keep expert weights in CPU memory and run attention on the GPU. That works because each token needs only a few experts per layer. Qwen3-30B-A3B at about 4.5 bits is roughly 17 GB, so it fits a 24 GB GPU outright.

**It hurts** in three places:

- **Low-traffic self-hosting.** You rent memory for the total parameters whether one user is active or a hundred. At a few requests a minute that GPU is mostly idle, and paying per token is cheaper ([choosing a model](lesson:aieng:01.05) has the total-cost comparison).
- **Models larger than one node.** DeepSeek-V3.2's roughly 690 GB of FP8 weights do not fit in 8 × 80 GB H100s (640 GB). You need 8 × H200 (1,128 GB) or two nodes, and then every MoE layer exchanges tokens between GPUs (expert parallelism, `--enable-expert-parallel` in vLLM). Interconnect bandwidth becomes the bottleneck ([expert parallelism](lesson:ieng:19.03)).
- **Fine-tuning.** The checkpoint is as big as the total, the router is usually frozen, and adapters need to cover the expert MLPs to work well (see [LoRA and QLoRA in practice](lesson:aieng:07.07)). Many hosted fine-tuning services offer only smaller dense models or a few MoE sizes.

## Common mistakes

- **Sizing memory from active parameters.** "3B active" does not run in 6 GB. Size from total parameters times the shipped bits, then add KV cache.
- **Doubling a parameter count to get bytes.** gpt-oss is not 234 GB; DeepSeek-V3.2 is not 1.3 TB. Check the shipped precision.
- **Benchmarking at batch 1 and extrapolating.** Single-stream speed is the best case for MoE. Measure at your real concurrency.
- **Assuming sparse attention shrinks the cache.** DSA saves attention compute. MLA is what shrinks DeepSeek's cache.

## What interviewers probe

- Total versus active parameters, and which one sets memory, compute and price.
- Why an MoE's decode gets relatively slower as the batch grows, and where the economics flip.
- How to estimate a model's memory from its card, including precision and KV cache.
- When you would self-host an MoE instead of calling an API, and why expert parallelism needs a fast interconnect.

<div class="take">
<div class="lab">Key takeaway</div>

An MoE model needs memory for its total parameters and does arithmetic for its active parameters. That suits hosted endpoints, which fill experts with many users' tokens, and punishes small self-hosted deployments that pay for idle memory. Read the card for experts, attention and shipped precision before you size hardware.

</div>
