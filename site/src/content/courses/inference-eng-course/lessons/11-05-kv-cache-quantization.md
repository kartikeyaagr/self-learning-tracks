---
id: "11.05"
course: "inference-eng-course"
module: "11"
slug: "kv-cache-quantization"
title: "KV Cache Quantization"
keyIdeas: ["KV cache often uses more memory than weights at high concurrency or long context.", "FP8 KV halves memory and attention bandwidth.", "Keys have per-channel outliers; values are smoother. Methods like KIVI quantize them differently.", "Scales: per tensor (simple), per head or per token (more accurate).", "Check quality on long-context tasks, where KV error accumulates."]
question: "What are the two benefits of an FP8 KV cache, and what can go wrong?"
answer: "It halves KV memory, roughly doubling the number of tokens that fit (more concurrency or longer context), and it halves bytes read by decode attention, speeding long-context decode. Things that can go wrong: quality loss if scales are poor (especially for keys with outlier channels), attention kernels without FP8 support falling back to slow paths, and calibration needed for per-tensor scales."
sources: [{"title": "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", "url": "https://arxiv.org/abs/2402.02750", "note": "Low-bit KV with per-channel keys."}, {"title": "vLLM: Quantized KV cache", "url": "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html", "note": "FP8 KV in practice."}]
tags: ["kv-cache", "quantization"]
---

## Why quantize the KV cache

At 128 KB per token in BF16, 200 sequences of 4,000 tokens hold 200 × 4,000 × 128 KB ≈ 102 GB of KV cache. That is more than six times the weights of an 8B model. At long context or high concurrency, the cache, not the weights, decides both capacity and decode speed.

Halving the bytes per element has two effects:

- **Capacity doubles.** The same pool holds twice as many tokens, so twice as many sequences fit.
- **Attention traffic halves.** Decode attention reads every cached key and value each step, so it becomes up to twice as fast at long context.

## FP8 KV cache

The common production option stores K and V in FP8 E4M3, with a scale per tensor, or per head in some engines. The attention kernel loads FP8 values, converts them to BF16 or FP16 in registers, and computes as usual. vLLM, SGLang and TensorRT-LLM support it on Ada, Hopper, Blackwell and recent AMD GPUs.

Scales come from one of three places:

- A default of 1.0, which works when K and V values stay within FP8's range.
- Scales stored in the checkpoint, computed during quantization.
- Scales computed from calibration data.

Calibrated scales are safer for models with large activation values.

~~~dg-raw
%%widget kv Compare the KV bar at 32k context, then halve it in your head for FP8. That is the capacity FP8 KV buys.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
t = 15
b = 12
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**FP8 KV at long context**

Llama 3 8B, 16 sequences at 32,000 tokens on one H100.

BF16 KV: 16 × 4 GB = 64 GB. It does not fit next to 16 GB of weights in the pool.

FP8 KV: 32 GB. It fits, with room to spare.

Each decode step reads 16 GB of weights plus 32 GB of KV, about 16 ms. In BF16, even if it fitted, it would read 80 GB, about 27 ms.

</div>

## Lower bits

Research methods push the cache to 4 or even 2 bits. They rely on what is known about the structure of K and V:

- **Keys have outlier channels.** A few dimensions are consistently large across tokens, so keys are quantized per channel.
- **Values are smoother across channels**, so they are quantized per token.
- **Recent tokens matter most.** Keeping the last few dozen or hundred tokens in full precision protects quality.
- **Rotation tricks** spread outliers across dimensions before quantizing.

Methods such as KIVI and KVQuant report usable quality at 2 to 4 bits. They need specialized kernels and are less common in production engines than FP8. Treat them as an option to evaluate when FP8 is not enough.

## How errors behave

KV quantization errors are different from weight errors. Each decode step's attention reads every quantized key and value in the context. Small errors in many keys can shift attention weights, especially when the model must find one specific fact in a long context. Short-context benchmarks can look perfect while long-context retrieval degrades.

## Measuring the impact

| Check | Why |
| --- | --- |
| KV capacity in the startup log | confirm the pool really doubled |
| Long-context accuracy (needle tests, long QA, multi-document tasks) | errors accumulate with length |
| Your product eval | catches task-specific regressions |
| Decode TPOT at long context | confirms the bandwidth saving |
| Kernel name in a profile | catches slow fallback paths |

## Deep dive: what KV precision buys

KV bytes per token scale with the bits per value, plus scales:

⟦M656⟧

~~~dg-raw
%%bars Tokens that fit in a 52 GB pool for Llama 3 8B, by KV precision (thousands).
unit = k tokens
BF16 (16 bits) | 397 | c4
FP8 (8 bits) | 794 | c1
INT4 with group scales (about 4.25 bits) | 1494 | wine
%%
~~~

## Deep dive: how KV error reaches the output

Quantizing ⟦m658⟧ with per-element error ⟦m659⟧ perturbs each attention score by ⟦m660⟧. Errors in independent elements partly cancel, so the score error grows with ⟦m661⟧, not ⟦m662⟧. The softmax then turns score errors into weight errors:

⟦M657⟧

Tokens with near-zero attention weight barely move. The risk is in sharp attention, where one or two tokens dominate: a long-context retrieval head pointing at one fact. That is why needle-in-a-haystack tests catch KV quantization problems that average benchmarks miss.

<div class="take">
<div class="lab">Key takeaway</div>

KV quantization is often the cheapest capacity win, because it doubles the pool and halves attention traffic. FP8 is the safe default; lower bits need per-channel keys, per-token values and long-context evaluation.

</div>
