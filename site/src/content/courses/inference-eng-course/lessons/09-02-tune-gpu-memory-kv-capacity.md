---
id: "09.02"
course: "inference-eng-course"
module: "09"
slug: "tune-gpu-memory-kv-capacity"
title: "Tune GPU Memory & KV Capacity"
keyIdeas: ["KV capacity decides max concurrency, which often decides throughput.", "Knobs: memory utilization fraction, KV dtype, weight precision, max model length, batched tokens, graph capture sizes.", "Each knob has a cost: OOM risk, quality, long-context support or speed.", "Measure KV tokens available (startup log) and max SLO-safe load after each change.", "Stop when KV is no longer the binding constraint."]
question: "List four ways to increase KV capacity on a single GPU and the cost of each."
answer: "Raise gpu_memory_utilization (less safety margin against OOM); quantize the KV cache to FP8 (small quality risk, kernel support needed); quantize weights to FP8 or INT4 to free space (quality risk); lower max_model_len or max_num_batched_tokens to reduce reserved or activation memory (limits long requests or TTFT). Disabling CUDA graphs saves a little memory but costs speed."
sources: [{"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Memory-related tuning."}, {"title": "vLLM: Quantized KV cache", "url": "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html", "note": "FP8 KV cache."}]
tags: ["memory", "tuning"]
---

## The knobs for KV capacity

When memory is the binding constraint, you need more KV tokens in the pool. There are six ways to get them, each with a cost.

| Knob | Gain | Cost |
| --- | --- | --- |
| gpu_memory_utilization 0.90 to 0.95 | about 5% of GPU memory moves to KV | less headroom for spikes and other processes |
| kv_cache_dtype fp8 | about 2× KV tokens | small accuracy risk, needs kernel support |
| FP8 or INT4 weights | frees half or more of weight memory | possible quality loss, kernel speed varies |
| Lower max_model_len | allows admission when long requests dominate | rejects longer requests |
| Lower max_num_batched_tokens | smaller activation peak, so a larger pool | slower prefill for long prompts |
| Fewer CUDA graph sizes | a few hundred MB | uncaptured batch sizes run more slowly |

## How much each knob buys

Work out the gain before you try it. For Llama 3 8B on an 80 GB H100:

- Raising utilization from 0.90 to 0.95 adds 4 GB to the pool: about 31,000 more tokens.
- FP8 KV halves the bytes per token from 128 KB to 64 KB: the same 52 GB pool holds 812,000 tokens instead of 406,000.
- FP8 weights free 8 GB: about 62,000 more BF16 tokens.

On this GPU, FP8 KV is by far the largest gain. On a smaller GPU with a larger model, weight quantization can matter more, because the weights take most of the memory.

~~~dg-raw
%%widget kv Try the same model on different GPUs. On the 48 GB card, weights take a third of memory and the pool fills quickly.
model = Llama 3 8B
gpu = L40S (48 GB)
t = 12
b = 32
%%
~~~

## FP8 KV cache in practice

FP8 KV stores each key and value element in 8 bits, with a scale factor per tensor or per head. Most engines support it on recent NVIDIA and AMD GPUs.

- **Quality**: on most chat and reasoning tasks, the difference is within noise. Long-context retrieval tasks and some math tasks can be more sensitive. Always run your quality check.
- **Speed**: decode attention reads half as many bytes, so it gets faster at long context. Some kernels dequantize on the fly with little overhead.
- **Scales**: engines can use a default scale of 1.0 or calibrated scales. Calibrated scales are safer for models with large activation values.

## The procedure

1. Read the KV capacity in tokens from the startup log.
2. Apply one change.
3. Read the new capacity from the log, and check it matches your estimate.
4. Rerun the expected and stress load points.
5. If you changed precision, rerun the quality check.
6. Keep the change only if the binding metric improved and quality held.

<div class="ex">
<div class="lab">Worked example</div>

**an 8B model on a 48 GB GPU**

Baseline: BF16 weights (16 GB) and BF16 KV. The log reports a pool of about 22 GB, or about 170,000 tokens.

At the stress load, KV usage hits 100% with 60 preemptions per minute, and p95 TTFT is 2.1 s.

With FP8 KV: about 340,000 tokens. Preemptions at the stress load drop to zero, and p95 TTFT falls to 1.2 s, a 43% improvement.

The quality check (500 task examples) moved from 81.4% to 81.1%, within run-to-run noise. The change is kept.

</div>

## Watch for side effects

- **Higher utilization and OOMs.** At 0.95, a long prefill or a burst of logprob requests can exceed the remaining headroom. Test the worst-case request after raising it.
- **More capacity, more batch, slower TPOT.** A larger pool lets the scheduler run more sequences, which can raise TPOT. If TPOT is close to its SLO, also lower max_num_seqs.
- **Shared GPUs.** If another process uses the same GPU, such as a monitoring agent or a second model, raising utilization can starve it.

## When to stop

Once KV usage at the stress load stays below about 85% and preemptions are zero, memory is no longer the constraint. Further capacity will not help until something else changes. Move to the next binding constraint from your baseline.

## Deep dive: what each knob buys, in tokens

Every capacity knob changes one term of the memory equation from 01.01. For an 8B model on a 48 GB L40S, starting from 191,000 tokens:

⟦M585⟧

~~~dg-raw
%%bars KV pool on a 48 GB GPU, Llama 3 8B, as knobs are applied (thousands of tokens).
unit = k tokens
Baseline (η 0.90, BF16 weights and KV) | 191 | c4
η 0.95 | 209 | c3
FP8 weights | 253 | c2
FP8 KV | 383 | c1
FP8 weights and FP8 KV | 506 | wine
%%
~~~

FP8 KV doubles the pool because it halves ⟦m586⟧. FP8 weights add a fixed 8 GB, worth 61,000 BF16 tokens. On a small GPU where weights take a third of memory, both matter. On an 80 GB GPU, FP8 weights add the same 8 GB, which is a much smaller share of a 52 GB pool.

## Deep dive: when more KV stops helping

Extra pool raises the KV ceiling ⟦m587⟧. It only adds throughput while ⟦m588⟧ is the smallest boundary (05.04). Past the point where ⟦m589⟧, the extra memory sits idle at the SLO, and is better spent on longer contexts or a larger prefix cache.

<div class="take">
<div class="lab">Key takeaway</div>

KV capacity is bought with memory fraction, precision or length limits. Estimate each knob's gain first, change one at a time, check quality after precision changes, and stop once memory no longer binds.

</div>
