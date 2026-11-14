---
id: "02.05"
course: "inference-eng-course"
module: "02"
slug: "build-a-tiny-autoregressive-generation-loop"
title: "Build a Tiny Autoregressive Generation Loop"
keyIdeas: ["Write the loop yourself: forward, take last logits, sample, append, check stop.", "Without a cache, every step re-runs the full sequence. Cost per step grows with length.", "Measure per-step latency and plot it against position: it rises steadily.", "This makes the case for the KV cache in Module 03.", "Keep the loop; you will extend it into a mini engine in Module 28."]
question: "Write pseudocode for a generation loop without a KV cache and state its cost per step as the sequence grows."
answer: "ids = prompt; for step in range(max_new): logits = model(ids)[-1]; tok = sample(logits); ids.append(tok); if tok == eos: break. Each step reprocesses the whole sequence, so step t costs work proportional to (prompt + t), and total cost grows quadratically with output length."
sources: [{"title": "nanoGPT", "url": "https://github.com/karpathy/nanoGPT", "note": "Minimal generate() implementation to compare against."}, {"title": "Hugging Face: Generation with LLMs", "url": "https://huggingface.co/docs/transformers/main/en/llm_tutorial", "note": "Reference for the full API."}]
tags: ["generation", "lab"]
---

## Goal

Write the generation loop yourself, without any cache, and measure how its step time grows. This makes the case for the KV cache with your own numbers.

## The loop

```
import time, torch

@torch.inference_mode()
def generate(model, tok, prompt, max_new=64, temp=0.8):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    times = []
    for _ in range(max_new):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits = model(ids, use_cache=False).logits[:, -1, :]
        probs = torch.softmax(logits / temp, dim=-1)
        nxt = torch.multinomial(probs, 1)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0]), times
```

Notes on the code:

- `use_cache=False` forces the model to recompute everything each step, which is what we want to measure.
- `torch.cuda.synchronize()` before each timer read makes sure the GPU has finished. Without it you measure Python, not the GPU.
- `.item()` also forces a sync. In a real engine you avoid calling it every step.

## Measure it

Run with prompts of 128, 512 and 2,048 tokens and generate 64 new tokens each. Plot step time against the current sequence length.

| Sequence length | Step time without cache (8B, H100, typical) | Relative |
| --- | --- | --- |
| 128 | about 25 ms | 1× |
| 512 | about 45 ms | about 2× |
| 2,048 | about 140 ms | about 6× |
| 8,192 | about 600 ms | about 25× |

The exact numbers depend on your GPU and library versions. The shape is what matters: step time climbs with length.

## Why it climbs

Step t runs the full network over all prompt + t tokens. The matmul work is proportional to the length, and attention work is proportional to its square. Summed over n new tokens, the total work grows like n² for the matmuls and n³ for attention, if you keep generating.

Almost all of that work is wasted. The keys and values for old tokens have not changed since they were first computed, as the causal mask guarantees. The model is recomputing them every step.

<div class="ex">
<div class="lab">Worked example</div>

**total work for one answer**

A 2,000-token prompt followed by 500 output tokens, with no cache.

The model processes 2,000 + 2,001 + ... + 2,499 tokens, which is about 1.12 million token passes, or roughly 18 PFLOP for an 8B model.

With a KV cache it processes 2,000 tokens once in prefill and 1 token per step after that: 2,500 token passes, about 40 TFLOP. That is about 450 times less work.

</div>

## Then compare with the cache

Run the same prompt with `model.generate(..., use_cache=True)` and time each step with a streamer or a logits processor that records timestamps. The step time stays nearly flat at around 20 to 30 ms, rising only slowly as the cache grows. The flat line is the KV cache at work. Module 03 builds that cache by hand.

## Things to check while you are here

- **Output equality.** With greedy decoding, the cached and uncached loops should produce the same tokens for the first few dozen steps. If they diverge immediately, you have a bug, such as wrong position ids.
- **Memory.** Uncached prefill at 8,192 tokens allocates large attention temporaries. Watch `torch.cuda.max_memory_allocated()` climb.
- **First step.** In both loops the first step is the same prefill. Only later steps differ.

## Write it up

Make a chart with two lines, uncached and cached step time against sequence length, and add one paragraph explaining the gap in terms of repeated work.

## Deep dive: the total work of a loop without a cache

Without a KV cache, step ⟦m268⟧ re-runs the whole sequence of length ⟦m269⟧. Linear-layer FLOPs over ⟦m270⟧ generated tokens are

⟦M266⟧

With a cache, the same loop does ⟦m271⟧ once and then ⟦m272⟧ per step:

⟦M267⟧

The ratio for ⟦m273⟧, ⟦m274⟧ is ⟦m275⟧. The uncached loop does about 375 times more linear-layer work.

~~~dg-raw
%%chart Cumulative linear-layer work (in units of 2P FLOPs) for a 500-token prompt.
x = 0, 100, 200, 300, 400, 500
xlab = Tokens generated
ylab = Work (thousands of 2P FLOPs)
No cache: 0, 55, 120, 195, 280, 375
With cache: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

a cache-free loop is correct but recomputes every past token on every step, so step time climbs with length. Watching that line climb is the clearest argument for the KV cache.

</div>
