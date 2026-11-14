---
id: "03.01"
course: "inference-eng-course"
module: "03"
slug: "why-autoregressive-generation-recomputes-work"
title: "Why Autoregressive Generation Recomputes Work"
keyIdeas: ["Each decode step depends on all previous tokens, so generation is sequential.", "Without caching, every step recomputes K and V for the entire prefix.", "Past K and V never change (causal mask), so recomputing them is pure waste.", "Caching turns quadratic total work into linear work at the cost of memory.", "That memory, the KV cache, becomes the main constraint on concurrency."]
question: "Without a cache, how many token-positions of work does generating 100 tokens after a 900-token prompt take, compared with a cached approach?"
answer: "Without a cache, step t processes 900 + t positions, so total is about 100 x 900 + (0+...+99), roughly 95,000 token-positions. With a cache, prefill processes 900 positions once and each of the 100 decode steps processes one new position, about 1,000 total. Roughly 95 times less work, ignoring attention reads."
sources: [{"title": "kipply: Transformer Inference Arithmetic", "url": "https://kipp.ly/transformer-inference-arithmetic/", "note": "KV cache sizing and cost."}, {"title": "Hugging Face: KV cache strategies", "url": "https://huggingface.co/docs/transformers/main/en/kv_cache", "note": "How transformers manages caches."}]
tags: ["kv-cache"]
---

## One token at a time, by construction

A decoder-only language model defines the probability of a text as a chain: the probability of token 1, times the probability of token 2 given token 1, and so on. Generating text means sampling from that chain one link at a time. Token t + 1 depends on token t, so you cannot compute step 50 before step 49 has finished. No amount of hardware removes this dependency within a single sequence.

This has two consequences that shape everything else:

- **Latency scales with output length.** A 1,000-token answer needs 1,000 sequential forward passes. If each takes 10 ms, the answer takes 10 s, however many GPUs you own.
- **Each pass does little work per sequence.** One new token per step is a tiny amount of math for a network with billions of weights.

## Where the naive loop wastes work

The simplest loop feeds the whole sequence into the model at every step, as you did in lesson 02.05. At each layer it recomputes keys and values for every earlier token. But the causal mask guarantees that position i's key and value depend only on tokens 0 to i. Those tokens have not changed since the last step, so every recomputation produces exactly the same numbers again.

| Approach | Tokens processed at step t | Total for n new tokens after a P-token prompt |
| --- | --- | --- |
| No cache | P + t | about n × P + n² ÷ 2 |
| KV cache | 1 (plus reading the cache) | about P + n |

<div class="ex">
<div class="lab">Worked example</div>

**how much is wasted**

A 1,000-token prompt and a 1,000-token answer.

Without a cache: 1,000 + 1,001 + ... + 1,999 ≈ 1.5 million token passes through the network.

With a cache: 1,000 tokens once in prefill, then 1,000 single-token steps, which is 2,000 token passes.

The cache removes about 99.9% of the work. For an 8B model that is the difference between about 24 PFLOP and 32 TFLOP.

</div>

## The trade: compute for memory

Caching is simple in principle. At each layer, store the K and V vectors for every token once, and reuse them at every later step. The new token still needs a full pass, and its attention still reads all cached K and V, but nothing is recomputed.

~~~dg-flow Caching trades compute for memory.
Recompute K and V -(cache them)-> *Store K, V per token -> Memory grows with context × users
~~~

The price is memory, and it grows during the request:

- Every new token adds one K and one V vector per layer per KV head.
- A request's cache is not known in advance, because you do not know how long the answer will be.
- Every active request holds its own cache for as long as it runs.

For Llama 3 8B, each token costs 128 KB of cache. A server with 200 active conversations averaging 3,000 tokens holds 600,000 tokens, or about 77 GB of KV cache. That is more than the weights.

## Why this makes inference a memory problem

Without a cache, inference would be a compute problem with quadratic waste. With a cache, it becomes a memory problem in two senses:

- **Capacity.** The KV cache competes with the weights for GPU memory, and its size decides how many requests can run at once.
- **Bandwidth.** Each decode step reads every weight and every active cache entry from memory, and that read sets the step time.

Most of the techniques in this course attack one of those two. Paged attention and prefix caching manage capacity. Quantization, GQA and better kernels reduce the bytes read. Batching shares the weight read across many sequences. Speculative decoding produces more than one token per weight read.

## Other ways to cut the sequential cost

The cache removes wasted recomputation, but the chain of steps remains. Two families of techniques attack the chain itself:

- **Speculative decoding** (Module 16) guesses several tokens cheaply and verifies them in one pass of the big model.
- **Multi-token prediction** heads, trained with some models, propose several future tokens per step.

Both still respect the dependency. They just do more useful work per sequential step.

## Deep dive: why the cache is exact, not an approximation

In a causal model, the key and value of token ⟦m279⟧ at layer ⟦m280⟧ depend only on tokens ⟦m281⟧:

⟦M276⟧

Adding token ⟦m282⟧ cannot change anything at positions ⟦m283⟧, because the causal mask stops position ⟦m284⟧ from looking forward. So ⟦m285⟧ and ⟦m286⟧ computed once are exactly what a full recomputation would produce. Caching them removes work without changing the output. In exact arithmetic the outputs are identical; in floating point they can differ in the last bits, because the kernels group the sums differently.

## Deep dive: total work with and without the cache, including attention

For prompt length ⟦m287⟧ and ⟦m288⟧ generated tokens, attention FLOPs per layer for one new token at context ⟦m289⟧ are about ⟦m290⟧. Summing over the generation:

⟦M277⟧

⟦M278⟧

Uncached decode re-runs causal attention over the whole sequence each step, so each step costs about ⟦m291⟧ per layer. Uncached work grows with the cube of the length; cached work grows with the square.

~~~dg-raw
%%chart Attention work per layer over a whole generation (p = 1,000), in units of d·10⁹ FLOPs.
x = 100, 250, 500, 1000, 2000
logy = 1
xlab = Tokens generated
ylab = Work
No cache: 0.22, 0.64, 1.58, 4.67, 17.3
With cache: 0.00042, 0.0011, 0.0025, 0.006, 0.016
%%
~~~

The gap is about 500 times at 100 tokens and 1,000 times at 2,000, and it keeps widening, because the ratio of the two is about ⟦m292⟧ per step. The log scale hides how large that is. The linear layers show the same pattern: uncached decode repeats ⟦m293⟧ FLOPs every step instead of ⟦m294⟧.

<div class="take">
<div class="lab">Key takeaway</div>

generation is sequential, and the naive loop recomputes the unchanged past at every step. The KV cache removes that waste and turns inference into a memory capacity and bandwidth problem.

</div>
