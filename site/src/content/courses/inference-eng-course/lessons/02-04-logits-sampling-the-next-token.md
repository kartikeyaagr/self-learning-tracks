---
id: "02.04"
course: "inference-eng-course"
module: "02"
slug: "logits-sampling-the-next-token"
title: "Logits, Sampling & the Next Token"
keyIdeas: ["The LM head outputs one logit per vocabulary entry for the last position.", "Logit processors run first: penalties, bans, grammar masks; then temperature, top-k, top-p, then sampling.", "Greedy (temperature 0) is deterministic; sampling needs a seed for reproducibility.", "Sampling over a 100k+ vocabulary for many sequences per step is not free. Engines use fused sampling kernels.", "Stop conditions: end-of-sequence token, stop strings, max tokens."]
question: "Explain what happens in order when you set temperature 0.7, top-p 0.9 and a repetition penalty, and which step can be expensive at large batch sizes."
answer: "The engine takes the logits for the last position, applies the repetition penalty to tokens already seen, divides by temperature 0.7, applies softmax, sorts tokens by probability and keeps the smallest set summing to 0.9, renormalizes and samples. Sorting the full vocabulary for top-p across many sequences can be noticeably expensive, so engines use specialized sampling kernels."
sources: [{"title": "The Curious Case of Neural Text Degeneration", "url": "https://arxiv.org/abs/1904.09751", "note": "Introduces nucleus (top-p) sampling."}, {"title": "Hugging Face: Text generation strategies", "url": "https://huggingface.co/docs/transformers/main/en/generation_strategies", "note": "All common decoding strategies."}]
tags: ["sampling"]
---

## From the last hidden state to a token

After the last layer and final norm, the model multiplies the hidden state by the LM head matrix:

```
logits = h_last @ W_vocab        # shape [vocab_size]
```

Each logit is an unnormalized score for one vocabulary entry. Softmax turns the scores into probabilities. During decode only the last position matters. During prefill, engines compute logits only for the final prompt token, which saves a matmul of [prompt_len, 4096] × [4096, 128256] that would otherwise be thrown away.

## The sampling pipeline

Turning logits into one token id takes several steps, applied in a fixed order:

~~~dg-flow The usual order of operations.
Logits -> Penalties and masks -> Temperature -> Top-k and top-p -> *Sample -> Token id
~~~

- **Penalties.** A repetition penalty divides positive logits of tokens already in the text by a factor such as 1.1, and multiplies negative ones. Frequency and presence penalties subtract an amount based on how often a token has appeared. They reduce loops but can hurt tasks that need repeated words, such as code.
- **Masks.** Banned tokens, and grammar constraints for structured output, set disallowed logits to minus infinity. A JSON grammar, for example, forbids any token that cannot continue valid JSON.
- **Temperature.** Divide the logits by T. T below 1 sharpens the distribution and makes the top token more likely. T above 1 flattens it. T = 0 is treated as greedy: always pick the top token.
- **Top-k.** Keep only the k highest-scoring tokens.
- **Top-p (nucleus).** Sort by probability and keep the smallest set whose probabilities add up to at least p, such as 0.9. When the model is confident, this keeps few tokens. When it is unsure, it keeps many.
- **Min-p.** Keep tokens whose probability is at least min_p times the top token's probability. Some find it more stable than top-p at high temperature.
- **Sample.** Draw one token from what remains, after renormalizing.

~~~dg-raw
%%widget sample Lower the temperature and the top token takes over. Tighten top-p and the long tail disappears. Draw samples to see the effect.
T = 0.8
k = 6
p = 0.9
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**temperature on three tokens**

Suppose the logits for "cat", "dog" and "car" are 2.0, 1.0 and 0.0.

At T = 1, softmax gives about 0.67, 0.24 and 0.09.

At T = 0.5 the logits become 4, 2 and 0, giving about 0.87, 0.12 and 0.02. The top token dominates.

At T = 2 the logits become 1, 0.5 and 0, giving about 0.51, 0.31 and 0.19. The choice is much more random.

With top-p = 0.9 at T = 1, the kept set is "cat" and "dog" (0.67 + 0.24 = 0.91), and "car" can never be chosen.

</div>

## Why sampling affects serving cost

Sampling settings change how long outputs are. A high temperature can produce rambling answers that never hit the stop token. A strong repetition penalty can prevent the model from closing a code block. Since decode cost is proportional to output length, these settings change your bill. When you benchmark, fix the sampling settings or use `ignore_eos` with a fixed output length.

## Determinism

Greedy decoding is deterministic in theory. In practice, floating-point sums depend on the order of operations, and on a batching server that order can depend on which other requests share the batch. Two logits that are nearly tied can swap places, and the outputs diverge from that token on. If you need repeatable outputs:

- Use a fixed seed per request for sampling.
- Expect exact bit-for-bit repeatability only with batch-invariant kernels, which some engines offer at a speed cost.
- For tests, compare outputs by quality metrics or by logit closeness, not by exact string equality.

## Sampling at scale

With a 128k vocabulary and 256 sequences in a step, the sampler handles 32.8 million logits. A naive top-p implementation sorts each row, which is slow on a GPU. Engines use faster methods:

- Apply top-k first to shrink the candidate set before any sort.
- Use rejection-based top-p sampling, which draws candidates and accepts or rejects them without a full sort. FlashInfer provides such kernels.
- Fuse temperature, softmax and sampling into one kernel.

With these, sampling stays at a few percent of step time even at large batch.

## Stopping

Generation ends when any of these happens:

- The model emits an end token, such as EOS or an end-of-turn marker.
- The text contains a user-supplied stop string.
- The output reaches `max_tokens`, or the sequence reaches the model's context limit.

Stop strings need care. "END" might arrive as "EN" in one token and "D" in the next. The engine must detokenize as it goes and hold back a few characters that could be the start of a stop string, so it does not stream text the user should never see.

## Deep dive: the cost of the LM head and sampling

The LM head multiplies the final hidden state by a ⟦m261⟧ matrix: ⟦m262⟧ for Llama 3 8B, 1.05 GB in BF16. At batch 1 that is about 6.5% of all weight bytes read per step, for a single matmul.

Sampling then works on ⟦m263⟧ logits per sequence. A full sort is ⟦m264⟧; good implementations avoid it:

⟦M259⟧

⟦M260⟧

<div class="ex">
<div class="lab">Worked example</div>

**does sampling show up in a profile?**

Batch 256, vocabulary 128,256: 32.8 million logits per step, 131 MB in FP32.

Softmax, top-p and the random draw read that a few times: about 0.5 GB of traffic, about 0.2 ms on an H100. Step time at that batch is perhaps 25 ms, so sampling is about 1%.

A naive per-sequence Python loop over sampling, by contrast, can take 10 ms or more at batch 256. That is why engines use batched sampling kernels.

</div>

## Deep dive: greedy versus sampled outputs, numerically

Greedy decoding takes ⟦m265⟧. Two tokens with logits that differ by less than BF16 rounding (about 0.4% of their size) can swap order between runs with different batch sizes, because matmuls split work differently. That is the main reason "temperature 0" output is not perfectly deterministic in batched engines.

~~~dg-raw
%%chart Logit gap between the top two tokens across 1,000 decode steps of a typical response (share of steps).
x = 0.01, 0.03, 0.1, 0.3, 1, 3
logx = 1
xlab = Gap between top two logits (less than)
ylab = Share of steps (%)
Steps with a gap below this: 0.4, 1.2, 4, 11, 30, 68
%%
~~~

A gap below about 0.03 happens on roughly 1 step in 100 in a typical response (an illustrative figure). Over a 500-token answer, a few such near-ties are likely, so outputs can diverge after a few hundred tokens even at temperature 0.

<div class="take">
<div class="lab">Key takeaway</div>

logits pass through penalties, masks, temperature and top-k or top-p before one token is drawn. Sampling is cheap per sequence but needs care at large batch, and its settings change output length and therefore cost.

</div>
