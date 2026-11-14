---
id: "01.03"
course: "inference-eng-course"
module: "01"
slug: "generate-stream-your-first-tokens"
title: "Generate & Stream Your First Tokens"
keyIdeas: ["model.generate runs the autoregressive loop: forward pass, pick next token, append, repeat.", "max_new_tokens bounds output length; stop tokens end generation early.", "Streaming sends tokens as they are produced. Users see the first token after TTFT instead of waiting for the end.", "Time the first token and the rest separately: they are different phases with different costs.", "Detokenizing token by token needs care: some tokens are partial characters."]
question: "Explain why streaming improves perceived latency without changing total generation time."
answer: "Without streaming the user waits for the whole response. With streaming, tokens are sent as soon as each is generated, so the user starts reading after the first token (TTFT) instead of after the last. Total end-to-end time is the same or slightly longer, but the wait the user feels is much shorter."
sources: [{"title": "Hugging Face: Generation with LLMs", "url": "https://huggingface.co/docs/transformers/main/en/llm_tutorial", "note": "generate() basics and pitfalls."}, {"title": "Hugging Face: Text generation strategies", "url": "https://huggingface.co/docs/transformers/main/en/generation_strategies", "note": "Greedy, sampling, beam search and streamers."}]
tags: ["generation"]
---

## Generation is a loop

A language model computes a probability for every possible next token. It does not produce a sentence. To get a sentence you run it in a loop: compute, pick a token, append it, compute again. This loop is called **autoregressive generation**, and nearly every performance problem in this course traces back to it.

~~~dg-flow The autoregressive loop. Press Play to step through one iteration.
Prompt ids -(forward)-> Logits -(sample)-> *Next token -(append)-> Longer sequence
~~~

Each iteration has three parts:

1. **Forward pass**: run the network on the sequence (or, with a cache, on just the newest token). The result is a vector of logits, one score per vocabulary entry, for the last position.
2. **Sampling**: turn the logits into one token id, using greedy choice or random sampling with temperature and top-p.
3. **Stopping check**: stop on an end-of-sequence token, a stop string or a length limit. Otherwise append the token and repeat.

## Using generate in Hugging Face

```
inputs = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=128,
                     do_sample=True, temperature=0.7, top_p=0.9)
new_tokens = out[0][inputs.input_ids.shape[1]:]
print(tok.decode(new_tokens, skip_special_tokens=True))
```

Three settings matter on day one:

- `max_new_tokens` bounds the output. Always set it. Without it a model that never emits its stop token will run until the context is full.
- `do_sample=False` gives greedy decoding, which is good for tests because it is mostly repeatable.
- `eos_token_id` should include every end token the model uses.

## Two phases with different costs

The first forward pass processes the whole prompt at once. This is **prefill**. Every later step processes one new token. This is **decode**. They behave very differently:

| Phase | Tokens per pass | Limited by | What the user feels |
| --- | --- | --- | --- |
| Prefill | the whole prompt | compute (big matmuls) | time to first token |
| Decode | 1 per sequence | memory bandwidth | speed of streaming |

That is why you always measure two latency numbers instead of one.

- **Time to first token (TTFT)**: from sending the request to receiving the first token. It includes queueing, tokenizing and prefill.
- **Time per output token (TPOT)**: the average gap between later tokens. Some tools call it inter-token latency (ITL).

End-to-end latency is approximately TTFT + (output tokens − 1) × TPOT.

<div class="ex">
<div class="lab">Worked example</div>

**what a user feels**

A request has a 1,000-token prompt and a 300-token answer. TTFT is 120 ms and TPOT is 25 ms.

End-to-end time: 0.12 + 299 × 0.025 ≈ 7.6 s.

Without streaming the user stares at a spinner for 7.6 s. With streaming they start reading after 0.12 s, and the text arrives at 40 tokens per second, faster than anyone reads.

</div>

## Streaming

Streaming sends each token to the client as soon as it is generated. It does not make generation faster, but it changes the wait the user feels from end-to-end latency to TTFT. In Hugging Face, a streamer object receives text pieces from a background thread:

```
from threading import Thread
from transformers import TextIteratorStreamer
streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
Thread(target=model.generate,
       kwargs=dict(**inputs, max_new_tokens=128, streamer=streamer)).start()
for piece in streamer:
    print(piece, end="", flush=True)
```

## The detokenization trap

You cannot simply decode each token on its own and print it. Two things go wrong:

- **Partial characters.** With byte-level BPE, one emoji or Chinese character can span several tokens. Decoding the first one alone gives an invalid byte sequence.
- **Spacing.** Some tokenizers mark word boundaries on the following token, so decoding token by token can drop or add spaces.

Streamers handle this by decoding a growing window and emitting only the new, valid text. Production engines do the same with incremental detokenizers. If you write your own, test with emoji and non-English text.

## Measuring your first timings

Wrap your timing in `torch.cuda.synchronize()` calls. GPU work is asynchronous, so without a sync your timer measures how fast Python queued the work, not how fast the GPU finished it.

On one H100 with an 8B model in plain Hugging Face you might see a TTFT of 50 to 100 ms for a short prompt and 20 to 30 ms per output token. A dedicated engine reaches about 6 ms per token at batch 1 on the same hardware. The gap comes from launch overhead, unfused kernels and a slow cache layout, all of which later modules remove.

## Deep dive: end-to-end time as a formula

For a prompt of ⟦m195⟧ tokens and ⟦m196⟧ output tokens:

⟦M194⟧

where ⟦m197⟧ is fixed overhead, ⟦m198⟧ and ⟦m199⟧ are achieved compute and bandwidth, and ⟦m200⟧ is the KV read per step. For short contexts the KV term is small, so TPOT is close to weight bytes divided by bandwidth.

<div class="ex">
<div class="lab">Worked example</div>

**predicting a run before you measure**

Llama 3 8B, BF16, one H100, batch 1, prompt 500 tokens, 300 output tokens.

TTFT: ⟦m201⟧ FLOPs. At an effective 400 TFLOP/s that is 20 ms, plus about 15 ms overhead: about 35 ms.

TPOT: 16 GB at an effective 2.8 TB/s is 5.7 ms. Plain Hugging Face generate adds Python overhead, so expect 20 to 30 ms. An optimized engine gets close to 6 to 7 ms.

E2E: about ⟦m202⟧ s with plain generate, about 2 s with an engine.

</div>

~~~dg-raw
%%chart Model: E2E time against output length for two TPOT values (prompt fixed at 500 tokens).
x = 0, 100, 200, 300, 400, 500
xlab = Output tokens
ylab = Seconds
Plain generate (25 ms TPOT): 0.04, 2.5, 5.0, 7.5, 10.0, 12.5
Optimized engine (6.5 ms TPOT): 0.04, 0.68, 1.33, 1.98, 2.63, 3.28
%%
~~~

## Deep dive: what streaming changes, and what it does not

Streaming does not make generation faster. It changes what the user waits for: the first visible text appears after TTFT instead of after E2E. Reading speed for most people is about 4 to 6 words per second, roughly 5 to 8 tokens per second. Any TPOT under about 125 ms keeps ahead of a reader. That is why chat products optimize TTFT hard and are relatively tolerant of TPOT, while agents, which read the whole answer before acting, care about E2E.

<div class="take">
<div class="lab">Key takeaway</div>

generation is a loop of forward passes with two phases, prefill and decode. Stream tokens, detokenize carefully, and always measure TTFT and TPOT separately.

</div>
