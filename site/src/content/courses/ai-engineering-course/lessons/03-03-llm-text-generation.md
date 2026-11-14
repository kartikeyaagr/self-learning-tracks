---
id: "03.03"
course: "ai-engineering-course"
module: "03"
slug: "llm-text-generation"
title: "LLM text generation"
keyIdeas: ["The model outputs logits; softmax turns them into probabilities over the vocabulary.", "Greedy decoding always picks the most likely token: deterministic, but tends to repeat itself in long text.", "Beam search keeps several candidate sequences; good for translation, poor for open-ended text.", "Temperature divides logits before softmax: below 1 is sharper and safer, above 1 is flatter and more varied.", "Top-k keeps the k most likely tokens; top-p (nucleus) keeps the smallest set with total probability p.", "For structured output use low temperature plus constrained decoding (JSON schema / grammar); for creative text use temperature 0.8-1.0 with top-p around 0.9."]
question: "You need reliable JSON extraction for one feature and varied marketing copy for another. What decoding settings would you use for each and why?"
answer: "JSON extraction: temperature 0 or near it (greedy-like, repeatable), plus schema/grammar-constrained decoding so every token keeps the output valid, and enough max_tokens to avoid truncation. Marketing copy: temperature ~0.8-1.0 with top-p ~0.9 (optionally a small repetition penalty) to sample diverse but coherent text, possibly generating several candidates. Greedy/beam search makes copy bland and repetitive; high temperature makes JSON unreliable."
sources: [{"title": "Hugging Face: How to generate text", "url": "https://huggingface.co/blog/how-to-generate", "note": "Greedy, beam search, top-k and top-p sampling with examples."}, {"title": "Holtzman et al.: The Curious Case of Neural Text Degeneration (2019)", "url": "https://arxiv.org/abs/1904.09751", "note": "The nucleus (top-p) sampling paper."}, {"title": "Hugging Face: Generation strategies", "url": "https://huggingface.co/docs/transformers/generation_strategies", "note": "Reference for every decoding option in transformers."}]
---

## From logits to a token

Each forward pass ends with a vector of logits, one per vocabulary entry (often 100k+). Softmax converts them to probabilities. A **decoding strategy** decides which token to emit. This choice changes output quality a lot, with no change to the model.

## Deterministic strategies

- **Greedy**: take the argmax every step. Simple and repeatable, but Hugging Face's generation guide shows it falls into loops ("I'm not sure. I'm not sure.") on long open-ended text.
- **Beam search**: keep the b highest-probability partial sequences and extend all of them. Works for tasks with one right answer (translation), but Holtzman et al. showed that maximising probability produces bland, repetitive text, because human text is not the most probable text.

## Sampling strategies

| Knob | What it does | Typical values |
| --- | --- | --- |
| Temperature T | Divide logits by T before softmax. T < 1 sharpens, T > 1 flattens | 0 to 1.2 |
| Top-k | Sample only from the k most likely tokens | 40-100 |
| Top-p (nucleus) | Sample from the smallest set whose probabilities add to p | 0.9-0.95 |
| Min-p | Drop tokens below a fraction of the top token's probability | 0.05-0.1 |
| Repetition / frequency penalty | Lower logits of tokens already used | Small values |

~~~dg-flow The usual sampling pipeline, applied every step.
Logits -(/ temperature)-> Scaled logits -(top-k / top-p filter)-> *Truncated distribution -(sample)-> Next token
~~~

Top-p adapts to the situation: when the model is confident, the nucleus holds a few tokens; when many continuations are plausible, it holds many. That is why Holtzman et al. proposed it over a fixed top-k.

## Stopping

Generation stops at the end-of-sequence token, a stop sequence you provide, or max_tokens. Hitting max_tokens mid-sentence is a common cause of truncated JSON.

## Structured output

For machine-read output, sampling settings are not enough. **Constrained decoding** masks out any token that would break a JSON schema or grammar, so output is always valid. Most APIs and servers (vLLM, llama.cpp, provider "structured output" modes) support it.

<div class="ex">
<div class="lab">Worked example</div>

**two features, two settings**

JSON extraction: temperature 0 (or close), schema-constrained decoding, generous max_tokens. You want the single most likely, valid answer every time.

Marketing copy: temperature around 0.9, top-p 0.9-0.95, maybe a small frequency penalty. You want variety without nonsense, and you may sample several drafts.

</div>

## Shape the distribution

Every decoding setting reshapes the same next-token distribution before one token is drawn. Change temperature, top-k, top-p and min-p below and watch which tokens stay eligible and how probability moves between them.

~~~dg-raw
%%widget sampling Temperature, top-k, top-p and min-p applied to one next-token distribution.
~~~

What each knob does, mechanically:

- **Temperature T** divides the logits before softmax. T below 1 sharpens the distribution (the top token gets more), T above 1 flattens it, T near 0 is greedy decoding.
- **Top-k** keeps the k most likely tokens and renormalizes. Simple, but the right k differs by context: after "The capital of France is" only one token makes sense, after "Once upon a" dozens do.
- **Top-p (nucleus)** keeps the smallest set of tokens whose probabilities add up to p, so the set shrinks when the model is confident and grows when it isn't. This is why it replaced top-k as the usual default.
- **Min-p** keeps tokens with at least min_p times the top token's probability. It adapts in the same direction as top-p and handles high temperatures more gracefully.

## Settings by task

| Task | Typical settings | Why |
| --- | --- | --- |
| Extraction, classification, code fixes | T = 0 to 0.2 | You want the single most likely answer |
| General chat | T = 0.7, top-p = 0.9 to 0.95 | Natural variety without nonsense |
| Brainstorming, fiction | T = 0.9 to 1.1, top-p = 0.95, or min-p = 0.05 | More diverse options |
| Self-consistency voting (08.02) | T = 0.7 to 1.0 | You need varied reasoning paths |

Temperature 0 is not fully deterministic on most hosted APIs: batching and floating-point order on GPUs can change near-ties between runs. If you need reproducibility, cache outputs.

## Structured output in more detail

Constrained decoding enforces a format by masking tokens. At each step, a grammar (often compiled from a JSON schema) tells the server which tokens could legally come next, and every other token's probability is set to zero before sampling. The model can't produce invalid JSON, but it can still produce valid JSON with wrong values, so validate the content too.

<div class="take">
<div class="lab">Key takeaway</div>

decoding is a product decision. Use low temperature and constrained decoding where correctness matters, and temperature plus top-p sampling where variety matters.

</div>
