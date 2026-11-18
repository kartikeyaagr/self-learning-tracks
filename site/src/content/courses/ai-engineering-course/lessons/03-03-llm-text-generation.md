---
id: "03.03"
course: "ai-engineering-course"
module: "03"
slug: "llm-text-generation"
title: "LLM text generation"
keyIdeas: ["The model outputs logits; a fixed pipeline of penalties, temperature, top-k, top-p or min-p and a final draw turns them into one token, and the order of those steps changes the result.", "Greedy and beam search maximise probability and suit tasks with one right answer; sampling suits open-ended text.", "Logprobs expose the model's confidence per token; logit bias and stop sequences edit the distribution and the stopping rule.", "Temperature 0 and a seed are not a determinism guarantee on hosted APIs: batching changes floating-point results.", "Reasoning models often fix or ignore temperature and top-p; control them with effort or thinking budgets instead."]
question: "You need reliable JSON extraction for one feature and varied marketing copy for another. What decoding settings would you use for each and why?"
answer: "JSON extraction: temperature 0 or near it, so the most likely token wins at each step, plus schema-constrained decoding so every emitted token keeps the output valid, and a max_tokens limit high enough that the object is never truncated (check the finish reason). Validate the values anyway, because constrained output can be valid but wrong. Marketing copy: temperature around 0.8 to 1.0 with top-p around 0.9 or min-p around 0.05, optionally a small frequency penalty, and generate several candidates to choose from. Greedy or beam search makes copy bland and repetitive; high temperature makes JSON unreliable. If either feature runs on a reasoning model, check which sampling parameters it accepts, since many fix temperature and top-p and expose an effort setting instead."
rubric: ["Low or zero temperature for extraction, with the reason (most likely token, repeatability)", "Schema-constrained decoding or a forced tool call for valid JSON, plus validating values", "Enough max_tokens and a finish-reason check to avoid truncated JSON", "Temperature near 1 with top-p or min-p for copy, possibly several samples", "Explains why greedy/beam is bland for copy and high temperature is risky for JSON", "Notes that reasoning models may ignore or reject sampling parameters"]
sources: [{"title": "Holtzman et al.: The Curious Case of Neural Text Degeneration (2019)", "url": "https://arxiv.org/abs/1904.09751", "note": "The nucleus (top-p) sampling paper; why maximising probability makes bland text."}, {"title": "Nguyen et al.: Turning Up the Heat, Min-p Sampling (2024)", "url": "https://arxiv.org/abs/2407.01082", "note": "Min-p truncation and why it stays coherent at higher temperatures."}, {"title": "Hugging Face: How to generate text", "url": "https://huggingface.co/blog/how-to-generate", "note": "Greedy, beam search, top-k and top-p sampling with examples."}, {"title": "Thinking Machines: Defeating Nondeterminism in LLM Inference (2025)", "url": "https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/", "note": "Why temperature 0 still varies: kernels that are not batch-invariant."}]
practice: ["text-generation-sampling", "temperature-top-k-and-top-p-sampling", "beam-search-decoding", "top-k-and-top-p-sampling", "gpt2-greedy-generate"]
---

## From logits to a token

Each forward pass ends with a vector of logits $z$, one per vocabulary entry (often 100k+). Softmax with temperature $T$ turns them into probabilities:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

A **decoding strategy** decides which token to emit from that distribution. It changes output quality a lot with no change to the model.

## Deterministic strategies

- **Greedy**: take the argmax every step. Simple and repeatable, but Hugging Face's generation guide shows it falls into loops ("I'm not sure. I'm not sure.") on long open-ended text.
- **Beam search**: keep the $b$ highest-probability partial sequences and extend all of them. Works for tasks with one right answer (translation, short extraction), but Holtzman et al. showed that maximising probability produces bland, repetitive text, because human text is not the most probable text.

## The sampling pipeline, in order

Sampling servers apply the same steps every token, and the order matters because each step changes what the next one sees:

~~~dg-flow The usual order in vLLM and Hugging Face transformers; llama.cpp lets you reorder its sampler chain.
Logits -(logit bias, repetition / frequency / presence penalties)-> Adjusted logits -(/ temperature)-> Scaled logits -(top-k, top-p, min-p cut-offs)-> *Truncated distribution -(renormalise, draw)-> Next token
~~~

| Step | What it does | Typical values |
| --- | --- | --- |
| Logit bias | Add a fixed amount to chosen token ids | −100 (ban) to +100 (force) |
| Penalties | Lower logits of tokens already generated: frequency scales with count, presence is a flat hit | 0 to 0.5 |
| Temperature $T$ | Divide logits by $T$: below 1 sharpens, above 1 flattens, near 0 is greedy | 0 to 1.2 |
| Top-k | Keep the $k$ most likely tokens | 20 to 100 |
| Top-p (nucleus) | Keep the smallest set whose probabilities add up to $p$ | 0.9 to 0.95 |
| Min-p | Keep tokens with at least min_p times the top token's probability | 0.05 to 0.1 |

Top-k is simple, but the right $k$ differs by context: after "The capital of France is" one token makes sense, after "Once upon a" dozens do. Top-p adapts, so the nucleus shrinks when the model is confident and grows when it isn't, which is why Holtzman et al. proposed it and it became the usual default. Min-p (Nguyen et al., 2024) adapts in the same direction and holds up better at high temperatures, because its cut-off scales with the top token.

~~~dg-raw
%%widget sampling Temperature, top-k, top-p and min-p applied to one next-token distribution.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why temperature comes before top-p**

Five candidate tokens have logits $[2.0, 1.0, 0.5, 0.0, -1.0]$.

At $T = 1$ softmax gives $[0.563, 0.207, 0.126, 0.076, 0.028]$. The running totals are 0.563, 0.770, 0.896, 0.972, so top-p 0.9 keeps **four** tokens (three only reach 0.896). Renormalised, the top token's share is 0.579.

At $T = 0.5$ the logits double to $[4, 2, 1, 0, -2]$ and softmax gives $[0.829, 0.112, 0.041, 0.015, 0.002]$. Now two tokens already pass 0.9 (0.941), so the nucleus holds **two**, and the top token gets 0.881.

Same top-p, half the temperature, half the candidates. Apply top-p first and temperature second and you would keep four tokens in both cases. Penalties come earlier still: if the second token had already appeared twice, a frequency penalty of 0.5 takes its logit from 1.0 to 0.0 before any of this.

</div>

## Logprobs

Many APIs can return the log probability of each generated token and its top alternatives (OpenAI's `logprobs` with `top_logprobs`, Gemini's `logprobs`, vLLM's `logprobs`; not every provider exposes them). Uses:

- **Confidence for classification.** Ask for a one-token label and read $\exp(\text{logprob})$ as a score you can threshold or calibrate on an eval set.
- **Spotting shaky spans.** A run of low-probability tokens inside a number or name flags a likely hallucination worth checking.
- **Scoring text.** Summed logprobs give a sequence's likelihood, the basis of perplexity.

Treat them as raw model confidence, not truth: post-trained models are often overconfident.

## Seeds and determinism

Some APIs take a `seed` so repeated requests draw the same random numbers; OpenAI documents this as best-effort and returns a `system_fingerprint` that changes when the backend does. Even temperature 0 varies on hosted APIs. Thinking Machines (2025) traced the main cause to kernels that are not batch-invariant: your request's numbers depend on how many other requests share the batch, so near-ties flip between runs. Self-hosted servers can be made reproducible at some speed cost; on an API, if you need the same answer twice, store it.

## Stopping

Generation stops at the end-of-sequence token, at a **stop sequence** you supply, or at max_tokens. Stop sequences are strings, matched even when they span several tokens, and the matched text is usually not returned. The response says which happened (a finish or stop reason such as `stop`, `end_turn`, `length` or `max_tokens`); always check it, because hitting the limit mid-sentence is the usual cause of truncated JSON.

## Logit bias

Logit bias adds a fixed value to specific token ids before sampling: −100 effectively bans a token, a large positive value nearly forces it. It is useful for restricting a classifier to its label tokens or banning a word. Two traps: the ids are tokenizer-specific, and one word has several tokens (" Yes", "Yes", " yes"), so a ban often leaks through a variant. Schema-constrained decoding ([Structured outputs](lesson:aieng:12.01)) is the general version of the same idea: a grammar masks every token that would break the format at each step.

## Reasoning models

Reasoning models change the knobs. OpenAI's o-series and GPT-5 reasoning models reject or ignore custom temperature, top-p and logit bias and expose a reasoning effort setting instead; Claude's extended thinking does not allow changing temperature or top-k; DeepSeek's reasoner API accepts sampling parameters but ignores them; Qwen3's thinking mode recommends temperature 0.6 with top-p 0.95 and warns against greedy decoding. Check the model's docs before porting settings from a chat model; [Using reasoning models in apps](lesson:aieng:08.06) covers budgets and effort.

## Settings by task

| Task | Typical settings | Why |
| --- | --- | --- |
| Extraction, classification, code fixes | $T$ = 0 to 0.2, constrained output | You want the single most likely answer |
| General chat | $T$ = 0.7, top-p = 0.9 to 0.95 | Natural variety without nonsense |
| Brainstorming, fiction | $T$ = 0.9 to 1.1, top-p = 0.95 or min-p = 0.05 | More diverse options |
| Self-consistency voting ([Chain of thought](lesson:aieng:08.02)) | $T$ = 0.7 to 1.0 | You need varied reasoning paths |

## Where this breaks

- Setting temperature and top-p both very low and wondering why outputs are stiff; tune one at a time.
- Assuming temperature 0 means reproducible tests; record outputs or run evals with repeats.
- Ignoring the finish reason, so a `length` stop ships half an answer.
- Banning one token id with logit bias and missing its other spellings.
- Copying sampling settings onto a reasoning model that silently ignores them.

<div class="take">
<div class="lab">Key takeaway</div>

Decoding is a product decision. Know the pipeline order, use low temperature and constrained decoding where correctness matters, temperature plus top-p or min-p where variety matters, check the finish reason, and do not expect temperature 0 or a seed to make a hosted model deterministic.

</div>
