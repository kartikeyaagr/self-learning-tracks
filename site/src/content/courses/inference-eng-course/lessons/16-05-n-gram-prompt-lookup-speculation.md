---
id: "16.05"
course: "inference-eng-course"
module: "16"
slug: "n-gram-prompt-lookup-speculation"
title: "N-Gram & Prompt-Lookup Speculation"
keyIdeas: ["N-gram speculation needs no draft model: it copies from existing text.", "Match the last n tokens against the prompt (or history); propose what followed.", "Nearly free to run; no extra GPU memory.", "Great when outputs repeat inputs: code edits, extraction, RAG answers, summaries.", "Weak for creative or novel text."]
question: "Why does prompt-lookup (n-gram) speculation work well for code editing and summarization, and poorly for open-ended chat?"
answer: "It drafts by finding the last few generated tokens elsewhere in the prompt and copying what followed. Code editing and summarization often reproduce long spans of the input, so drafts are frequently right. Open-ended chat produces new text not present in the prompt, so matches are rare and drafts are mostly rejected."
sources: [{"title": "Prompt Lookup Decoding", "url": "https://github.com/apoorvumang/prompt-lookup-decoding", "note": "The original prompt lookup method."}, {"title": "vLLM: Speculative decoding", "url": "https://docs.vllm.ai/en/latest/features/spec_decode.html", "note": "N-gram speculator configuration."}]
tags: ["speculative"]
---

## Drafting by copying

Many outputs repeat text that already appears in the prompt. A code assistant editing a file copies most of the file back. A RAG answer quotes the retrieved documents. A rewrite keeps much of the original. For these tasks, the best guess for the next few tokens is often "whatever followed the same words last time".

**N-gram speculation**, also called **prompt lookup decoding**, uses exactly that guess, with no draft model at all:

```
suffix = generated[-n:]                                   # the last n tokens
pos = find_last_occurrence(prompt + generated[:-n], suffix)
if pos is not None:
    draft = (prompt + generated)[pos + n : pos + n + k]   # what followed last time
else:
    draft = []                                            # no guess this step
```

The target model verifies the draft exactly as in any speculative scheme. If there is no match, the step proceeds as normal decoding.

## Why it is attractive

- **Drafting is free.** A string search on the CPU or GPU, with no neural network.
- **No extra memory.** No draft weights and no draft KV cache.
- **No training.** Works with any model.
- **No tokenizer constraints.** The draft comes from the target's own tokens.

The only cost is verification compute for rejected guesses, which matters only at high load.

## Where it shines

| Task | Why it works |
| --- | --- |
| Code editing | the output copies large parts of the input file |
| Extraction and RAG | answers quote documents verbatim |
| Summaries with quotations | repeated phrases from the source |
| Rewriting or reformatting structured text | most tokens carry over |
| Multi-turn chat that repeats earlier content | reuse of earlier answers |

<div class="ex">
<div class="lab">Worked example</div>

**editing a file**

A user asks to rename a variable in a 400-line file, and the model outputs the whole file again with the change.

With n = 3 and k = 10, most of the output matches the input exactly. Measured mean accepted length: 7.8 tokens per round.

At batch 1 on an 8B model, TPOT falls from 6.0 ms to about 1.0 ms per token, a 6 times speedup on this request, at almost no cost.

On an open-ended chat answer, the same setting accepted 1.1 tokens per round on average, about the same speed as normal decoding.

</div>

## Where it does not help

Open-ended chat, creative writing, reasoning traces and translation into another language rarely repeat the prompt's token sequences. Matches are rare, and when they occur, they are often wrong. Because drafting is free, the loss at low load is small. At high load, the wasted verification compute can cost some throughput, so combine it with the load-aware policies from lesson 16.04.

## Tuning

- **n, the match length.** Longer matches are more reliable but occur less often. Values of 2 to 4 are typical. Some implementations try a longer n first and fall back to shorter ones.
- **k, the draft length.** Long drafts pay off for highly repetitive outputs, where a match often continues for many tokens. 5 to 10 is common.
- **Search scope.** Searching the prompt only, or the prompt plus the generated text. Including generated text helps with outputs that repeat themselves, such as lists.

## Availability

vLLM and SGLang support n-gram speculation through configuration flags, and Hugging Face Transformers supports prompt lookup in `generate`. Because it needs no extra model, it is the easiest form of speculation to try: enable it for endpoints whose outputs reuse their inputs and measure.

## Deep dive: steps saved by copying

When a share of the output repeats spans already in the context (code edits, extraction, rewriting), prompt lookup proposes the continuation of the matching span. If ⟦m850⟧ output tokens are copied in spans long enough to fill whole drafts of length ⟦m851⟧, and ⟦m852⟧ are new:

⟦M849⟧

<div class="ex">
<div class="lab">Worked example</div>

**a 1,000-token file edit**

800 tokens are unchanged lines copied from the original, 200 are new. With ⟦m853⟧:

Steps ⟦m854⟧, against 1,000 without speculation. Speed-up about 3.7 times at batch 1, with no draft model at all.

On free-form chat with the same setup, copied spans are rare. Most drafts are rejected at the first token, and the gain is close to zero.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

n-gram speculation drafts for free by copying what followed the same words earlier in the context. Turn it on where outputs reuse inputs, such as code editing and extraction, since it costs almost nothing to try.

</div>
