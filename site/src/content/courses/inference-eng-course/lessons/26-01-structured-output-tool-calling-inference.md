---
id: "26.01"
course: "inference-eng-course"
module: "26"
slug: "structured-output-tool-calling-inference"
title: "Structured Output & Tool-Calling Inference"
keyIdeas: ["Structured output forces outputs to match a schema, regex or grammar.", "Implementation: token masks from a grammar state machine applied before sampling.", "Mask computation must be fast and overlapped with the GPU step.", "Tool calling relies on parsing model output into function calls, often with constrained decoding.", "Measure overhead: TPOT with and without constraints, and validity rate."]
question: "How does grammar-constrained decoding guarantee valid JSON, and what does it cost per step?"
answer: "A grammar (from a JSON schema or regex) is compiled into a state machine. At each step, the engine computes which vocabulary tokens are allowed from the current state and masks the rest to minus infinity before sampling, so only valid continuations can be chosen. The cost is computing that mask each step (precomputation and caching keep it small) and occasionally constraining the model away from its preferred tokens, which can affect quality."
sources: [{"title": "XGrammar", "url": "https://arxiv.org/abs/2411.15100", "note": "Efficient grammar-constrained decoding."}, {"title": "Outlines", "url": "https://github.com/dottxt-ai/outlines", "note": "Structured generation library."}, {"title": "vLLM: Structured outputs", "url": "https://docs.vllm.ai/en/latest/features/structured_outputs.html", "note": "Using guided decoding in vLLM."}]
tags: ["structured-output"]
---

## How constrained decoding works

At every step, the engine knows which tokens would keep the output valid under a grammar or JSON schema. All other tokens are masked out before sampling.

~~~dg-flow One constrained decoding step.
Grammar state -> *Allowed token mask -> Apply mask to logits -> Sample -> Advance grammar state
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one step inside a JSON object**

The schema says the object has a "city" string and an "age" integer. The output so far is {"city": "Lyon", "age":

The only valid continuations are whitespace or the start of an integer: tokens such as " 3", "4" or "12". Every token that starts with a letter, a quote or a brace is masked to −∞.

If the model's top choice was " \"", which would start a string, it can no longer be picked. The sampler chooses among the digits instead.

</div>

## Making it fast

The mask covers the whole vocabulary, often 128,000 tokens or more, and is needed at every step. A naive check of every token against the grammar would be far too slow.

- **Precompile** the grammar into a state machine before decoding starts.
- **Cache masks** for common states. Most tokens are either always allowed or never allowed in a given state, so only a small set needs checking at run time.
- **Overlap:** compute the next mask on the CPU while the GPU runs the forward pass.

Modern backends such as XGrammar keep the overhead to a small share of step time for typical JSON schemas.

## Tool calling

Models emit tool calls in a model-specific format, often JSON inside special tags. The server parses them into the API's tool_calls field. Constraining the output to the tool's schema prevents malformed calls, such as a missing required argument or a string where a number belongs.

## Risks

- **Overly strict grammars** can push the model into tokens it would not naturally choose, which can lower answer quality. Let the model reason in free text first, then constrain only the final answer.
- **Complex schemas** (deep recursion, very large enums) cost more per step and take longer to compile.
- **Prefix caching still works**, because only sampling changes, not the prompt.

| Metric | Unconstrained | Constrained |
| --- | --- | --- |
| JSON validity | 95 to 99% | 100% |
| TPOT overhead | none | small, with a good backend |
| First-request latency | none | grammar compile time, unless cached |

## Deep dive: what a grammar does to the distribution

Constrained decoding multiplies the model's distribution by a mask ⟦m1100⟧ of tokens the grammar allows in the current state, and renormalizes:

⟦M1098⟧

The output is always valid, but if the model put most of its probability on forbidden tokens, the renormalized choice can be a poor one: the denominator tells you how much the model "wanted" to leave the grammar. Logging it per step is a cheap way to find prompts that fight the schema.

The mask must be ready every step. Precompiling the grammar into a finite-state machine over tokens makes it a lookup:

⟦M1099⟧

Computing masks on the fly instead walks the grammar for each candidate token. For complex schemas that can take longer than a decode step, which is why mask computation is overlapped with the GPU forward pass.

<div class="take">
<div class="lab">Key takeaway</div>

constrained decoding masks invalid tokens at every step, which guarantees the format at a small cost when masks are precomputed and overlapped with the GPU.

</div>
