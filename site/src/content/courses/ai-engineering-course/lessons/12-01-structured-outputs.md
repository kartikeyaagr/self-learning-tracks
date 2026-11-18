---
id: "12.01"
course: "ai-engineering-course"
module: "12"
slug: "structured-outputs"
title: "Structured outputs"
keyIdeas: ["Prompt-only JSON, JSON mode, schema-constrained decoding and a forced tool call give different guarantees: valid syntax is not the same as your schema, and your schema is not the same as correct values.", "Constrained decoding compiles the schema into a grammar and, at every step, masks every token that could not continue a valid output; the model's probabilities are renormalized over what is left.", "Schema design is prompt design: closed sets as enums with an escape value, every field required (nullable when optional), flat objects, and fields ordered so evidence is generated before the verdict.", "Keep a validate, repair, retry loop even with strict mode: truncation, refusals and constraints the grammar cannot express still produce bad objects.", "Strict formats can cost reasoning quality; let the model think first and constrain only the final answer, and measure the difference on your own eval."]
question: "You are building invoice extraction for an accounts-payable system: PDFs in, a JSON record out (supplier, invoice number, date, currency, line items, total). Describe how you would get reliable structured output, what the schema looks like, and what still has to be checked in code."
answer: "Use schema-constrained decoding (a strict response format or a forced tool call with a strict schema) so every reply parses and has the right keys and types. Design the schema flat and explicit: required fields throughout, nullable where the document may lack a value, currency as an ISO-code enum plus an \"other\" value, line items as an array of small objects, and a short evidence or notes field before the totals. Grammar-level guarantees stop at syntax and types, so validate in code: ranges and formats the grammar dropped, line items summing to the total within a rounding tolerance, dates being real, the supplier matching a known vendor. On failure, repair mechanical problems, otherwise retry with the specific errors, at most once or twice, then route to human review. Handle max_tokens truncation and refusals separately, log repair and retry rates, and check field-level accuracy on a labelled set, since a valid object can still hold wrong values."
rubric: ["Picks schema-constrained decoding or a strict forced tool call and says what it guarantees (syntax, keys, types) and what it does not (correct values).", "Designs the schema deliberately: required plus nullable, enums with an escape value, flat structure, sensible field order.", "Validates business rules in code (sums, ranges, formats, cross-field checks) because the grammar cannot express them.", "Has a bounded repair then retry loop with error feedback and a human-review fallback.", "Handles truncation and refusals, and measures field accuracy and retry rates rather than only validity."]
sources: [{"title": "Willard and Louf: Efficient Guided Generation for Large Language Models (2023)", "url": "https://arxiv.org/abs/2307.09702", "note": "Regex and grammar constraints as a finite-state machine indexed over the vocabulary; the basis of Outlines."}, {"title": "Tam et al.: Let Me Speak Freely? Format restrictions and LLM performance (2024)", "url": "https://arxiv.org/abs/2408.02442", "note": "Measures reasoning loss under strict JSON and XML formats."}, {"title": "OpenAI: Structured Outputs guide", "url": "https://platform.openai.com/docs/guides/structured-outputs", "note": "Strict JSON schema response format and strict function calling, supported schema subset, refusals."}, {"title": "Anthropic: Structured outputs", "url": "https://docs.claude.com/en/docs/build-with-claude/structured-outputs", "note": "JSON outputs and strict tool use, grammar compilation and caching, schema limitations, property ordering."}]
practice: ["repair-and-validate-json", "validate-tool-arguments", "record-schema-validation", "parse-jsonrpc-message"]
tools: ["JSON Schema", "Outlines", "XGrammar", "Pydantic"]
---

## Four ways to ask for JSON

Most LLM features end in code that reads the model's answer: a router reads a label, an extractor fills a database row, an agent reads a tool call. There are four ways to ask, and they give different guarantees.

| Approach | What you send | Guarantees | Still your problem |
| --- | --- | --- | --- |
| Prompt only | "Reply with JSON like {...}" | Nothing | Fences, preamble, trailing commas, wrong keys |
| JSON mode | A flag asking for any valid JSON | Parses (unless truncated) | Keys, types, enums |
| Schema-constrained output | A JSON Schema with a strict flag | Parses and matches the supported schema subset | Values, ranges, cross-field rules, truncation, refusals |
| Forced tool call | One tool, `tool_choice` set to it, strict schema | Same as above, delivered as tool arguments | Same as above |

Schema-constrained output and a forced tool call use the same machinery; pick by where you want the object to land. A response format is cleaner when the object is the answer. A tool call fits when the model might also reply in text or call other tools ([Tool calling](lesson:aieng:08.03) covers `tool_choice` and strict tool schemas). OpenAI, Anthropic and Gemini all offer schema-constrained output in their APIs; vLLM, SGLang and llama.cpp offer the same for open-weight models through grammar backends such as XGrammar and Outlines.

## How constrained decoding works

[Text generation](lesson:aieng:03.03) ends with a sampler choosing a token from a probability distribution. Constrained decoding adds a single step before sampling: remove every token that could not be part of a valid output.

~~~dg-flow Schema-constrained decoding, once per request and then every step.
JSON Schema -(compile once, cache)-> Grammar -(index over vocabulary)-> Token automaton -(current state)-> *Allowed-token mask -(zero the rest, renormalize)-> Sample -(advance state)-> Token automaton
~~~

The schema is compiled into a grammar. A flat object with string and enum fields is a regular language, so a finite-state machine is enough. Nested objects and arrays need a context-free grammar with a stack. The hard part is that the model emits tokens, not characters, and one token can span grammar pieces: `",` or `"}` is often a single token. So the engine asks, for each token in a vocabulary of 100k+ entries, "can every character of this token be consumed from the current grammar state?" Willard and Louf showed this can be precomputed as an index from automaton states to allowed tokens. XGrammar splits the vocabulary into tokens whose validity never depends on the stack (prechecked) and the few that do (checked at run time). How engines make this fast and overlap it with the GPU is covered in [Structured output and tool-calling inference](lesson:ieng:26.01). What matters for an app engineer is what the mask does to the model's choices.

With $m(t) \in \{0, 1\}$ the mask and $p(t)$ the model's probability for token $t$, the sampler draws from

$$
p'(t) = \frac{p(t)\, m(t)}{Z}, \qquad Z = \sum_u p(u)\, m(u)
$$

$Z$ is the share of probability the model put on valid continuations. When $Z$ is small, the model wanted to write something your schema forbids, and the grammar is overriding it.

<div class="ex">
<div class="lab">Worked example</div>

**An enum the data does not fit**

A support classifier's schema has `"category": {"enum": ["billing", "bug", "account"]}`. A customer writes "I want my money back for the annual plan". After `{"category": "` the model's next-token probabilities are (illustrative): `refund` 0.45, `billing` 0.30, `bug` 0.10, `account` 0.05, everything else 0.10.

The mask keeps three tokens, so $Z = 0.30 + 0.10 + 0.05 = 0.45$. Renormalized: billing $\tfrac{0.30}{0.45} = 0.667$, bug $0.222$, account $0.111$.

The output is valid every time, and one time in three it is wrong (bug or account) on a ticket the model understood perfectly. 55% of the model's probability was on a label the schema did not offer. Two fixes: add the missing label (or an `"other"` escape value) so the model has somewhere honest to go, and log $Z$, or the logprob of each constrained field where your API exposes logprobs. A field whose $Z$ is often below 0.5 is a schema that fights your data.

</div>

## Designing schemas that work

The schema is part of the prompt. The model reads it, and the grammar enforces it token by token, so its shape changes answers.

- **Enums for closed sets, with an escape.** Labels, statuses and currencies belong in enums. Add `"other"` or `"unknown"` so the grammar never forces a guess.
- **Every field required; optional means nullable.** Strict modes generally require every property in `required` and `additionalProperties: false`. Express "may be absent" as `{"type": ["string", "null"]}`. The model then has to decide explicitly, instead of you guessing whether a missing key means "no value" or "forgot".
- **Flat beats deep.** Every nesting level and `anyOf` branch adds grammar states, compile time and places to go wrong. Recursive schemas are unsupported on some providers. Two small calls often beat one deeply nested one.
- **Field order is generation order.** The model writes fields in schema order (Anthropic's docs note that required properties come out before optional ones). Put `evidence` or a short `reason` before `verdict`, and the verdict is conditioned on it. Put the verdict first and the reason becomes a justification written after the fact.
- **Descriptions are instructions.** A `description` such as "ISO 4217 code as printed on the invoice; null if not shown" does more than a paragraph in the system prompt, because it sits where the model is writing.
- **Don't ask for what code can compute.** Totals, counts, date arithmetic and IDs you already have should be computed or looked up, not generated.
- **Know what the grammar drops.** Many providers do not enforce `minimum`, `maximum`, `minLength` or `pattern`; SDK helpers may move them into descriptions. Anything the grammar cannot express is enforced by your validator.

## Validate, repair, retry

Strict mode makes bad objects rare, not impossible. They still come from replies cut off at `max_tokens`, refusals (the reply is a refusal, not your schema), providers or models without strict mode, schema features the provider ignores, and values that are well-typed but wrong. Production code keeps one small loop:

```python
import json

def call_structured(messages, schema, check, max_retries=2):
    """check(obj) -> list of error strings: full schema plus business rules."""
    for attempt in range(1 + max_retries):
        resp = llm(messages, response_schema=schema)          # strict mode if the API has it
        if resp.stop_reason == "max_tokens":
            raise Truncated(resp)                              # raise the limit or split the task; don't retry blindly
        if resp.stop_reason == "refusal":
            return Refused(resp.text)                          # a product decision, not a parse error
        try:
            obj = json.loads(repair(resp.text))                # fences, preamble, trailing commas
            errors = check(obj)
        except json.JSONDecodeError as e:
            errors = [f"not valid JSON: {e.msg} at character {e.pos}"]
        if not errors:
            return obj
        messages = messages + [assistant(resp.text),
                               user("Fix these problems and return only the corrected JSON:\n- " + "\n- ".join(errors))]
    raise NeedsReview(messages, errors)                        # human queue or fallback model
```

Three details matter. Repair is for mechanical faults only. Code that "fixes" a wrong enum value is guessing. Error messages should name the field and the rule ("`total` is 118.00 but line items sum to 108.00"), because that is what makes the retry succeed. And retries need a hard cap: if a schema fails twice on the same input, a third identical call rarely helps. Fall back to a stronger model, a simpler schema or a person. Count repairs and retries per schema version. A rising retry rate is often the first sign that a prompt or model change broke something ([Observability](lesson:aieng:12.07)).

## Streaming partial JSON

Users should not wait for the closing brace. Partial-JSON parsers return the object so far from an incomplete string. For example, `pydantic_core.from_json(text, allow_partial=True)` turns `{"name": "Ada", "tags": ["x", "y` into `{"name": "Ada", "tags": ["x"]}`, dropping the unfinished string. Tool-call arguments stream the same way, as string deltas. Render fields as they complete, but act only on the validated final object: a half-streamed `amount` of `1` may be on its way to `1500`. Order the schema so fields the UI needs first are generated first.

## The quality cost of strict formats

Constraining the output can make answers worse. Tam et al. found that forcing JSON or XML reduced accuracy on reasoning tasks, and stricter formats hurt more. The Outlines authors replied that much of the gap came from prompts that did not match the format. Both effects are plausible: a grammar removes the space where the model would have worked through the problem, and a mask can push it off the tokens it preferred, as in the worked example. The practical rules are to let the model reason before it commits, keep the constrained part small, and measure. Reasoning can go in a thinking block, a free-text first call, or a `reasoning` field placed first. Some providers discourage asking for step-by-step reasoning inside the schema, so check their guidance. Measuring means running your eval set ([Building an eval set](lesson:aieng:10.02)) with and without the constraint, comparing field accuracy, not just validity.

## Where this breaks

- **Valid is not correct.** The commonest production bug is a well-formed object with a hallucinated invoice number. Validity rate is a health metric; field accuracy on labelled data is the quality metric.
- **Truncation.** A long array hits `max_tokens` and the object never closes. Set limits from the largest real output, and split very long extractions.
- **First-call latency.** Grammars are compiled on first use and cached. Anthropic documents a 24-hour cache that changing the schema invalidates, so generating schemas per request defeats the cache.
- **Prompt and schema disagree.** The prompt says "list up to three tags", the schema allows any number. The grammar wins on structure; the prompt loses quietly.
- **Structured is not safe.** An object extracted from a hostile web page can carry hostile strings. Parsing it does not make it trusted ([Prompt injection](lesson:aieng:12.03)).

## What interviewers probe

Expect to be asked what JSON mode does not guarantee, how a grammar mask handles tokens that cross field boundaries, why reasoning should come before the answer in the schema, what you do when the model's answer is valid but wrong, how you would stream a structured answer to a UI, and how you would detect that a schema hurts quality. Strong answers separate the three layers (syntax, schema, truth) and say which tool enforces each.

<div class="take">
<div class="lab">Key takeaway</div>

Schema-constrained decoding guarantees the shape of the output by masking invalid tokens. The schema's design decides whether that shape helps or forces bad answers, and only your validator and eval set can tell you the values are right.

</div>
