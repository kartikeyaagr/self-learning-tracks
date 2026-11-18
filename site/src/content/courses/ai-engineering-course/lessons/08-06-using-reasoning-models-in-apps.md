---
id: "08.06"
course: "ai-engineering-course"
module: "08"
slug: "using-reasoning-models-in-apps"
title: "Using reasoning models in apps"
keyIdeas: ["Thinking tokens are hidden but billed as output tokens, count against the output cap, and delay the first visible token.", "Current APIs have converged on a named effort level (low to max, model-dependent) instead of a fixed token budget; effort is a behavioural signal, not a hard limit.", "Cost per request is input plus cached input plus (thinking + visible output) at the output price; thinking usually dominates at medium effort and above.", "Route by difficulty: low effort or no reasoning for the easy bulk, high effort only where an eval shows it pays.", "With tools, pass reasoning items back unchanged, drop sampling parameters the model rejects, and log thinking tokens, effort and truncations per request."]
question: "You move a support-ticket classifier from a non-reasoning model to a reasoning model at medium effort. Accuracy rises 2 points, the bill rises about 4x and p95 latency goes from 3 s to 20 s. What do you measure and change before deciding?"
answer: "First split the cost: log thinking tokens separately from visible output and check how many requests ran out of output budget. Then run an effort sweep (none or low, medium, high) on a labelled set, per ticket slice, recording accuracy, cost per correct answer and time to first visible token. Most classification is easy, so the likely result is that low effort keeps nearly all the gain. Route: send the bulk to the cheapest setting that holds accuracy and escalate only the slices where higher effort measurably helps (ambiguous or multi-issue tickets), or use a verifier to trigger a retry at higher effort. Set the output cap well above typical thinking plus answer so truncation does not waste whole requests, and keep latency-critical paths on a non-reasoning setting."
rubric: ["Separates thinking tokens from visible output when diagnosing cost", "Proposes an effort sweep on a labelled set rather than a blanket setting", "Uses cost per correct answer or per successful task, not per request", "Routes or escalates by difficulty instead of one effort for all traffic", "Accounts for latency (hidden thinking delays the first visible token) and for output-cap truncation"]
sources: [{"title": "OpenAI: Reasoning models guide", "url": "https://developers.openai.com/api/docs/guides/reasoning", "note": "reasoning.effort values, reasoning tokens billed as output, max_output_tokens and incomplete responses, encrypted reasoning items."}, {"title": "Anthropic: Building with thinking", "url": "https://platform.claude.com/docs/en/build-with-claude/thinking", "note": "Adaptive thinking, summarized versus omitted display, billing for full thinking, passing thinking blocks back with tool results."}, {"title": "Anthropic: Effort parameter", "url": "https://platform.claude.com/docs/en/build-with-claude/effort", "note": "Effort levels per model and why effort is a behavioural signal rather than a token budget."}, {"title": "Google: Gemini thinking", "url": "https://ai.google.dev/gemini-api/docs/thinking", "note": "thinking_level per model, thought signatures, thinking tokens priced as output."}]
practice: ["request-cost-with-thinking-tokens", "optimize-throughput-and-cost-under-latency-slo", "ttft-tpot-itl-and-token-throughput", "pass-at-k-estimator"]
tools: ["OpenAI Responses API", "Claude API", "Gemini API", "gpt-oss", "Qwen3"]
tags: ["Reasoning", "Cost"]
---

## What changes when the model thinks

A reasoning model writes a chain of thought before (and between) its visible answer. Lesson [08.01](lesson:aieng:08.01) covers how that behaviour is trained and [08.02](lesson:aieng:08.02) why extra tokens buy accuracy. This lesson is about the bill, the latency and the integration details once you call one from an app.

Four facts hold across every major API today:

1. **Thinking tokens are billed as output tokens**, at the output price, whether or not you ever see them. Anthropic, OpenAI and Google all state this in their docs.
2. **They count against the output cap.** `max_output_tokens` (OpenAI), `max_tokens` (Anthropic) and the Gemini output limit cover thinking plus the answer. Hit the cap while the model is still thinking and you get an incomplete response with no answer, and you still pay for the thinking.
3. **They are hidden.** You get a summary at most (opt-in on all three vendors), never the raw chain. What you can log is the count.
4. **They delay the first visible token.** A streamed answer cannot start until the model stops thinking, so time to first visible token grows with effort.

## The controls, across APIs

The industry moved from "give the model N thinking tokens" to "tell the model how hard to try". A named level is easier to tune and lets the model spend little on easy inputs and a lot on hard ones.

| API | Control | Values | Notes |
| --- | --- | --- | --- |
| OpenAI Responses / Chat Completions | `reasoning.effort` / `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | Model-dependent: some models reject `none`; defaults differ per model |
| Anthropic Messages | `output_config.effort` with adaptive thinking | `low`, `medium`, `high`, `xhigh`, `max` | Default is `high` on most models, `medium` on some; the older `budget_tokens` (minimum 1,024, below `max_tokens`) is deprecated and rejected by the newest models |
| Gemini | `thinking_level` | `minimal` to `xhigh`, per model | Older Gemini releases took a token budget instead |
| gpt-oss (open weights) | "Reasoning: low / medium / high" in the system prompt | 3 levels | Set through the chat format, not an API field |
| Qwen3 (open weights) | `enable_thinking` in the chat template | on / off | Hybrid model: one checkpoint, thinking toggled per request |

Two things to internalise. First, **effort is not a budget**. Anthropic's docs describe it as a behavioural signal: at low effort the model still thinks on a hard problem, just less. If you need a hard ceiling on spend, that is the output cap, and hitting it truncates. Second, **levels are not portable**. "Medium" on one model is not "medium" on another, and a vendor can recalibrate levels between model versions. Re-run the sweep after every model change.

## Worked cost and latency

The per-request cost formula has one new term:

$$
C = \frac{(n_{\text{in}} - n_{\text{cache}})\,p_{\text{in}} + n_{\text{cache}}\,p_{\text{cache}} + (n_{\text{think}} + n_{\text{out}})\,p_{\text{out}}}{10^6}
$$

with prices per million tokens. [Cost budgeting](lesson:aieng:12.04) covers caching and batch discounts; here the point is the size of $n_{\text{think}}$.

<div class="ex">
<div class="lab">Worked example</div>

**A ticket assistant at 1 million requests a month.** Illustrative prices: \$2 per million input tokens, \$0.20 per million cached input tokens, \$10 per million output tokens. Each request has 3,000 input tokens (2,000 of them a cached system prompt) and a 300-token visible reply. Input costs \$0.002 + \$0.0004 = \$0.0024 per request at every setting. Thinking counts are illustrative averages from a sweep.

| Setting | Thinking tokens | Cost per request | Monthly | Streamed reply time at 100 tokens/s |
| --- | --- | --- | --- | --- |
| No thinking | 0 | \$0.0054 | \$5,400 | 3 s |
| Low effort | 200 | \$0.0074 | \$7,400 | 5 s |
| Medium effort | 1,500 | \$0.0204 | \$20,400 | 18 s |
| High effort | 6,000 | \$0.0654 | \$65,400 | 63 s |

At high effort, thinking is 92% of the bill. The user also waits through 60 seconds of tokens they never see before the reply starts.

**Routing instead of one setting.** If an eval shows 80% of tickets are as good at low effort and only the hard 20% gain from high: $0.8 \times 0.0074 + 0.2 \times 0.0654 = 0.0190$ dollars per request, \$19,000 a month instead of \$65,400.

**The truncation trap.** Set `max_output_tokens` to 4,000 at high effort and a request that wants 6,000 thinking tokens stops mid-thought: you pay \$0.0024 + 4,000 × \$10 per million = \$0.0424 for no answer, then pay again on the retry.

</div>

## Picking a setting

Treat effort as a hyperparameter you tune on an eval, per route. The sweep is cheap: the same 200–500 labelled cases at each level, recording accuracy, **cost per correct answer** and time to first visible token. Published sweeps suggest the shape varies by workload. Anthropic reports research-style tasks with nearly flat curves (lower effort costs a few points at most) and long-horizon coding with a real trade-off at each step down. Only your sweep tells you which shape you have.

Patterns that work in production:

- **Escalate on failure.** Run at low effort, check the answer with something external (a schema, a test, a verifier from [08.04](lesson:aieng:08.04)), and retry the failures at high effort. You pay high-effort prices only on the hard tail.
- **Route by request type.** The app often knows the difficulty already: a password-reset question is not a refund dispute.
- **Keep latency-critical paths off thinking.** Voice, autocomplete and inline UI hints should use no reasoning or the lowest level.

A reasoning model is the wrong choice when the task has no multi-step structure (extraction, simple classification, rewriting), when latency is the product, when volume is huge and margins thin, or when you depend on sampling controls the model no longer accepts.

## Thinking with tools

Current reasoning models think **between** tool calls, not only before the first one: read a result, reason about it, choose the next call. OpenAI and Anthropic call this interleaved thinking. The integration rule follows from it: the reasoning that led to a tool call must still be there when the result comes back.

~~~dg-seq Interleaved thinking: reasoning items travel back with each tool result.
App > Model: user task + tools
Model -> App: reasoning item (opaque) + tool call search(...)
App > Tool: run search
App > Model: history + reasoning item unchanged + tool result
Model -> App: reasoning item + final answer
~~~

- **Pass reasoning back unchanged.** Anthropic thinking blocks carry an encrypted `signature`; within a tool-use turn you must return them complete and unmodified. OpenAI reasoning items carry `encrypted_content` when you run stateless (`store=false` or zero data retention). Gemini returns thought signatures that must be resent. Strip or edit them and the model loses its plan, or the request is rejected.
- **Do not rewrite history under a reasoning model.** Editing earlier turns, reordering tool results or splicing in summaries can invalidate stored reasoning and the prompt cache. Append instead.
- **Forced tool choice may be restricted.** With thinking on, some models accept only `auto` or `none` for tool choice. Use a strict schema and an instruction instead of forcing.

## Parameters they ignore or reject

- **Temperature, top_p, top_k, logprobs.** OpenAI's newest reasoning models tell you to omit `temperature`, `top_p` and `logprobs`; Anthropic's current models return a 400 for non-default sampling values. Code that sets `temperature=0` "for determinism" breaks on migration. You cannot get determinism this way anyway: control variance with evals and repeated trials.
- **Prefill.** Starting the assistant turn for the model is rejected on current Claude models; use structured outputs ([12.01](lesson:aieng:12.01)).
- **"Think step by step".** Redundant and sometimes harmful on a model that already reasons. State the goal, constraints and output format instead.

## What to log

Per request: input, cached input, **thinking** and visible output tokens as separate fields (`output_tokens_details.reasoning_tokens` on OpenAI, `usage.output_tokens_details.thinking_tokens` on Anthropic, `total_thought_tokens` on Gemini), the effort level, model version, stop reason or status (count `incomplete` and `max_tokens` stops), time to first visible token and total latency. Store reasoning summaries for debugging if you opted in, but remember from [08.05](lesson:aieng:08.05) that a summary of reasoning is not a faithful explanation of the answer. [Observability](lesson:aieng:12.07) covers the tracing side.

<div class="take">
<div class="lab">Key takeaway</div>

Thinking tokens are hidden output tokens: they set the bill and the wait. Tune effort per route on an eval, measure cost per correct answer, escalate only the hard cases, size the output cap above thinking plus answer, and pass reasoning items back untouched when tools are involved.

</div>

## Common mistakes

- **Budgeting on visible output.** The dashboard shows a 300-token reply; the invoice bills 6,300.
- **A tight output cap as a cost control.** It truncates, bills the thinking and returns nothing. Lower effort instead.
- **One global effort level.** Easy traffic pays hard-traffic prices, or hard traffic fails.
- **Carrying settings across models.** Levels, defaults and accepted parameters change between versions.
- **Hiding latency.** With thinking hidden, the UI shows nothing for many seconds. Stream a status or a summary, or use a lower level.

## What interviewers probe

Expect "how would you cut the cost of this reasoning feature by half without losing quality?" (sweep, route, escalate, cache the prefix, cost per success), "why did cost jump after a model upgrade?" (thinking defaults and level calibration changed), and "what happens if you drop the reasoning items between tool calls?". Be ready to do the cost formula above on a whiteboard with thinking as its own term.
