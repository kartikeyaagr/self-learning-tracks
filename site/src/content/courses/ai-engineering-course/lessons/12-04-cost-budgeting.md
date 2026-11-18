---
id: "12.04"
course: "ai-engineering-course"
module: "12"
slug: "cost-budgeting"
title: "Cost budgeting"
keyIdeas: ["A request's cost is input, cached input, cache writes and output (thinking tokens bill as output), each at its own price. Write the formula down per feature before optimising anything.", "Prompt caching reuses an exact token prefix up to a breakpoint. Writes cost a premium (1.25x for 5 minutes, 2x for 1 hour on Anthropic), reads a fraction (0.1x), so a prefix pays off from its second use and the break-even hit rate is about 22% for a 5-minute entry.", "Most cache misses are self-inflicted: a timestamp or user name early in the prompt, unstable tool order, changing tools or thinking settings, prompt A/B variants, truncating old history, or parallel fan-out before the first write lands.", "Batch APIs (about half price, results within 24 hours) suit evals, backfills and nightly jobs. Routing and cascades cut the per-call price but must be judged on task success.", "Budget on cost per successful task, including what failures cost downstream (retries, human handoffs), not on cost per call."]
question: "A support assistant sends a 6,000-token static prefix (tools and system prompt), 4,100 tokens of per-request context and gets 350 output tokens, 1 million times a month. Input costs \\$3 and output \\$15 per million tokens (illustrative). Estimate the monthly bill, what prompt caching saves at a 90% hit rate with a 1.25x write premium and 0.1x reads, and how you would decide whether to route 60% of traffic to a model one third the price."
answer: "Without caching each call is 10,100 x \\$3/M + 350 x \\$15/M = \\$0.0303 + \\$0.00525 = \\$0.0356, so about \\$35,550 a month. With caching the prefix costs 0.9 x 6,000 x \\$0.30/M + 0.1 x 6,000 x \\$3.75/M = \\$0.00387 instead of \\$0.018, so the bill falls to about \\$21,400 (40% less) with no quality risk. Routing 60% to the cheaper model brings it to about \\$12,900, but it must be judged on cost per resolved ticket: if resolution drops from 72% to 70% and an unresolved ticket costs a \\$4 human handoff, the extra 20,000 handoffs cost \\$80,000, far more than the \\$8,600 saved. So cache first, then route only the slices where an eval shows the small model resolves as well."
rubric: ["Computes per-request cost from input and output tokens at their separate prices and scales to a month", "Prices the cached prefix as a mix of reads and writes at the given hit rate", "States that caching changes cost but not output quality", "Judges routing on cost per successful task, including the cost of failures", "Proposes measuring the small model per slice on an eval set before routing"]
sources: [{"title": "Anthropic: Prompt caching", "url": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching", "note": "Breakpoints, the tools-system-messages order, TTLs, write and read multipliers, lookback window and what invalidates the cache."}, {"title": "OpenAI: Prompt caching", "url": "https://platform.openai.com/docs/guides/prompt-caching", "note": "Automatic and explicit caching, cache keys, routing by prefix hash and retention options."}, {"title": "Chen, Zaharia and Zou: FrugalGPT (2023)", "url": "https://arxiv.org/abs/2305.05176", "note": "Prompt adaptation, model approximation and LLM cascades for cutting API cost."}, {"title": "Anthropic: Batch processing", "url": "https://platform.claude.com/docs/en/build-with-claude/batch-processing", "note": "50% discount, 24-hour window, stacking with prompt caching."}]
practice: ["prompt-cache-cost", "model-cascade-routing", "optimize-throughput-and-cost-under-latency-slo", "prefix-cache-matching-and-reuse"]
tools: ["Anthropic API", "OpenAI API", "Gemini API"]
---

## The cost of one request

Every hosted model bills tokens by type. For one call,

$$
C = \frac{T_{\text{in}}\,p_{\text{in}} + T_{\text{read}}\,r\,p_{\text{in}} + T_{\text{write}}\,w\,p_{\text{in}} + (T_{\text{out}} + T_{\text{think}})\,p_{\text{out}}}{10^6}
$$

where prices $p$ are per million tokens, $r$ and $w$ are the cache read and write multipliers, and $T_{\text{in}}$ counts only the uncached input. Thinking tokens are billed as output even when you never see them; [Using reasoning models in apps](lesson:aieng:08.06) covers budgeting them. Output usually costs several times more than input per token (why is in [Cost per request](lesson:ieng:25.01)), but most apps send far more input, so input often dominates the bill. Write this formula down per feature, fill it from real usage logs, and only then decide what to optimise. The provider-side view of the same maths, GPU-hours per token, is in the [cost-optimised inference case study](lesson:mlsd:11.04).

## Prompt caching

When requests share a prefix, the provider can keep that prefix's KV cache (see [KV cache](lesson:aieng:04.04)) and skip its prefill. You pay less and the first token arrives sooner. The mechanics are worth knowing exactly, because most of the savings are won or lost in small details.

**Prefix matching.** A hit requires the request to start with exactly the same tokens as a cached entry, up to some boundary. Anthropic renders the request in the order tools, system, messages, and hashes the prefix cumulatively: change one byte of a tool description and everything after it misses. OpenAI routes requests by a hash of the first part of the prompt, so requests that differ early land on machines that never saw the prefix; its `prompt_cache_key` parameter groups requests that should share a cache.

**Breakpoints.** On Anthropic you mark up to four `cache_control` breakpoints (or one top-level field that follows the conversation). A write happens only at a breakpoint; a read looks backwards from the breakpoint for an entry an earlier request wrote, at most 20 blocks back. OpenAI caches implicitly by default, and its newer models also accept explicit breakpoints. Below a model-specific minimum (512 to 4,096 tokens on current Anthropic models, 1,024 on OpenAI's newest) nothing is cached and no error is returned, so check the usage fields (`cache_read_input_tokens`, `cache_creation_input_tokens`, `cached_tokens`).

**TTLs.** An Anthropic entry lives 5 minutes by default and every hit refreshes it for free; a 1-hour entry costs more to write. The lifetime runs from the start of the request, so a long streamed answer eats into it. OpenAI's retention depends on the model (minutes in memory, or extended retention measured in hours). Gemini offers implicit caching plus explicit cache objects whose cost grows with how long you keep them.

**Write premium versus read discount.** With Anthropic's write multipliers ($w = 1.25$ for 5 minutes, $w = 2$ for 1 hour) and its standard read multiplier $r = 0.1$ (its newest models discount reads further, to 0.05 or 0.025, so check the pricing page for your model), a prefix used $n$ times within its TTL costs $w + r(n-1)$ times its uncached price instead of $n$. One use loses 25%; two uses cost 1.35 instead of 2. With hit rate $h$ the expected price per prefix token is $h r + (1-h) w$, which beats 1 when

$$
h > \frac{w - 1}{w - r}
$$

That is 22% for a 5-minute entry and 53% for a 1-hour entry. Choose the 1-hour TTL only when gaps between uses are mostly between 5 and 60 minutes. Cache reads also stop counting against Anthropic's input-tokens-per-minute limit, so caching raises throughput as well ([Reliability](lesson:aieng:12.06)).

~~~dg-layers Order the prompt from most stable to most volatile, and put the breakpoint after the last stable block.
Tools | identical for every request; sorted, versioned with the deployment
System prompt | instructions and policies; no dates, names or request ids
Static context | product docs, few-shot examples, a long document being discussed
*Breakpoint | cache entry written here; everything above must be byte-identical
History | earlier turns; append-only, so the previous turn's entry is a prefix
Request | retrieved chunks, user message, today's date, per-user data
~~~

### Cache-breaking mistakes

| Mistake | Why it misses | Fix |
| --- | --- | --- |
| Current time or user name in the system prompt | prefix differs per request | move it after the breakpoint |
| Tools serialized from a dict or set | order or JSON keys change | sort tools and keys; snapshot-test the rendered prompt |
| Toggling tools, thinking or effort settings per request | these are rendered into the prefix | keep them fixed per feature |
| Prompt A/B test with many variants | each variant has its own, colder cache | fewer variants, or accept the cost in the test budget |
| Dropping the oldest turns to fit the window | the start of the prompt changes | summarise in chunks at stable points |
| Fanning out 50 parallel calls on a cold prefix | an entry is readable only after the first response starts | send one, then the rest |
| Breakpoint on a block that changes every request | writes every time, never reads | put it on the last stable block |

## Semantic caching

A semantic cache returns a stored *answer* when a new query embeds close to an old one. It skips the model entirely, so it saves more than prompt caching, but it changes behaviour. "Cancel my order" and "cancel my subscription" can sit above any similarity threshold you pick. Stored answers go stale when the documents behind them change, and a cache shared across users can leak one user's personalised answer to another. If you use one, key it by tenant, user permissions and index version; restrict it to non-personal, high-repeat questions; and measure its false-hit rate on labelled query pairs before launch. An exact-match cache on normalised queries is the safe first step.

## Batch APIs

For work nobody waits on (evals, backfills, nightly summaries, embedding a corpus), batch endpoints charge about half price in exchange for results within 24 hours. Anthropic's batch discount is 50% and stacks with prompt caching, though cache hits in batches are best effort because requests run concurrently. Moving the eval suite to batch alone often halves the eval bill.

## Routing and cascades

A router sends each request to one model, chosen by rules or a trained classifier; RouteLLM trains it on preference data. A cascade calls the cheap model first and escalates when its confidence is low, as in FrugalGPT. With escalation rate $e$, a cascade costs $c_s + e\,c_\ell$ per request, so it saves money only while $e < 1 - c_s/c_\ell$. Both need an eval set with per-slice results, and both change quality, which is why the worked example judges them on task success.

<div class="ex">
<div class="lab">Worked example</div>

**A monthly bill before and after (illustrative prices)**

A support assistant answers 1 million questions a month with one model call each: 6,000 tokens of tools and system prompt, 4,100 tokens of account context, retrieved chunks and question, and 350 output tokens. Prices: large model \$3 input and \$15 output per million tokens, small model \$1 and \$5. An unresolved conversation goes to a human at \$4.

| Step | LLM cost per call | LLM bill | Resolved | Total with handoffs | Per resolved ticket |
| --- | --- | --- | --- | --- | --- |
| Baseline, large model | \$0.0356 | \$35,550 | 72% | \$1,155,550 | \$1.60 |
| + prompt caching, 90% hits | \$0.0214 | \$21,420 | 72% | \$1,141,420 | \$1.59 |
| + route 60% to small model | \$0.0129 | \$12,852 | 70% | \$1,212,852 | \$1.73 |

Baseline: $10{,}100 \times 3 + 350 \times 15 = 35{,}550$ micro-dollars per call. Caching prices the prefix at $0.9 \times 6{,}000 \times 0.3 + 0.1 \times 6{,}000 \times 3.75 = 3{,}870$ instead of $18{,}000$. The small model costs $7{,}140$ per call with caching; $0.6 \times 7{,}140 + 0.4 \times 21{,}420 = 12{,}852$.

Caching cuts the LLM bill by 40% at no quality cost. Routing cuts another \$8,568, but two points of resolution add 20,000 handoffs worth \$80,000. If the large model also thinks for 800 tokens per call, that adds \$12,000 a month, a third of the baseline.

</div>

## Where this breaks

- **Budgeting per call.** Retries, failed tool loops and agent steps multiply calls per task. Track tokens per successful task, per feature and per tenant.
- **Trusting list prices.** Long-context surcharges, regional endpoints, priority tiers and per-model cache multipliers all change the bill. Reconcile your estimate with the invoice monthly.
- **Context bloat.** Retrieval that adds ten 500-token chunks to every call costs more than the answer. Fewer, better chunks (module 11) is a cost lever too.
- **No owner for the cache hit rate.** A deploy that adds a timestamp to the system prompt can quietly double the bill. Alert on cache-read tokens as a share of input.

## What interviewers probe

Expect to compute a bill from token counts on a whiteboard, explain the caching break-even and where the breakpoint goes, list cache-breaking mistakes, argue when a semantic cache is unsafe, and say how you would prove that a cheaper model or router does not lose more in failures than it saves.

<div class="take">
<div class="lab">Key takeaway</div>

write the per-request cost formula, then take the safe savings first: a stable prompt prefix with prompt caching, batch pricing for offline work, shorter context. Treat routing, cascades and semantic caches as quality changes, and judge them on cost per successful task.

</div>
