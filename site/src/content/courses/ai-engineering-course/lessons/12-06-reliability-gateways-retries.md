---
id: "12.06"
course: "ai-engineering-course"
module: "12"
slug: "reliability-gateways-retries"
title: "Reliability: gateways, retries and fallbacks"
keyIdeas: ["LLM APIs fail in their own ways: 429s from requests-per-minute and tokens-per-minute limits, 5xx and overloaded errors, stalls mid-stream, quota and spend caps, refusals and malformed output. Each needs a different response, and some must never be retried.", "Rate limits are token buckets on requests, input tokens and output tokens. Find the binding one per request shape; prompt caching can raise input-token headroom because cached reads may not count.", "Retry with capped exponential backoff and full jitter, honour Retry-After, stop at a deadline, and cap retries with a budget. Retries at several layers multiply.", "A gateway centralises keys, client-side rate limiting, retries, fallback chains, circuit breakers, caching and cost logging. Fallback models need their own evaluated prompts and will start with a cold cache.", "Pin exact model versions, track retirement dates, and roll out any change to model, prompt or parameters as one versioned config through evals, shadow traffic and a canary."]
question: "Your product calls one hosted model. Design how it handles a provider incident (rising 529s and timeouts) and the provider's notice that your model retires in 60 days. Include retry policy, fallbacks and rollout."
answer: "Incident: a gateway retries 5xx, 529 and timeouts with capped exponential backoff and full jitter (for example base 0.5 s, cap 8 s), honours Retry-After on 429s, never retries 400s or spend-cap errors, and stops at the request deadline. A retry budget (at most about 10% extra requests) and a per-provider circuit breaker stop retries from amplifying the outage. When the breaker opens, traffic fails over to the same model in a second region or cloud, then to a different model with its own prompt that passed the eval suite, then to a degraded answer. Retirement: the model id is pinned, so nothing changes silently. Run the regression eval set against candidate replacements, adjust prompts, shadow real traffic, then canary 1%, 10%, 50% with automatic rollback on quality, latency and cost metrics, well before the retirement date."
rubric: ["Distinguishes retryable errors (429 with Retry-After, 5xx, overloaded, timeouts) from non-retryable ones (400, quota or spend cap, refusals)", "Specifies capped exponential backoff with jitter, a deadline and a retry budget or circuit breaker", "Describes a fallback chain and notes that fallback models need evaluated prompts", "Pins model versions and treats model, prompt and parameters as one versioned config", "Rolls out the migration through evals, shadow and canary with rollback"]
sources: [{"title": "Anthropic: Rate limits", "url": "https://platform.claude.com/docs/en/api/rate-limits", "note": "Token-bucket limits on RPM, input and output tokens, cache-aware input limits, ramp limits and response headers."}, {"title": "AWS Builders' Library: Timeouts, retries and backoff with jitter", "url": "https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/", "note": "Retry amplification, retry budgets and choosing timeouts."}, {"title": "AWS Architecture Blog: Exponential backoff and jitter", "url": "https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/", "note": "Simulation comparing no jitter, full, equal and decorrelated jitter."}, {"title": "Anthropic: Model deprecations", "url": "https://platform.claude.com/docs/en/about-claude/model-deprecations", "note": "Deprecation and retirement lifecycle, notice period and migration advice."}]
practice: ["retry-backoff-schedule", "latency-budget-allocation", "dynamic-request-batching"]
tools: ["LiteLLM", "Envoy AI Gateway", "Anthropic API", "OpenAI API"]
---

## How LLM calls fail

A model call is a remote call to a shared, capacity-constrained service. The general rules for timeouts, retries and idempotency are in [Retries, timeouts, and idempotency](lesson:sdsd:02.09) and [Circuit breakers](lesson:sdsd:06.10); the inference-server view is in [Timeouts, cancellation, retries](lesson:ieng:24.05). What changes for LLM apps is the mix of failures and their price: a retry re-bills every input token, and a dropped stream is paid for up to the point it dropped.

| Failure | Signal | Retry? | Response |
| --- | --- | --- | --- |
| Rate limit | 429 with `retry-after` | yes, after the header | throttle on the client, spread across keys or regions |
| Quota or spend cap | 429 without `retry-after`, a specific error code | no | alert a human; retrying only adds load |
| Server error, overload | 500, 502, 503, 504, 529 | yes, with backoff | then fail over |
| No first token | first-token timeout | once, or fail over | cancel the stalled request |
| Stream drops mid-answer | connection error after partial output | restart, billed again | or keep the partial text and show an error |
| Bad request | 400 (schema, context too long) | no | fix: truncate, validate before sending |
| Refusal or safety block | stop reason, filtered content | not with the same input | handle in the UI ([Guardrails](lesson:aieng:12.02)) |
| Malformed output | schema validation fails | repair or re-ask with the error | [Structured outputs](lesson:aieng:12.01) |

## Rate limits are token buckets

Providers limit requests per minute (RPM) and tokens per minute, usually input (ITPM) and output (OTPM) separately, per model. Anthropic documents its limits as token buckets that refill continuously, so a short burst is allowed if the bucket is full, and adds "ramp limits" that can return 429 on a sudden jump in traffic even under the nominal limit. Responses carry headers with what remains (`anthropic-ratelimit-input-tokens-remaining`, `x-ratelimit-remaining-tokens` on OpenAI). A good client reads them and throttles itself before the provider does.

<div class="ex">
<div class="lab">Worked example</div>

**Which limit binds? (illustrative limits)**

A workspace allows 4,000 RPM, 2,000,000 ITPM and 400,000 OTPM. Each request sends 10,100 input tokens, of which a 6,000-token prefix is cached with a 90% hit rate, and returns 350 output tokens. On Anthropic, cache reads do not count toward ITPM, but cache writes and uncached input do.

| Limit | Tokens counted per request | Requests per minute it allows |
| --- | --- | --- |
| RPM | 1 request | 4,000 |
| ITPM, no caching | 10,100 | 198 |
| ITPM, with caching | $4{,}100 + 0.1 \times 6{,}000 = 4{,}700$ | 425 |
| OTPM | 350 | 1,142 |

Input tokens bind. Caching more than doubles throughput before any limit increase. A backfill of 50,000 such requests would hold the whole input bucket for two hours, so batch work belongs in the batch API or on a separate key with its own limit, never in the same bucket as user traffic.

</div>

## Backoff, jitter and the retry budget

After the $k$-th failed attempt (0-based), wait a random time below an exponential ceiling:

$$
e_k = \min(\text{cap},\ \text{base}\cdot 2^k), \qquad d_k \sim \text{Uniform}(0, e_k)
$$

That is full jitter. Without jitter, every client that failed together retries together, and the overload returns in waves; the AWS simulation found full jitter does the least total work under contention. With base 0.5 s and cap 8 s the ceilings run 0.5, 1, 2, 4, 8 s, so five retries wait 7.75 s on average and 15.5 s at most. A `retry-after` header overrides the formula when it asks for longer.

Retries add load exactly when the provider is struggling. If each attempt fails with probability $p$ and you allow $R$ retries, the expected number of attempts per request is

$$
\sum_{k=0}^{R} p^k = \frac{1 - p^{R+1}}{1 - p}
$$

At $p = 0.5$ and $R = 3$ that is 1.875 times normal load. Layers multiply: an SDK that retries twice inside your loop that retries three times can send twelve requests per user action. Retry at one layer (the gateway), turn off the others, cap retries with a budget (at most about 10% extra requests), and stop at the request's deadline rather than an attempt count.

## Gateways, fallbacks and failover

An LLM gateway sits between your services and every provider: one place for API keys, client-side rate limiting, retries, fallback chains, circuit breakers, exact-match caching, per-team budgets and a cost and latency log for every call. Open-source proxies such as LiteLLM and Envoy AI Gateway, and hosted gateways, provide most of this; the [AI engineering stack](lesson:aieng:01.02) shows where it sits.

~~~dg-flow A fallback chain in the gateway. Each hop is taken only on retryable failures, after the retry budget or when the circuit breaker is open.
Request -(budget check)-> *Gateway -(call)-> Primary model -(breaker open)-> Same model, other region or cloud -(still failing)-> Fallback model, own prompt -(all down)-> Cached or degraded answer
~~~

Fallbacks are less portable than they look:

- **Prompts do not transfer.** A different model family needs its own prompt, tool definitions and output parsing, and must pass the same eval suite. The same model served from another cloud is the cheapest safe first hop.
- **Caches start cold.** The fallback pays full prefill and cache writes, so cost and TTFT jump during an incident.
- **Streams can fail over only before the first token.** After that, the user has seen text from the primary.
- **Availability gains assume independence.** Two providers at 99.5% each are both down 0.0025% of the time only if their failures are unrelated; a shared cloud region or DNS provider breaks that.

A circuit breaker per provider and model stops paying a full timeout on every request during an outage: after a run of failures, send traffic straight to the fallback and probe the primary occasionally.

## Idempotency

A model call has no side effects, so retrying it is safe, just costly and not guaranteed to return the same answer. The danger is in what surrounds it. An agent step that sends an email or issues a refund must not run twice because the step that followed it timed out. Give every side-effecting tool call an idempotency key derived from the conversation, step and tool-call id, and have the tool store results by key ([Stripe's idempotency keys](lesson:sdsd:18.02)). Charge users or deduct credits after a successful response, not per attempt.

## Pinning, deprecation and rollout

Use exact model ids, either dated snapshots or ids the provider documents as fixed, never a moving alias, in production. Providers retire models on a schedule (Anthropic commits to at least 60 days' notice for publicly released models), so keep a registry of every model in use with its owner and retirement date, and start migrating when a deprecation is announced, not when calls start failing.

Treat the model id, prompt template, sampling parameters, tool definitions and retrieval settings as one versioned config, and log its version on every trace ([Observability](lesson:aieng:12.07)). Change any part of it like code: run the regression eval suite ([Regression testing in CI](lesson:aieng:10.06)), shadow real traffic, canary at 1%, 10% and 50% with automatic rollback on quality, latency, error and cost metrics ([A/B testing and safe rollout](lesson:mlsd:05.05), [Fallbacks and model rollouts](lesson:mlsd:06.04)).

## Common mistakes

- Retrying 400s and spend-cap 429s, which can never succeed.
- Fixed retry delays without jitter, which synchronise clients into retry storms.
- A single total timeout on streams, which kills long, healthy answers.
- Failing over to a model nobody evaluated, which turns an outage into a quality incident.
- Pointing production at a "latest" alias, so behaviour changes on the provider's schedule.

## What interviewers probe

Expect questions on which errors to retry, the backoff formula and why jitter matters, how retries amplify outages and what bounds them, which rate limit binds for a given request shape, what a gateway owns, why a fallback model is not free, how to make agent tool calls safe to retry, and how you would migrate off a retiring model.

<div class="take">
<div class="lab">Key takeaway</div>

classify failures before retrying, back off with jitter under a deadline and a retry budget, put keys, limits, retries and fallbacks in one gateway, evaluate every fallback model, and ship model or prompt changes as pinned, versioned configs through evals and a canary.

</div>
