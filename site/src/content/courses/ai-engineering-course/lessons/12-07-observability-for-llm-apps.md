---
id: "12.07"
course: "ai-engineering-course"
module: "12"
slug: "observability-for-llm-apps"
title: "Observability for LLM apps"
keyIdeas: ["Trace every request as a tree of spans: the agent or request root, model calls, tool calls, retrieval and guardrails. OpenTelemetry's GenAI conventions name them (chat {model}, execute_tool {tool}, invoke_agent {agent}) and their attributes (provider, model, token usage by type, finish reasons, conversation id, prompt version).", "Record ids, versions, token counts, latencies, costs and outcomes on every span. Keep prompts, completions and tool payloads out of telemetry by default; when you need them, redact and store them separately with access control and retention, and reference them from the span.", "Dashboards need four views: cost (per feature, tenant, model, cache hit rate), latency (TTFT and end-to-end per stage), errors (provider errors, truncations, tool and schema failures, guardrail blocks) and quality (online eval scores, feedback, escalations).", "Capture explicit and implicit user feedback against the trace id; it tells you where to look, not what is right.", "Close the loop: sample failing and random traces, label failure categories, turn the top category into replayable eval cases with frozen tool and retrieval fixtures, gate the fix in CI, and watch the metric move."]
question: "You are launching an LLM support agent. What do you put in its traces, what do you keep out, which dashboards do you build, and how do production failures become eval cases?"
answer: "Traces: one root span per user turn, child spans per model call, tool call, retrieval and guardrail, following OpenTelemetry GenAI conventions: provider, requested and returned model, input, cached, output and reasoning tokens, finish reasons, conversation id, prompt or config version, latency, cost and error type. Keep content out of telemetry by default; store redacted prompts, outputs and tool payloads in a separate access-controlled store with short retention, referenced by id, and never log secrets. Dashboards: cost per feature, tenant and model with cache hit rate; TTFT and end-to-end p50/p95 per stage; error rates by type including truncations, tool and schema failures; quality from sampled LLM-judge scores, thumbs and escalations. Loop: each week sample traces with errors, negative feedback, low judge scores and a random slice, label failure categories, turn the largest category into eval cases with frozen tool and retrieval outputs, fix, gate in CI and monitor."
rubric: ["Describes a span tree with model, tool, retrieval and guardrail spans and the key GenAI attributes", "Keeps prompt and completion content out of default telemetry and explains redaction, access control and retention", "Names dashboards covering cost, latency, errors and quality", "Links user feedback to traces and treats it as a signal rather than a label", "Describes turning sampled failing traces into replayable eval cases and gating fixes in CI"]
sources: [{"title": "OpenTelemetry: GenAI client inference conventions", "url": "https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md", "note": "Span name, kind and attributes for model calls, including token usage by type and opt-in content attributes."}, {"title": "OpenTelemetry: GenAI spans and content capture", "url": "https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md", "note": "Tool and retrieval spans; content is not captured by default, with external storage recommended in production."}, {"title": "Hamel Husain: A Field Guide to Rapidly Improving AI Products", "url": "https://hamel.dev/blog/posts/field-guide/", "note": "Error analysis on real traces, data viewers and turning failures into evals."}, {"title": "Google SRE book: Monitoring distributed systems", "url": "https://sre.google/sre-book/monitoring-distributed-systems/", "note": "The four golden signals and alerting on symptoms."}]
practice: ["trace-critical-path", "latency-budget-allocation", "prompt-cache-cost", "p50-p95-and-p99-inference-latency"]
tools: ["OpenTelemetry", "Langfuse", "Arize Phoenix"]
---

## What is different about LLM telemetry

Standard observability ([Observability for distributed systems](lesson:sdsd:14.01)) answers "is it up, is it fast, is it erroring". An LLM app adds three problems. The worst failures return HTTP 200: a confident wrong answer, a skipped tool call, a reply in the wrong language. Cost varies a hundredfold between requests, depending on context size, cache hits, thinking tokens and how many agent steps ran. And one user turn fans out into retrieval, several model calls and tool calls, so a single latency number explains nothing. The answer is a trace per request with LLM-specific attributes, and a path from traces into the eval set.

## Spans and the GenAI semantic conventions

OpenTelemetry's GenAI semantic conventions define a shared vocabulary so that SDK instrumentations, gateways and observability back ends agree. They are still marked Development, they now live in their own repository, and names have changed between releases (the provider attribute used to be `gen_ai.system`, now `gen_ai.provider.name`), so pin the version your instrumentation emits and normalise at ingestion.

| Span | Name pattern | Key attributes |
| --- | --- | --- |
| Agent or request root | `invoke_agent {agent}` | `gen_ai.agent.name`, `gen_ai.conversation.id` |
| Model call | `chat {model}` | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.reasoning.output_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.prompt.version`, `error.type` |
| Tool call | `execute_tool {tool}` | `gen_ai.tool.name`, tool call id, outcome |
| Retrieval, embeddings | `embeddings {model}`, retrieval spans | index, top-k, number of results |

Three details catch people out. A model-call span covers the whole logical call including automatic retries, so record retry counts as an attribute. `gen_ai.usage.input_tokens` should *include* cached tokens, while some provider APIs report input tokens excluding cache reads and writes; mix the two and your cost dashboard is wrong. And `gen_ai.request.model` (what you asked for) can differ from `gen_ai.response.model` (the snapshot that answered), which is how you notice an alias moving under you ([Reliability](lesson:aieng:12.06)). The conventions also define metrics such as `gen_ai.client.operation.duration` and server-side `gen_ai.server.time_to_first_token`. The inference-server side of the same trace, queueing and prefill per replica, is in [Distributed request tracing](lesson:ieng:24.03).

<div class="ex">
<div class="lab">Worked example</div>

**Reading one agent trace (illustrative prices: large model \$3 input, \$0.30 cached, \$15 output per million tokens; small model \$1, \$0.10, \$5)**

| Span | Time (ms) | Tokens in / cached / out | Cost | On critical path |
| --- | --- | --- | --- | --- |
| `invoke_agent support` | 0 to 4,200 | | | root |
| `chat small` (guardrail) | 5 to 180 | 900 / 0 / 5 | \$0.0009 | no, hidden by retrieval |
| `retrieve kb` | 5 to 420 | | | yes, 415 ms |
| `chat large` (plan) | 430 to 1,300 | 9,000 / 8,000 / 120 | \$0.0072 | yes, 870 ms |
| `execute_tool get_order` | 1,310 to 1,650 | | | no, hidden by get_policy |
| `execute_tool get_policy` | 1,310 to 2,100 | | | yes, 790 ms |
| `chat large` (answer) | 2,110 to 4,150 | 10,500 / 9,000 / 380 | \$0.0129 | yes, 2,040 ms |
| `write audit log` | 4,150 to 4,400 | | | no, after the root ends |

The critical path, found by walking back from the root's end to the last child that finished, is retrieval, plan, get_policy, answer. The two large-model calls take 2.9 of the 4.2 seconds, so shorter outputs or a smaller planning model help; speeding up `get_order` or the guardrail does nothing. The answer call costs $(1{,}500 \times 3 + 9{,}000 \times 0.3 + 380 \times 15)/10^6 = 0.0129$, so \$0.0129; output tokens are 44% of it. The trace costs \$0.021, which at 1 million turns a month is \$21,000.

</div>

## What to log, and what not to

The conventions say instrumentations should not capture instructions, inputs or outputs by default, and recommend storing content outside the telemetry system with a reference on the span (`gen_ai.input.messages_ref`) in production. The reasons are privacy, volume and access: traces are read by every engineer on call, while conversations may hold health details, account numbers and other people's data.

| Always on the span | Separate store, redacted, access-controlled, short retention | Never |
| --- | --- | --- |
| trace, conversation and response ids | prompts and completions | API keys, auth tokens, passwords |
| model ids, prompt or config version | retrieved chunks, tool arguments and results | full card or bank numbers, government ids |
| token counts by type, cost, latency, TTFT | user-uploaded files | content from users or tenants who opted out |
| finish reasons, error types, retry counts | judge rationales that quote the user | |
| tool names and outcomes, guardrail verdicts | | |
| hashed user id, tenant, region | | |

Run PII redaction before content is written anywhere ([Guardrails](lesson:aieng:12.02)), keep the store in the region the data must stay in, and make deletion requests reach traces, stored content and any eval cases built from them. Sample by value: keep every trace with an error, a slow end-to-end time or negative feedback, and a small random share of the rest.

## Dashboards and alerts

| View | What it shows | Slice by | Alert on |
| --- | --- | --- | --- |
| Cost | spend, tokens by type, cache-read share of input, cost per successful task | feature, tenant, model, prompt version | daily spend or cache hit rate jumps |
| Latency | TTFT and end-to-end p50/p95, per-stage time on the critical path | feature, model, region | p95 over the SLO ([Latency budgeting](lesson:aieng:12.05)) |
| Errors | 429s, 5xx, timeouts, `length` finish reasons, tool errors, schema failures, guardrail blocks | provider, model, tool | error rate, fallback rate |
| Quality | online LLM-judge scores on a sample, thumbs, regenerations, human escalations | intent, prompt version | drops after a release |

Alert on symptoms users feel, as the SRE book advises, and keep the cause panels for diagnosis. A rise in `length` finish reasons, for example, means answers are being cut off, which no HTTP error rate will show.

## User feedback

Explicit feedback (thumbs, a reason picker, a comment) is sparse and skewed toward angry users. Implicit signals are plentiful: regenerate clicks, copying the answer, how much a user edits a drafted reply before sending it, abandoning the conversation, asking for a human. Record both as events tied to the trace and response ids, so every signal points to the exact prompt version, context and output that caused it. Treat feedback as a pointer to traces worth reading, not as a label: users downvote correct answers they dislike.

## From traces to eval cases

~~~dg-flow The loop from production back into the eval set.
Production traces -(sample errors, thumbs-down, low judge scores, random)-> Review queue -(read and label)-> *Failure categories -(largest first)-> Eval cases with frozen fixtures -(gate in CI)-> Fix and release -(watch the metric)-> Production traces
~~~

Each week, a person reads a sample of traces and writes down what went wrong in plain words, then groups those notes into categories. Suppose 150 reviewed traces split into retrieval missed the document (52), wrong policy reasoning (31), tool error (24), tone or format (18) and correct but the user was unhappy (25). Retrieval is a third of the failures, so it gets the next fix, and 30 of those traces become eval cases. A trace is only useful as a case if it is replayable: freeze the retrieved chunks and tool results from the spans as fixtures, so the case tests the model's behaviour rather than whether the order database still holds that order. How to label, size, split and version the set is in [Building an eval set](lesson:aieng:10.02); gating the fix is in [Regression testing in CI](lesson:aieng:10.06). The same trace store supplies judge calibration data ([Graders and LLM-as-judge](lesson:aieng:10.03)).

## Where this breaks

- **Logging everything.** Full prompts in a shared log index leak personal data and cost more to store than the model calls.
- **Logging nothing useful.** Without the prompt version and model snapshot on each span, you cannot tell which change caused a regression.
- **Unlinked feedback.** A thumbs-down without a trace id cannot be investigated.
- **Dashboards no one reads.** Error analysis is the work; the platform only makes it fast.

## What interviewers probe

Expect to sketch a span tree for an agent turn, list the attributes you would record, explain where prompt content goes and why, design cost, latency, error and quality dashboards, say which signals you would alert on, and walk through how a production failure becomes a regression test.

<div class="take">
<div class="lab">Key takeaway</div>

trace every request with OpenTelemetry GenAI spans carrying ids, versions, tokens, cost and latency; keep content out of telemetry and in a redacted, access-controlled store; watch cost, latency, errors and quality; and turn sampled failing traces into replayable eval cases.

</div>
