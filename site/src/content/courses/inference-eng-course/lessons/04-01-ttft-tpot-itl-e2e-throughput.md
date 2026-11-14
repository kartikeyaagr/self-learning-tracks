---
id: "04.01"
course: "inference-eng-course"
module: "04"
slug: "ttft-tpot-itl-e2e-throughput"
title: "TTFT, TPOT, ITL, E2E & Throughput"
keyIdeas: ["TTFT: request sent to first token received. Includes queueing and prefill.", "TPOT: average time per output token after the first, per request.", "ITL: each individual gap between tokens; a distribution, shows stalls.", "E2E latency: request sent to last token. Roughly TTFT + TPOT x (tokens - 1).", "Throughput: output tokens per second (or requests per second) across all requests. Say which."]
question: "A request has TTFT 300 ms, produces 200 tokens, and finishes at 4.3 s. Compute TPOT and explain how ITL differs from TPOT."
answer: "TPOT = (4.3 - 0.3) / (200 - 1), about 20 ms per token. ITL is the gap between each pair of consecutive tokens; TPOT is their average for the request. ITL keeps the distribution, so you can see stalls (for example a 400 ms gap when a long prefill ran) that the average hides."
sources: [{"title": "NVIDIA: LLM inference benchmarking fundamentals", "url": "https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/", "note": "Definitions of TTFT, ITL and throughput."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Benchmark tool reporting these metrics."}]
tags: ["metrics"]
---

## Why one number is never enough

"The server does 5,000 tokens per second" and "the model responds in 200 ms" are both incomplete. A streamed LLM response has a shape in time, and different products care about different parts of that shape. Benchmarking starts with naming those parts precisely.

~~~dg-flow The timeline of one streamed request.
Sent -(TTFT)-> First token -(ITL gaps)-> *Last token -> E2E = TTFT + decode time
~~~

## The five core metrics

- **TTFT (time to first token).** From the moment the client sends the request until the first token arrives. It includes network time, queueing, tokenization, prefill and the first sampling step. Users experience it as "it started answering".
- **TPOT (time per output token).** For one request: (E2E − TTFT) ÷ (output tokens − 1). It is the average streaming speed for that request.
- **ITL (inter-token latency).** Every individual gap between consecutive tokens. Where TPOT is one average per request, ITL is a distribution over all gaps. It reveals stalls that averages hide.
- **E2E latency.** Total time from send to the last token. It matters most for non-streaming calls, short completions and agent tool calls where the next step needs the full output.
- **Throughput.** Work completed per second across the whole server. Usually output tokens per second, sometimes requests per second.

## Throughput has several meanings

Tools disagree on what "throughput" means, and the differences are large.

- **Output tokens per second**: tokens generated across all requests, divided by wall-clock time. The most common meaning.
- **Total tokens per second**: input plus output tokens per second. With 2,000-token prompts and 200-token outputs, this is 11 times larger than output throughput. It flatters long-prompt workloads.
- **Requests per second**: completed requests per second. Useful when outputs are short and fixed.
- **Per-GPU throughput**: any of the above divided by the number of GPUs. Needed to compare setups of different sizes.

Always state which one you mean, and the latency at which it was measured.

<div class="ex">
<div class="lab">Worked example</div>

**computing the metrics for one request**

A request is sent at t = 0. Tokens arrive at 180 ms, 205 ms, 229 ms, 300 ms and 324 ms, then the stream ends.

TTFT = 180 ms. E2E = 324 ms. Output tokens = 5.

TPOT = (324 − 180) ÷ 4 = 36 ms.

The ITL values are 25, 24, 71 and 24 ms. The 71 ms gap is a stall, perhaps a long prefill from another user. TPOT averages it into 36 ms and hides it; ITL shows it.

</div>

## Which metric for which product

| Product | Primary metric | Secondary |
| --- | --- | --- |
| Chat interface | p95 TTFT | p95 TPOT, p99 ITL |
| Code completion in an editor | p95 E2E (short outputs) | TTFT |
| Agents and tool calls | E2E per call | total cost per task |
| Voice assistants | TTFT and the first sentence | ITL jitter |
| Offline batch jobs | throughput, cost per token | completion deadline |

A voice assistant is a good example of why the shape matters. Text-to-speech needs a few words before it can start talking, so the time to the first sentence matters more than TTFT alone. Steady ITL also matters, because a stall in the text becomes a pause in the voice.

## How human perception maps to the numbers

- People read about 4 to 6 words per second, or about 5 to 8 tokens per second. A TPOT under 50 ms (20 tokens per second) feels instant for reading.
- A TTFT under about 300 ms feels responsive. Above 1 second, users notice the wait. Above 3 seconds, many assume something broke.
- Stalls hurt more than slow, steady streams. A stream at 30 tokens per second with no pauses feels better than one at 60 tokens per second with a one-second freeze.

## Common measurement mistakes

- **Measuring TTFT without streaming.** If the client reads the whole response before timing, it records E2E and calls it TTFT.
- **Timing on the server only.** Server-side TTFT leaves out network time and any proxy buffering. Measure at the client too.
- **Counting chunks as tokens.** Some servers send several tokens in one SSE chunk. Count tokens from the usage report or by tokenizing the text, not by counting chunks.
- **Averaging TPOT across requests of very different lengths.** A 5-token answer and a 2,000-token answer get the same weight. Decide whether you care about requests or tokens, and weight accordingly.
- **Ignoring errors.** A request that failed after 30 s did not have good latency. It must count as a failure.

## Deep dive: the metrics as formulas

A request arrives at ⟦m348⟧. Its output tokens arrive at ⟦m349⟧. Every latency metric is a function of these ⟦m350⟧ timestamps:

⟦M345⟧

⟦M346⟧

For one request, TPOT is exactly the mean ITL. Across many requests they differ, because they are averaged over different things: TPOT gives each request one vote, and pooled ITL gives each token one vote.

<div class="ex">
<div class="lab">Worked example</div>

**two requests, two different "averages"**

Request A: 10 tokens at 20 ms TPOT. Request B: 1,000 tokens at 10 ms TPOT.

Mean TPOT (per request) ⟦m351⟧ ms.

Mean ITL (per token) ⟦m352⟧ ms.

Neither is wrong, but a dashboard that mixes them can show a 50% "regression" when nothing changed except the mix of short and long answers.

</div>

~~~dg-raw
%%timeline Anatomy of one streamed request (ms). TTFT covers queueing, prefill and the first sample; after that, each tick is one ITL.
unit = ms
tick = 50
Server | 0-40 queue c4 ; 40-170 prefill c1 ; 170-180 sample gold ; 180-324 decode c3
Client | 0-180 waiting flame ; 180-182 t1 go ; 205-207 t2 go ; 229-231 t3 go ; 300-302 t4 go ; 322-324 t5 go
%%
~~~

## Deep dive: three throughputs from one log

Over a measurement window of length ⟦m353⟧ containing completed requests ⟦m354⟧ with ⟦m355⟧ prompt and ⟦m356⟧ output tokens:

⟦M347⟧

⟦m357⟧ flatters prompt-heavy workloads: a RAG server with 4,000-token prompts and 200-token answers reports 21 times its output rate. Always say which one you mean.

<div class="take">
<div class="lab">Key takeaway</div>

TTFT measures responsiveness, TPOT and ITL measure streaming speed, E2E measures whole calls, and throughput measures capacity. Name the metric, its percentile and the load it was measured at, every time.

</div>
