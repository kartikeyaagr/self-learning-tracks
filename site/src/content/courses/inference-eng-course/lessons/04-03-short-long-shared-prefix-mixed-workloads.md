---
id: "04.03"
course: "inference-eng-course"
module: "04"
slug: "short-long-shared-prefix-mixed-workloads"
title: "Short, Long, Shared-Prefix & Mixed Workloads"
keyIdeas: ["Workload shape (input length, output length, prefix sharing) changes results more than most tuning.", "Short in, long out stresses decode. Long in, short out stresses prefill.", "Shared-prefix workloads reward prefix caching; random prompts get no benefit.", "Mixed workloads expose interference: long prefills stall decode for everyone.", "Best: sample lengths from your own production logs."]
question: "Why can a benchmark of 128-token prompts with 128-token outputs mislead you about a RAG product, and what workload would you use instead?"
answer: "RAG prompts are long (often 2k to 8k tokens of retrieved context) with short outputs, so prefill dominates and TTFT is the key metric; a short-in, short-out benchmark mostly measures decode and hides prefill cost and interference. Use prompt and output lengths sampled from real logs, include shared system prompts so prefix caching is exercised, and test a mix with chat traffic."
sources: [{"title": "ShareGPT dataset (on Hugging Face)", "url": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered", "note": "Real chat lengths often used in benchmarks."}, {"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "Supports random, ShareGPT and prefix workloads."}]
tags: ["workloads"]
---

## The workload decides the winner

An engine configuration that looks best on one workload can look worst on another. The two things that change results most are the lengths of inputs and outputs, and how much of each prompt is shared with other requests. Benchmark the workload you will serve, or your numbers will not predict production.

## The basic shapes

| Shape | Example product | Main bottleneck | Metric to watch |
| --- | --- | --- | --- |
| Short in, short out | classification, routing, autocomplete | scheduling and launch overhead | requests/s, E2E |
| Short in, long out | creative writing, chat answers | decode bandwidth | TPOT, output tok/s |
| Long in, short out | RAG, summarization, extraction | prefill compute | TTFT |
| Long in, long out | document rewriting, reasoning models | prefill, decode and KV memory | everything |
| Shared prefix | one system prompt for all users, few-shot templates | prefix cache hit rate | TTFT with a warm cache |
| Mixed | a real product | interference between phases | p99 ITL, p95 TTFT |

## How shape changes the bottleneck

- **Short in, short out.** Each request is cheap, so fixed costs dominate: HTTP parsing, tokenization, scheduler overhead, kernel launches. A Python-heavy engine may be the limit, not the GPU.
- **Short in, long out.** Almost all time is decode. Throughput depends on how large a batch you can run, and KV memory caps that.
- **Long in, short out.** Almost all time is prefill. The GPU is compute bound, and TTFT grows with prompt length. Prefix caching and FP8 compute help most.
- **Long in, long out.** Every request holds a large KV cache for a long time, so concurrency is low and memory is the limit.

<div class="ex">
<div class="lab">Worked example</div>

**the same engine, two workloads**

An 8B model on one H100, benchmarked at 16 requests per second.

Workload A: 200 tokens in, 200 out. p95 TTFT 60 ms, p95 ITL 18 ms. Everyone is happy.

Workload B: the same, except 5% of requests carry a 30,000-token document. Each such prefill takes about 1.5 s. Without chunked prefill, every decoding stream stalls during it. p99 ITL jumps to 1.4 s, and p95 TTFT for short requests rises to 900 ms because they queue behind the long ones.

Same engine, same rate, very different user experience.

</div>

## Prefix sharing

Many products send nearly identical prompts: a long system prompt, a set of tool definitions, few-shot examples, or earlier turns of the same conversation. With prefix caching, the engine stores the KV cache of those shared parts and skips their prefill.

- A benchmark with a shared 2,000-token system prompt can show a 5 to 10 times TTFT improvement with prefix caching on.
- A benchmark with random prompts shows no improvement at all.

Both results are true. Only one of them describes your product. Measure your real prefix sharing: what fraction of requests share a prefix, and how long that prefix is.

## Building a realistic workload

1. **Collect lengths.** Pull input and output token counts from production logs. If you have none, use a public conversation dataset such as ShareGPT, and say so.
2. **Fit distributions, not averages.** Real lengths are skewed: many short requests and a long tail. A workload where every prompt is the mean length hides the tail that causes trouble.
3. **Measure prefix sharing.** Count shared prefixes and their lengths, including multi-turn conversations, where turn n contains all of turns 1 to n − 1.
4. **Keep the mix.** For example 70% chat, 20% RAG with 8k-token contexts and 10% long documents.
5. **Fix the seed.** The same workload file and seed should produce exactly the same sequence of requests every run.

## Controlling output length

Output length depends on the model and the sampling. If two engines sample slightly differently, one may produce longer answers and look slower per request. Two ways to control this:

- **Fixed lengths.** Set `max_tokens` to the target and use `ignore_eos` (supported by vLLM, SGLang and most benchmark tools) so every request generates exactly that many tokens. Best for comparing engines and settings.
- **Natural lengths.** Let the model stop on its own. Best for estimating production cost, but then compare per token, not per request.

## Deep dive: what a request costs, split into phases

On one GPU, a request's prefill takes about ⟦m380⟧ of exclusive compute time. Its decode shares each step with ⟦m381⟧ others, so each output token costs about ⟦m382⟧ of GPU time:

⟦M377⟧

For Llama 3 8B on an H100 at 50% of peak, prefill is about 30 µs per prompt token. At batch 64 and 13 ms steps, decode is about 0.2 ms per output token.

~~~dg-raw
%%bars Approximate GPU time per request by workload shape, Llama 3 8B, H100, batch 64 (ms).
unit = ms
stack = Prefill, Decode
Chat (500 in, 500 out) | 15+100
Code completion (1,500 in, 50 out) | 45+10
RAG (4,000 in, 200 out) | 120+40
Summarization (8,000 in, 300 out) | 240+60
Agent step (12,000 in, 150 out) | 360+30
%%
~~~

Chat is decode-bound and benefits from bigger batches. Agent and RAG traffic is prefill-bound and benefits from prefix caching and chunked prefill. A benchmark built on chat-shaped prompts says little about an agent workload.

## Deep dive: the arithmetic of prefix sharing

If a fraction ⟦m383⟧ of each prompt is a cached prefix, only ⟦m384⟧ tokens need prefill:

⟦M378⟧

<div class="ex">
<div class="lab">Worked example</div>

**a shared 2,000-token system prompt**

Prompts are 2,000 tokens of system prompt plus 500 of user text. ⟦m385⟧, so prefill work drops 5 times: from about 75 ms to 15 ms on the setup above.

A benchmark that randomizes every prompt measures ⟦m386⟧ and understates this server by 60 ms of TTFT per request. One that repeats an identical prompt measures ⟦m387⟧ and overstates it.

</div>

## Deep dive: output length sets slot occupancy

A sequence holds a batch slot for ⟦m388⟧ steps. With output lengths drawn from a heavy-tailed distribution, the mean slot time is set by the tail:

⟦M379⟧

If 5% of answers are 4,000 tokens and the rest 300, ⟦m389⟧. The rare long answers fill 41% of all slot time. Fix output lengths in a benchmark (with ignore-EOS and a max-tokens cap) or sample them from real logs. Never leave them to chance.

<div class="take">
<div class="lab">Key takeaway</div>

benchmark the workload you will serve. Input and output lengths, their distributions and prefix sharing change results more than most engine flags, and averages hide the tails that matter.

</div>
