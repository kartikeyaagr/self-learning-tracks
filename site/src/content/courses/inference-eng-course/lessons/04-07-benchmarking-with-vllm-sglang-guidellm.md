---
id: "04.07"
course: "inference-eng-course"
module: "04"
slug: "benchmarking-with-vllm-sglang-guidellm"
title: "Benchmarking With vLLM, SGLang & GuideLLM"
keyIdeas: ["Use an existing tool before writing your own; they handle streaming timing and arrival patterns.", "vLLM and SGLang ship serving benchmarks that hit any OpenAI-compatible server.", "GuideLLM sweeps rates automatically and reports TTFT, ITL and throughput per rate.", "Point all tools at the same endpoint type with the same dataset to compare engines.", "Check the tool's definitions: some count input tokens in throughput."]
question: "You want to compare two engines fairly with an existing tool. What settings must match across both runs?"
answer: "Same model weights and dtype, same tokenizer and chat template, same dataset with identical prompt and output lengths (fixed output via ignore_eos or max_tokens), same arrival pattern and rate or concurrency, same hardware, same warm-up and duration, and the same client tool measuring through the same OpenAI-compatible endpoint. Also align prefix caching and sampling settings."
sources: [{"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "benchmark_serving and related scripts."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Sweeps rates and reports SLO-oriented metrics."}, {"title": "SGLang benchmark docs", "url": "https://docs.sglang.ai/", "note": "bench_serving and engine docs."}]
tags: ["tools"]
---

## Why use an established tool

Writing a load generator looks easy. Doing it correctly is hard: timestamping each SSE chunk as it arrives, generating Poisson arrivals on the clock, computing percentiles correctly, handling errors, and parsing different response formats. Established tools have solved these problems, and their results are what other people will compare yours with.

## The common tools

| Tool | Good for | Notes |
| --- | --- | --- |
| vLLM benchmark (vllm bench serve) | quick rate or concurrency runs | random, ShareGPT, prefix and custom datasets |
| SGLang bench_serving | similar, with SGLang-friendly defaults | also works against vLLM and other OpenAI endpoints |
| GuideLLM | automatic rate sweeps and SLO views | finds the knee by sweeping rates |
| NVIDIA GenAI-Perf and AIPerf | Triton, TensorRT-LLM, Dynamo | detailed token-level metrics |
| LLMPerf | multi-provider API comparisons | useful for hosted APIs |

## A typical run

```
vllm bench serve --backend openai-chat \
  --base-url http://localhost:8000 --endpoint /v1/chat/completions \
  --model qwen2.5-7b --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --ignore-eos \
  --request-rate 8 --num-prompts 2000 --seed 0
```

Flags change between versions. Check `--help` for the version you have, and save the exact command with the results.

## Reading the output

A typical summary reports request throughput, output token throughput, total token throughput, and mean, median and p99 for TTFT, TPOT and ITL. Check:

- **Which throughput is which.** Total token throughput includes input tokens and will look much larger.
- **Whether p99 means over requests or over tokens.** For ITL it is over individual gaps.
- **How many requests failed.** Some tools report failures separately and exclude them from latency.
- **Whether `ignore_eos` was on.** Without it, output lengths vary and so does everything else.

## Comparing engines fairly

When you compare vLLM with SGLang or TensorRT-LLM, keep these identical:

- The model weights and dtype, including quantization.
- Tensor-parallel size and GPU type.
- Maximum context length and any memory fraction.
- Prefix caching on or off.
- The workload file, seed, rate and duration.
- The client tool and its version.

Then run each engine with its recommended settings for that workload, and also with matched settings. Report both, because "engine X with defaults" and "engine X tuned" are different claims.

<div class="ex">
<div class="lab">Worked example</div>

**a misleading comparison**

A blog post claims engine A is twice as fast as engine B.

A closer look shows engine A had prefix caching on, while engine B had it off, and the workload used one shared 3,000-token system prompt.

With prefix caching on for both, the gap shrank to 7%. Matching settings is most of the work in a fair comparison.

</div>

## Pitfalls

- **Client CPU.** One Python process can become the bottleneck above a few hundred concurrent streams. Watch its CPU usage, or run several clients.
- **Co-located clients.** Running the client on the GPU host steals CPU from the engine's scheduler and tokenizer.
- **Tokenizer mismatch.** Some tools count tokens with their own tokenizer. Make sure it is the model's tokenizer, or token counts and throughput will be off.
- **Network path.** Benchmarking through a load balancer, gateway or VPN adds latency and sometimes buffering. Know which path you measured.

## Deep dive: turning benchmark numbers into cost

For one GPU costing ⟦m417⟧ dollars per hour at throughput ⟦m418⟧ output tokens per second:

⟦M416⟧

| Setup | Output tok/s | Cost per M tokens at \$2.50/h |
| --- | --- | --- |
| Batch 1, plain generate | 40 | \$17.36 |
| Batch 1, engine | 150 | \$4.63 |
| Batch 64, engine | 4,900 | \$0.14 |
| Batch 64, engine, FP8 KV | 6,500 | \$0.11 |

The batch-1 and batch-64 rows differ by 33 times. A comparison of two engines that does not state batch size, load mode and SLO cannot be converted into cost.

<div class="ex">
<div class="lab">Worked example</div>

**the ignore-EOS trap**

Engine A is run with ignore-EOS and max tokens 500, so it always produces 500 tokens. Engine B stops at EOS, averaging 180 tokens.

B finishes each request 2.8 times sooner and reports more requests per second, with lower E2E. It has done 64% less work. Compare token throughput at fixed output length, or nothing.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

start with an established benchmark tool, match every setting across engines, save the exact command, and read each tool's metric definitions before comparing numbers.

</div>
