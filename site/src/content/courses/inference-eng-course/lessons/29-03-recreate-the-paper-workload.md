---
id: "29.03"
course: "inference-eng-course"
module: "29"
slug: "recreate-the-paper-workload"
title: "Recreate the Paper Workload"
keyIdeas: ["Workloads drive results; recreate them precisely.", "Pin dataset version, filtering, prompt construction and tokenization.", "Match output length handling and arrival process.", "Plot length distributions and compare with the paper's figures.", "Save the exact workload file for others."]
question: "A paper evaluates on \"ShareGPT at 4 requests per second.\" What details must you pin down to recreate this workload?"
answer: "Which ShareGPT version and filtering, how conversations were turned into prompts (first turn only or full history), tokenizer used to compute lengths, any length truncation or filtering, output length handling (use dataset output lengths with ignore_eos or let the model decide), number of requests, arrival distribution (Poisson with which seed), and warm-up."
sources: [{"title": "ShareGPT dataset (on Hugging Face)", "url": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered", "note": "Commonly used dataset."}, {"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "Common workload generators."}]
tags: ["research", "workloads"]
---

## Details to pin

| Detail | Example choices |
| --- | --- |
| Dataset | which ShareGPT version, filtered by language or length |
| Prompt | first user turn only, or the full conversation |
| Lengths | counted with the model's own tokenizer |
| Truncation | capped at the context limit |
| Output | the dataset's reply length with ignore_eos, or free generation |
| Arrival | Poisson at rate R, with a fixed seed |
| Count | number of requests, or a duration |

Each of these choices can change the result by tens of percent. Papers often leave some unstated, so record what you chose and why.

## Validate

Plot histograms of input and output lengths. Compare them with any distribution the paper reports: mean, median, maximum.

<div class="ex">
<div class="lab">Worked example</div>

**one detail, a big difference**

A paper uses ShareGPT with outputs fixed to the dataset's reply lengths, with a mean of about 340 tokens.

Your first attempt lets the model generate freely until it stops. The mean output length comes out at 610 tokens.

Longer outputs mean more decode, more KV per request and fewer concurrent requests. A scheduling technique that helps prefill will look much weaker on your workload. Fixing output lengths to the dataset's brings your baseline close to the paper's.

</div>

## Publish the workload

Save the workload as a file: prompt text or token ids, target output lengths and arrival times. Anyone can then rerun exactly what you ran.

<div class="take">
<div class="lab">Key takeaway</div>

most reproduction gaps come from workload differences. Pin every detail and compare length distributions with the paper.

</div>
