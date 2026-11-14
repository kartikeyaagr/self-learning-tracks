---
id: "29.02"
course: "inference-eng-course"
module: "29"
slug: "choose-a-paper-reconstruct-the-baseline"
title: "Choose a Paper & Reconstruct the Baseline"
keyIdeas: ["Choose a paper with a clear mechanism, available code or enough detail, and feasible hardware.", "Reconstruct the baseline exactly: engine version, flags, model, hardware class.", "Match the paper's baseline numbers within a tolerance before going further.", "If you cannot match, document why (hardware, missing details).", "Also run a modern baseline to see if the gap still exists today."]
question: "Why is reconstructing the paper's baseline the most important step of a reproduction?"
answer: "The headline result is a ratio against a baseline. If your baseline differs (version, flags, hardware), the ratio is meaningless: you might show a large gain against a weak baseline or none against a strong one. Matching the paper's baseline numbers first proves your setup is comparable before you test the technique."
sources: [{"title": "ML Reproducibility Challenge", "url": "https://reproml.org/", "note": "Community reproduction practice."}, {"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Common baseline engine."}]
tags: ["research"]
---

## Picking a paper

| Criterion | Why |
| --- | --- |
| Clear mechanism | easier to implement and isolate |
| Code available | saves weeks |
| Hardware within reach | you can actually run it |
| Headline result in a measurable metric | you can check it |

Good candidates:

- **Chunked prefill** (Sarathi-Serve): one clear mechanism, now in most engines.
- **Prefix sharing** (SGLang's RadixAttention): easy to build a workload that shows it.
- **Speculative decoding variants**: measurable on one GPU.
- **KV quantization** (KIVI): memory and quality, both measurable.
- **Disaggregation** (DistServe), at small scale on 2 GPUs.

## Reconstruct the baseline

1. **Extract every setup detail** from the paper and its appendix: model, GPU, engine version, flags, dataset, rates.
2. **Install the baseline engine version** they used, if possible.
3. **Rebuild their workload:** datasets, lengths and arrival rates.
4. **Run it** and compare your baseline numbers with theirs.

| Metric | Paper baseline | Yours | Difference |
| --- | --- | --- | --- |
| Throughput | | | |
| p99 latency | | | |

<div class="ex">
<div class="lab">Worked example</div>

**why the baseline comes first**

A paper reports the baseline at 1,800 tokens/s and the technique at 4,100 tokens/s, a 2.3× gain.

Your baseline on similar hardware reaches 2,900 tokens/s, because your engine version is newer. Your technique run reaches 4,300 tokens/s.

Your gain is 1.5×, not 2.3×. The technique still helps, but part of the paper's gain came from a weaker baseline. Without measuring the baseline first, you might conclude the technique was implemented wrongly.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

reproduce the baseline before the technique. Without a matching baseline, a reproduced speedup proves nothing.

</div>
