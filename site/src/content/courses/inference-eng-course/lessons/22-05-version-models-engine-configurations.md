---
id: "22.05"
course: "inference-eng-course"
module: "22"
slug: "version-models-engine-configurations"
title: "Version Models & Engine Configurations"
keyIdeas: ["A deployment is weights + tokenizer/template + quantization + engine + flags + sampling defaults.", "Pin weights by revision hash, not by name.", "Version all of it as one config artifact.", "Log the version id with every request so behavior changes can be traced.", "Changes to any part need the same testing as a model change."]
question: "What should a \"deployment version\" of an LLM service pin, beyond the model name?"
answer: "The exact weights revision (hash), tokenizer and chat template, quantization recipe and calibration, engine version and image hash, all engine flags (parallelism, scheduler, KV dtype, speculation), and default sampling parameters. Any of these can change outputs or performance, so they should be versioned together and recorded with every result and log."
sources: [{"title": "Hugging Face Hub: Model revisions", "url": "https://huggingface.co/docs/hub/models-downloading", "note": "Downloading a specific revision."}, {"title": "vLLM: Engine arguments", "url": "https://docs.vllm.ai/en/latest/configuration/engine_args.html", "note": "Flags to pin."}]
tags: ["versioning"]
---

## What to pin

| Item | Example |
| --- | --- |
| Weights | model id and commit hash |
| Tokenizer and chat template | from the same revision as the weights |
| Quantization | method, bits, calibration set id |
| Engine | version and image digest |
| Flags | the full flag set, not only the ones you changed |
| Sampling defaults | temperature, top-p, max tokens |
| Speculation | method and draft model revision |

## Store it as config

```
deploy_id: chat-v42
model: org/model@abc123
quant: fp8-w8a8 (calib: chat-mix-2k)
engine: vllm@<version> image@sha256:...
flags: {tp: 2, max_num_seqs: 256, kv_cache_dtype: fp8, ...}
sampling: {temperature: 0.7, top_p: 0.9, max_tokens: 2048}
```

Tag every request log line with deploy_id. When someone reports that answers got worse on Tuesday, you can see exactly what changed on Tuesday.

## Why every item matters

- **Engine upgrade:** can change numerics, defaults or kernels. Outputs at temperature 0 may differ token by token.
- **Chat template change:** can change output style, length or tool-call format, with the same weights.
- **Quantization recalibration:** can shift quality on a specific domain.
- **One flag:** for example max_model_len, changes which requests are accepted at all.

<div class="ex">
<div class="lab">Worked example</div>

**the template that changed behavior**

A team updates the model revision to pick up a tokenizer fix. The new revision also ships an updated chat template that adds a default system prompt.

Average output length rises 25%, and so does cost. The weights did not change, so nobody looked at the template.

Pinning the template separately, and diffing it on every revision change, would have caught it.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

version the whole serving stack (engine, kernels, config, template and model), and tag every request with that version.

</div>
