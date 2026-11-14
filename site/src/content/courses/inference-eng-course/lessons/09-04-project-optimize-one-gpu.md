---
id: "09.04"
course: "inference-eng-course"
module: "09"
slug: "project-optimize-one-gpu"
title: "Project: Optimize One GPU"
keyIdeas: ["Goal: maximize SLO-safe capacity of one GPU for a fixed workload.", "Start from the tunable baseline; change one variable per experiment.", "Track each change's effect in a table.", "Explain gains with engine metrics, verify quality where precision changed.", "End with a capacity statement including headroom."]
question: "What would a convincing before-and-after report for single-GPU optimization contain?"
answer: "The baseline and final configs (full flags), the workload and SLO, a goodput vs load chart for both, a table of each change with its individual effect, engine metrics explaining each gain, quality checks for any precision change, and the final SLO-safe capacity with planned headroom."
sources: [{"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Tuning reference."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Sweep tool."}]
tags: ["project"]
tools: ["Project"]
---

## The task

You are given a model, one GPU, a workload and an SLO. Raise the maximum request rate this GPU can serve while meeting the SLO, without hurting quality beyond an agreed threshold. This project applies the whole method of Modules 04 to 09.

## The process

1. **Baseline.** Follow lesson 09.01: pinned config, three load points, binding constraint, quality baseline, SLO-safe capacity.
2. **Hypothesis.** Pick the change most likely to relieve the binding constraint, and write down which metric it should move and by roughly how much.
3. **Change one thing.** Apply it.
4. **Measure.** Rerun the expected load point and a fine rate sweep near the knee. Rerun the quality check if outputs could change.
5. **Decide.** Keep the change if SLO-safe capacity rose and quality held. Revert it otherwise, and note why.
6. **Repeat** from step 2 with the new binding constraint.

Changing several things at once is faster but leaves you unable to say which change helped, or whether one helped and another hurt.

## Candidate changes, by constraint

- **Memory**: FP8 KV cache, higher memory utilization, FP8 weights.
- **TTFT**: prefix caching if prompts share prefixes, a larger token budget, ordering prompts so shared content comes first.
- **TPOT**: a lower max_num_seqs, FP8 weights for faster decode, speculative decoding at low load.
- **ITL tails**: a smaller token budget, chunked prefill.
- **Overhead at low batch**: confirm CUDA graphs are on and cover your batch sizes.

## The change log

The change log is the real deliverable. It shows what you tried, what happened and why.

| # | Change | SLO-safe req/s | Change | Why |
| --- | --- | --- | --- | --- |
| 0 | baseline | 5.0 | | KV full at peak, preemptions |
| 1 | FP8 KV cache | 6.8 | +36% | preemptions gone, queue shorter |
| 2 | token budget 2,048 | 7.4 | +9% | p95 TTFT improved for long prompts |
| 3 | max_num_seqs 384 | 7.9 | +7% | larger decode batch fits now |
| 4 | FP8 weights | 7.6 | −4% | slower kernel on this GPU, reverted |

Row 4 is as useful as the others. It records that a plausible change did not help here, so nobody tries it again without a new reason.

<div class="ex">
<div class="lab">Worked example</div>

**explaining a result with metrics**

After change 1, a reviewer asks why capacity rose 36% when the KV pool doubled.

The answer from the metrics: at 6.8 requests per second, KV usage peaks at 74%, so memory is no longer binding. The new limit is p95 TPOT, which reaches 38 ms against a 40 ms SLO as the batch grows.

So the pool doubled, but the TPOT SLO capped how much of it could be used. That also explains why change 3, a larger batch, only gained 7%.

</div>

## Deliverables

- **Config diffs** from the baseline to the final config.
- **The change log**, including reverted changes.
- **Before and after charts**: p95 TTFT, p95 TPOT and goodput against request rate.
- **Quality results** for every change that could affect outputs.
- **The final capacity statement**, in the one-line form from lesson 09.03, with recommended headroom.
- **A next step**: which constraint now binds, and what you would try next with more time.

## What good looks like

A strong result is not the largest number. It is a number someone else can reproduce from your config and workload, with each gain explained by a metric, and with quality checked. A 40% improvement with a clear change log is worth more than a 2 times claim nobody can rerun.

## Deep dive: predict before you measure

For each change in the tuning log, write the expected effect from the formulas first, then measure. A large gap between prediction and result means either the model or the measurement is wrong, and both are worth knowing.

| Change | Predicted | Measured | Explained? |
| --- | --- | --- | --- |
| FP8 KV | KV pool 2×, decode step at batch 128 and 2k context −32% | Pool 2.0×, step −28% | Yes, within 15% |
| max-num-seqs 128 → 192, 1k context | Throughput +21%, p95 TPOT +24% | +18%, +29% | Yes |
| Prefix caching on | TTFT −60% (80% of a prefill that is 75% of TTFT) | −12% | No: hit rate was 18%, prompts differed in a timestamp at the top |

The third row is the useful one. The formula said 80% of prefill should disappear. The measurement said it did not. The reason turned out to be a timestamp at the start of the system prompt, which changed every block hash after it (08.04). Moving the timestamp to the end gave the predicted gain.

⟦M597⟧

Check ⟦m598⟧ from the engine's prefix-cache metrics before trusting any caching result.

<div class="take">
<div class="lab">Key takeaway</div>

optimize one GPU methodically: baseline, one change at a time, measured against the SLO, explained with metrics and checked for quality. The change log, including the changes you reverted, is the real result.

</div>
