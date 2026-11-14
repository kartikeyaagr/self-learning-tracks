---
id: "29.04"
course: "inference-eng-course"
module: "29"
slug: "reproduce-the-main-result"
title: "Reproduce the Main Result"
keyIdeas: ["Implement or enable the technique on your reconstructed baseline.", "Run the paper's main experiment with repeated trials.", "Compare direction, magnitude and conditions with the paper.", "Explain any gap with specific differences.", "Report confidence intervals, not single runs."]
question: "Your reproduced speedup is 1.6 times where the paper claims 2.3 times. How do you decide if this counts as a successful reproduction?"
answer: "Check whether the direction and rough magnitude hold and whether the gap can be explained: different hardware, a stronger baseline, workload differences, or missing tuning. If you can trace the gap to a documented difference and the mechanism still shows a clear, repeatable gain, it is a partial reproduction worth reporting. If the gain disappears entirely under matched conditions, report that too."
sources: [{"title": "ML Reproducibility Challenge", "url": "https://reproml.org/", "note": "Reporting standards."}, {"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Example of a reproducible systems result."}]
tags: ["research"]
---

## Running it

1. **Same workload, same baseline config,** with the technique turned on.
2. **At least 3 runs of each,** interleaved (baseline, technique, baseline, technique...), so that drift in the machine affects both equally.
3. **Report the mean and the spread.**

## Comparing

| Result | Paper | Yours |
| --- | --- | --- |
| Main metric gain | 2.3× | 1.6× (1.5 to 1.7 across runs) |
| Conditions | A100, engine version X | H100, engine version Y |

## Explaining gaps

| Possible cause | How to check |
| --- | --- |
| Hardware | the ratio of bandwidth to compute differs between GPUs |
| A stronger baseline | a newer engine already includes part of the gain |
| Workload | the length distribution differs |
| Tuning | the paper's flags were not documented |

<div class="ex">
<div class="lab">Worked example</div>

**a hardware explanation**

A technique cuts memory traffic during decode. On an A100, decode is strongly memory bound, so cutting traffic gives a large speedup.

An H100 has about 3 times the compute of an A100 but only about 1.6 times the bandwidth. At the same batch, decode is even more memory bound there, so you might expect a bigger gain, not a smaller one.

If your gain is smaller, the hardware is not the explanation. Look at the baseline and the workload instead.

</div>

## Verdicts

- **Reproduced:** within the paper's stated variance.
- **Partially reproduced:** a clear gain, but smaller, with the differences explained.
- **Not reproduced:** no significant gain under matched conditions.

## Deep dive: did the reproduction agree?

Compare speed-ups, not absolute numbers, because hardware and software versions differ. With the paper's speed-up ⟦m1149⟧ and yours ⟦m1150⟧, each with a 95% interval from repeated runs:

⟦M1147⟧

Papers rarely report run-to-run variation. When they do not, assume at least 3 to 5% per reported number, and treat ⟦m1151⟧ between about 0.85 and 1.15 as agreement.

If ⟦m1152⟧ is outside that band, predict it before guessing. A technique that removes a memory-bound cost scales with bandwidth; one that removes compute scales with FLOPs; one that removes host overhead shrinks as GPUs get faster:

⟦M1148⟧

A paper measured on A100s, where host overhead was a smaller share of a slower step, will often show a smaller gain on H100s for host-side optimizations, and a similar or larger gain for bandwidth-side ones.

<div class="take">
<div class="lab">Key takeaway</div>

a reproduction compares direction, magnitude and conditions. Explain gaps with specific differences and report the spread across repeated runs.

</div>
