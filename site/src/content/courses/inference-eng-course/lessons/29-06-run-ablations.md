---
id: "29.06"
course: "inference-eng-course"
module: "29"
slug: "run-ablations"
title: "Run Ablations"
keyIdeas: ["Ablate each component to see its contribution.", "Vary key parameters (chunk size, draft length, block size) to see sensitivity.", "Check interactions: do components help more together?", "Ablations often show one part gives most of the benefit.", "Report ablations in a table with spread."]
question: "What is an ablation, and design one for a technique that combines three ideas."
answer: "An ablation removes or varies one component at a time to measure its individual contribution. For a technique combining A, B and C: run baseline, each of A, B and C alone, each pair, and all three, on the same workload, so you can see which components matter, whether they interact, and whether one gives most of the gain."
sources: [{"title": "S. Keshav: How to Read a Paper", "url": "https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf", "note": "Critical reading of evaluations."}, {"title": "SGLang paper", "url": "https://arxiv.org/abs/2312.07104", "note": "Example with component ablations."}]
tags: ["research"]
---

## Component ablation

| Config | A | B | C | Gain |
| --- | --- | --- | --- | --- |
| Baseline | | | | 1.0× |
| A only | ✓ | | | |
| B only | | ✓ | | |
| C only | | | ✓ | |
| A + B | ✓ | ✓ | | |
| All | ✓ | ✓ | ✓ | |

Many techniques bundle several ideas. An ablation turns them on one at a time and in combination, to see which ones carry the gain.

## Parameter sweeps

Pick the most important parameter and sweep it over a wide range. Plot gain against the parameter.

- **A flat region** means the technique is robust: most reasonable settings work.
- **A sharp peak** means fragile tuning: the headline number may only hold at one carefully chosen value.

<div class="ex">
<div class="lab">Worked example</div>

**a three-part technique**

A paper combines (A) a new scheduling policy, (B) a custom attention kernel and (C) a cache admission rule, for a 1.9× total gain.

The ablation shows A alone gives 1.7×, B alone 1.1×, C alone 1.05×, and all three 1.9×.

A gives most of the benefit and is a few hundred lines of scheduler code. B requires maintaining a custom CUDA kernel for a 10% gain. Many teams would adopt A alone.

</div>

## Why it matters

If one component gives 90% of the gain, you can adopt it alone with less complexity. If the components only help together, adopt them as a unit.

## Deep dive: ablations that add up

For a technique with parts ⟦m1154⟧, ⟦m1155⟧ and ⟦m1156⟧, measure all ⟦m1157⟧ on/off combinations if you can afford it, and report each part's effect in log space:

⟦M1153⟧

If ⟦m1158⟧ matches the full technique's ⟦m1159⟧ gain, the parts are independent. If the sum is much larger, they overlap: two parts fix the same bottleneck, and the paper's attribution to each one alone is overstated.

<div class="take">
<div class="lab">Key takeaway</div>

ablations tell you which parts of a technique matter. Often the simplest part gives most of the benefit.

</div>
