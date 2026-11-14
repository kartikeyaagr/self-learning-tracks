---
id: "23.08"
course: "inference-eng-course"
module: "23"
slug: "lab-hold-an-slo-under-changing-traffic"
title: "Lab: Hold an SLO Under Changing Traffic"
keyIdeas: ["Replay a trace: slow ramp up, plateau, sudden 2x spike, ramp down.", "Run with static peak capacity, reactive autoscaling, and autoscaling plus warm pool.", "Measure SLO attainment per minute and GPU-hours.", "Tune thresholds, headroom and scale-down delay.", "Report the cost vs SLO trade-off of each policy."]
question: "How do you judge whether your autoscaling setup passed the lab?"
answer: "Replay a traffic trace with a ramp and a spike, and measure SLO attainment over time (for example per minute), total GPU-hours used, and cold starts. It passes if SLO attainment stays above target (say 99%) through the trace except a bounded window during the spike, with GPU-hours lower than a static deployment sized for peak."
sources: [{"title": "KEDA", "url": "https://keda.sh/", "note": "Autoscaling on custom metrics."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Load generation."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Compare autoscaling policies on the same changing traffic, and judge them on SLO attainment and GPU-hours together.

## Trace

| Phase | Duration | Rate |
| --- | --- | --- |
| Ramp | 10 min | 1× to 3× |
| Plateau | 10 min | 3× |
| Spike | 5 min | 6× |
| Ramp down | 10 min | 6× to 1× |

## Policies

| Policy | Description |
| --- | --- |
| Static | sized for the 6× peak all the time |
| Reactive | autoscale on queue and KV |
| Reactive plus warm pool | the same, plus N warm standby replicas |

## Results table

| Policy | SLO attainment | Worst minute | GPU-hours |
| --- | --- | --- | --- |
| Static | | | |
| Reactive | | | |
| Reactive plus warm | | | |

Plot SLO attainment and replica count over time on one chart.

## What to expect

- **Static** has the best attainment and by far the most GPU-hours.
- **Reactive** handles the ramp well but misses the SLO during the spike, for about as long as one cold start.
- **Reactive plus warm** holds the spike for a modest extra cost. Choose N from the spike's size within one cold-start window.

<div class="take">
<div class="lab">Key takeaway</div>

judge autoscaling by SLO attainment over time and by GPU-hours spent. Warm capacity is usually what saves the spike.

</div>
