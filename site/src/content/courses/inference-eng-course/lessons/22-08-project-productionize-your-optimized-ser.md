---
id: "22.08"
course: "inference-eng-course"
module: "22"
slug: "project-productionize-your-optimized-server"
title: "Project: Productionize Your Optimized Server"
keyIdeas: ["Take your optimized single-GPU config from Module 09 to a production-shaped deployment.", "Containerize, externalize weights and config, add probes and warm-up.", "Add graceful drain, metrics and dashboards.", "Run a rolling update under load and confirm no failed requests.", "Document cold start time and capacity."]
question: "List the production-readiness checks your server must pass."
answer: "Pinned, versioned config and image; weights cached for fast start; startup, readiness and liveness probes with warm-up; graceful drain within the grace period; metrics exported and dashboards built; request logging with version id; load test at expected and peak load meeting the SLO; a canary and rollback procedure tested; and a cold start time measured and documented."
sources: [{"title": "Kubernetes: Liveness, readiness and startup probes", "url": "https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/", "note": "Probes."}, {"title": "vLLM: Production stack", "url": "https://github.com/vllm-project/production-stack", "note": "Reference deployment for vLLM."}]
tags: ["project"]
tools: ["Project"]
---

## Goal

Take the optimized server from earlier modules and make it production-ready: reproducible, observable, drainable and tested under a live rollout.

## Checklist

| Item | Done when |
| --- | --- |
| Image | pinned versions, digest recorded |
| Config | in git, versioned, tagged with a deploy id |
| Weights | cached near the GPU; cold start measured per phase |
| Probes | startup probe, readiness after warm-up, simple liveness |
| Drain | a rolling update under load drops zero requests |
| Metrics | engine and GPU metrics on a dashboard |
| Logs | request id, deploy id, tokens in and out, latency |
| Capacity | SLO-safe request rate documented, with headroom |
| Rollback | tested; the old version stays warm during rollout |

## The rolling update test

1. Run your load harness at expected production load.
2. Trigger a rolling update to a new deploy id.
3. Record errors, TTFT p99 during the rollout, and the number of streams that did not finish.

Success means no errors, a TTFT p99 spike within a bound you state in advance (for example under 2 times the normal value), and every stream completed.

## Common findings

- Readiness flips before warm-up finishes: TTFT spikes as each new pod takes traffic.
- The grace period is shorter than the longest streams: a few streams are cut off.
- The rollout replaces pods faster than new ones warm up: capacity dips and queues build.

<div class="take">
<div class="lab">Key takeaway</div>

production-ready means reproducible, observable, drainable and tested under a live rollout. Speed alone does not count.

</div>
