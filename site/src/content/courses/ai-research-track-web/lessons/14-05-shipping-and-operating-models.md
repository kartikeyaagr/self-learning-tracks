---
id: "14.05"
course: "ai-research-track-web"
module: "14"
slug: "shipping-and-operating-models"
title: "Shipping and operating models"
keyIdeas: ["A shippable model is more than weights: tokenizer, config, inference code, a pinned environment (an image digest) and a declared input and output contract, all tied to one registry version.", "Release patterns answer different questions. Shadow asks whether the new version runs correctly on real traffic; canary asks whether it is safe, in minutes; an A/B test asks whether it is better, over days; blue/green and alias flips make rollback a pointer move.", "Service metrics (latency, errors, throughput, saturation) detect a bad release within minutes; model metrics (output distribution, drift, quality on labelled samples) take hours to weeks. Gate the canary on the first and decide the launch on the second.", "Every alert needs a playbook whose first step is mitigation: roll the alias back, fall back to the previous model or a heuristic, then diagnose.", "Retraining is triggered by schedule, drift or quality loss, and its output is a candidate that goes through the same gates and rollout as any other release."]
question: "You are replacing a production model behind an API serving 2,000 requests per second with an error rate of 0.2 percent. The new version passed offline evals. Design the rollout, say how quickly a canary on 5 percent of traffic can detect the error rate doubling, and explain why the canary cannot tell you whether the new model improves a 5 percent conversion rate by 1 percent relative."
answer: "Package the new version as a registry version with a pinned image digest, run it in shadow on mirrored traffic to compare outputs and catch crashes, then canary it on 5 percent with automatic rollback on guardrails, then ramp through an A/B test, keeping the old version warm so rollback is an alias or traffic flip. Detection: the canary sees 100 requests per second. Testing 0.4 against a known 0.2 percent at three standard errors needs $n$ with $0.002\\sqrt{n} / \\sqrt{0.002} \\ge 3$, about 4,500 requests, or 45 seconds. Conversion is different: a 1 percent relative change on 5 percent is 0.0005 absolute, and a standard rule of thumb of $16p(1-p)/\\delta^2$ gives about three million users per arm. That takes an A/B test with a large share of traffic over days or weeks, so the canary gates on service health and the A/B test decides on value."
rubric: ["Includes a shadow stage, a canary with automatic rollback, and an A/B or ramp stage", "Keeps the previous version warm and makes rollback a pointer or traffic change", "Computes canary detection time from the request rate and error rates", "Explains with a sample-size estimate why quality and business metrics need an A/B test", "Separates service-health guardrails from model-quality decisions"]
sources: [{"title": "Google SRE Workbook: Canarying Releases", "url": "https://sre.google/workbook/canarying-releases/", "note": "Canary population size, duration, metric selection and comparison against a control."}, {"title": "Shankar et al.: Operationalizing Machine Learning: An Interview Study (2022)", "url": "https://arxiv.org/abs/2209.09125", "note": "How practitioners deploy, monitor and respond to failures of production models."}, {"title": "Sculley et al.: Hidden Technical Debt in Machine Learning Systems (NeurIPS 2015)", "url": "https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "Why the model is a small part of a production ML system, and where the debt accumulates."}, {"title": "Kubernetes documentation: Deployments", "url": "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/", "note": "Rolling updates, revision history and rolling back, as one example of the mechanics."}]
practice: ["canary-mann-whitney", "shadow-deployment-comparison", "model-registry-promotion", "retraining-trigger-policy", "consecutive-breach-alerts"]
tags: ["MLOps", "deployment", "monitoring", "incident response"]
tools: ["Docker", "Kubernetes", "vLLM", "NVIDIA Triton Inference Server", "KServe", "Prometheus", "Grafana", "Evidently"]
---

## What you will learn

- What has to be packaged with a model so it runs the same way in production as in evaluation.
- What shadow, canary, blue/green and rollback each answer, and how long each takes to answer it.
- The difference between service metrics and model metrics, and which one gates a release.
- How to write an incident playbook for a model, and where retraining fits.

*Before you start:* lessons 14.02 (registry versions and aliases) and 14.03 (gates). Monitoring and drift are taught in [Monitoring models](lesson:mlcore:09.03) and [Monitoring and drift](lesson:mlsd:07.01); experiment design in [A/B testing and safe rollout](lesson:mlsd:05.05). This lesson connects them into an operating routine.

## Packaging

The model a user hits must be the model you evaluated. That means shipping one versioned unit containing:

- **Weights, tokenizer and model config**, pulled from a registry version (14.02), never from a path someone overwrites.
- **Inference code and preprocessing.** A chat template, a normalisation step or a feature transform that differs from the eval harness is training/serving skew; [From notebook to service](lesson:mlcore:09.01) shows how it creeps in.
- **A pinned environment.** A container image referenced by digest, built from a lockfile, with the CUDA and driver requirements stated. Rebuilding "the same" image a month later can pull different library versions.
- **A model server.** Rarely hand-written today: vLLM or SGLang for LLMs, NVIDIA Triton Inference Server for mixed model types, with KServe or a similar layer on Kubernetes for scaling and routing. [Package the inference service](lesson:ieng:22.01) and [Model weights, caches and cold starts](lesson:ieng:22.02) cover the details, including whether to bake weights into the image or fetch them at startup.
- **A contract.** Declared input and output schemas, limits (context length, batch size) and a version endpoint that reports the registry version and image digest, so every log line and incident can name exactly what was serving.

## Release patterns

~~~dg-seq A canary with automatic rollback. The registry alias moves only after the guardrails hold.
Deployer > Router: send 5% of traffic to v14, 95% to v13
Router > v14: canary requests
v14 -> Monitor: latency, errors, saturation, output stats
Monitor -> Deployer: compare v14 against v13 over the bake time
Deployer > Router: guardrails held, ramp to 25%, 50%, 100% (or breached, back to 0%)
Deployer > Registry: move alias champion to v14 after full ramp
~~~

| Pattern | Question it answers | Typical duration | Cost
| --- | --- | --- | --- |
| Shadow | Does it run correctly on real inputs? How different are its outputs? | hours to days | double compute for mirrored traffic, no user impact
| Canary | Is it safe? (errors, latency, crashes, gross output changes) | minutes to hours | a small slice of users exposed
| A/B test | Is it better on the metric that matters? | days to weeks | traffic split for the duration
| Blue/green | Can we switch, and switch back, at once? | the switch is seconds | two full fleets for a while
| Rollback | Can we undo? | seconds if the old version is warm | keeping the previous version deployed

The rollout ladder and test design are in [Experimenting with models](lesson:mlcore:09.04) and [A/B testing and safe rollout](lesson:mlsd:05.05); deployment safety in general, including schema changes that make rollback unsafe, in [Deployment and migration safety](lesson:sdsd:14.04). Two model-specific points. Rollback is only fast if the previous version stays warm and the inputs it expects (feature schema, tokenizer, prompt template) are still produced; [Fallbacks and model rollouts](lesson:mlsd:06.04) covers keeping it that way. And for LLMs, compare canary **outputs** as well as health: response length, refusal rate and format-validity rate can move sharply while latency and errors look fine.

<div class="ex">
<div class="lab">Worked example</div>

An API serves 2,000 requests per second with a 0.2 percent error rate. A canary takes 5 percent: 100 requests per second.

**Can the canary catch a doubled error rate?** Test the canary's rate against the known baseline $p_0 = 0.002$. With $n$ requests the standard error under the baseline is $\sqrt{p_0/n}$ (ignoring the tiny $1 - p_0$), so a rise of 0.002 is three standard errors when $0.002\sqrt{n}/\sqrt{0.002} = 3$, that is $n = 9 \times 0.002 / 0.002^2 = 4{,}500$ requests. At 100 per second that is 45 seconds. Latency regressions show up on a similar timescale. A bake time of 15 to 30 minutes leaves room for slower problems such as memory leaks.

**Can it tell whether the model is better?** Suppose the model should raise a 5 percent conversion rate by 1 percent relative, to 5.05 percent. The rule of thumb $16p(1-p)/\delta^2$ for 80 percent power at 5 percent significance gives $16 \times 0.05 \times 0.95 / 0.0005^2 \approx 3.0$ million users per arm. A 5 percent canary would need weeks; an A/B test with half the traffic still needs days.

So the canary gates on service health and gross output checks, and the A/B test decides on value. Mixing the two leads either to canaries that run for weeks or to launches decided on noise.

</div>

## Operating the model

Watch two families of signals, on separate dashboards with separate owners.

| | Service metrics | Model metrics
| --- | --- | --- |
| Examples | p50 and p99 latency, error rate, throughput, GPU memory, queue depth; for LLMs time to first token and inter-token latency | input drift, prediction or output distribution, output length, refusal rate, quality on sampled and labelled traffic, label-based metrics once labels arrive
| Visible after | seconds to minutes | hours to months
| Typical tools | Prometheus and Grafana, tracing | Evidently or custom jobs over logged inputs and outputs
| Taught in | [Observability](lesson:sdsd:14.01), [SLOs and error budgets](lesson:sdsd:14.02), [Inference metrics](lesson:ieng:24.01) | [Monitoring models](lesson:mlcore:09.03), [Monitoring and drift](lesson:mlsd:07.01)

Log enough to recompute any model metric later: inputs (or their features), outputs, the registry version and a request ID. Drift statistics without that log can raise an alarm but cannot explain it.

## Incident playbooks

Every alert should point to a playbook whose first step restores service and whose later steps find the cause; the general process (roles, timeline, blameless review) is in [Incident response](lesson:sdsd:14.03).

| Symptom | First action | Then check
| --- | --- | --- |
| Errors or latency jump right after a release | roll back: move the alias or traffic to the previous version | the diff between versions: image, config, limits
| Out-of-memory or crash loops on the GPU servers | revert the last config change; lower batch or context limits | input length distribution, memory headroom
| Output distribution shifts with no release | fall back to the previous model or a safe default if quality is at risk | upstream data or feature pipeline changes, input drift
| Users report bad answers, dashboards normal | sample and read outputs on the affected slice | prompt or template version, tokenizer, a slice the evals miss
| Quality decays slowly over weeks | none urgent; open a retraining ticket | staleness, drift, label delays

## Retraining

Slow decay is handled by retraining, on a schedule, on a trigger (drift, a quality drop, enough new labelled data) or continuously; [When and how to retrain](lesson:mlsd:07.02) covers choosing among them and measuring staleness. In this module's terms, retraining is the continuous-training loop of 14.03: a pipeline (14.04) produces a candidate version with full lineage (14.01, 14.02), the candidate passes regression gates, and it is released through the same shadow, canary and A/B ladder as anything else. A model that retrains itself and deploys without gates is an incident waiting for a bad data day; deployed models also change the data they are later trained on, which [Feedback loops and exposure bias](lesson:mlsd:07.03) explains.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Rebuilding images instead of pinning digests | production differs from what was evaluated | deploy by digest, from a lockfile
| Preprocessing reimplemented in the server | training/serving skew, quiet quality loss | one shared code path, tested in CI
| Canary judged on business metrics | canaries that never finish, or decisions on noise | canary on health, A/B on value
| Previous version torn down at launch | rollback takes hours | keep it warm until the new one has baked
| No version in logs | incidents where nobody knows what was serving | version endpoint and version in every log line
| Auto-retrain straight to production | a bad data day reaches users | retrained models are candidates behind gates

<div class="take">
<div class="lab">Key takeaway</div>

Ship one pinned, versioned unit; release it through shadow, a canary gated on service health and an A/B test that decides on value, with the old version warm so rollback is a pointer move; watch service and model metrics separately, start every playbook with mitigation, and send retrained models back through the same gates.

</div>
