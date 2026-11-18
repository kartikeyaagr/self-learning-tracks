---
id: "14.04"
course: "ai-research-track-web"
module: "14"
slug: "pipelines-and-orchestration"
title: "Pipelines and orchestration"
keyIdeas: ["A pipeline is a DAG of steps with declared inputs and outputs. The orchestrator decides what runs and when; a cluster scheduler decides where and when resources are free. Keep the two roles separate.", "Every step should be idempotent: rerunning it with the same inputs produces the same output and no side effects. Read pinned partitions or snapshots, write to a location derived from the inputs, and publish atomically.", "Cache a step by a key built from its code version, its parameters and the keys of its upstream steps. A change upstream then invalidates exactly the downstream steps, and reverting to an old configuration is a cache hit.", "Retry transient failures (preemption, a lost node, a network blip) with backoff and a limit; never retry deterministic ones such as out-of-memory. For training, a retry is a resume from the last complete checkpoint.", "Slurm allocates all of a job's nodes at once and suits large batch training; Kubernetes schedules pods one by one and needs a gang-scheduling queue (Kueue, Volcano) for multi-node jobs, but runs services and batch side by side. Idle allocated GPUs, not compute, are usually the biggest waste."]
question: "Your lab has 64 GPUs on a shared cluster. Researchers keep interactive notebooks holding 8 GPUs all month, eval jobs request whole 8-GPU nodes but use one GPU, and multi-node training jobs sometimes start on part of their nodes and wait for the rest. Utilisation is about 55 percent. Estimate the idle cost and propose scheduling and pipeline changes."
answer: "Price the waste first. Notebooks: 8 GPUs x 720 hours at 5 percent use idles about 5,500 GPU-hours a month. Evals: 300 jobs of 2 hours each leaving 7 of 8 GPUs idle is 4,200 GPU-hours. Partial allocations: 20 incidents of 24 GPUs waiting 6 hours is 2,880. Together about 12,500 of 46,080 GPU-hours, 27 percent, about \\$31,000 a month at an illustrative \\$2.50 per GPU-hour. Fixes: idle timeouts or one-GPU limits on interactive sessions; eval jobs that request one GPU each and are packed eight to a node; gang scheduling so a multi-node job starts only when all its nodes are free (Slurm does this by default; on Kubernetes use a queue such as Kueue or Volcano); a low-priority preemptible queue that fills remaining gaps with sweeps and evals. On the pipeline side, trigger evals from new checkpoints, cache steps by input hash so a changed eval setting reruns only the eval, and make every step idempotent so retries after preemption are safe."
rubric: ["Quantifies idle GPU-hours per cause and converts to cost with a labelled price", "Proposes right-sizing or packing small jobs instead of whole-node requests", "Proposes gang scheduling for multi-node jobs and names how Slurm and Kubernetes differ", "Proposes limits or timeouts on interactive GPU sessions and a preemptible backfill queue", "Mentions pipeline-level fixes: caching by input hash, idempotent steps, event-triggered evals"]
sources: [{"title": "Apache Airflow documentation: Best Practices", "url": "https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html", "note": "Idempotent tasks, reading fixed partitions instead of the latest data, and avoiding duplicate writes on rerun."}, {"title": "Kubeflow Pipelines documentation: Caching", "url": "https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/caching/", "note": "How a pipeline reuses a component's outputs when its inputs and parameters are unchanged."}, {"title": "Slurm Workload Manager: Overview", "url": "https://slurm.schedmd.com/overview.html", "note": "Allocation, queueing, priorities, preemption and backfill scheduling in Slurm."}, {"title": "Kueue documentation: Overview", "url": "https://kueue.sigs.k8s.io/docs/overview/", "note": "Job queueing, quotas and all-or-nothing admission for batch and training jobs on Kubernetes."}]
practice: ["dag-cache-invalidation", "topologically-sort-computation-dag", "retry-backoff-schedule", "content-hash-data-manifest"]
tags: ["MLOps", "pipelines", "orchestration", "scheduling"]
tools: ["Airflow", "Dagster", "Kubeflow Pipelines", "Slurm", "Kubernetes", "Kueue"]
---

## What you will learn

- How to turn a sequence of scripts into a DAG an orchestrator can run, retry and cache.
- What makes a step idempotent, and why every retry and backfill depends on it.
- How caching by input hash decides which steps rerun after a change.
- How Slurm and Kubernetes differ for research workloads, and where idle GPUs come from.

*Before you start:* lessons 14.01 to 14.03. Batch inference as a serving mode (versus online and streaming) is covered in [Batch, online and streaming inference](lesson:mlsd:06.01); here it is just another step in a pipeline.

## From scripts to a DAG

Most research pipelines start as a README: "run `prepare.py`, then `tokenize.py`, then launch training, then run the evals on the final checkpoint." That works until a step fails halfway, someone reruns step two with different settings, or forty checkpoints need evaluating. A **pipeline** writes the same steps down as a directed acyclic graph (DAG): nodes are steps, edges are data dependencies, and each step declares its inputs and outputs.

~~~dg-flow A typical research pipeline. Evals fan out from each checkpoint and can run in parallel.
Data snapshot -(filter, dedup)-> Tokenized shards -(training job)-> *Checkpoints -(per checkpoint)-> Eval suite -(best checkpoint)-> Export + register -(nightly)-> Batch inference
~~~

An **orchestrator** (Airflow, Dagster, Kubeflow Pipelines, Argo Workflows, Prefect) reads the DAG and decides what runs and when: it starts steps whose inputs are ready, records each attempt, retries failures and shows the state of every run. It does not usually own the GPUs. It submits work to a **cluster scheduler** (Slurm, Kubernetes) that decides where the work runs and when resources free up. Some tools model steps as tasks ("run this"), others as assets ("this table or checkpoint should exist"); the second view, used by Dagster for example, makes caching and lineage more natural, but both describe the same graph.

## Idempotent steps

A step is **idempotent** if running it twice with the same inputs leaves the world in the same state as running it once. Retries, backfills and caching all assume it. Three rules get you most of the way:

1. **Read pinned inputs.** Read a named snapshot or a fixed date partition, never "the latest". If someone updates the input between an attempt and its retry, a step that reads the latest produces a different output under the same name.
2. **Write to a location derived from the inputs**, for example `outputs/<step>/<cache key>/`, and overwrite or upsert rather than append. An `INSERT` that ran before a crash runs again on retry and duplicates rows.
3. **Publish atomically.** Write to a temporary path and rename, or write a completion marker last, so a half-written output is never mistaken for a finished one. Checkpoints need the same rule (14.02).

## Caching by input hash

If a step is idempotent, its output is a function of its inputs, so a stored output can be reused whenever the inputs match. The usual **cache key** is a hash of the step's code version, its parameters and the cache keys of its upstream steps. Two consequences follow. A change anywhere upstream changes every downstream key, so invalidation propagates without anyone tracking it by hand. And because outputs are stored under their keys, not overwritten, returning to an earlier configuration is a cache hit. Kubeflow Pipelines turns caching on by default for this reason; DVC pipelines and Dagster offer the same idea.

The payoff is large in research. Changing the few-shot count of one benchmark should rerun that benchmark on the stored checkpoints, minutes of one GPU, not tokenization and training. The trap is a step that is not really a function of its key: one that reads the current time, an unpinned package or a mutable table. Pin those inputs and put them in the key, or turn caching off for that step.

## Retries and backfills

Failures come in two kinds. **Transient** ones (preemption, a node that dies, a storage timeout) succeed on retry, so retry with exponential backoff and a cap on attempts. **Deterministic** ones (out-of-memory, a shape error, a bad config) fail every time, so retrying them only burns GPU-hours and hides the error. Classify by exit code or error type and retry only the first kind. For a training job, a retry is a resume from the last complete checkpoint, which is why resume equivalence is worth a test (14.03).

A **backfill** runs a pipeline over past inputs: re-score the last 90 days with a new model, or evaluate every saved checkpoint on a new benchmark. Backfills are just many idempotent runs with different pinned inputs, and the orchestrator's job is to run them with a concurrency limit so they do not starve the cluster. (Slurm's "backfill scheduling" is unrelated: it lets small jobs run in gaps that do not delay higher-priority ones.)

Scheduling can be by time (nightly batch inference, weekly retraining) or by event (a new checkpoint triggers its evals, a new data snapshot triggers a retraining candidate). Event triggers keep eval results in step with training without anyone polling.

## Cluster schedulers for research

~~~dg-layers Who decides what. Each layer can be swapped without changing the others.
Orchestrator | what runs and when: DAG, triggers, retries, caching (Airflow, Dagster, Kubeflow Pipelines)
Job spec | how one job runs: nodes, GPUs per node, image, launcher (sbatch script, TrainJob, JobSet)
*Cluster scheduler | where and when: queues, quotas, priorities, preemption (Slurm, Kubernetes with Kueue or Volcano)
Hardware | nodes, GPUs, interconnect, shared storage
~~~

| | Slurm | Kubernetes
| --- | --- | --- |
| Built for | batch jobs on HPC clusters | long-running services, later batch
| Multi-node jobs | allocates every requested node before the job starts | schedules pods individually; needs a queue with all-or-nothing admission (Kueue, Volcano) to avoid partial starts
| Queueing and fairness | priorities, fair-share, preemption, job arrays, dependencies built in | quotas and queues via add-ons
| Environment | modules or containers (Pyxis, Apptainer) on shared filesystems | containers native, everything declared as objects
| Strength | large training runs, simple mental model | mixing serving, evals and training on one pool; cloud elasticity

Many labs run both: Slurm for big pretraining jobs, Kubernetes for serving and pipeline steps. Serving on Kubernetes, with GPU scheduling and KServe, is covered in [Kubernetes, GPU scheduling and KServe](lesson:ieng:23.07).

## The cost of idle GPUs

A GPU is billed or depreciated whether it computes or not, and in research clusters idle allocations usually waste more than slow kernels.

<div class="ex">
<div class="lab">Worked example</div>

A lab reserves 64 GPUs for a month: $64 \times 720 = 46{,}080$ GPU-hours, at an illustrative \$2.50 per GPU-hour about \$115,000. Three habits are measured:

| Cause | Arithmetic | Idle GPU-hours
| --- | --- | --- |
| Interactive notebooks hold 8 GPUs all month at 5 percent use | $8 \times 720 \times 0.95$ | 5,472
| 300 eval jobs of 2 hours each request a full 8-GPU node, use 1 GPU | $300 \times 2 \times 7$ | 4,200
| 20 multi-node jobs a month hold 24 GPUs for 6 hours waiting for the rest | $20 \times 24 \times 6$ | 2,880
| **Total** | | **12,552**

That is 27 percent of the reservation, about \$31,000 a month. Fixes that cost no compute: release interactive GPUs after an idle timeout; make eval jobs request one GPU and pack eight per node; admit multi-node jobs all or nothing. Then put a low-priority, preemptible queue of sweeps and evals behind the main queue, so remaining gaps do useful work and leave the moment a priority job needs the GPUs. That only works if those jobs checkpoint and are idempotent.

</div>

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Steps read "latest" data | a retry or backfill gives different results | read pinned snapshots or partitions
| Appending outputs | duplicate rows after a retry | write by key, overwrite or upsert, publish atomically
| Cache key without upstream keys | stale model reused after a tokenizer change | include upstream keys in every key
| Retrying out-of-memory | the same failure five times, hours of GPU | retry only transient errors
| Pods of one training job start separately | GPUs held while waiting for peers | gang scheduling with a queue
| Whole nodes for one-GPU jobs | high allocation, low utilisation | right-size requests, pack small jobs

## Key terms

| Term | Meaning
| --- | --- |
| DAG | Steps as nodes, data dependencies as edges, no cycles
| Orchestrator | Runs a DAG: triggers, ordering, retries, caching, run history
| Idempotent step | Same inputs, same result, however many times it runs
| Cache key | Hash of code version, parameters and upstream keys
| Backfill | Running a pipeline over past inputs or partitions
| Gang scheduling | Starting a multi-node job only when all its resources are available

<div class="take">
<div class="lab">Key takeaway</div>

Write the pipeline as a DAG of idempotent steps keyed by a hash of their code, parameters and upstream keys, so retries, backfills and partial reruns are safe and cheap; then let a scheduler with gang admission and right-sized requests keep the GPUs busy, because idle allocations cost more than slow code.

</div>
