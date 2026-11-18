---
id: "14.01"
course: "ai-research-track-web"
module: "14"
slug: "reproducibility-and-experiment-tracking"
title: "Reproducibility and experiment tracking"
keyIdeas: ["A training run is fixed by six things: code, data, config, environment, random seeds and the hardware layout. Record an identifier for each one (git SHA, data snapshot hash, resolved config, image digest, seeds, GPU type and count) or the run cannot be rebuilt.", "Aim for statistical reproducibility, not bitwise: floating-point addition is not associative, so atomics, autotuned kernels and a different GPU count change the bits. A rebuild succeeds if it lands inside the seed-to-seed spread of the original.", "Compare runs by diffing their records before their curves. When the configs match, the metric gap must be larger than seed noise (and, for LLM benchmarks, larger than eval-set noise) before it means anything.", "At LLM scale a run is a tree of restarts and forks, logged for weeks. Track lineage between segments, throughput and loss spikes per step, and run the eval suite asynchronously on saved checkpoints with item-level results stored.", "A sweep is a parent record with child runs. Decide the search space, the budget and the selection metric before launching, and report the winner with seeds, not the best single run."]
question: "A colleague's fine-tuning change raises validation accuracy from 71.2 to 71.9 on one run each. The run records show the same config, but different git SHAs and different data snapshot hashes. Five-seed baselines have a standard deviation of about 0.5 points. What do you conclude, and what would you run before believing the gain?"
answer: "Nothing yet. Two things changed besides the intended edit: the code (the SHA diff may contain more than the change) and the data (a different snapshot can move the metric on its own). And one run per arm cannot separate a 0.7 point gap from seed noise of 0.5: a single difference of two runs has a standard deviation of about $0.5\\sqrt{2} \\approx 0.7$. Pin the data snapshot, put the change behind a config flag on one commit so the diff is exactly the change, then run both arms with the same set of seeds (say 8 to 10 each, which gives a standard error of the difference near 0.2 to 0.25). Report the mean difference with its interval. If the eval set is small, also estimate eval noise: with 1,000 items at 71 percent, binomial noise alone is about 1.4 points, so a paired comparison on the same items is needed."
rubric: ["Notices that code and data both changed, not only the intended edit", "Compares the gap with seed noise and recognises one run per arm is not enough", "Proposes pinning the data snapshot and isolating the change on one commit", "Proposes multiple seeds per arm and reporting a mean difference with uncertainty", "Mentions eval-set sampling noise or a paired comparison on the same items"]
sources: [{"title": "PyTorch documentation: Reproducibility", "url": "https://pytorch.org/docs/stable/notes/randomness.html", "note": "Seeding, cuDNN benchmarking, use_deterministic_algorithms and DataLoader worker seeds, and what is not guaranteed."}, {"title": "Pineau et al.: Improving Reproducibility in Machine Learning Research (2020)", "url": "https://arxiv.org/abs/2003.12206", "note": "The NeurIPS 2019 reproducibility programme and its checklist of what a paper must report."}, {"title": "Bouthillier et al.: Accounting for Variance in Machine Learning Benchmarks (2021)", "url": "https://arxiv.org/abs/2103.03098", "note": "Measures how much seeds, data order and splits move benchmark results, and how to compare methods despite it."}, {"title": "MLflow documentation: Tracking", "url": "https://mlflow.org/docs/latest/ml/tracking/", "note": "One concrete implementation of runs, parameters, metrics, artifacts and tags."}]
practice: ["self-join", "summarize-across-seeds", "learning-rate-sweep", "hashed-dataset-split"]
tags: ["MLOps", "reproducibility", "experiment tracking"]
tools: ["MLflow", "Weights & Biases", "PyTorch", "Git"]
---

## What you will learn

- The six inputs that fix a training run, and which identifier to record for each.
- Why GPU training is rarely bitwise reproducible, and what "reproduced" should mean instead.
- What a run record should contain, and how to compare two runs honestly.
- What changes when one run lasts weeks and spans restarts, checkpoints and an eval suite.
- How to organise and report a hyperparameter sweep.

*Before you start:* [Reproducible data](lesson:mlcore:09.02) covers hashed splits, point-in-time joins and data snapshots. This lesson takes a pinned dataset as given and covers everything else about a run.

## Module 14 at a glance

Module 13 was about making one large run possible. This module is about making many runs trustworthy and getting the good ones into use: tracking (this lesson), versioning data and models (14.02), testing and CI (14.03), pipelines and schedulers (14.04), and shipping and operating models (14.05). Tools appear as examples. MLflow, Weights & Biases, DVC or Airflow can each be replaced; the questions they answer cannot.

## What fixes a run

| Input | What can drift | Record
| --- | --- | --- |
| Code | uncommitted edits, a dependency's code | git SHA, plus a dirty flag or the diff itself
| Data | a rebuilt table, a new shard, a changed filter | snapshot ID or content hash of the manifest (14.02)
| Config | defaults that changed between versions | the fully resolved config, not the command line
| Environment | CUDA, PyTorch and library versions | container image digest, or a lockfile hash
| Randomness | init, data order, dropout, augmentation | every seed, and RNG states at each checkpoint
| Hardware layout | GPU type, GPU count, parallelism plan | device name, world size, data/tensor/pipeline degrees

Leave out any row and a later rebuild can differ without anyone touching the code. The common failure is the config row: a run launched with `lr=3e-4` on the command line also inherited forty defaults from a file that has since changed. Log the config after every default and override has been applied.

## Bitwise versus statistical reproducibility

Floating-point addition is not associative. In float64, `(0.1 + 0.2) + 0.3` gives `0.6000000000000001` and `0.1 + (0.2 + 0.3)` gives `0.6`. A GPU sums millions of numbers per step, and the order is not always fixed:

- **Atomic adds** in some kernels (scatter, index-add, certain backward passes) accumulate in whatever order threads finish.
- **Autotuning** (`torch.backends.cudnn.benchmark = True`) times several algorithms and may pick a different one on the next run.
- **Topology** changes the order outright. Moving from 64 to 32 GPUs changes how the all-reduce of [data parallelism](lesson:air-fanout-v1:13.02) groups its sums and which rank reads which examples.

PyTorch can be asked to use deterministic kernels (`torch.use_deterministic_algorithms(True)`, which raises an error for operations that have no deterministic version), and its documentation still makes no promise across releases or platforms. Deterministic mode also costs speed. A useful rule: use it in CI and when debugging a divergence, where bitwise equality is a test, and turn it off for real runs.

For real runs, the target is **statistical reproducibility**: a rebuild from the same record lands within the seed-to-seed spread of the original. That spread is your **seed noise**, and it is worth measuring once per project.

Seeds need care too. One global seed does not cover DataLoader workers (each needs its own seed derived from the base), and resuming from a checkpoint must restore the RNG states and the data position, or the resumed run sees a different stream of batches from the one it would have seen.

## The run record

Every tracker stores the same few kinds of thing: **parameters** (fixed inputs), **metrics** (numbers over steps), **artifacts** (files such as checkpoints, plots and eval outputs) and **tags** (labels for search). Whatever the tool, the record should let a stranger rebuild the run:

```yaml
run_id: 7f3c2a
parent: null              # or the run this one resumed or forked from
git: {sha: 9e1b4d0, dirty: false}
image: registry/train@sha256:4be1...
data: {snapshot: pretrain-mix-v12, manifest_hash: 5d0c91...}
config: {model: 1.3b, lr: 3.0e-4, warmup: 2000, global_batch_tokens: 2097152, seed: 17}
hardware: {gpu: H100-80GB, world_size: 64, dp: 64, tp: 1, pp: 1}
metrics: [train/loss, val/loss, tokens_per_s, mfu, grad_norm, lr]
artifacts: [config.resolved.yaml, ckpt/step_*/, evals/step_*/items.jsonl]
```

~~~dg-flow Everything needed to rebuild a run is captured at launch; outputs are linked back to it.
Git SHA + resolved config + data snapshot + image digest -(launch)-> *Run record -(logs during training)-> Metrics, checkpoints, eval outputs -(each tagged with run ID and step)-> Comparable later
~~~

Capture the inputs automatically at launch, in the training script, not by hand. A launch wrapper that refuses to start on a dirty working tree, or that saves the diff as an artifact, removes the most common cause of "I can't get that number again".

## Comparing runs

Diff the records first. Two runs whose configs match but whose data hashes differ are not a test of the code change. Only when the diff is exactly the intended change do the curves mean something, and then the question is whether the gap beats noise.

<div class="ex">
<div class="lab">Worked example</div>

A fine-tuning change is tested with five seeds per arm on one commit and one data snapshot.

| Arm | Validation accuracy, seeds 1 to 5 | Mean | Std
| --- | --- | --- | --- |
| Baseline | 71.2, 70.6, 71.9, 70.9, 71.4 | 71.20 | 0.49
| Change | 71.8, 71.1, 72.3, 71.5, 71.6 | 71.66 | 0.44

Single runs could tell any story: baseline seed 1 against change seed 2 shows a loss of 0.1 points; baseline seed 2 against change seed 3 shows a gain of 1.7.

The mean difference is 0.46 points. Its standard error is $\sqrt{0.49^2/5 + 0.44^2/5} = 0.30$, so the gap is about 1.5 standard errors: suggestive, not established. To bring the standard error down to 0.23, half the gap, with a per-run standard deviation near 0.47 needs $n$ with $0.47\sqrt{2/n} \le 0.23$, that is about 9 seeds per arm.

At LLM scale you cannot afford nine pretraining runs, so measure the other noise source you can: the eval set. A benchmark of 1,000 items scored at 71 percent has a binomial standard error of $\sqrt{0.71 \times 0.29 / 1000} = 1.4$ points. Two checkpoints scored on the same items should be compared item by item (paired), which removes the shared item difficulty and is far tighter than comparing two means.

</div>

## Tracking at LLM scale

A pretraining or long post-training run differs from a notebook experiment in four ways.

**A run is a tree.** Jobs crash, get preempted, or are deliberately forked ("restart from step 40k with a lower learning rate"). Record each segment as its own run with a parent and a resume step, so the loss curve of the final model can be stitched from its true ancestors and not from an abandoned branch.

**Per-step health metrics matter as much as loss.** Log tokens per second and MFU (from [compute accounting](lesson:air-fanout-v1:13.01)), gradient norm, learning rate, and the loss scale if you use fp16. A loss spike preceded by a growing gradient norm reads differently from one that coincides with a new data shard; you need both series at step resolution to tell. Histograms and per-layer statistics are expensive, so log them every few hundred steps.

**Evaluation is decoupled from training.** Training writes checkpoints; a separate job picks each one up, runs the eval suite and logs results against the training run's ID and step. Store item-level outputs, not just averages, so a regression can be traced to the questions that flipped and so paired comparisons are possible later.

**The record outlives the cluster.** Weeks later someone will ask which data mixture produced checkpoint 120k. Keep records in a tracker or object store with retention, not on a cluster's scratch disk.

## Sweeps

A sweep is a set of runs that differ only in chosen hyperparameters. Treat it as one parent record (search space, search method, budget, selection metric) with child runs. [Hyperparameter search](lesson:mlcore:06.05) covers grid, random and Bayesian search; random search is the usual default because a few hyperparameters dominate and random sampling covers each of them more densely than a grid.

Three habits keep sweeps honest. Fix the selection metric and validation split before launching. Stop bad trials early with a rule set in advance (successive halving and its variants), not by eye. And rerun the winner with fresh seeds: the best of 30 trials is partly the luckiest of 30. If all 30 were equally good, the maximum would still sit about two seed-noise standard deviations above their mean.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Logging the command line, not the resolved config | old run cannot be rebuilt after a defaults change | log the merged config as an artifact
| Launching from a dirty tree | SHA points at code that never ran | refuse to launch, or save the diff
| Comparing single runs | "wins" that vanish on rerun | several seeds per arm, report mean and interval
| Resuming without RNG and data state | loss jumps or the model sees data twice | checkpoint RNG states and the sampler position
| Expecting bitwise equality across GPU counts | false alarms when hardware changes | define success as within seed noise
| Reporting the best trial of a sweep | published number does not reproduce | rerun the winner with new seeds

## Key terms

| Term | Meaning
| --- | --- |
| Run record | Parameters, metrics, artifacts and tags that identify and describe one run
| Resolved config | The configuration after every default and override is applied
| Seed noise | The spread of a metric across runs that differ only in seed
| Statistical reproducibility | A rebuild lands within seed noise of the original
| Run lineage | Parent links between resumed or forked segments of one training job
| Sweep | A parent record with child runs over a defined search space and budget

<div class="take">
<div class="lab">Key takeaway</div>

Record an identifier for code, data, config, environment, seeds and hardware at launch, and compare runs by diffing those records before reading their curves; a gain counts only when it beats seed noise and eval noise measured the same way.

</div>
