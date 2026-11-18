---
id: "14.02"
course: "ai-research-track-web"
module: "14"
slug: "versioning-data-and-models"
title: "Versioning data and models"
keyIdeas: ["Content addressing names a file by the hash of its bytes, so a name always means the same content, identical files are stored once, and a changed file cannot hide under an old name.", "A dataset version is a manifest: every file's path, hash and size plus the recipe that produced it (filters, mixture weights, tokenizer). Hashing the canonical manifest gives one ID for the whole dataset; changing one shard changes the ID and uploads only that shard.", "A model registry stores immutable model versions with lineage back to the run, the data manifest and the code. Environments point at versions through movable aliases (champion, candidate), so promotion and rollback are pointer moves, not copies.", "A resumable checkpoint for mixed-precision Adam is about 12 bytes per parameter (fp32 master weights and two moments), roughly 84 GB for a 7B model; a weights-only bf16 export is 14 GB. A retention policy and the checkpoint interval matter more than raw storage price.", "A model card records intended use, training data, evaluation results by slice and known limitations, and belongs next to the registered version it describes."]
question: "Your team trains a 7B model on 256 GPUs; a step takes about 1.7 s and the job fails roughly once a day. Checkpoints are saved every 1,000 steps and kept forever, and the serving team copies 'the latest good weights' by hand into a bucket called prod-model. Propose a checkpoint, retention and model-versioning scheme, with numbers."
answer: "Checkpoint size: fp32 master weights plus two Adam moments is 12 bytes per parameter, about 84 GB, written as sharded files from every rank. At an illustrative 10 GB/s aggregate write bandwidth one save blocks about 8.4 s. Young's approximation gives an interval near $\\sqrt{2 \\times 8.4 \\times 86{,}400} \\approx 1{,}200$ s, about 20 minutes or 700 steps, so every 1,000 steps (28 minutes) is close enough; asynchronous saving hides most of the write. Retention: keep the last three full checkpoints for resuming and a 14 GB bf16 weights-only export every 10,000 steps for evals, instead of every full checkpoint (250 of them, 21 TB, over a 250,000-step run). Versioning: register each candidate export as an immutable model version whose record links the run ID, data manifest hash, git SHA and eval results, attach a model card, and let serving resolve an alias such as champion. Promotion and rollback then move the alias; nobody copies files into prod-model."
rubric: ["Computes checkpoint size from bytes per parameter and distinguishes full from weights-only", "Chooses an interval by trading save cost against expected lost work on failure", "Proposes a retention policy with an estimate of the storage it saves", "Replaces the hand-copied bucket with immutable registered versions and movable aliases", "Links each version to its run, data manifest and code, with a model card"]
sources: [{"title": "DVC documentation: Internal directories and files", "url": "https://dvc.org/doc/user-guide/project-structure/internal-files", "note": "How a content-addressable cache stores files and directories by hash."}, {"title": "MLflow documentation: Model Registry workflows", "url": "https://mlflow.org/docs/latest/ml/model-registry/workflow/", "note": "Versions, aliases and tags, and the deprecation of fixed model stages."}, {"title": "Mitchell et al.: Model Cards for Model Reporting (2019)", "url": "https://arxiv.org/abs/1810.03993", "note": "The sections a model card should contain, with disaggregated evaluation."}, {"title": "PyTorch documentation: Distributed Checkpoint", "url": "https://pytorch.org/docs/stable/distributed.checkpoint.html", "note": "Sharded saving from every rank, load-time resharding and asynchronous saves."}]
practice: ["content-hash-data-manifest", "model-registry-promotion", "common-table-expressions", "row-number"]
tags: ["MLOps", "data versioning", "model registry", "checkpointing"]
tools: ["DVC", "lakeFS", "MLflow", "PyTorch"]
---

## What you will learn

- How content addressing gives data an identity that cannot silently change.
- How to describe a dataset version with a manifest, and why hashing the manifest gives a dataset ID.
- What a model registry adds on top of a folder of weights: immutable versions, lineage and aliases.
- What goes in a model card.
- How big checkpoints are, how often to save them and how many to keep.

*Before you start:* lesson 14.01 (the run record) and [Reproducible data](lesson:mlcore:09.02), which introduces snapshots and lineage for tabular data. This lesson generalises both to large unstructured datasets and to models.

## Content addressing

A path such as `s3://corpus/web/shard-0042.jsonl` says where bytes are, not what they are. Someone can rewrite the shard and every pointer to it now means different data. **Content addressing** names the bytes by their hash instead: SHA-256 (or MD5 in older tools) of the file content becomes the key in an object store.

Three properties follow. A key always refers to the same bytes. Two identical files, under any names, are stored once. And you can verify a download by rehashing it. DVC's cache is exactly this: a file whose hash starts `ec1d...` is stored under `cache/files/md5/ec/1d...`, and the small `.dvc` file committed to git holds only the hash. lakeFS applies the same idea to whole buckets, adding git-style branches and commits over object storage. Table formats such as Iceberg and Delta give tables time travel by version number. The tools differ; the identity rule is the same.

## Dataset manifests

A dataset is many files plus the recipe that chose them. A **manifest** writes both down:

```json
{
  "name": "pretrain-mix",
  "recipe": {"filters": "quality-v3", "dedup": "minhash-0.8", "tokenizer": "bpe-64k-v2",
             "mixture": {"web": 0.72, "code": 0.15, "papers": 0.08, "books": 0.05}},
  "files": [
    {"path": "code/shard-0000.bin", "sha256": "9c41...", "bytes": 2147483648},
    {"path": "web/shard-0000.bin",  "sha256": "a07e...", "bytes": 2147483648}
  ]
}
```

Serialise it canonically (sorted keys, sorted file list, fixed separators) and hash the result: that hash is the **dataset ID** the run record stores. Change one shard and the ID changes; rebuild from the same files in another order and it does not. Uploading the new version moves only the blobs whose hashes the store has not seen. A snapshot is then just a named, immutable manifest, and "what did this model train on?" has an exact answer.

The recipe fields matter as much as the hashes. Two manifests with identical files but different mixture weights produce different models; a tokenizer change makes every shard different even though the source text is not.

## Model registry and lineage

A model registry is a database of **model versions**. Each version is immutable: an artifact (weights, tokenizer, config) plus metadata. The metadata that earns the registry its keep is **lineage**: the training run ID, and through it the data manifest hash, git SHA, image digest and eval results.

~~~dg-flow Lineage from data to a serving pointer. Every arrow is a stored ID, so any step can be traced backwards.
Data manifest hash -(read by)-> Training run -(saves)-> Checkpoint -(export + evals)-> *Model version 14 -(alias "champion")-> Serving
~~~

Environments refer to versions through **aliases**, movable named pointers such as `champion` or `candidate`. Promotion moves `champion` from version 13 to 14 once 14 passes its gates (14.03), and rollback moves it back in seconds, because version 13 was never overwritten. Older registries used a fixed set of stages (Staging, Production, Archived) per version; MLflow, for one, now marks stages as deprecated in favour of aliases and tags, which are more flexible when there are several environments or several serving variants. What to avoid is the hand-copied "prod" folder: it has no history, no lineage and no safe rollback.

## Model cards

A **model card** is the human-readable side of a version: what the model is for and not for, what data it was trained on (by dataset ID), evaluation results broken down by relevant slices, known failure modes, and licence or usage limits. Mitchell et al. proposed the format; its key idea is disaggregated evaluation, since an average can hide a slice where the model fails. Generate the numeric parts from the registry and eval outputs, and keep the prose short and maintained alongside the version.

## Checkpoints

A checkpoint written to resume training must hold everything the step function reads: model weights, optimiser state, learning-rate scheduler state, the step counter, RNG states and the data loader position. A checkpoint written for evaluation or serving needs only the weights.

<div class="ex">
<div class="lab">Worked example</div>

A 7B model trained with mixed-precision Adam ([compute and memory accounting](lesson:air-fanout-v1:13.01)) on one trillion tokens, global batch 4M tokens, 256 H100s at 40 percent MFU.

| Item | Bytes per parameter | Size for 7B
| --- | --- | --- |
| fp32 master weights | 4 | 28 GB
| Adam first and second moments (fp32) | 8 | 56 GB
| **Resumable checkpoint** | 12 | **84 GB** (98 GB if the bf16 copy is saved too)
| bf16 weights-only export | 2 | 14 GB

**Run length.** One trillion tokens at 4M per step is 250,000 steps. Each step costs $6 \times 7 \times 10^9 \times 4 \times 10^6 = 1.68 \times 10^{17}$ FLOPs; the cluster delivers $256 \times 989 \times 10^{12} \times 0.4 = 1.01 \times 10^{17}$ FLOP/s, so a step takes 1.66 s and the run about 4.8 days.

**Keep everything.** Saving every 1,000 steps gives 250 checkpoints, $250 \times 84 = 21{,}000$ GB. At an illustrative \$0.02 per GB-month that is about \$420 a month, for as long as nobody deletes it.

**A retention policy.** Keep the last three full checkpoints (252 GB) for resuming, plus a bf16 export every 10,000 steps (25 exports, 350 GB) for evals and analysis: 602 GB, about 35 times less.

**How often to save.** With sharded files written from all ranks at an illustrative 10 GB/s aggregate, one save blocks for $84 \div 10 = 8.4$ s. If the job fails about once a day (also illustrative; measure your own rate), Young's approximation for the interval that minimises save time plus lost work is $\sqrt{2 \times \text{save cost} \times \text{MTBF}} = \sqrt{2 \times 8.4 \times 86{,}400} \approx 1{,}200$ s, about 20 minutes or 725 steps. Saving every 1,000 steps (28 minutes) is close to optimal and costs $8.4 \div 1{,}660 \approx 0.005$ of run time, half a percent; an asynchronous save, which blocks only while state is copied to host memory, cuts that further.

</div>

Two practical points. Sharded checkpoints are written as one or more files per rank, so loading them on a different GPU count needs resharding; PyTorch's Distributed Checkpoint does this at load time, and other frameworks ship conversion scripts. And a checkpoint is only valid once every shard is written: write to a temporary location, then mark it complete (a final metadata file or an atomic rename), so a crash mid-save never leaves a half-written "latest".

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Overwriting data at the same path | old runs no longer reproduce, nobody knows why | content-addressed storage, immutable snapshots
| Manifest without the recipe | same files, different mixtures, one dataset ID | hash the recipe into the manifest
| Hand-copied "prod" weights | no record of what is serving, slow rollback | registry versions plus aliases
| Saving only weights during training | resume restarts Adam from zero and the loss jumps | full state, including RNG and data position
| Treating a partial save as latest | resume fails or loads corrupt shards | write then mark complete
| Keeping every full checkpoint | tens of terabytes per run | retention policy: recent full, sparse weights-only

## Key terms

| Term | Meaning
| --- | --- |
| Content addressing | Naming stored bytes by the hash of their content
| Manifest | List of files with hashes and sizes, plus the recipe that produced them
| Dataset ID | Hash of the canonical manifest
| Model version | Immutable registered artifact with lineage metadata
| Alias | Movable named pointer to a model version
| Model card | Document of intended use, data, sliced evals and limitations
| Resumable checkpoint | Weights, optimiser, scheduler, RNG and data state needed to continue training

<div class="take">
<div class="lab">Key takeaway</div>

Give every dataset and model an identity that cannot change underneath you: content hashes and a hashed manifest for data, immutable registry versions with lineage for models, aliases for what is serving, and a checkpoint policy sized from bytes per parameter and failure rate.

</div>
