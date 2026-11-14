---
id: "04.04"
course: "ml-system-design"
module: "04"
slug: "training-pipelines-and-versioning"
title: "Training pipelines and versioning"
keyIdeas: ["A training pipeline is code that turns versioned data into a versioned, evaluated model artifact without manual steps. Notebooks are for exploration, not for production models.", "Version everything that affects the model: data snapshot, feature definitions, code commit, configuration, random seeds, and the environment.", "Every trained model goes through automatic **evaluation gates** before it can be deployed: metrics against the current model, slice checks, calibration and sanity tests.", "A **model registry** records each model's lineage and status, so you can answer \"what is in production, how was it trained, and what do we roll back to?\""]
question: "A model was retrained last night and click-through dropped 4% after deploy. Nobody can tell what changed. Describe the pipeline and records that would have let you diagnose and roll back within an hour, and the gates that should have stopped the deploy."
answer: "The pipeline should be a scheduled job that (1) reads a pinned data snapshot ID, (2) builds features from versioned definitions, (3) trains with a configuration file and seed stored in the repo, (4) evaluates against the current production model on the same holdout and on key slices, and (5) registers the artifact with its lineage: data snapshot, code commit, config, metrics and environment. With that, you diff the two models' records to see what changed (for example a new data source, a changed feature definition or a different number of training days) and roll back by redeploying the previous registered version in minutes. Gates that should have blocked it: an offline metric no worse than production beyond a tolerance; no slice (new users, each country, each platform) worse by more than a set amount; calibration error within bounds; prediction distribution on recent traffic close to the current model's; and a canary with automatic rollback on online metrics before full rollout."
sources: [{"title": "Google Cloud, MLOps: continuous delivery and automation pipelines", "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", "note": "Levels of pipeline automation."}, {"title": "Sato, Wider and Windheuser, Continuous Delivery for Machine Learning (2019)", "url": "https://martinfowler.com/articles/cd4ml.html", "note": "Versioning data, models and code together."}, {"title": "MLflow documentation", "url": "https://mlflow.org/docs/latest/index.html", "note": "Experiment tracking and a model registry."}]
practice: ["hashed-dataset-split", "learning-rate-sweep", "early-stopping", "k-fold-cross-validation"]
tags: ["training pipeline", "reproducibility"]
---

## From notebook to pipeline

A model trained by hand in a notebook cannot be reproduced reliably. Someone ran cells in a particular order, on a particular extract, with a library version that has since changed. When that model misbehaves, nobody can say why or recreate it. A training pipeline replaces those steps with code that runs the same way every time and records what it did.

~~~dg-flow A training pipeline. Every stage reads versioned inputs and writes versioned outputs.
Data snapshot -(feature definitions vN)-> Training set -(config + seed)-> Train -(holdout + slices)-> Evaluate -(gates pass)-> *Model registry -(canary)-> Production
~~~

## What to version

| Item | Why | How
| --- | --- | --- |
| Data | the same code on different data gives a different model | snapshot IDs, partitioned tables
| Feature definitions | a changed feature silently changes the model | versioned feature registry or code
| Code | training logic and preprocessing | git commit hash
| Configuration | hyperparameters, date ranges, sampling rates | config file in the repo
| Random seeds | initialisation and sampling | recorded per run
| Environment | library versions, hardware | container image digest
| Artifact | the model itself | registry entry with all of the above

The Machine Learning course's "Reproducible data" lesson (mlcore 09.02) covers data versioning in detail.

## Evaluation gates

Automatic checks stop bad models before users see them:

- **comparison with production**: the new model must not be worse on the main offline metric beyond a tolerance, measured on the same recent holdout;
- **slices**: no important segment (new users, each market, each platform, rare classes) may regress more than a set amount;
- **calibration**: if scores are used as probabilities, calibration error stays within bounds;
- **sanity**: prediction distribution on recent traffic is close to the current model's; no NaNs; latency and size within budget;
- **invariance tests**: predictions do not change when irrelevant inputs change (for example a name in a text classifier).

Only then does the model go to a canary or A/B test (05.05).

## The model registry

A registry is the source of truth for models. Each entry stores the artifact, lineage (data, code, config), evaluation results, approval status and deployment history. It answers operational questions: which model is serving, what changed between two versions, and which version to roll back to. Deployment reads from the registry; nothing is deployed by copying files by hand.

<div class="ex">
<div class="lab">Worked example</div>

**an invisible data change**

A churn model's quality dropped after a routine retrain. The registry showed the data snapshot had a different row count. The upstream team had changed how cancelled trials were recorded, turning many former churners into "never subscribed" and removing them. The slice gate on "trial users" would have caught it; the team added it after the incident and set up an alert on row counts per label.

</div>

## Retraining schedules

Pipelines run on a schedule (daily, weekly), on a trigger (drift detected, new labelled data), or both. Each run produces a candidate, not an automatic deployment. Lesson 07.02 covers when and how to retrain.

## Experiments versus production runs

Track every experiment, including failed ones, with its configuration and metrics; tools such as MLflow make this cheap. Promote an experiment to production only by running it through the pipeline, so the production model is always the output of reproducible code rather than a one-off run.

## Costs and speed

Long training jobs slow every iteration. Keep a fast path for experiments (a sampled dataset, fewer epochs) and a full path for production candidates. Cache intermediate datasets, such as built training tables, so retraining does not rebuild them from raw logs each time.

## Testing the pipeline itself

Treat pipeline code like any other production code. Unit-test feature transformations on small hand-made inputs, run the whole pipeline end to end on a tiny sample in continuous integration, and check that two runs with the same inputs and seed produce the same metrics. These tests catch the most common breakages, such as a renamed column or a changed default, before they reach a nightly run.

<div class="take">
<div class="lab">Key takeaway</div>

train production models only through a pipeline that versions data, features, code, config, seeds and environment, gates every candidate on metrics, slices, calibration and sanity checks, and records lineage in a registry for diagnosis and rollback.

</div>
