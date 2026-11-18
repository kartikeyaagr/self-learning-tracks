---
id: "14.03"
course: "ai-research-track-web"
module: "14"
slug: "testing-and-ci-cd-for-ml"
title: "Testing and CI/CD for ML"
keyIdeas: ["ML bugs rarely crash; they train a slightly worse model. Tests must therefore check behaviour that a correct pipeline guarantees: masked tokens add no loss, initial loss is near ln(V), one batch can be memorised, resuming from a checkpoint changes nothing.", "The ML test pyramid, cheapest first: unit tests for transforms, losses and custom ops; data validation against a contract; training smoke tests on a tiny model; and regression evals against a baseline as the gate at the top.", "CI tests every code change, CD delivers a validated artifact through staged environments, and CT (continuous training) reruns the training pipeline on new data. Each needs its own gates; automated training never means automated release without them.", "Split the suite by cost and latency: CPU tests in minutes on every push, GPU smoke tests when model code changes, multi-GPU runs and full eval suites nightly or before release.", "GPU CI looks expensive until you price the alternative: a month of tiered CI typically costs less than one lost day of a large training run."]
question: "Your team's pretraining codebase has only a test that imports every module. Last month a change to the padding mask silently added padded positions to the loss, and the bug was found after a 3-day, 128-GPU run. Design the test suite and CI tiers that would have caught it, and say what each tier costs in time."
answer: "Unit tier on every push (CPU, under 10 minutes): a masked cross-entropy test that builds a batch with padding and checks the loss equals the loss computed on the unpadded tokens only, and that changing padded logits leaves the loss unchanged; this alone catches the bug. Add tests for tokenizer round-trips, shapes, and finite-difference gradient checks on custom ops. Data tier: validate shards against a contract (token ranges, lengths, padding fraction, no eval-set overlap). Smoke tier when model or training code changes (one GPU or CPU, a tiny config, a few minutes): initial loss near ln(V), overfit one batch to near zero, loss decreases over 50 steps, and train-20-steps equals train-10-save-load-10 in deterministic mode. Nightly: a short multi-GPU run and a regression eval against the last good checkpoint with a non-inferiority gate. The run that was lost cost 128 x 72 = 9,216 GPU-hours, far more than a year of such CI."
rubric: ["Proposes a unit test that directly checks padded positions do not contribute to the loss", "Includes training smoke tests such as initial loss, overfit-one-batch and resume equivalence", "Includes data validation against a contract", "Organises tests into tiers by cost and trigger (push, model-code change, nightly)", "Uses regression evals against a baseline as the release gate", "Compares CI cost with the cost of the lost run"]
sources: [{"title": "Breck et al.: The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction (2017)", "url": "https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/", "note": "Google's checklist of data, model, infrastructure and monitoring tests."}, {"title": "Karpathy: A Recipe for Training Neural Networks (2019)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "Verify the loss at init, overfit one batch, and other checks before scaling up."}, {"title": "Google Cloud: MLOps: Continuous delivery and automation pipelines in machine learning", "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", "note": "Defines CI, CD and continuous training, and the levels of pipeline automation."}, {"title": "Sato, Wider and Windheuser: Continuous Delivery for Machine Learning (2019)", "url": "https://martinfowler.com/articles/cd4ml.html", "note": "End-to-end CD4ML: versioned data, model tests and deployment pipelines."}]
practice: ["record-schema-validation", "finite-difference-gradient-checking", "bootstrap-confidence-interval", "train-serving-skew-check", "mini-batch-training-loop"]
tags: ["MLOps", "testing", "CI/CD"]
tools: ["pytest", "GitHub Actions", "PyTorch"]
---

## What you will learn

- Why ML code needs tests of behaviour, not just tests that it runs.
- The four layers of the ML test pyramid and a concrete test for each.
- The difference between CI, CD and continuous training.
- Which tests to run on every push, which nightly, and what GPU CI costs.

*Before you start:* lessons 14.01 and 14.02. The statistics of regression gates (paired comparisons, bootstrap intervals, flakiness) are in [Regression testing prompts and agents in CI](lesson:aieng:10.06); this lesson uses them without re-deriving them.

## Silent bugs

A web service with a bug usually returns an error. A training pipeline with a bug usually returns a model, and the model is a little worse. A padding mask applied one position late, a label shifted by one token, augmentation applied to the validation set, a learning-rate warmup that never ends: each produces a plausible loss curve. You notice weeks later, if ever.

So ML tests check **invariants**: properties a correct pipeline must have whatever the data. Each test below exists because the bug it catches has happened to someone at the cost of a full run.

## The ML test pyramid

~~~dg-layers Cheapest and most numerous at the bottom; each layer catches what the one below cannot.
Regression evals | candidate vs baseline checkpoint on fixed eval sets; the release gate; minutes to hours of GPU
Training smoke tests | tiny config, a few steps: initial loss, overfit one batch, resume equivalence; minutes
Data validation | schema, ranges, null and padding rates, eval-set overlap; runs on every new shard
*Unit tests | transforms, tokenizer, losses, masks, custom ops; CPU, seconds each, on every push
~~~

### Unit tests for transforms and losses

Test the function, not the training run. Good unit tests in ML assert an invariant:

```python
import torch, torch.nn.functional as F

def masked_ce(logits, targets, mask):            # the function under test
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (loss * mask).sum() / mask.sum()

def test_padding_adds_no_loss():
    torch.manual_seed(0)
    logits, targets = torch.randn(2, 6, 11), torch.randint(0, 11, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float)
    base = masked_ce(logits, targets, mask)
    logits2 = logits.clone(); logits2[0, 4:] = 100.0   # change only padded positions
    assert torch.allclose(masked_ce(logits2, targets, mask), base)
```

Others in the same spirit: the tokenizer decodes what it encodes; the causal mask lets position $t$ see only positions up to $t$ (perturb a future token and check earlier outputs do not move); a custom kernel matches a slow reference implementation within tolerance; a custom autograd function passes a finite-difference gradient check. Run unit tests with deterministic algorithms on, so a failure reproduces.

### Data validation and contracts

A **data contract** states what a batch or shard must look like: field types, token-ID ranges, sequence-length limits, padding fraction, label distribution. Validate each new shard against it before training reads it, and against a reference shard for distribution shifts. For LLM work add one more check: overlap between training data and eval sets (n-gram or exact-match), since contamination inflates every downstream number. [Monitoring models](lesson:mlcore:09.03) covers the drift statistics if you want distribution checks rather than hard rules.

### Training smoke tests

Run the real training loop on a tiny config (two layers, small width, a few hundred tokens) on CPU or one GPU:

- **Initial loss.** With a near-uniform output at initialisation, cross-entropy should be close to $\ln V$: 10.82 for a 50,257-token vocabulary, 11.76 for 128,256. A loss of 30 at step 0 means a scaling or init bug.
- **Overfit one batch.** Train on a single fixed batch; the loss should fall to near zero. If it cannot, the model, loss or optimiser wiring is broken, and no amount of data will fix it.
- **Loss decreases.** Over 50 steps on real data the loss should fall by a set margin.
- **Resume equivalence.** Train 20 steps straight, and separately 10 steps, save, load, 10 more. In deterministic mode the weights must match bit for bit. This catches missing optimiser, scheduler, RNG or data-loader state (14.02).

### Regression evals as gates

At the top, a candidate checkpoint or model is scored against the current baseline on fixed eval sets. Two kinds of gate work together: hard assertions on must-pass cases, and a statistical gate on the aggregate, usually non-inferiority (pass if the lower bound of the paired difference is above minus a margin). The [CI regression testing lesson](lesson:aieng:10.06) derives the bootstrap gate and how to control flakiness; the same machinery applies to a pretrained or fine-tuned checkpoint.

## CI, CD and CT

~~~dg-flow Three loops with different triggers. Every path into production passes the same gates.
Code change -(CI: tests, smoke run)-> Merged code -(CT: new data or schedule triggers training)-> Candidate model -(regression evals + registry)-> *Approved version -(CD: shadow, canary, ramp)-> Serving
~~~

- **Continuous integration (CI)** tests every code change before it merges.
- **Continuous delivery (CD)** takes a validated artifact (a service image, a registered model version) through staging and production with automated checks at each stage. Lesson 14.05 covers the rollout patterns.
- **Continuous training (CT)** reruns the training pipeline automatically, on a schedule or when new data arrives. When to trigger it is the subject of [When and how to retrain](lesson:mlsd:07.02).

The rule that ties them together: an automatically trained model is a candidate, not a release. It goes through the same eval gates and staged rollout as a hand-trained one.

## What runs when

| Trigger | What runs | Hardware | Time budget
| --- | --- | --- | --- |
| Every push | lint, type checks, unit tests, data-contract tests on fixtures, CPU smoke test | CPU | under 10 minutes
| Change to model, kernel or training code | GPU smoke tests, kernel-vs-reference tests | 1 GPU | 10 to 20 minutes
| Nightly on main | short multi-GPU run (catches distributed bugs), regression evals on the last checkpoint | 8 GPUs | 1 to 2 hours
| Release or new base checkpoint | full eval suite, long-context and safety evals, model card refresh | as needed | hours

Path filters decide the second row: a documentation change should not wait for a GPU.

<div class="ex">
<div class="lab">Worked example</div>

A team opens 40 pull requests a day with about three pushes each: 120 CI runs a day. Prices are illustrative, \$3 per GPU-hour.

**GPU smoke on every push.** 10 GPU-minutes each: $120 \times 10 = 1{,}200$ GPU-minutes, 20 GPU-hours a day, \$60 a day, about \$1,320 over 22 working days. Serially that is 20 hours of work a day, so at peak hours a single GPU runner builds a queue; you would need three or more.

**With a path filter.** If 30 percent of pushes touch model or training code, that drops to 6 GPU-hours, \$18 a day, about \$400 a month.

**Nightly tier.** 8 GPUs for 1.5 hours is 12 GPU-hours, \$36 a night, about \$1,080 over 30 nights.

Tiered CI costs roughly \$1,500 a month. One lost day of a 256-GPU run costs $256 \times 24 = 6{,}144$ GPU-hours, about \$18,400. The suite pays for itself if it prevents one bad day of a big run every year.

</div>

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Only import or "it runs" tests | silent quality regressions | assert invariants: masks, initial loss, overfit, resume
| Smoke tests with random data only | tokenizer and data bugs pass | include a small fixture of real data
| GPU tests on every push | slow PRs, people skip CI | path filters, CPU smoke on push, GPU on model changes
| Evals gate on a single run's average | flaky gates, ignored failures | paired comparison, repeats, non-inferiority margin
| Automated retraining deploys directly | a bad data day reaches users | CT produces candidates; CD gates release
| Train and eval sets never checked for overlap | benchmark gains from contamination | overlap check in data validation

<div class="take">
<div class="lab">Key takeaway</div>

Test ML code by asserting what a correct pipeline guarantees (masked tokens add no loss, initial loss near ln V, one batch can be memorised, resume changes nothing), tier the suite by cost from CPU unit tests on every push to regression evals as the release gate, and keep CI, CD and continuous training behind the same gates.

</div>
