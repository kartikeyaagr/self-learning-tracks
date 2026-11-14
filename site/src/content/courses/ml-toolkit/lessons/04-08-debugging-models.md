---
id: "04.08"
course: "ml-toolkit"
module: "04"
slug: "debugging-models"
title: "Debugging models"
keyIdeas: ["Before training for real, overfit one small batch: a correct model and loop drive the loss on 2 to 16 examples to near zero within a few hundred steps. If it cannot, the bug is in the code, not the data or the hyperparameters.", "Check the loss at initialisation against theory: with $C$ balanced classes and small initial logits, cross-entropy should start near $\\ln C$ (2.30 for 10 classes). A much larger value means overconfident initial logits or a mismatched target format.", "Most shape bugs that do not crash come from broadcasting; assert shapes at module boundaries and be suspicious of any loss whose inputs have different numbers of dimensions.", "NaNs come from a short list: log or sqrt of zero, division by a zero norm, softmax over a fully masked row, fp16 overflow, or a learning rate that blows up. `torch.isfinite` checks and `torch.autograd.detect_anomaly` localise them.", "Exploding gradients show up as gradient-norm spikes before loss spikes. Log the norm per layer to find where they start."]
question: "A Transformer language model (vocabulary 32,000) trains fine for 3,000 steps, then the loss becomes nan at step 3,140 and stays nan. The gradient norm log shows values around 1 to 2, then 900 at step 3,138 and inf at 3,139. Describe a systematic way to find the cause, and name two plausible root causes and their fixes."
answer: "First reproduce cheaply: reload the last checkpoint before the spike, with the same data order, and replay steps 3,130 to 3,140 with per-layer gradient-norm logging, forward hooks asserting finite outputs, and anomaly detection on. That identifies the first layer and op that produce huge or non-finite values, and the exact batch. Inspect the batch: an all-padding or garbage sequence points at the data. Plausible causes: (1) attention logits growing until softmax saturates or fp16 overflows; fixes are bfloat16, query-key normalisation, lower peak learning rate or longer warmup. (2) A fully masked row producing softmax over all -inf, which gives nan; fix the masking or filter empty sequences. Also add global-norm clipping with `error_if_nonfinite=True` or skip non-finite steps so one bad batch cannot destroy the weights."
sources: [{"title": "Karpathy: A recipe for training neural networks (2019)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "The overfit-one-batch, check-init-loss and visualise-the-data habits in detail."}, {"title": "PyTorch docs: torch.autograd (anomaly detection and gradcheck)", "url": "https://pytorch.org/docs/stable/autograd.html", "note": "detect_anomaly, set_detect_anomaly and gradcheck usage."}, {"title": "PyTorch docs: Frequently asked questions", "url": "https://pytorch.org/docs/stable/notes/faq.html", "note": "Out-of-memory causes such as accumulating history across iterations, and other common errors."}, {"title": "PyTorch docs: Reproducibility", "url": "https://pytorch.org/docs/stable/notes/randomness.html", "note": "Seeding, deterministic algorithms and DataLoader worker seeding, needed to replay a failing step."}]
practice: ["finite-difference-gradient-checking", "gradient-clipping", "softmax-numerical-stability-and-log-sum-exp", "mutation-trap", "tensor-shape-manipulation"]
tags: ["debugging", "NaNs", "exploding gradients"]
tools: ["PyTorch"]
---

## Neural network bugs are quiet

A typo in a web service throws an exception. A typo in a model usually produces a model that trains a bit worse. Wrong axis in a softmax, dropout left on at evaluation, labels shuffled separately from inputs, a frozen layer, a loss computed on probabilities: all of them run, and most still reduce the loss. Debugging therefore means building checks that fail loudly. The Deep Learning course covers reading loss curves and diagnosing dead or saturated units; this lesson is the PyTorch-side toolkit.

~~~dg-flow A debugging ladder: each rung is cheap and rules out a whole class of bugs.
Inspect data and labels -> Loss at init = ln C? -> *Overfit one batch -> Gradient norms per layer -> Scale up
~~~

## Sanity checks before any real run

1. **Look at the data after the loader.** Decode a batch back to images or text and print the labels next to it. Augmentation bugs, label misalignment and normalisation mistakes show up here and nowhere else.
2. **Check the loss at initialisation.** For $C$ balanced classes, a freshly initialised classifier should give roughly uniform predictions and loss $\approx \ln C$. For a language model with vocabulary $V$, the first loss should be close to $\ln V$ (10.8 for 50,000 tokens). A value of 40 means huge initial logits; initialise the final layer smaller.
3. **Overfit a single batch.** Take 2 to 16 examples, disable augmentation and regularisation, and train on that batch repeatedly. Loss should approach zero and training accuracy reach 100 percent. Failure isolates the problem to the model, loss or optimiser wiring: frozen parameters, a detached tensor in the forward pass, a learning rate of zero after a scheduler mistake, targets in the wrong format.
4. **Check that every parameter gets a gradient.** After one `backward()`, list parameters whose `.grad` is `None` or all zeros. Those layers are unused, detached or frozen.
5. **Check batch independence.** Compute the gradient of one output example's loss with respect to the input batch; only that example's input row should be non-zero. A reshape that mixes the batch axis with another axis (for example `view` after `transpose` without care) fails this test while still training.

```python
for name, p in model.named_parameters():
    if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
        print("no gradient:", name)
```

## Shape errors and silent broadcasting

Crashing shape errors are the easy kind: read the two shapes in the message (`mat1 and mat2 shapes cannot be multiplied (64x512 and 256x10)`) and trace which dimension is wrong. The dangerous ones broadcast:

- Loss between `(N, 1)` and `(N,)` becomes `(N, N)`.
- A per-feature mean computed with the wrong `dim` broadcasts back without error.
- A mask of shape `(B, T)` applied to scores of shape `(B, H, T, T)` broadcasts along the wrong axes unless reshaped to `(B, 1, 1, T)`.

Assert shapes at module boundaries during development (`assert x.shape == (B, T, D), x.shape`), give dimensions names in comments, and prefer `einsum` or einops-style rearranges, which state the axes explicitly.

## Finding NaNs and infinities

| source | typical trigger | fix
| --- | --- | --- |
| `log(0)`, `log` of a negative | probabilities computed then logged | work in log space; `log_softmax`, `logsigmoid`
| `sqrt(0)` in backward | norms of zero vectors | add epsilon inside the sqrt
| division by zero | normalising by a sum or std that can be 0 | clamp the denominator
| softmax over all `-inf` | a fully masked row (an all-padding sequence) | ensure one unmasked position or use a large finite negative
| fp16 overflow | activations above 65504 | bfloat16, or GradScaler for gradients
| optimisation blow-up | learning rate too high, no warmup | lower LR, warmup, clipping

To find the first bad operation, enable anomaly detection for a short debugging run (it is slow): `torch.autograd.set_detect_anomaly(True)`. When backward produces NaN it raises an error that includes the traceback of the forward op responsible. For forward NaNs, register forward hooks that assert `torch.isfinite(out).all()` and report the module name. `clip_grad_norm_(..., error_if_nonfinite=True)` turns a silent nan step into an exception.

## Exploding and vanishing gradients

Log the global gradient norm every step and the per-layer norms occasionally. Explosions usually appear first as spikes in a few layers (often the last layers, embeddings, or attention logits in Transformers), a few steps before the loss spikes. Remedies, roughly in order: lower the peak learning rate or extend warmup; clip the global norm; check initialisation and normalisation placement (pre-norm residual blocks are far more stable than post-norm); for attention, normalise queries and keys or cap logits. Vanishing gradients show up as per-layer norms that shrink by orders of magnitude towards the input.

When you suspect a custom backward, compare it with finite differences using `torch.autograd.gradcheck(fn, inputs)` on small float64 inputs.

<div class="ex">
<div class="lab">Worked example</div>

**a classifier stuck at 10 percent**

A 10-class CNN trains for an epoch and stays at 10 percent accuracy, loss flat at 2.302.

Loss at init is correct, but the single-batch overfit test also stays at 2.302. The no-gradient check prints every convolution weight: the forward pass ran `features = self.backbone(x).detach()` left over from a frozen-feature experiment.

Removing the detach, the single batch reaches loss 0.01 in 200 steps and the full run trains normally. Total debugging time: two minutes, against a day of hyperparameter search that could never have worked.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

make bugs loud. Check data, initial loss and single-batch overfitting before every new model, assert shapes, and localise NaNs and gradient spikes with hooks and anomaly detection instead of guessing hyperparameters.

</div>
