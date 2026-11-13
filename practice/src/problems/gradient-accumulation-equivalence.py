import numpy as np

META = dict(title='Token-weighted gradient accumulation with autograd', tags=['Training', 'Language Models'], diff='Easy',
            runtime='torch', lessons=['air-fanout-v1:03.09', 'air-fanout-v1:10.04'],
            related=['gradient-accumulation', 'accumulate-shared-path-gradients', 'synchronize-data-parallel-gradients'])

STATEMENT = r"""
Language-model losses are a mean over **tokens**, not over sequences. When you split a padded batch into micro-batches and call `backward()` on each, PyTorch adds the gradients into `.grad`, so each micro-batch loss must be scaled to make the sum equal the full-batch gradient. The common shortcut, `loss_k.mean() / n_micro`, is only right when every micro-batch has the same number of real tokens.

Write `accumulate_grads(X, targets, W, b, n_micro)` for a linear next-token head.

## Inputs
- `X`: `float64` `(B, T, D)` hidden states. `W`: `float64` `(D, V)`. `b`: `float64` `(V,)`.
- `targets`: `int64` `(B, T)`, token ids in $[0, V)$, or `-100` for padding (ignored).
- `n_micro`: int, $1 \le$ `n_micro` $\le B$. Split the batch along dimension 0 with `torch.tensor_split(..., n_micro)` (contiguous chunks, earlier chunks one larger when $B$ is not divisible).

The full-batch loss is the mean cross-entropy over all real tokens of the batch:
$$\mathcal L = \frac{1}{N_\text{tok}} \sum_{(i,t)\,:\,y_{it} \ne -100} -\log \operatorname{softmax}(x_{it} W + b)_{y_{it}}$$

The batch has at least one real token, but a micro-batch may have none.

## What to compute
1. **Correct accumulation**: for each micro-batch $k$, compute its summed token loss, divide by the **full-batch** token count $N_\text{tok}$, and call `backward()`, accumulating into `.grad` of leaf copies of `W` and `b` (gradients start at zero).
2. **Naive accumulation**: for each micro-batch with at least one real token, call `backward()` on (its mean token loss) $/\,$`n_micro`. Skip micro-batches with no real tokens (their mean would be NaN), but still divide by `n_micro`.

## Output
A tuple `(gW, gb, naive_gW, naive_gb)` of `float64` tensors shaped like `W`, `b`, `W`, `b`. Do not modify the inputs.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
import torch.nn.functional as F

def accumulate_grads(X, targets, W, b, n_micro):
    """Return (gW, gb, naive_gW, naive_gb) from micro-batch backward passes."""
    W = W.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    for Xk, yk in zip(torch.tensor_split(X, n_micro), torch.tensor_split(targets, n_micro)):
        logits = Xk @ W + b
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yk.reshape(-1), ignore_index=-100)
        (loss / n_micro).backward()          # TODO: right only if all micro-batches have equal token counts
    return W.grad, b.grad, W.grad, b.grad
'''

SOLUTION = '''import torch
import torch.nn.functional as F

def _grads(X, targets, W, b, n_micro, naive):
    W = W.detach().clone().requires_grad_(True)       # fresh leaves: .grad starts as None (= zero)
    b = b.detach().clone().requires_grad_(True)
    n_tok = (targets != -100).sum()                   # real tokens in the WHOLE batch
    for Xk, yk in zip(torch.tensor_split(X, n_micro), torch.tensor_split(targets, n_micro)):
        logits = (Xk @ W + b).reshape(-1, W.shape[1])
        yk = yk.reshape(-1)
        if naive:
            if (yk != -100).sum() == 0:
                continue                              # mean over zero tokens would be NaN
            loss = F.cross_entropy(logits, yk, ignore_index=-100) / n_micro
        else:
            # sum, then divide by the global count: the k pieces add up to exactly the full-batch mean
            loss = F.cross_entropy(logits, yk, ignore_index=-100, reduction='sum') / n_tok
        loss.backward()                               # adds into W.grad and b.grad
    return W.grad, b.grad

def accumulate_grads(X, targets, W, b, n_micro):
    gW, gb = _grads(X, targets, W, b, n_micro, naive=False)
    nW, nb = _grads(X, targets, W, b, n_micro, naive=True)
    return gW, gb, nW, nb
'''

HINTS = [
    '`backward()` **adds** into `.grad`. So if the micro-batch losses sum to the full-batch loss, the accumulated gradient is the full-batch gradient, by linearity.',
    'Count the real tokens of the whole batch first: `(targets != -100).sum()`. Then use `reduction="sum"` per micro-batch and divide by that global count.',
    'For the naive version, the default `reduction="mean"` already ignores `-100`; skip micro-batches with no real tokens before calling it. Build fresh leaf tensors for each version so their `.grad` do not mix.',
]


def _case(seed, B, T, D, V, pad_rows):
    r = np.random.default_rng(seed)
    X = r.normal(size=(B, T, D))
    y = r.integers(0, V, size=(B, T))
    for i, n_pad in pad_rows.items():
        y[i, T - n_pad:] = -100
    return X, y.astype(np.int64), r.normal(size=(D, V)) * 0.5, r.normal(size=V) * 0.1


_a = _case(1, 2, 3, 2, 3, {1: 2})
_b = _case(2, 4, 5, 3, 4, {0: 0, 1: 4, 2: 1, 3: 5})
_c = _case(3, 5, 6, 4, 7, {0: 6, 1: 6})
_d = _case(4, 3, 4, 2, 5, {})
_e = _case(5, 7, 9, 6, 11, {0: 3, 2: 8, 4: 1, 5: 9, 6: 2})
TESTS = [
    (*_a, 2),
    (*_a, 1),
    (*_b, 2),
    (*_b, 4),
    (*_c, 3),
    (*_d, 3),
    (*_e, 3),
    (*_e, 7),
]

EXPLANATION = r"""
## Why it works
The full-batch loss is $\frac{1}{N}\sum_{\text{tokens}} \ell$. Split the sum by micro-batch: $\mathcal L = \sum_k \frac{S_k}{N}$, where $S_k$ is micro-batch $k$'s summed loss. Gradients are linear, and `backward()` accumulates, so calling it on each $S_k / N$ leaves exactly $\nabla \mathcal L$ in `.grad`.

The naive version computes $\frac{1}{n}\sum_k \frac{S_k}{N_k}$: every micro-batch gets equal weight regardless of its token count $N_k$. Tokens in sparse micro-batches are overweighted. In test 4, micro-batch 2 has a single real token, so that token gets as much weight as all the tokens of another micro-batch combined.

## Cost
Accumulation trades memory for time: peak activation memory is that of one micro-batch, compute is unchanged, and there is one optimizer step per full batch. The only extra work is counting tokens up front, which needs the targets (cheap) but not the forward pass.

## Common bugs
- Calling `optimizer.zero_grad()` inside the micro-batch loop: only the last micro-batch survives.
- Dividing by the per-micro-batch count (the default mean) instead of the global one.
- With data parallelism the global count must include all ranks' tokens (an all-reduce of the count), or each rank's gradient is again weighted equally.

## Interview follow-ups
- Hugging Face Trainer once had exactly this bug for causal LM fine-tuning. How would you detect it from loss curves alone? (Loss depends on `gradient_accumulation_steps` even with the same effective batch.)
- How do you get the global token count when samples are packed and the count is only known after the data loader runs? (Prefetch all micro-batches of the step, or all-reduce the count before the first backward.)
"""
