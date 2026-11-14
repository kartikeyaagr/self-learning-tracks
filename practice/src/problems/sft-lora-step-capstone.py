import numpy as np

META = dict(title='Capstone: one LoRA fine-tuning step with an assistant-only loss', tags=['Language Models', 'Training'], diff='Hard', runtime='numpy',
            related=['chat-template-format', 'build-assistant-only-sft-loss-mask', 'lora-finetune-and-merge'])

STATEMENT = r"""
A tiny language model maps token $x_t$ to logits $\ell_t = E[x_t]\,(W_0 + s\,AB)$, where `E (V, d)` and `W0 (d, V)` are frozen, and the LoRA adapter has `A (d, r)` and `B (r, V)` with scale $s = \alpha / r$.

Write `lora_sft_step(ids, mask, E, W0, A, B, alpha, lr)`:

1. Position $t$ predicts `ids[t+1]`. Only targets with `mask[t+1] == 1` count (the assistant tokens). The loss is the mean cross-entropy over those $n$ targets.
2. Compute gradients for **A and B only**. With $G = \partial L / \partial \ell$ (softmax minus one-hot, divided by $n$, zero rows for unmasked positions) and $H = E[x_{0..T-2}]$: $\partial L/\partial W = H^\top G$, $\partial L/\partial A = s\,(\partial L/\partial W) B^\top$, $\partial L/\partial B = s\,A^\top (\partial L/\partial W)$.
3. Take one SGD step on A and B with learning rate `lr`.

Return `(loss_before, loss_after, A_new, B_new)`. If no target is masked in, return `(0.0, 0.0, A, B)` unchanged.
"""

STARTER = '''import numpy as np

def lora_sft_step(ids, mask, E, W0, A, B, alpha, lr):
    """One SGD step on the LoRA factors with an assistant-only loss."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _loss_grad(H, tgt, keep, W):
    z = H @ W
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    n = keep.sum()
    loss = float(-np.log(p[keep, tgt[keep]]).sum() / n)
    G = p.copy(); G[np.arange(len(tgt)), tgt] -= 1
    G[~keep] = 0
    return loss, G / n

def lora_sft_step(ids, mask, E, W0, A, B, alpha, lr):
    E, W0, A, B = (np.asarray(a, float) for a in (E, W0, A, B))
    ids, mask = np.asarray(ids), np.asarray(mask)
    keep = mask[1:] == 1
    if not keep.any():
        return 0.0, 0.0, A, B
    s = alpha / A.shape[1]
    H, tgt = E[ids[:-1]], ids[1:]
    loss0, G = _loss_grad(H, tgt, keep, W0 + s * A @ B)
    dW = H.T @ G
    A2, B2 = A - lr * s * dW @ B.T, B - lr * s * A.T @ dW
    loss1, _ = _loss_grad(H, tgt, keep, W0 + s * A2 @ B2)
    return loss0, loss1, A2, B2
'''

HINTS = [
    'Compute the gradient with respect to the merged weight W once, then push it through the product AB with the chain rule.',
    'Use the old A when updating B and the old B when updating A.',
    'With the standard initialisation B = 0, the gradient for A is zero on the first step; only B moves.',
]

def _case(seed, V, d, r, T, b_zero):
    g = np.random.default_rng(seed)
    ids = g.integers(0, V, T)
    mask = np.r_[np.zeros(T // 2, int), np.ones(T - T // 2, int)]
    B = np.zeros((r, V)) if b_zero else 0.1 * g.normal(size=(r, V))
    return (ids, mask, g.normal(size=(V, d)), 0.5 * g.normal(size=(d, V)), 0.1 * g.normal(size=(d, r)), B)

TESTS = [
    _case(1, 6, 4, 2, 8, True) + (16.0, 0.5),
    _case(2, 6, 4, 2, 8, False) + (16.0, 0.5),
    _case(3, 10, 8, 4, 12, False) + (8.0, 0.1),
    _case(4, 5, 3, 1, 6, True) + (1.0, 1.0),
    (np.array([1, 2, 3]), np.array([0, 0, 0]), np.eye(4), np.ones((4, 4)), np.ones((4, 1)), np.zeros((1, 4)), 1.0, 0.1),
]

EXPLANATION = r"""
## What LoRA trains
The frozen weight $W_0$ gets a low-rank correction $sAB$. Only $r(d + V)$ numbers are trained instead of $dV$, so optimiser state and gradients shrink by orders of magnitude, and the adapter can be merged into $W_0$ afterwards at no inference cost.

## The zero start
$B = 0$ makes the adapted model identical to the base model at step 0, so fine-tuning starts from the pretrained behaviour. The price is that $A$ gets no gradient until $B$ moves, which the first test shows.

## The assistant-only mask
Without the mask, the loss also trains the model to predict user and system text, which wastes capacity and can teach it to write the user's side of a conversation. The mask comes from the chat template's assistant spans after tokenisation.

## QLoRA
QLoRA stores $W_0$ in 4-bit NF4 and dequantises on the fly, training the same $A$ and $B$ in higher precision. The gradient computation is unchanged; only the memory for $W_0$ falls.

## Interview follow-ups
- Why does the learning rate for LoRA usually need to be higher than for full fine-tuning?
"""
