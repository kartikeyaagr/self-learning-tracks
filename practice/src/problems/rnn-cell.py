import numpy as np

META = dict(title='Train an RNN with truncated backprop through time', tags=['Neural Networks', 'Training', 'Time Series'],
            diff='Medium', runtime='torch', lessons=['ieng:00.05'],
            related=['vanilla-rnn-cell', 'language-modeling', 'mini-training-loop'])

STATEMENT = r"""
Long sequences don't fit in one backward pass. Truncated BPTT cuts the sequence into chunks, carries the hidden state **forward** across chunks but stops gradients at chunk boundaries with `detach()`.

Write `tbptt_train(x, y, W_ih, W_hh, b_h, W_out, b_out, h0, chunk, lr)`:

- `x`: float64 `(T, B, D)` inputs, `y`: float64 `(T, B, K)` targets (time first).
- Cell (one bias, `nn.RNNCell` weight layout): $h_t = \tanh(x_t W_{ih}^\top + h_{t-1} W_{hh}^\top + b_h)$, with `W_ih` `(H, D)`, `W_hh` `(H, H)`, `b_h` `(H,)`.
- Readout: $\hat y_t = h_t W_{out}^\top + b_{out}$, with `W_out` `(K, H)`, `b_out` `(K,)`.
- `h0`: `(B, H)` initial state. `chunk`: int $\ge 1$. `lr`: float.

## The loop
Split time into consecutive chunks `[0, chunk), [chunk, 2*chunk), ...`; the last one may be shorter. For each chunk:

1. Start from the hidden state carried over from the previous chunk, **detached** (the first chunk starts from `h0`).
2. Run the cell over the chunk and compute the loss: mean squared error over all `len * B * K` entries of the chunk.
3. Zero the gradients, backward, and apply one plain SGD step $\theta \leftarrow \theta - \eta \nabla\theta$ to all five parameters.
4. Carry the chunk's last hidden state to the next chunk. It was computed with the weights from before this step; don't recompute it.

## Return
A tuple `(losses, W_ih, W_hh, b_h, W_out, b_out, h_last)`: `losses` is float64 `(n_chunks,)` with each chunk's loss before its update, then the five trained parameters and the final hidden state `(B, H)`. Don't modify the inputs.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def tbptt_train(x, y, W_ih, W_hh, b_h, W_out, b_out, h0, chunk, lr):
    """Return (losses, W_ih, W_hh, b_h, W_out, b_out, h_last)."""
    params = [p.clone().requires_grad_(True) for p in (W_ih, W_hh, b_h, W_out, b_out)]
    Wih, Whh, bh, Wo, bo = params
    h = h0
    losses = []
    for s in range(0, x.shape[0], chunk):
        # TODO: run the cell over x[s:s+chunk], MSE loss, backward, SGD step, carry h
        pass
    return (torch.tensor(losses, dtype=torch.float64), *[p.detach() for p in params], h)
'''

SOLUTION = '''import torch

def tbptt_train(x, y, W_ih, W_hh, b_h, W_out, b_out, h0, chunk, lr):
    params = [p.clone().requires_grad_(True) for p in (W_ih, W_hh, b_h, W_out, b_out)]
    Wih, Whh, bh, Wo, bo = params
    h = h0.clone()
    losses = []
    for s in range(0, x.shape[0], chunk):
        h = h.detach()                                   # cut the graph: no gradient into earlier chunks
        preds = []
        for t in range(s, min(s + chunk, x.shape[0])):
            h = torch.tanh(x[t] @ Wih.T + h @ Whh.T + bh)
            preds.append(h @ Wo.T + bo)
        loss = ((torch.stack(preds) - y[s:s + chunk]) ** 2).mean()
        for p in params:
            p.grad = None                                # zero_grad
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
        losses.append(loss.item())
    return (torch.tensor(losses, dtype=torch.float64), *[p.detach().clone() for p in params], h.detach())
'''

HINTS = [
    'The outer loop is `for s in range(0, T, chunk)`; slicing `y[s:s + chunk]` handles the short last chunk for free.',
    'Call `h = h.detach()` at the **start** of each chunk. Without it the second `backward()` tries to go through the first chunk\'s freed graph and raises an error (and with `retain_graph=True` it silently computes full-BPTT gradients through stale weights).',
    'Make the parameters leaf tensors with `.clone().requires_grad_(True)`, set `p.grad = None` before each backward, and update inside `torch.no_grad()`.',
]

_r = np.random.default_rng(5)

def _case(T, B, D, H, K, chunk, lr):
    x = _r.normal(size=(T, B, D))
    y = np.sin(np.cumsum(x[..., :1], axis=0)) * np.ones((1, 1, K))
    return (x, y, 0.5 * _r.normal(size=(H, D)), 0.5 * _r.normal(size=(H, H)) / np.sqrt(H), np.zeros(H),
            0.5 * _r.normal(size=(K, H)), np.zeros(K), np.zeros((B, H)), chunk, lr)

TESTS = [
    (np.array([[[1.0]], [[0.0]], [[-1.0]]]), np.array([[[0.5]], [[0.5]], [[0.0]]]), np.array([[1.0]]), np.array([[0.5]]),
     np.array([0.0]), np.array([[1.0]]), np.array([0.0]), np.zeros((1, 1)), 2, 0.1),
    (np.ones((2, 1, 2)), np.zeros((2, 1, 1)), np.array([[0.5, -0.5], [1.0, 0.0]]), np.eye(2) * 0.5, np.zeros(2),
     np.array([[1.0, 1.0]]), np.array([0.1]), np.array([[0.2, -0.2]]), 1, 0.5),
    _case(8, 2, 3, 4, 1, 3, 0.1),
    _case(6, 1, 2, 3, 2, 6, 0.2),
    _case(7, 3, 2, 5, 1, 1, 0.05),
    _case(10, 2, 4, 6, 3, 4, 0.1),
    _case(5, 2, 1, 2, 1, 2, 0.0),
    _case(12, 4, 3, 8, 2, 5, 0.03),
]

EXPLANATION = r"""
## What detach buys you
Without truncation, the graph for a sequence of length $T$ holds every hidden state until `backward()`: memory grows as $O(T \cdot B \cdot H)$ and one update needs the whole sequence. Detaching at chunk boundaries keeps the **value** of $h$ (so the model still sees long-range context in the forward pass) but makes it a constant for autograd. Each update only needs the last `chunk` steps of the graph, and you get $\lceil T/\text{chunk}\rceil$ updates per sequence instead of one.

## What it costs
Gradients can't flow further back than one chunk, so the model can only *learn* dependencies up to about `chunk` steps long, even though the forward pass carries information further. The carried state was produced by the previous weights; that small inconsistency is accepted in practice.

## Complexity
Time $O(T \cdot B\, (DH + H^2 + HK))$ in total. Peak activation memory $O(\text{chunk} \cdot B \cdot H)$ instead of $O(T \cdot B \cdot H)$.

## Common bugs
- No `detach()`: the second backward raises "Trying to backward through the graph a second time". Adding `retain_graph=True` to silence it is the worst fix: memory grows without bound and gradients pass through stale weights.
- Detaching at the end of the chunk *before* using `h` for the readout, which zeroes the gradient into the cell.
- Averaging the chunk loss over `chunk` instead of the real chunk length on the last, short chunk.

## Interview follow-ups
- Why do LSTMs and GRUs still need truncated BPTT, and how do you pick the chunk length relative to the dependency length in your data?
- In a stateful language-model training loop over a long text stream, how do you carry hidden states across batches when the batch is a set of parallel streams?
"""
