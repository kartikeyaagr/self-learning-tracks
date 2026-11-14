---
id: "05.01"
course: "deep-learning"
module: "05"
slug: "recurrent-networks"
title: "Recurrent networks"
keyIdeas: ["An RNN applies the same cell at every time step, carrying a hidden state: $h_t = \\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$. Weight sharing over time lets one model handle any sequence length.", "Unrolled over $T$ steps, an RNN is a $T$-layer deep network with tied weights. Backprop through time (BPTT) is ordinary backprop on that unrolled graph, with gradients for the shared weights summed over steps.", "The gradient from step $t$ to step $k$ is a product of $t-k$ Jacobians. If their norms sit below 1 the signal vanishes exponentially; above 1 it explodes.", "Clip gradients by global norm to tame explosions; use gated cells (next lesson) to fight vanishing; use truncated BPTT to bound memory on long sequences.", "Recurrence is inherently sequential in time, which is the main reason transformers replaced RNNs for training on large datasets."]
question: "You train a character-level RNN with truncated BPTT, chunk length k = 35, carrying the hidden state across chunks. Can the model use information from 100 characters back? Explain what the truncation does and does not prevent."
answer: "It can use such information only by accident, not by design. Truncation stops gradients at chunk boundaries, so no loss term ever adjusts the weights to store information in h for longer than 35 steps; the model is never trained to keep a fact for 100 characters. But the forward pass does carry h across boundaries, so whatever the network has learned to keep for short-range reasons (for example 'inside a quotation', or the current topic) persists and can influence predictions far later. So dependencies longer than k are not impossible, just not directly learnable. To learn them you need longer chunks (more memory), a gated cell whose state decays slowly, or an architecture with direct access to earlier positions, such as attention."
sources: [{"title": "Dive into Deep Learning: Recurrent Neural Networks", "url": "https://d2l.ai/chapter_recurrent-neural-networks/rnn.html", "note": "The RNN cell with runnable code and shapes."}, {"title": "Dive into Deep Learning: Backpropagation Through Time", "url": "https://d2l.ai/chapter_recurrent-neural-networks/bptt.html", "note": "Derives the Jacobian product and compares full, truncated and randomised truncation."}, {"title": "On the difficulty of training Recurrent Neural Networks (Pascanu et al., 2012)", "url": "https://arxiv.org/abs/1211.5063", "note": "The analysis of vanishing and exploding gradients and the case for norm clipping."}, {"title": "The Unreasonable Effectiveness of Recurrent Neural Networks (Karpathy)", "url": "https://karpathy.github.io/2015/05/21/rnn-effectiveness/", "note": "Character-level RNNs in practice, with visualisations of what hidden units track."}]
practice: ["vanilla-rnn-cell", "rnn-cell", "language-modeling", "gradient-clipping-global-norm-rescaling"]
tags: ["RNN", "BPTT", "vanishing gradients"]
tools: ["NumPy", "PyTorch"]
---

## Why sequences need their own architecture

This module covers models that read ordered data: text, audio, sensor streams, event logs. An MLP needs a fixed input size and treats position 3 and position 30 as unrelated features. A sequence model should handle any length and reuse what it learns about a pattern wherever the pattern occurs. Recurrent networks do this by sharing weights across time. Later lessons add gates (05.02), attention between two sequences (05.03), and finally replace recurrence with attention altogether (05.04). The NLP module that follows uses all of these.

## The RNN cell

At each step the cell combines the current input $x_t \in \mathbb{R}^d$ with the previous hidden state $h_{t-1} \in \mathbb{R}^H$:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h), \qquad y_t = W_{hy} h_t + b_y$$

The same $W_{xh}$ ($H \times d$), $W_{hh}$ ($H \times H$) and $b_h$ are used at every step, so the cell has $H(d + H + 1)$ parameters regardless of length. With $d = 128$ and $H = 256$ that is 98,560. The hidden state is the network's entire memory of the past: a fixed-size summary it must keep updating.

```python
import numpy as np
def rnn_forward(X, Wxh, Whh, b, h0):       # X: (T, B, d)
    h, hs = h0, []                          # h: (B, H)
    for x in X:                             # strictly one step after another
        h = np.tanh(x @ Wxh.T + h @ Whh.T + b)
        hs.append(h)
    return np.stack(hs)                     # (T, B, H)
```

For batches of different-length sequences, pad to the longest and mask: padded steps must not update the hidden state that later steps or the loss rely on, and padded outputs must be excluded from the loss. PyTorch's packed sequences do this by skipping padded steps entirely.

## Backpropagation through time

Unroll the loop and you have a $T$-layer feed-forward network whose layers share weights. BPTT is backprop on that graph. Because $W_{hh}$ appears at every step, its gradient is the sum of the contributions from every step.

~~~dg-flow The unrolled RNN. Forward runs left to right; the gradient of a late loss must travel back through every intermediate state.
[Forward] h0 -(x1)-> h1 -(x2)-> h2 -(x3)-> h3 -(x4)-> *h4 -> Loss
[Backward] Loss -> *dL/dh4 -(× J4)-> dL/dh3 -(× J3)-> dL/dh2 -(× J2)-> dL/dh1
~~~

Each backward hop multiplies by the step Jacobian $J_i = \partial h_i / \partial h_{i-1} = \text{diag}(1 - h_i^2)\, W_{hh}$. So the gradient that a loss at step $t$ sends to the state at step $k$ is

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \text{diag}(1 - h_i^2)\, W_{hh}$$

Memory is the other cost: BPTT must store every hidden state from the forward pass, so activation memory grows linearly in $T$.

## Vanishing and exploding gradients

The product of $t - k$ matrices behaves roughly like the largest singular value raised to the power $t-k$. The tanh derivative $1 - h^2$ is at most 1 and often much smaller, which pushes toward vanishing.

<div class="ex">
<div class="lab">Worked example</div>

**what 50 steps does to a gradient**

If each step scales the gradient by 0.9, after 50 steps it is $0.9^{50} \approx 0.005$: the loss barely affects what happened 50 steps earlier.

If each step scales it by 1.1, it is $1.1^{50} \approx 117$, and a slightly larger spectral radius gives inf or NaN.

A plain RNN in practice learns dependencies of about 10 to 20 steps, not hundreds.

</div>

Vanishing is silent: training runs, loss falls, and the model just never learns long-range structure. Exploding is loud: loss spikes or becomes NaN. The fixes differ:

- **Gradient clipping** for explosions. Compute the global norm over all parameter gradients and, if it exceeds a threshold (commonly 1 or 5), rescale all of them by threshold / norm. Direction is kept, step size bounded.
- **Gated cells** (LSTM, GRU) for vanishing: they add a path through time that is not multiplied by $W_{hh}$ at every step.
- **Initialisation**: orthogonal $W_{hh}$ starts with all singular values at 1, which delays both problems.

## Truncated BPTT

For long streams (a book, a year of sensor data) you cannot backprop through the entire history. Truncated BPTT splits the stream into chunks of $k$ steps (e.g. 35 or 128). The hidden state is carried from one chunk to the next, but detached, so gradients stop at the chunk boundary.

```python
h = h0
for chunk in chunks(stream, k=128):
    out, h = rnn(chunk, h)
    loss = criterion(out, targets_for(chunk))
    loss.backward(); clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
    h = h.detach()                       # keep the state, drop the graph
```

The model still sees information from earlier chunks through $h$, but it is never trained by gradient to put information into $h$ for use more than $k$ steps later.

## The sequential bottleneck

Step $t$ cannot start until step $t-1$ finishes, so a 1,000-step sequence is 1,000 dependent matrix multiplies, regardless of how many GPU cores are idle. Batching across sequences helps, but not across time. This is the main reason transformers, which process all positions of a training sequence in parallel, took over (lesson 05.04). RNNs remain useful for streaming inference with constant memory per step, and modern state-space models revive the recurrent form for exactly that.

<div class="take">
<div class="lab">Key takeaway</div>

an RNN is a deep network unrolled in time with shared weights. Gradients through it are products of Jacobians, so they vanish or explode with distance; clip for explosions, gate for vanishing, truncate for memory.

</div>
