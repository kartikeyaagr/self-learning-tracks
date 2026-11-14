---
id: "05.01"
course: "deep-learning"
module: "05"
slug: "recurrent-networks"
title: "Recurrent networks"
keyIdeas: ["An RNN applies the same cell at every time step, carrying a **hidden state**: $h_t = \\tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$. Sharing weights over time lets one model handle any sequence length, and the state is its only memory of the past.", "Unrolled over $T$ steps, an RNN is a $T$-layer deep network with tied weights. Backprop through time (BPTT) is ordinary backprop on that unrolled graph, with gradients for the shared weights summed over steps, and its memory grows with $T$.", "The gradient from step $t$ back to step $k$ is a product of $t-k$ Jacobians $\\text{diag}(1-h_i^2)W_{hh}$. If they shrink vectors it vanishes exponentially (silently: the model just ignores the distant past); if they stretch them it explodes (loudly: spikes and NaNs).", "In a small experiment a plain tanh RNN recalled a clue perfectly after 20 steps but was at chance after 40: plain RNNs learn dependencies of tens of steps, not hundreds.", "Clip gradients by global norm to tame explosions; use gated cells (05.02) and orthogonal initialisation against vanishing; use truncated BPTT to bound memory on long streams.", "Recurrence is sequential in time, so training cannot be parallelised across steps, which is the main reason transformers replaced RNNs for training on large datasets."]
question: "You train a character-level RNN with truncated BPTT, chunk length k = 35, carrying the hidden state across chunks. Can the model use information from 100 characters back? Explain what the truncation does and does not prevent, and what you would change if the task really needs 100-character dependencies."
answer: "It can use such information only by accident, not by design. Truncation stops gradients at chunk boundaries (the state is detached), so no loss term ever adjusts the weights to store something in h for longer than 35 steps: the model is never trained to keep a fact for 100 characters. But the forward pass does carry h across boundaries, so whatever the network has learned to keep for short-range reasons (for example 'inside a quotation', 'inside a parenthesis', or the current topic) persists and can influence predictions much later. So dependencies longer than k are not impossible, just not directly learnable. Even inside a chunk, a plain tanh RNN would struggle: gradients shrink exponentially with distance, and plain RNNs typically learn dependencies of only tens of steps. To learn 100-character dependencies: increase k towards 100 or more (memory grows linearly with k, which you can offset with a smaller batch), switch to a gated cell (LSTM or GRU) whose additive state path lets gradients survive that distance, keep gradient clipping on because longer unrolls make explosions more likely, or use an architecture with direct access to earlier positions, such as attention. Check the effect by measuring accuracy as a function of how far back the needed information is, not just the overall loss."
sources: [{"title": "Dive into Deep Learning: Recurrent Neural Networks", "url": "https://d2l.ai/chapter_recurrent-neural-networks/rnn.html", "note": "The RNN cell with runnable code and shapes."}, {"title": "Dive into Deep Learning: Backpropagation Through Time", "url": "https://d2l.ai/chapter_recurrent-neural-networks/bptt.html", "note": "Derives the Jacobian product and compares full, truncated and randomised truncation."}, {"title": "On the difficulty of training Recurrent Neural Networks (Pascanu et al., 2012)", "url": "https://arxiv.org/abs/1211.5063", "note": "The analysis of vanishing and exploding gradients and the case for norm clipping."}, {"title": "The Unreasonable Effectiveness of Recurrent Neural Networks (Karpathy)", "url": "https://karpathy.github.io/2015/05/21/rnn-effectiveness/", "note": "Character-level RNNs in practice, with visualisations of what hidden units track."}]
practice: ["vanilla-rnn-cell", "rnn-cell", "language-modeling", "gradient-clipping-global-norm-rescaling"]
tags: ["RNN", "BPTT", "vanishing gradients"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why ordered data (text, audio, sensor streams) needs its own architecture.
- How a recurrent cell carries a **hidden state** forward as its memory, by hand and in PyTorch.
- What backpropagation through time is, and why its gradients shrink or blow up exponentially with distance.
- The standard remedies: gradient clipping, gated cells, truncated BPTT, and when each applies.
- Why recurrence is slow to train, which is the reason transformers took over.

*Before you start:* MLP layers and activations (01.01, 01.02), backpropagation on a computational graph (01.04) and initialisation (01.05).

## Where this module starts

This module is about models that read **sequences**: data where order carries meaning, such as text, audio frames or event logs. An MLP handles them badly: it needs a fixed input size, and it treats position 3 and position 30 as unrelated inputs, so a pattern learned at one position must be relearned at every other. A CNN shares weights across positions, but each output only sees a fixed window.

**Recurrent neural networks (RNNs)** handle any length and reuse what they learn everywhere, by applying the same small network at every step and passing a running summary forward. The module then adds gates that let memory last longer (05.02), attention between two sequences (05.03), and transformers, which replace recurrence with attention (05.04). The NLP module that follows uses all of these.

## The idea in one picture

Take the smallest possible RNN: one input number and one memory number. At every step it mixes the new input with what it remembered from the previous step, then squashes the result with tanh:

$$h_t = \tanh(x_t + w_h\, h_{t-1}).$$

Here $x_t$ is the input at step $t$, $h_t$ is the **hidden state** (the network's memory after reading step $t$), and $w_h$ is the **recurrent weight**: how much of the old memory is carried into the new one. The state starts at $h_0 = 0$. Feed it twelve inputs, a $+1$ at step 1, a $-1$ at step 4 and zeros everywhere else:

| step $t$ | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| input $x_t$ | 1 | 0 | 0 | −1 | 0 | 0 | 0 | 0
| $h_t$ with $w_h = 0.5$ | 0.762 | 0.363 | 0.180 | −0.721 | −0.346 | −0.171 | −0.043 | −0.003
| $h_t$ with $w_h = 0.9$ | 0.762 | 0.595 | 0.490 | −0.508 | −0.427 | −0.367 | −0.279 | −0.173

~~~dg-fig 05.01-trace.svg
The same cell with three memory weights. With w_h = 0 the state just copies the input. With 0.5 each pulse fades by about half per step; with 0.9 it is still visible eight steps later. The memory weight sets how long the past survives.
~~~

With $w_h = 0.5$ the $+1$ is down to 0.18 by step 3, and the $-1$ at step 4 overwrites it. With $w_h = 0.9$ eight steps after the $-1$ the state is still $-0.17$. That is the whole idea of an RNN: one fixed rule applied again and again, with a hidden state carrying information forward. Real RNNs use vectors, but this trade-off remains: the multiplication that carries memory forward also makes it fade (or grow) exponentially.

## Step 1: the recurrent cell

In a real network the input at each step is a vector $x_t \in \mathbb{R}^d$ (for example a 128-number word embedding) and the hidden state is a vector $h_t \in \mathbb{R}^H$. The cell is a one-layer network with two inputs, the current input and the previous state:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h), \qquad y_t = W_{hy} h_t + b_y.$$

Here $W_{xh}$ ($H \times d$) brings in what is new at this step, $W_{hh}$ ($H \times H$) decides what to keep from before (the matrix version of $w_h$), $b_h$ is a bias, and tanh keeps each unit between −1 and 1. The optional output $y_t$ (a tag, a next-character prediction) is an ordinary linear layer on the state.

The key design choice is **weight sharing over time**: the same weights are used at every step, so the cell has $H(d + H + 1)$ parameters regardless of the sequence length. With $d = 128$ and $H = 256$ that is $256 \times (128 + 256 + 1) = 98{,}560$. (PyTorch's `nn.RNN` keeps two bias vectors, one per matrix, so it reports 98,816; the extra 256 numbers are redundant.)

The hidden state is the network's *entire* memory of the past: everything about steps $1..t$ must fit in $H$ numbers, rewritten at every step. Lesson 05.03 is largely about escaping this bottleneck.

## Step 2: running it forward

The forward pass is a loop. The state from one step is an input to the next, so the steps must run in order:

```python
import numpy as np

def rnn_forward(X, Wxh, Whh, b, h0):       # X: (T, B, d)  time, batch, features
    h, hs = h0, []                          # h: (B, H)
    for x in X:                             # strictly one step after another
        h = np.tanh(x @ Wxh.T + h @ Whh.T + b)
        hs.append(h)
    return np.stack(hs)                     # (T, B, H): the state after every step
```

To classify a whole sequence, put a classifier on the last state $h_T$; to tag every step or predict the next character, put an output layer on every $h_t$; to translate, read into a state and let a second RNN write (05.03).

For batches of different lengths, **pad** to the longest and keep a **mask**: padded steps must not update the state that later steps or the loss rely on, and padded outputs must be excluded from the loss. PyTorch's packed sequences do this by skipping padded steps entirely.

## Step 3: backpropagation through time

**Unroll** the loop: one copy of the cell per step, each feeding its state to the next. A $T$-step RNN becomes a $T$-layer feed-forward network whose layers share weights.

~~~dg-flow The unrolled RNN. Forward runs left to right; the gradient of a late loss must travel back through every intermediate state.
[Forward] h0 -(x1)-> h1 -(x2)-> h2 -(x3)-> h3 -(x4)-> *h4 -> Loss
[Backward] Loss -> *dL/dh4 -(× J4)-> dL/dh3 -(× J3)-> dL/dh2 -(× J2)-> dL/dh1
~~~

**Backpropagation through time (BPTT)** is just backprop (01.04) on that unrolled graph. Two things are special:

- Because $W_{hh}$ appears at every step, its gradient is the *sum* of the contributions from every step.
- To reach an early step, the gradient passes back through every state in between, and each hop multiplies it by the step's **Jacobian** ("how much $h_i$ moves when $h_{i-1}$ moves"):

$$J_i = \frac{\partial h_i}{\partial h_{i-1}} = \text{diag}(1 - h_i^2)\, W_{hh}.$$

The $\text{diag}(1 - h_i^2)$ factor is the tanh derivative at each unit (tanh′(a) = 1 − tanh(a)²). Chaining the hops, the gradient that a loss at step $t$ sends back to the state at step $k$ passes through

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \text{diag}(1 - h_i^2)\, W_{hh}.$$

BPTT must also store every hidden state, so activation memory grows linearly in $T$.

<div class="ex">
<div class="lab">Worked example</div>

**forward and backward through four steps by hand**

Use the one-number cell from the picture: $h_t = \tanh(x_t + 0.5\, h_{t-1})$, inputs $x = (1, 0, 0, -1)$, start $h_0 = 0$.

Step 1: $h_1 = \tanh(1 + 0.5 \times 0) = \tanh(1) = 0.7616$.

Step 2: $h_2 = \tanh(0 + 0.5 \times 0.7616) = \tanh(0.3808) = 0.3634$.

Step 3: $h_3 = \tanh(0.5 \times 0.3634) = \tanh(0.1817) = 0.1797$.

Step 4: $h_4 = \tanh(-1 + 0.5 \times 0.1797) = \tanh(-0.9101) = -0.7212$.

Now the backward direction. How much would $h_4$ change if $h_1$ were nudged? Each hop multiplies by $J_i = (1 - h_i^2) \times 0.5$.

$J_2 = (1 - 0.3634^2) \times 0.5 = 0.8679 \times 0.5 = 0.4340$. $J_3 = (1 - 0.1797^2) \times 0.5 = 0.9677 \times 0.5 = 0.4838$. $J_4 = (1 - 0.7212^2) \times 0.5 = 0.4799 \times 0.5 = 0.2399$.

$\partial h_4 / \partial h_1 = 0.4340 \times 0.4838 \times 0.2399 = 0.0504$.

Three steps back, only 5% of the signal survives: $0.5^3 = 0.125$ from the weight, cut by a further factor of about 2.5 by the tanh slopes.

</div>

## Step 4: vanishing and exploding gradients

A product of $t - k$ Jacobians behaves roughly like one number raised to the power $t - k$: the size of $W_{hh}$ (its largest singular value, or roughly its **spectral radius**, the size of its largest eigenvalue) times the typical tanh slope. The tanh slope is at most 1, so the default drift is toward shrinking.

<div class="ex">
<div class="lab">Worked example</div>

**what 50 steps does to a gradient**

If each step scales the gradient by 0.9, after 50 steps it is $0.9^{50} \approx 0.005$: the loss barely affects what happened 50 steps earlier.

If each step scales it by 1.1, it is $1.1^{50} \approx 117$, and a slightly larger factor quickly gives inf or NaN.

</div>

The next figure measures this in a 64-unit tanh RNN whose $W_{hh}$ is a random orthogonal matrix times a scale (so the scale is its spectral radius), averaged over five random input sequences.

~~~dg-fig 05.01-vanish.svg
Size of the gradient ∂h_t/∂h_k as it travels back through a 64-unit tanh RNN (log scale). At spectral radius 0.8 it falls to about 0.001 after 20 steps; at 1.0 the tanh slopes still make it shrink; at 2.0 it grows to about 20,000 after 59 steps.
~~~

At radius 0.8 the gradient is 0.048 after 10 steps. Even radius 1.0 vanishes, to 0.056 after 20 steps, because the tanh slopes are below 1. At 2.0 it reaches about 85 after 20 steps. The radius-1.5 curve stays flat only because saturated units cancel the growth, a fragile balance.

The two failures look very different during training:

- **Vanishing is silent.** The loss falls, and the model simply never learns long-range structure, because that signal is drowned by short-range gradients.
- **Exploding is loud.** One huge gradient throws the weights far away, and the loss spikes or becomes NaN.

To see the silent failure, train a plain RNN on a memory test: the first input is a clue, $+1$ or $-1$, then come $L$ steps of noise, and at the end the network must name the clue's sign. Scoring above 50% requires carrying one bit across $L$ steps.

~~~dg-fig 05.01-memory.svg
A 32-unit tanh RNN trained for 400 Adam steps per run, four runs per delay. Up to 20 steps every run is perfect; at 25 and 30 steps some runs succeed and others never leave chance; from 40 steps on, every run ends at a coin flip.
~~~

No error appears: failed runs just settle at the guessing loss. This matches the rule of thumb that a plain RNN learns dependencies of roughly 10 to 20 steps, not hundreds. It describes what gradient descent finds within a budget, not a hard limit; a longer run or a lucky seed sometimes succeeds.

## Step 5: the three remedies

The fixes differ because the problems differ.

**Gradient clipping** handles explosions. After the backward pass, compute the **global norm** of all parameter gradients together, $\|g\| = \sqrt{\sum_j \|g_j\|^2}$. If it exceeds a threshold $c$ (1 and 5 are common choices), rescale every gradient by $c / \|g\|$:

$$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|}\right).$$

The direction is kept and only the step size is capped. The next figure trains the same RNN twice in a deliberately unstable setup: $W_{hh}$ initialised with spectral radius 1.5, SGD with learning rate 0.5, delay 50.

~~~dg-fig 05.01-clip.svg
Left: without clipping the loss climbs above the guessing level (ln 2 ≈ 0.69) and never settles; with clipping it falls to almost zero. Right: the unclipped run's gradient norm spikes above 1,000, and its typical value stays near 4.
~~~

Without clipping the gradient norm spikes to 1,124, and the loss over the last 50 steps averages 3.75, *worse* than guessing (0.69). With clipping at 1 the largest raw norm was 13 and the final loss is 0.0002. (Notice too that this saturating initialisation solves a 50-step delay the default one could not: initialisation matters a lot for RNNs.)

**Gated cells** (LSTM, GRU, lesson 05.02) handle vanishing. They add a path through time where the state is updated by addition and learned scaling instead of being multiplied by $W_{hh}$ and squashed, so gradients can survive hundreds of steps.

**Initialisation** delays both: an orthogonal $W_{hh}$ starts with every singular value at 1, neither shrinking nor stretching any direction.

## Step 6: truncated BPTT

For very long streams (a book, a year of sensor data) you cannot backprop through the whole history. **Truncated BPTT** splits the stream into chunks of $k$ steps (35 and 128 are common). The hidden state is carried from one chunk to the next, but **detached**, so gradients stop at the chunk boundary.

```python
# sketch: chunks() and targets_for() stand for your data pipeline
h = h0
for chunk in chunks(stream, k=128):
    out, h = rnn(chunk, h)
    loss = criterion(out, targets_for(chunk))
    loss.backward(); clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
    h = h.detach()                       # keep the state, drop the graph
```

The forward pass still sees the past through $h$. What is lost is *training signal*: no gradient ever tells the network to store something for use more than $k$ steps later. Memory costs $O(k)$ instead of $O(T)$.

## Recurrent networks in code

The built-in layers run the same loop in optimised code. This script checks shapes, then trains the memory test from Step 4 at three delays.

```python
import torch, torch.nn as nn
torch.manual_seed(0)

rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)
print(sum(p.numel() for p in rnn.parameters()))          # 98816 (two bias vectors)
x = torch.randn(8, 50, 128)                               # (batch, time, features)
out, h_last = rnn(x)
print(out.shape, h_last.shape)                            # (8, 50, 256)  (1, 8, 256)
print(torch.allclose(out[:, -1], h_last[0]))              # True

def make_batch(B, L):                                     # remember the first sign for L steps
    bit = torch.randint(0, 2, (B,))
    x = 0.3 * torch.randn(B, L, 1)                        # distracting noise
    x[:, 0, 0] = 2.0 * bit - 1                            # the clue: +1 or -1 at step 1
    return x, bit

class Recall(nn.Module):
    def __init__(self, H=32):
        super().__init__()
        self.rnn = nn.RNN(1, H, batch_first=True)          # tanh by default
        self.head = nn.Linear(H, 2)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])                       # read the last hidden state

for L in (10, 40, 80):
    torch.manual_seed(0)
    model = Recall(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(400):
        x, y = make_batch(64, L)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    x, y = make_batch(2000, L)
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    print(L, round(acc, 3))
# 10 1.0
# 40 1.0
# 80 0.493
```

`out` holds the top layer's state at *every* step; `h_last` holds the final state of *every layer*. The 40-step run happened to succeed with this seed, while all four runs in the figure failed at 40: near the limit, results swing from run to run. At 80 steps it fails.

## Checking that it worked

- **Log the gradient norm** (`clip_grad_norm_` returns it before clipping). Noisy but bounded is healthy; repeated spikes of 100× the median mean explosions.
- **Truncate the input.** If accuracy barely drops when you feed only the last 20 steps, the model is not using long-range information.
- **Plot accuracy against distance**, as in the memory figure. A cliff at 10 to 30 steps is the signature of vanishing gradients.
- **Inspect hidden units.** On text, some units track quotes or nesting depth (see the Karpathy post); units stuck at ±1 suggest recurrent weights that are too large.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss spikes, then NaN | exploding gradients | clip by global norm (1 to 5), lower learning rate
| Loss stuck at the guessing level on long-delay tasks | vanishing gradients | LSTM/GRU (05.02), shorter dependency path, attention
| Results change when batch padding changes | padded steps leak into state or loss | mask the loss, use packed sequences
| Out of memory on long sequences | BPTT stores every state | truncated BPTT, smaller batch
| Model ignores context beyond the chunk length | truncation stops gradients at $k$ | longer chunks, gated cell, attention

## Going deeper

**Bounds.** Since $\|AB\| \le \|A\|\,\|B\|$, the norm of $\partial h_t / \partial h_k$ is at most $(\gamma \sigma_{\max})^{t-k}$, where $\sigma_{\max}$ is the largest singular value of $W_{hh}$ and $\gamma \le 1$ bounds the tanh slope. So $\sigma_{\max} < 1/\gamma$ is *sufficient* for vanishing. Pascanu et al. (in the sources) show the matching condition for explosions is only necessary, and describe the loss surface behind it: steep "cliffs" where the gradient is right in direction but far too large, which is exactly what clipping fixes.

**Why vanishing is silent.** The gradient for $W_{hh}$ sums one term per pair of steps $(k, t)$, each scaled by $\partial h_t / \partial h_k$. Short-range terms are exponentially larger, so there is always a healthy-looking gradient; it just only talks about the recent past.

**Dynamical systems view.** Without inputs, $h_t = \tanh(W_{hh} h_{t-1})$ is a dynamical system. Small $W_{hh}$ pulls every state to zero (forgetting); large $W_{hh}$ creates stable attractors that can hold a bit indefinitely, but near their boundaries gradients explode. Storing and training stably pull against each other; gates make "keep" an explicit, learnable choice.

**The sequential bottleneck.** Step $t$ cannot start until step $t - 1$ finishes, so a 1,000-step sequence is 1,000 dependent matrix multiplies, however many cores sit idle. Batching helps across sequences, not across time. This is the main reason transformers, which process all positions of a training sequence in parallel, took over (05.04). RNNs remain attractive for streaming inference at constant memory per step, and modern state-space models revive the recurrent form for exactly that.

**Variants.** `num_layers=2` stacks RNNs. A **bidirectional** RNN adds a second cell running backwards and concatenates both states; it helps tagging, but cannot be used when predicting the future.

## Key terms

| Term | Meaning
| --- | --- |
| Hidden state $h_t$ | The fixed-size vector the RNN carries forward; its only memory of the past
| Recurrent weight $W_{hh}$ | The matrix that maps the previous state into the new one, shared by every step
| Weight sharing over time | Using the same cell parameters at every step, so any length works
| Unrolling | Drawing the loop as one copy of the cell per step, giving a deep network with tied weights
| BPTT | Backpropagation through the unrolled network; shared-weight gradients are summed over steps
| Vanishing / exploding gradients | Exponential shrinking or growth of gradients as they travel back through many steps
| Gradient clipping | Rescaling all gradients when their global norm exceeds a threshold
| Truncated BPTT | Carrying the state across fixed-length chunks but stopping gradients at the boundaries

<div class="take">
<div class="lab">Key takeaway</div>

an RNN applies one shared cell at every step and carries a hidden state as memory; unrolled, it is a deep network with tied weights, so gradients through it are products of Jacobians that vanish or explode with distance. Clip for explosions, use gated cells for vanishing, truncate for memory, and expect a plain RNN to learn dependencies of tens of steps, not hundreds.

</div>
