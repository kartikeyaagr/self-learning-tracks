---
id: "05.02"
course: "deep-learning"
module: "05"
slug: "lstms-and-grus"
title: "LSTMs and GRUs"
keyIdeas: ["An LSTM keeps a separate **cell state** updated by addition: $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$. Along this path the gradient is scaled by the forget-gate values, not by a repeated weight matrix, so it survives when $f_t \\approx 1$.", "Gates are sigmoids in $(0, 1)$ computed from $x_t$ and $h_{t-1}$: forget ($f$, what to keep), input ($i$, what to write), output ($o$, what to expose), plus a tanh candidate $g$. An LSTM has 4× the parameters of a plain RNN of the same width.", "Initialise the forget-gate bias to about 1 or more so the cell starts out remembering. At bias 0 every unit halves its memory each step and the LSTM starts out as forgetful as a plain RNN; PyTorch does not set this for you.", "A GRU merges cell and hidden state and uses two gates (update $z$, reset $r$): $h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}$. It has 3× the RNN's parameters and usually similar accuracy. Check which side of the interpolation $z$ multiplies in your framework.", "Gates fix vanishing, not exploding gradients, and they do not enlarge the fixed-size memory: keep gradient clipping, check accuracy against distance, and use attention when one vector cannot hold everything."]
question: "You train a one-layer `nn.LSTM` (hidden size 128, default PyTorch initialisation) to classify server logs as normal or faulty, where the deciding event usually appears 60 to 200 lines before the end of each sequence. After 10 epochs the training loss is stuck at about 0.69 and validation accuracy is 50%. A colleague proposes doubling the hidden size and training for 50 epochs. What is most likely going on, what would you try first, and how would you check that your change worked?"
answer: "A loss of 0.69 is $\\ln 2$, the cross-entropy of guessing on a two-class problem, so the model has learned nothing; the information it needs is 60 to 200 steps back. With PyTorch's default initialisation the forget-gate bias is near 0 (each of `bias_ih` and `bias_hh` is drawn from $\\pm 1/\\sqrt{128}$), so $f \\approx 0.5$ and the gradient along the cell path is multiplied by roughly 0.5 per step: after 60 steps that is about $10^{-18}$, so no useful signal reaches the deciding event and the gates never learn to open. A bigger hidden size does not change this, and more epochs of a near-zero gradient will not either. First fix: set the forget-gate bias to a larger value, for example `bias_ih_l0[128:256] = 1` to 3 and `bias_hh_l0[128:256] = 0` (rows 128:256 are the forget gate in PyTorch's i, f, g, o order), or use chrono initialisation with $T_{\\max} \\approx 200$. Keep gradient clipping on, and check that sequences are packed or masked so padding does not overwrite the state. To verify: the training loss should leave 0.69 within the first epochs; log the mean forget-gate value (it should start near $\\sigma(b_f)$ and stay high in the units that carry memory); measure the gradient norm with respect to inputs 60 to 200 steps back; and report validation accuracy bucketed by the distance between the deciding event and the end of the sequence, so you can see whether long cases improved and not only the average. If long cases still fail, consider a model that can look back directly (attention or a transformer) or truncating the logs closer to the event."
sources: [{"title": "Understanding LSTM Networks (Olah, 2015)", "url": "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", "note": "The standard visual walkthrough of each gate, plus GRU and peephole variants."}, {"title": "Learning Phrase Representations using RNN Encoder-Decoder (Cho et al., 2014)", "url": "https://arxiv.org/abs/1406.1078", "note": "Introduces the GRU (section 2.3) alongside the encoder-decoder."}, {"title": "LSTM: A Search Space Odyssey (Greff et al., 2015)", "url": "https://arxiv.org/abs/1503.04069", "note": "Large ablation of LSTM variants; read the conclusions on which components matter."}, {"title": "PyTorch nn.LSTM", "url": "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html", "note": "Exact equations, gate order and weight shapes used by PyTorch."}]
practice: ["lstm-cell", "gru-cell", "lstm-cell-2"]
tags: ["LSTM", "GRU", "gating"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why a plain RNN forgets after a few dozen steps, and the one design change (an additive memory lane) that fixes it.
- What each LSTM gate does, in plain words and in symbols, and how to compute one LSTM step by hand.
- Why the gradient survives along the cell state, and why the starting forget-gate bias decides whether it does.
- How the GRU does the same job with two gates instead of three, and how to choose between them.
- How to use `nn.LSTM` and `nn.GRU` in PyTorch without the classic bugs.

*Before you start:* the plain RNN, backpropagation through time and vanishing gradients (05.01), and the sigmoid and tanh functions (module 01). $\odot$ means "multiply matching entries".

Lesson 05.01 ended with a problem: the gradient a late loss sends back to an early step is a product of one Jacobian per step, so it shrinks exponentially, and a plain RNN learns dependencies of only about 10 to 20 steps. This lesson covers the gated cells that pushed that horizon to hundreds of steps and ran most sequence work until transformers (05.04).

## The idea in one picture

The simplest memory test: each sequence starts with one bit, $+1$ or $-1$, followed by a gap of pure noise steps. At the end the network must say what the first bit was. The only way to succeed is to carry the bit across the gap.

I trained six small networks (16 hidden units, 600 Adam updates, two seeds each) and measured test accuracy on 2,000 fresh sequences. "Forget bias" and "update bias" are starting values explained in Step 3; read bias 3 as "set up to remember from the start".

| Model (gap →) | 10 | 20 | 40 | 80 | 160
| --- | --- | --- | --- | --- | --- |
| Plain RNN | 100% | 75% | 50% | 50% | 50%
| GRU, default init | 100% | 100% | 50% | 50% | 51%
| LSTM, forget bias 0 | 100% | 75% | 50% | 50% | 50%
| LSTM, forget bias 1 | 100% | 100% | 100% | 50% | 50%
| LSTM, forget bias 3 | 100% | 100% | 100% | 100% | 50%
| GRU, update bias 3 | 100% | 100% | 100% | 100% | 100%

A 75% means one seed solved the task and the other stayed at guessing.

~~~dg-fig 05.02-gap.svg
Accuracy on the remember-the-first-bit task for each model and gap (mean of two seeds; 50% is guessing). Moving down the rows, the starting setup favours remembering more and more, and the green region reaches further right.
~~~

Every network stores a bit for 10 steps. By a gap of 40 the plain RNN is at a coin flip, and so are the gated networks whose memory starts out leaky (default GRU, LSTM with forget bias 0). Gated networks set up to remember keep going to 40, 80 and, for the GRU, 160 steps. All of them could *represent* "copy the bit forward"; the question is whether training can *find* it, which depends on whether a gradient from the end still reaches step 1.

Two terms: a **gate** is a vector of sigmoid outputs in $(0, 1)$ that multiplies another vector elementwise (1 passes a signal, 0 blocks it, 0.3 lets 30% through), computed fresh at each step from the input and previous state. The **cell state** $c_t$ is the LSTM's long-term memory, updated mostly by addition.

## Step 1: give the memory its own lane

In a plain RNN every step rewrites $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$ completely, so keeping a value means the matrix and tanh must reproduce it exactly, every step.

The **LSTM** (long short-term memory; Hochreiter and Schmidhuber, 1997, with the forget gate added by Gers et al., 2000) adds a second state, the cell $c_t$, and updates it with an almost trivial rule:

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t.$$

In words: keep a fraction $f_t$ of the old memory, and add a fraction $i_t$ of some new content $g_t$. There is no weight matrix between $c_{t-1}$ and $c_t$, and no squashing. If $f_t = 1$ and $i_t = 0$, the memory passes through untouched, for as many steps as you like.

~~~dg-flow The cell-state path through one LSTM step. Nothing on it multiplies by a weight matrix.
c(t-1) -(× forget gate)-> kept memory -(+ input gate × candidate)-> *c(t) -(tanh, × output gate)-> h(t)
~~~

Hochreiter called this path the **constant error carousel**: an error signal rides along it backwards without being rescaled. The same additive path reappears in ResNets (module 03) and transformers.

## Step 2: the gates that run the lane

The LSTM computes $f_t$, $i_t$ and the new content $g_t$ at every step from the input $x_t$ and previous hidden state $h_{t-1}$. With $[h_{t-1}, x_t]$ the two stacked into one vector and $\sigma$ the sigmoid:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

$$g_t = \tanh(W_g [h_{t-1}, x_t] + b_g), \quad c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t)$$

- $x_t$ is the input (size $d$); $h_t$ and $c_t$ are the hidden and cell states (size $H$).
- The **forget gate** $f_t$: fraction of each cell unit to keep (near 1 remember, near 0 wipe).
- The **input gate** $i_t$: how much of the candidate to write.
- The **candidate** $g_t$: proposed new content, in $(-1, 1)$ so it can push memory up or down.
- The **output gate** $o_t$: how much memory to expose as $h_t$, which the next layer and the next step's gates see.
- Each $W$ is $H \times (H + d)$; four of them give four times a plain RNN's parameters.

Below is the forget-bias-1 LSTM trained at a 40-step gap, showing the cell unit whose final value best separates the two classes.

~~~dg-fig 05.02-trace.svg
Left: the chosen cell unit moves one way or the other at step 1, depending on the bit, and settles near plus or minus 4. Right: averaged over 200 sequences, the forget gate sits near 0.82, the input gate near 0.71, and the candidate always agrees in sign with the stored bit, so each step tops the memory back up.
~~~

Surprise: the forget gate is not at 1 but about 0.82, and the input gate stays open at about 0.71. The memory survives because the candidate agrees with what is stored (sign-matched value about $+1.0$), so each step leaks 18% and writes back the same amount. The balance point is $c^* = i\,g/(1 - f) \approx 0.71 \times 1.0 / 0.18 \approx 4.0$, matching the measured 4.03. Trained networks often find such self-reinforcing loops rather than perfectly closed gates.

<div class="ex">
<div class="lab">Worked example</div>

**one LSTM step by hand (one unit)**

Take a single cell with old memory $c_{t-1} = 2$. Suppose the weights and inputs produce these pre-activations (the numbers inside $\sigma$ and tanh): forget 2, input 0, candidate 1, output 1.

Gates: $f = \sigma(2) = 0.881$, $i = \sigma(0) = 0.5$, $o = \sigma(1) = 0.731$. Candidate: $g = \tanh(1) = 0.762$.

Kept memory: $f \cdot c_{t-1} = 0.881 \times 2 = 1.762$. New write: $i \cdot g = 0.5 \times 0.762 = 0.381$.

New cell: $c_t = 1.762 + 0.381 = 2.142$. The memory moved only a little.

Output: $\tanh(2.142) = 0.973$, so $h_t = 0.731 \times 0.973 = 0.711$. Note that $c_t$ can grow beyond 1 (it is a sum), while $h_t$ is always in $(-1, 1)$.

</div>

## Step 3: why the gradient survives, and the forget-gate bias

Take the derivative of the cell update with respect to the old cell, treating the gates as given:

$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t) \quad (\text{plus smaller terms through the gates}).$$

In a plain RNN each backward hop multiplies by $\text{diag}(1 - h_t^2)\, W_{hh}$; here it is just the forget-gate values, so across $k$ steps the gradient is scaled by $f_{t} f_{t-1} \cdots f_{t-k+1}$. Where the network learns $f \approx 1$, gradient flows back hundreds of steps almost unchanged.

That only helps if $f$ is near 1 *before* training, or no gradient arrives to teach it. With small random weights, $f \approx \sigma(b_f)$:

- $b_f = 0$ gives $f \approx 0.5$: after 10 steps $0.5^{10} \approx 0.001$ remains, the plain-RNN problem again.
- $b_f = 1$ gives $f \approx 0.73$: after 10 steps, 4.4% remains; after 50 steps, $1.6 \times 10^{-7}$.
- $b_f = 3$ gives $f \approx 0.95$: after 10 steps, 62% remains; after 50 steps, 8.8%.
- $b_f = -3$ gives $f \approx 0.05$: memory is wiped almost completely every step.

~~~dg-fig 05.02-forget.svg
Left: the fraction of a stored value left after k steps is f to the power k (log scale). Right: the measured gradient of the final hidden state with respect to each earlier input, in untrained networks with 32 units. The plain RNN, the default GRU and the zero-bias LSTM lose about 20 orders of magnitude over 100 steps; the LSTM with forget bias 3 barely loses one.
~~~

The right panel is measured, not drawn. Fifty steps back, the gradient in the plain RNN is about $2 \times 10^{-11}$ of its size at the last step, the same as the zero-bias LSTM; with forget bias 1 it is $7 \times 10^{-5}$, and with forget bias 3 it is 0.28. The GRU with update bias 3 keeps $10^{-2}$ even 100 steps back. This is why the bias-3 models learned the longest gaps. A forget bias of 1 is the classic recommendation (Jozefowicz et al., 2015); long-range tasks need more, or the *chrono* initialisation under Going deeper. PyTorch does **not** do this for you: its default draws every bias uniformly from $\pm 1/\sqrt{H}$.

## Step 4: the GRU, the same idea with fewer parts

The **gated recurrent unit** (Cho et al., 2014) drops the separate cell and keeps one state $h_t$. It has two gates. The **update gate** $z_t$ decides how much of the old state to keep; the **reset gate** $r_t$ decides how much of the old state the candidate may look at:

$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r), \quad z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$

$$n_t = \tanh(W_n x_t + r_t \odot (U_n h_{t-1}) + b_n), \quad h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$

$W$ acts on the input, $U$ on the previous state, $n_t$ is the candidate. The last equation is a weighted average: with $z = 0.9$ the new state is 90% old, 10% candidate. The LSTM's forget and input gates are tied into $z$ and $1 - z$, and $\partial h_t / \partial h_{t-1}$ contains $\text{diag}(z_t)$: the same additive path.

In this convention (Cho's and PyTorch's) $z \approx 1$ keeps the old state; some textbooks swap $z$ and $1 - z$, so check before porting weights. An update bias of 3 ($z \approx 0.95$ at the start) is the GRU's forget-gate trick, and that GRU alone solved the 160-step gap. One toy task does not show GRUs remember better in general; it does show the starting bias matters more than the cell type.

| | Plain RNN | LSTM | GRU
| --- | --- | --- | --- |
| States | $h$ | $h$ and $c$ | $h$
| Gates | none | forget, input, output | update, reset
| Parameters (relative) | 1× | 4× | 3×
| Long-range memory | poor (about 10-20 steps) | good | good

<div class="ex">
<div class="lab">Worked example</div>

**parameter counts for one layer**

Input $d = 300$ (word vectors), hidden $H = 512$. One block of weights plus bias has $H(d + H + 1) = 512 \times 813 = 416{,}256$ numbers.

Plain RNN: one block, $416{,}256$.

LSTM: four blocks (i, f, g, o), $4 \times 416{,}256 = 1{,}665{,}024$. PyTorch keeps two bias vectors per block, adding $4 \times 512 = 2{,}048$, for $1{,}667{,}072$ in total.

GRU: three blocks (r, z, n), $3 \times 416{,}256 = 1{,}248{,}768$, a quarter fewer than the LSTM ($1{,}250{,}304$ in PyTorch).

</div>

## LSTMs and GRUs in code

The code sets the forget-gate bias, runs a batch, and re-implements the loop by hand to confirm the gate order:

```python
import torch, torch.nn as nn
torch.manual_seed(0)
lstm = nn.LSTM(input_size=2, hidden_size=16, batch_first=True)
# weight_ih_l0 (64, 2), weight_hh_l0 (64, 16), bias_ih_l0 (64,), bias_hh_l0 (64,)
H = 16
with torch.no_grad():                       # rows are [input | forget | cell | output]
    lstm.bias_ih_l0[H:2*H].fill_(1.0)       # forget-gate bias = 1 ...
    lstm.bias_hh_l0[H:2*H].zero_()          # ... in total (the two biases are added)

x = torch.randn(8, 40, 2)                   # (batch, time, features)
out, (h_T, c_T) = lstm(x)
print(out.shape, h_T.shape, c_T.shape)
# torch.Size([8, 40, 16]) torch.Size([1, 8, 16]) torch.Size([1, 8, 16])
print(torch.allclose(out[:, -1], h_T[0]))   # True: last output = final hidden state

# Re-implement the loop by hand and check it matches PyTorch
W = torch.cat([lstm.weight_ih_l0, lstm.weight_hh_l0], 1)
b = lstm.bias_ih_l0 + lstm.bias_hh_l0
h = c = torch.zeros(8, H)
for t in range(40):
    z = torch.cat([x[:, t], h], 1) @ W.T + b
    i, f, g, o = z.chunk(4, 1)
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h = torch.sigmoid(o) * torch.tanh(c)
print(torch.allclose(h, h_T[0], atol=1e-6))  # True
```

Things to notice:

- LSTM blocks are ordered input, forget, cell, output; GRU blocks reset, update, new. In both, rows `H:2H` are the "keep" gate.
- The two bias vectors are added: setting both to 1 gives a forget bias of 2.
- `out` holds $h_t$ for every step (for tagging, or as encoder memory in 05.03); `h_T`, `c_T` are final states only.

## Reading the result

A recurrent model that "trains" may have learned nothing long-range. The curves at a 40-step gap:

~~~dg-fig 05.02-curves.svg
Training loss at a 40-step gap (smoothed over 15 updates). Models that cannot carry the bit sit at ln 2 = 0.69, the loss of guessing; the others drop to near zero once the gradient finds the bit, after very different waiting times.
~~~

The models without a remembering start never leave the guessing line. The bias-3 models leave it within about 100 updates; the bias-1 LSTM waits about 300 updates and then drops abruptly. That plateau-then-cliff shape is typical: a tiny gradient slowly opens the gates, then the loss falls fast. A flat loss may be a plateau, not a dead end.

Checks that tell you whether memory is working:

- **Loss at the guessing level** ($\ln K$ for $K$ classes): no gradient reaches the information. Try a larger forget bias before a bigger model.
- **Accuracy by distance**: bucket validation examples by how far back the needed information is; averages hide failing long cases.
- **Gate statistics**: mean forget gate near 0.5 means memory is not in use yet; everything saturated at 0 or 1 means blown-up pre-activations.
- **Gradient norm**: log and clip it (`clip_grad_norm_(params, 1.0)`). Gates fix vanishing, not exploding.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss flat at $\ln K$ for many epochs | forget gate starts near 0.5, no gradient reaches early steps | forget bias 1 or more (GRU: update bias), or chrono init
| Loaded weights give nonsense outputs | gate order differs between formats (PyTorch: i, f, g, o; ONNX: i, o, f, c) | permute the row blocks; test against a hand-written step
| Forget bias seems to have no effect | set on both `bias_ih` and `bias_hh`, or overwritten by a custom init later | set the sum once, after all other initialisation
| GRU "remember" init makes things worse | used the textbook convention where $1 - z$ keeps the state | check which side $z$ multiplies in your framework
| Results depend on padding length | padded steps update the state | pack sequences (`pack_padded_sequence`) or mask the loss and states
| Memory destroyed by dropout | fresh dropout mask on the recurrent state at every step | dropout between layers, or one mask for all steps

## Practical choices

- **Which cell?** Greff et al. (2015) compared eight LSTM variants over thousands of runs: none beat the standard LSTM significantly, and removing the forget gate or the output activation hurt most. GRUs perform comparably on most tasks; pick the GRU when parameters or speed matter.
- **Stacking**: 2 to 4 layers, dropout between them (`num_layers=2, dropout=0.2`).
- **Bidirectional**: a forward and a backward LSTM, states concatenated. Right for tagging (06.05) and encoding (05.03), wrong for generation, which must not see the future.
- **Speed**: the $T$ steps still run one after another. This, more than accuracy, is why transformers (05.04) replaced LSTMs for large-scale text.

## Going deeper

**The full Jacobian.** The gates are computed from $h_{t-1} = o_{t-1} \odot \tanh(c_{t-1})$, so $c_t$ also depends on $c_{t-1}$ through weight matrices and squashing slopes; those terms shrink like a plain RNN's. Only the direct term $\text{diag}(f_t)$ can stay near the identity. The 1997 LSTM truncated the indirect terms; full BPTT keeps them, and they matter for learning *when* to open gates.

**Time constants.** A unit with a constant forget value $f$ remembers with a characteristic time of about $1/(1-f)$ steps: $f = 0.73$ gives about 4 steps, $f = 0.95$ about 20, $f = 0.99$ about 100. Tallec and Ollivier (2018) turned this into the **chrono initialisation**: if dependencies can span up to $T_{\max}$ steps, draw $b_f = \log(u)$ with $u$ uniform on $[1, T_{\max} - 1]$ and set $b_i = -b_f$, so units start with a spread of memory lengths up to $T_{\max}$. (Recipe stated from memory; check the paper for exact constants.)

**Variants.** *Peephole* connections let the gates read $c_{t-1}$ directly. *Coupled* input-forget gates (CIFG) set $i = 1 - f$, which is half-way to a GRU. Greff et al. found neither changes much on average.

**Memory capacity.** An LSTM's memory is still a fixed-size vector of $H$ numbers. Gates choose what to keep but cannot add room; squeezing a long sentence into one final vector is the bottleneck attention (05.03) removes.

## Key terms

| Term | Meaning
| --- | --- |
| LSTM | Recurrent cell with a separate, additively updated cell state and three gates
| Cell state $c_t$ | The LSTM's long-term memory vector, updated by $f \odot c + i \odot g$
| Forget gate $f$ | Fraction of each cell unit kept from the previous step
| Input / output gate | Fractions of the candidate written into the cell / of the squashed cell exposed as $h_t$
| Constant error carousel | The cell path, along which gradients are scaled only by forget-gate values
| Forget-gate bias | Bias of the forget gate; about 1 or more at initialisation keeps memory open
| GRU | Two-gate cell (update $z$, reset $r$) that interpolates between old state and candidate
| Update gate $z$ | In PyTorch's convention, the fraction of the old GRU state kept

<div class="take">
<div class="lab">Key takeaway</div>

gates give recurrent networks an additive memory path whose gradient is scaled by learned gate values instead of a repeated weight matrix. LSTMs and GRUs differ in bookkeeping, not in principle; initialise the forget (or update) gate to remember, check accuracy against distance, and still clip gradients.

</div>
