---
id: "05.02"
course: "deep-learning"
module: "05"
slug: "lstms-and-grus"
title: "LSTMs and GRUs"
keyIdeas: ["An LSTM keeps a separate cell state updated additively: $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$. Gradients along $c$ are scaled by the forget gate, not by a repeated weight matrix, so they survive when $f_t \\approx 1$.", "Gates are sigmoids in $(0, 1)$ computed from $x_t$ and $h_{t-1}$: forget ($f$), input ($i$), output ($o$), plus a tanh candidate $g$. An LSTM has 4× the parameters of a plain RNN of the same width.", "Initialise the forget-gate bias to about 1 so the cell starts out remembering; at zero it starts out forgetting half its state every step.", "A GRU merges cell and hidden state and uses two gates (update, reset): 3× the RNN's parameters, usually similar accuracy. Watch the convention for which side of the interpolation $z$ multiplies.", "In large comparisons, the forget gate and the output activation matter most; most other LSTM variants make little difference."]
question: "Why does initialising the LSTM forget-gate bias to 1 help early training, and what would you expect if it were initialised to -3 instead?"
answer: "With bias 0 the forget gate starts at about 0.5, so each cell unit loses half its content per step and the gradient along the cell path also shrinks by half per step; information and credit vanish over about 10 steps, exactly the plain-RNN problem the LSTM was meant to avoid, and the network must first learn to open the gate before it can learn long dependencies. Bias 1 gives f ≈ 0.73 initially (higher bias gives more), so memory and gradients persist much longer from the first update, and the model can discover long-range patterns early. With bias -3, f ≈ 0.05: the cell is wiped almost completely every step, the gradient along c is multiplied by about 0.05 per step, and the LSTM behaves like a memoryless RNN until (and unless) training drives the bias up."
sources: [{"title": "Understanding LSTM Networks (Olah, 2015)", "url": "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", "note": "The standard visual walkthrough of each gate, plus GRU and peephole variants."}, {"title": "Learning Phrase Representations using RNN Encoder-Decoder (Cho et al., 2014)", "url": "https://arxiv.org/abs/1406.1078", "note": "Introduces the GRU (section 2.3) alongside the encoder-decoder."}, {"title": "LSTM: A Search Space Odyssey (Greff et al., 2015)", "url": "https://arxiv.org/abs/1503.04069", "note": "Large ablation of LSTM variants; read the conclusions on which components matter."}, {"title": "PyTorch nn.LSTM", "url": "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html", "note": "Exact equations, gate order and weight shapes used by PyTorch."}]
practice: ["lstm-cell", "gru-cell", "lstm-cell-2"]
tags: ["LSTM", "GRU", "gating"]
tools: ["NumPy", "PyTorch"]
---

## The fix for vanishing gradients

The previous lesson showed that in a plain RNN the gradient across $n$ steps is a product of $n$ Jacobians, each containing $W_{hh}$ and a tanh derivative, so it decays or blows up exponentially. The LSTM (Hochreiter and Schmidhuber, 1997; forget gate added by Gers et al., 2000) adds a second state, the cell $c_t$, whose update is mostly additive. That changes the gradient path from a matrix product into an elementwise product of gate values that the network can choose to keep near 1.

## LSTM equations

With $[h_{t-1}, x_t]$ the concatenation of the previous hidden state and the input, and $\sigma$ the sigmoid:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

$$g_t = \tanh(W_g [h_{t-1}, x_t] + b_g), \quad c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t)$$

- The **forget gate** $f$ decides what fraction of each cell unit to keep.
- The **input gate** $i$ decides how much of the new candidate $g$ to write.
- The **output gate** $o$ decides how much of the cell to expose as $h_t$, which is what the next layer and the next step's gates see.

~~~dg-flow The cell-state path through one LSTM step. Nothing on it multiplies by a weight matrix.
c(t-1) -(× forget gate)-> kept memory -(+ input gate × candidate)-> *c(t) -(tanh, × output gate)-> h(t)
~~~

Along this path, $\partial c_t / \partial c_{t-1} = \text{diag}(f_t)$ (plus smaller terms through the gates). If the network learns $f \approx 1$ for a unit, the gradient flows back through hundreds of steps almost unchanged. Hochreiter called this the constant error carousel. The same idea, an identity-like additive path, reappears in ResNets (module 03) and in the residual stream of transformers.

```python
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
def lstm_step(x, h, c, W, b):             # W: (4H, d+H), rows in order i, f, g, o
    z = W @ np.concatenate([x, h]) + b
    i, f, g, o = np.split(z, 4)
    c = sig(f) * c + sig(i) * np.tanh(g)
    h = sig(o) * np.tanh(c)
    return h, c
```

PyTorch stacks the four blocks of `weight_ih_l0` in the order input, forget, cell (candidate), output, and keeps two bias vectors (`bias_ih` and `bias_hh`) that are simply added. Loading weights from another framework with a different gate order is a classic silent bug.

## Forget-gate bias

At initialisation with zero bias, $f_t \approx \sigma(0) = 0.5$: every unit halves its memory each step, so after 10 steps only $0.5^{10} \approx 0.001$ of the original remains. The gradient path the LSTM was designed to provide is closed until training opens it. Setting $b_f \approx 1$ gives $f \approx 0.73$ at the start, and many implementations go higher. Setting it strongly negative would make the LSTM behave like a forgetful plain RNN.

<div class="ex">
<div class="lab">Worked example</div>

**parameter counts for one layer**

Input $d = 300$ (word vectors), hidden $H = 512$.

Plain RNN: $512 \times (300 + 512 + 1) = 416{,}256$.

LSTM: $4 \times 416{,}256 = 1{,}665{,}024$ (PyTorch's two bias vectors add another 2,048).

GRU: $3 \times 416{,}256 = 1{,}248{,}768$, a quarter fewer than the LSTM.

</div>

## The GRU

The gated recurrent unit (Cho et al., 2014) has no separate cell. An update gate $z$ interpolates between the old state and a candidate; a reset gate $r$ controls how much of the old state the candidate may read:

$$r_t = \sigma(W_r x_t + U_r h_{t-1}), \quad z_t = \sigma(W_z x_t + U_z h_{t-1})$$

$$n_t = \tanh(W_n x_t + r_t \odot (U_n h_{t-1})), \quad h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$

In this convention (Cho's and PyTorch's), $z \approx 1$ keeps the old state. Some textbooks swap $z$ and $1 - z$; check before porting weights. The interpolation is again an additive path, so the GRU fights vanishing gradients for the same reason as the LSTM.

| | Plain RNN | LSTM | GRU |
| --- | --- | --- | --- |
| States | $h$ | $h$ and $c$ | $h$ |
| Gates | none | forget, input, output | update, reset |
| Parameters (relative) | 1× | 4× | 3× |
| Long-range memory | poor (about 10-20 steps) | good | good |

## Practical choices

- **Which cell?** Greff et al. (2015) compared eight LSTM variants over thousands of runs: none beat the standard LSTM significantly, and removing the forget gate or the output activation hurt most. GRUs perform comparably on most tasks; pick the GRU when parameters or speed matter.
- **Stacking**: 2 to 4 layers are typical, with dropout between layers (not on the recurrent connection, unless you use the same mask at every step).
- **Bidirectional**: run one LSTM forward and one backward and concatenate their states. Each position then sees both past and future, which is right for tagging (06.05) and encoding (05.03), and wrong for generation, which must not see the future.
- **Gradient clipping** is still needed; gates mitigate vanishing, not exploding.

<div class="take">
<div class="lab">Key takeaway</div>

gates give recurrent networks an additive memory path whose gradient is scaled by learned gate values instead of a repeated weight matrix. LSTMs and GRUs differ in bookkeeping, not in principle; initialise the forget gate to remember.

</div>
