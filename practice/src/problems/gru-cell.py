import numpy as np

META = dict(title='GRU over a sequence (PyTorch gate convention)', tags=['Neural Networks', 'NLP', 'Time Series'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:04.24'],
            related=['vanilla-rnn-cell', 'lstm-cell', 'sequence-to-sequence-forward-pass'])

STATEMENT = r"""
The gated recurrent unit adds two gates to the plain RNN: a **reset** gate $r$ that decides how much of the old state feeds the candidate, and an **update** gate $z$ that interpolates between old state and candidate. Use the convention of `torch.nn.GRU`:

$$\begin{aligned}
r_t &= \sigma(x_t W_{xr} + b_{xr} + h_{t-1} W_{hr} + b_{hr}) \\
z_t &= \sigma(x_t W_{xz} + b_{xz} + h_{t-1} W_{hz} + b_{hz}) \\
n_t &= \tanh\big(x_t W_{xn} + b_{xn} + r_t \odot (h_{t-1} W_{hn} + b_{hn})\big) \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}
\end{aligned}$$

Note the reset gate multiplies $h_{t-1} W_{hn} + b_{hn}$, after the hidden projection and its bias.

Write `gru_forward(x, h0, Wx, Wh, bx, bh)` that returns all hidden states.

## Packed weights
- `x`: `(N, T, D)`; `h0`: `(N, H)`.
- `Wx`: `(D, 3H)`, `Wh`: `(H, 3H)`, `bx`, `bh`: `(3H,)`. Columns are packed in the order **r, z, n**: columns `0:H` belong to $r$, `H:2H` to $z$, `2H:3H` to $n$.

## Output
Float array `(N, T, H)` whose step $t$ is $h_t$ (so the last step is $h_T$).
"""

STARTER = '''import numpy as np

def gru_forward(x, h0, Wx, Wh, bx, bh):
    """x: (N, T, D); h0: (N, H); Wx: (D, 3H); Wh: (H, 3H); bx, bh: (3H,). Return (N, T, H)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sigmoid(a):
    e = np.exp(-np.abs(a))
    return np.where(a >= 0, 1 / (1 + e), e / (1 + e))

def gru_forward(x, h0, Wx, Wh, bx, bh):
    x, h = np.asarray(x, dtype=float), np.asarray(h0, dtype=float)
    N, T, _ = x.shape
    H = h.shape[1]
    gx = x @ Wx + bx                                   # (N, T, 3H): input side for every step
    out = np.zeros((N, T, H))
    for t in range(T):
        gh = h @ Wh + bh                               # (N, 3H): hidden side, including b_hn
        r = _sigmoid(gx[:, t, :H] + gh[:, :H])
        z = _sigmoid(gx[:, t, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gx[:, t, 2 * H:] + r * gh[:, 2 * H:])   # reset applied after the hidden bias
        h = (1 - z) * n + z * h
        out[:, t] = h
    return out
'''

HINTS = [
    'Precompute the input side `x @ Wx + bx` for all steps, shape `(N, T, 3H)`. Inside the loop compute the hidden side `h @ Wh + bh`, shape `(N, 3H)`.',
    'Slice both into thirds `[:H]`, `[H:2H]`, `[2H:]` for r, z, n. r and z simply add the two sides before the sigmoid.',
    'For the candidate, multiply r with the **whole** hidden-side slice (it already includes $b_{hn}$), add the input side, then tanh. Finally $h = (1 - z) n + z h$: $z$ is the weight on the **old** state.',
]

_r = np.random.default_rng(67)
def _p(D, H, s=0.5):
    return _r.normal(size=(D, 3 * H)) * s, _r.normal(size=(H, 3 * H)) * s, _r.normal(size=3 * H) * 0.3, _r.normal(size=3 * H) * 0.3
TESTS = [
    (np.array([[[1.0]]]), np.zeros((1, 1)), np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros(3), np.zeros(3)),
    (np.array([[[0.0], [0.0]]]), np.ones((1, 1)), np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.zeros(3), np.array([0.0, 0.0, 1.0])),
    (_r.normal(size=(2, 4, 3)), _r.normal(size=(2, 5)), *_p(3, 5)),
    (_r.normal(size=(3, 6, 2)), np.zeros((3, 4)), *_p(2, 4)),
    (_r.normal(size=(1, 3, 4)) * 10, _r.normal(size=(1, 2)), *_p(4, 2, s=5.0)),
    (_r.normal(size=(2, 1, 1)), _r.normal(size=(2, 3)), *_p(1, 3)),
]

EXPLANATION = r"""
## What the gates do
- $z_t \approx 1$: copy the old state forward unchanged. This is the additive, near-identity path that lets gradients survive many steps, the same idea as an LSTM's forget gate or a residual connection.
- $r_t \approx 0$: compute the candidate as if the past didn't exist, which helps the unit reset at boundaries in the input.

## Two conventions
The original paper applies the reset gate **before** the hidden projection, $\tanh(x W_{xn} + (r \odot h) W_{hn})$. cuDNN and PyTorch use $r \odot (h W_{hn} + b_{hn})$ because it lets all three hidden projections be one matrix product per step. The two are not equivalent, so pretrained weights only work with the convention they were trained under. Test 2 has a nonzero $b_{hn}$, which separates them.

## Complexity
Per step $O(N(3DH + 3H^2))$: about 3x a vanilla RNN, and 3/4 of an LSTM.

## Common bugs
- Swapping the roles of $z$ and $1 - z$ (PyTorch's $z$ keeps the **old** state).
- Applying $r$ to $h W_{hn}$ but adding $b_{hn}$ outside the product.
- Packing the gates in the order z, r, n (Keras's order) when reading these weights.

## Interview follow-ups
- GRU or LSTM: what does the LSTM's separate cell state buy you, and when does it matter?
- Count the parameters of a GRU layer with input size $D$ and hidden size $H$ (with both bias vectors).
"""
