import numpy as np

META = dict(title='Elman RNN over padded sequences', tags=['Neural Networks', 'NLP', 'Time Series'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:04.24', 'air-fanout-v1:04.27'],
            related=['gru-cell', 'lstm-cell', 'sequence-to-sequence-forward-pass'])

STATEMENT = r"""
A vanilla (Elman) RNN reads a sequence one step at a time, carrying a hidden state:

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b).$$

Batches mix sequences of different lengths, padded to a common $T$. Padding steps must not change a sequence's state.

Write `rnn_forward(x, h0, Wx, Wh, b, lengths=None)` that returns `(outputs, h_last)`.

## Shapes (batch first)
- `x`: `(N, T, D)`; `h0`: `(N, H)`; `Wx`: `(D, H)`; `Wh`: `(H, H)`; `b`: `(H,)`.
- `lengths`: int array `(N,)` with $0 \le$ `lengths[n]` $\le T$, or `None` meaning every sequence has length $T$.

## Output
- `outputs`: `(N, T, H)`. For sample $n$, row $t$ is $h_t$ for $t <$ `lengths[n]` and **zeros** for padded steps.
- `h_last`: `(N, H)`, each sample's state after its **last valid** step (`h0[n]` itself if its length is 0).
"""

STARTER = '''import numpy as np

def rnn_forward(x, h0, Wx, Wh, b, lengths=None):
    """x: (N, T, D); h0: (N, H). Return (outputs (N, T, H), h_last (N, H))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rnn_forward(x, h0, Wx, Wh, b, lengths=None):
    x, h = np.asarray(x, dtype=float), np.asarray(h0, dtype=float)
    N, T, _ = x.shape
    lengths = np.full(N, T) if lengths is None else np.asarray(lengths)
    xw = x @ Wx + b                                    # input projections for all steps at once
    out = np.zeros((N, T, h.shape[1]))
    for t in range(T):                                 # the recurrence itself is sequential
        h_new = np.tanh(xw[:, t] + h @ Wh)
        live = (t < lengths)[:, None]                  # which sequences are still running
        h = np.where(live, h_new, h)                   # padded steps keep the old state
        out[:, t] = np.where(live, h_new, 0.0)
    return out, h
'''

HINTS = [
    'The input projection $x_t W_x + b$ doesn\'t depend on the state, so compute it for every step with one product before the loop. Only $h_{t-1} W_h$ must be sequential.',
    'At step $t$ a sequence is still running iff `t < lengths[n]`. Make that a `(N, 1)` boolean column.',
    'Use it twice with `np.where`: to keep the old state for finished sequences, and to write zeros into their output rows. Then `h` after the loop is already each sample\'s last valid state.',
]

_r = np.random.default_rng(61)
_p = lambda D, H: (_r.normal(size=(D, H)) * 0.5, _r.normal(size=(H, H)) * 0.5, _r.normal(size=H) * 0.1)
TESTS = [
    (np.array([[[1.0], [0.0]]]), np.zeros((1, 1)), np.array([[1.0]]), np.array([[1.0]]), np.zeros(1)),
    (np.array([[[1.0], [2.0]], [[1.0], [5.0]]]), np.zeros((2, 1)), np.array([[0.5]]), np.array([[1.0]]), np.zeros(1), np.array([2, 1])),
    (_r.normal(size=(3, 5, 4)), _r.normal(size=(3, 6)), *_p(4, 6)),
    (_r.normal(size=(3, 5, 4)), _r.normal(size=(3, 6)), *_p(4, 6), np.array([5, 2, 0])),
    (_r.normal(size=(2, 1, 3)), np.zeros((2, 2)), *_p(3, 2), np.array([1, 1])),
    (_r.normal(size=(4, 7, 2)), np.zeros((4, 3)), *_p(2, 3), np.array([3, 7, 1, 6])),
    (_r.normal(size=(1, 12, 3)) * 3, _r.normal(size=(1, 5)), *_p(3, 5)),
]

EXPLANATION = r"""
## What can be parallel, what can't
$x_t W_x$ for all $t$ is one big matrix product, and frameworks do exactly that (cuDNN fuses it). The $h_{t-1} W_h$ term creates a dependency chain of length $T$, which is the fundamental reason RNNs train slower than transformers on long sequences.

## Masking
Freezing the state on padded steps makes the result for a sequence independent of how much padding it got, so `h_last` is its true final state. This is what `pack_padded_sequence` achieves in PyTorch. Without it, a short sequence keeps evolving on padding inputs (often zeros, which still change $h$ through $h W_h + b$).

## Vanishing and exploding gradients
$\partial h_t/\partial h_{t-1} = \operatorname{diag}(1 - h_t^2)\,W_h^\top$. Backprop through $k$ steps multiplies $k$ of these, so gradients shrink or blow up geometrically depending on the spectral radius of $W_h$. Clipping handles the explosion; gating (LSTM, GRU) handles the vanishing.

## Complexity
$O(N T (DH + H^2))$ time, $O(NTH)$ memory to keep the outputs for backprop.

## Common bugs
- Writing $W_h h$ with row-vector states: shapes only work by accident when $W_h$ is square, and the result is $W_h^\top$.
- Returning `outputs[:, -1]` as the last state: that's a zero row for any padded sequence.
- Off-by-one in the mask (`t <= lengths`).

## Interview follow-ups
- Derive backprop through time for this cell and say where truncated BPTT cuts the graph.
- How does a bidirectional RNN combine states, and why can't it be used for autoregressive generation?
"""
