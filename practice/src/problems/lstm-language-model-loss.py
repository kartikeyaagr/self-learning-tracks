import numpy as np

META = dict(title='LSTM language model over a padded batch', tags=['Neural Networks', 'Language Models'], diff='Hard',
            runtime='numpy', lessons=['dlcore:05.02'],
            related=['lstm-cell', 'lstm-cell-2', 'language-modeling', 'perplexity-computation'])

STATEMENT = r"""
Put an LSTM to work as a next-token language model over a batch of sequences of different lengths.

Write `lstm_lm(ids, lengths, p)` returning `(loss, h_last, c_last)`.

## Inputs
- `ids` `(B, L)` int, `lengths` `(B,)`: row $b$ has real tokens in positions $0..\text{lengths}_b - 1$ (at least 2) and padding after.
- `p`: `E` `(V, D)`, `W_ih` `(D, 4H)`, `W_hh` `(H, 4H)`, `b` `(4H,)`, `W_out` `(H, V)`, `b_out` `(V,)`.

## Model
- Step $t$ reads the embedding of `ids[:, t]` for $t = 0..L-2$, starting from zero $h$ and $c$.
- Gates, PyTorch order: $[i, f, g, o] = $ split of $x_tW_{ih} + h_{t-1}W_{hh} + b$ into four blocks of $H$; $i, f, o$ through a sigmoid, $g$ through tanh. $c_t = f\odot c_{t-1} + i\odot g$, $h_t = o\odot\tanh(c_t)$.
- **Padding**: if position $t$ is not a real token of row $b$ ($t \ge \text{lengths}_b$), that row's $h$ and $c$ stay unchanged.
- Logits at step $t$: $h_tW_{out} + b_{out}$, predicting `ids[:, t+1]`. A prediction counts only if $t + 1 < \text{lengths}_b$.

## Output
- `loss`: mean cross-entropy over all counted predictions in the batch (token-weighted, not per-sequence).
- `h_last`, `c_last`: `(B, H)`, each row's state after its last real token.
"""

STARTER = '''import numpy as np

def lstm_lm(ids, lengths, p):
    """Return (loss, h_last (B, H), c_last (B, H))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))

def lstm_lm(ids, lengths, p):
    ids, lengths = np.asarray(ids), np.asarray(lengths)
    B, L = ids.shape
    H = p['W_hh'].shape[0]
    h, c = np.zeros((B, H)), np.zeros((B, H))
    total, count = 0.0, 0
    for t in range(L):
        real = (t < lengths)[:, None]
        a = p['E'][ids[:, t]] @ p['W_ih'] + h @ p['W_hh'] + p['b']
        i, f, g, o = _sig(a[:, :H]), _sig(a[:, H:2 * H]), np.tanh(a[:, 2 * H:3 * H]), _sig(a[:, 3 * H:])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        c, h = np.where(real, c_new, c), np.where(real, h_new, h)
        if t < L - 1:
            scored = t + 1 < lengths
            if scored.any():
                z = h[scored] @ p['W_out'] + p['b_out']
                z = z - z.max(1, keepdims=True)
                logp = z - np.log(np.exp(z).sum(1, keepdims=True))
                total -= logp[np.arange(scored.sum()), ids[scored, t + 1]].sum()
                count += int(scored.sum())
    return total / count, h, c
'''

HINTS = [
    'One loop over time with the whole batch at once. Compute the new $(h, c)$ for every row, then keep the old values where the position is padding: `np.where(real[:, None], new, old)`.',
    'The gate block order is input, forget, cell candidate, output. Only the candidate uses tanh.',
    'Accumulate the summed negative log-likelihood and the number of counted predictions separately, and divide once at the end.',
]

_r = np.random.default_rng(1997)
def _p(V, D, H):
    return dict(E=_r.normal(size=(V, D)), W_ih=_r.normal(size=(D, 4 * H)) / np.sqrt(D), W_hh=_r.normal(size=(H, 4 * H)) / np.sqrt(H),
                b=np.concatenate([np.zeros(H), np.ones(H), np.zeros(2 * H)]) + 0.1 * _r.normal(size=4 * H),
                W_out=_r.normal(size=(H, V)) / np.sqrt(H), b_out=0.1 * _r.normal(size=V))
_p1, _p2 = _p(5, 3, 4), _p(9, 4, 6)
TESTS = [
    (np.array([[1, 2, 3]]), np.array([3]), _p1),
    (np.array([[1, 2, 3, 4], [4, 3, 0, 0]]), np.array([4, 2]), _p1),
    (_r.integers(1, 9, size=(3, 7)), np.array([7, 5, 2]), _p2),
    (_r.integers(1, 9, size=(4, 6)), np.array([6, 6, 6, 6]), _p2),
    (_r.integers(1, 9, size=(2, 9)), np.array([3, 9]), _p2),
]

EXPLANATION = r"""
## Why the LSTM helps
The cell state is updated additively, $c_t = f\odot c_{t-1} + i\odot g$. When the forget gate is near 1 the gradient flows back through $c$ almost unchanged, so information (and gradient) can survive many steps, which a tanh RNN's repeated matrix products destroy. That is why the forget-gate bias is usually initialised to 1, as in the test weights.

## Padding done right
Freezing the state at padded positions means `h_last` is the state after the last real token, which is what a classifier or a decoder should read, and the loss counts only real next-token predictions. Frameworks do the same with packed sequences.

## Token-weighted loss
Averaging over all counted tokens weights long sequences more than short ones. Averaging per sequence first is the other common choice; perplexity is defined with the token-weighted version.

## Complexity
$O(BL(4DH + 4H^2 + HV))$; the output projection dominates for large vocabularies.

## Common bugs
- Letting padding tokens update the state.
- Predicting `ids[:, t]` from $h_t$ (that is copying, not language modelling).
- Mixing gate orders (TensorFlow/Keras uses $i, f, c, o$ too, but some code uses $i, f, o, g$).

## Interview follow-ups
- How many parameters does an LSTM layer with $D = H = 512$ have, and how does that compare with a GRU?
- Why does initialising the forget-gate bias to 1 help early training?
"""
