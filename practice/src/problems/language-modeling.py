import numpy as np

META = dict(title='Score a padded batch with an RNN language model', tags=['Language Models', 'Neural Networks'], diff='Medium', runtime='numpy',
            lessons=['aieng:03.02', 'ieng:02.04'], related=['n-gram-language-model', 'perplexity-computation', 'cross-entropy-logits'])

STATEMENT = r"""
A recurrent language model reads tokens one at a time and, after each one, predicts the next. Write `rnn_lm_nll(ids, lengths, E, Wxh, Whh, bh, Why, by)` that runs an Elman RNN over a padded batch and returns its negative log-likelihood.

## Model
For each sequence, with $h_{-1} = 0$ and $x_t$ the token at position $t$:
$$h_t = \tanh\big(E_{x_t} W_{xh} + h_{t-1} W_{hh} + b_h\big), \qquad z_t = h_t W_{hy} + b_y$$
$z_t$ are the logits for the token at position $t + 1$.

## Inputs
- `ids`: int array `(B, T)`. Row $b$ holds a real sequence of length `lengths[b]`, followed by padding (any valid ids).
- `lengths`: int array `(B,)` with $0 \le$ `lengths[b]` $\le T$. At least one sequence has length $\ge 2$.
- `E` `(V, D)`, `Wxh` `(D, H)`, `Whh` `(H, H)`, `bh` `(H,)`, `Why` `(H, V)`, `by` `(V,)`.

## Loss
The targets of sequence $b$ are positions $1, \dots, L_b - 1$ (the first token is never predicted), each predicted from $z_{t-1}$:
$$\text{nll}_b = -\sum_{t=1}^{L_b - 1} \ln \mathrm{softmax}(z_{t-1})_{x_t}$$
A sequence of length 0 or 1 has $\text{nll}_b = 0$.

## Output
`(mean_nll, per_seq)`: `per_seq` is the float array of $\text{nll}_b$, and `mean_nll` is $\sum_b \text{nll}_b$ divided by the total number of targets $\sum_b \max(L_b - 1, 0)$, as a float.
"""

STARTER = '''import numpy as np

def rnn_lm_nll(ids, lengths, E, Wxh, Whh, bh, Why, by):
    """Return (mean NLL per predicted token, per-sequence summed NLL (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rnn_lm_nll(ids, lengths, E, Wxh, Whh, bh, Why, by):
    ids, lengths = np.asarray(ids), np.asarray(lengths)
    B, T = ids.shape
    h = np.zeros((B, Whh.shape[0]))
    per_seq = np.zeros(B)
    for t in range(T - 1):
        h = np.tanh(E[ids[:, t]] @ Wxh + h @ Whh + bh)       # whole batch at once
        z = h @ Why + by
        logp = z - z.max(axis=1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))   # stable log-softmax
        nll = -logp[np.arange(B), ids[:, t + 1]]
        per_seq += np.where(t + 1 < lengths, nll, 0.0)       # padding targets don't count
    n_targets = np.maximum(lengths - 1, 0).sum()
    return float(per_seq.sum() / n_targets), per_seq
'''

HINTS = [
    'Run the recurrence for the whole batch at once: `h` has shape `(B, H)` and `E[ids[:, t]]` gathers one embedding per row.',
    'The logits computed after reading position `t` score the token at `t + 1`. Use a stable log-softmax and pick the target column per row with `logp[np.arange(B), ids[:, t + 1]]`.',
    'Mask instead of slicing: a target at `t + 1` counts only if `t + 1 < lengths[b]`. Padding after a sequence ends can run through the RNN harmlessly because nothing reads its outputs.',
]

_r = np.random.default_rng(37)
_V, _D, _H = 7, 4, 5
_P = (_r.normal(scale=0.6, size=(_V, _D)), _r.normal(scale=0.6, size=(_D, _H)), _r.normal(scale=0.6, size=(_H, _H)),
      _r.normal(scale=0.1, size=_H), _r.normal(scale=0.8, size=(_H, _V)), _r.normal(scale=0.1, size=_V))
_ids = _r.integers(0, _V, size=(4, 9))
TESTS = [
    (np.array([[0, 1]]), np.array([2]), np.eye(2), np.zeros((2, 1)), np.zeros((1, 1)), np.zeros(1),
     np.zeros((1, 2)), np.array([0.0, 1.0])),
    (np.array([[1, 0, 1]]), np.array([3]), np.eye(2), np.array([[1.0], [-1.0]]), np.array([[0.5]]), np.zeros(1),
     np.array([[2.0, -2.0]]), np.zeros(2)),
    (np.array([[2, 3], [1, 1]]), np.array([2, 1]), *_P),
    (np.array([[4, 2, 6, 0]]), np.array([4]), *_P),
    (_ids, np.array([9, 3, 1, 6]), *_P),
    (_ids, np.array([0, 9, 9, 2]), *_P),
]

EXPLANATION = r"""
## Why it works
Teacher forcing feeds the true previous tokens, so all predictions for a sequence come out of one left-to-right pass. The loss at position $t$ is the cross-entropy between the softmax of $z_{t-1}$ and the true token, and summing these is exactly $-\ln P(x_1, \dots, x_{L-1} \mid x_0)$ by the chain rule of probability.

## Masking instead of slicing
Every row runs for the full length $T$, which keeps the computation batched. Positions past a sequence's end produce garbage hidden states, but no loss term reads them and the hidden state never flows backwards, so they cannot affect the real tokens. Dividing by the number of real targets (not $B \cdot (T-1)$) keeps the mean comparable between batches with different amounts of padding.

## Complexity
$O(T \cdot B \cdot (DH + H^2 + HV))$. The $HV$ output projection dominates for real vocabularies.

## Common bugs
- Off-by-one alignment: scoring `ids[:, t]` with the logits computed after reading `ids[:, t]`.
- Including padding positions in the mean.
- Computing `np.log(softmax(z))`, which gives `-inf` when a probability underflows.
- Starting the loop at $t = 0$ with a target at position 0 (the first token has no prediction).

## Interview follow-ups
- Why does an RNN's gradient vanish or explode over long sequences, and how do LSTMs and gradient clipping help?
- How does a Transformer LM compute all $T$ predictions in parallel during training, and why can't it do the same when generating?
"""
