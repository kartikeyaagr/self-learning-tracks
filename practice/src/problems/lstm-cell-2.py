import numpy as np

META = dict(title='Bidirectional LSTM over a padded batch', tags=['Neural Networks', 'NLP'], diff='Hard', runtime='torch',
            lessons=['ieng:00.05'],
            related=['lstm-cell', 'gru-cell', 'vanilla-rnn-cell'])

STATEMENT = r"""
Reproduce what `nn.LSTM(bidirectional=True, batch_first=True)` computes on a padded batch fed through `pack_padded_sequence`, from the raw weights and without `nn.LSTM`, `nn.LSTMCell` or the packing utilities. The hard part is padding: the backward direction must start at each sequence's **last real token**, not at position `T - 1`.

Write `bilstm_padded(x, lengths, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_b, W_hh_b, b_ih_b, b_hh_b)`:

- `x`: float64 `(B, T, D)`, batch first. Positions `t >= lengths[b]` are padding and may hold any values.
- `lengths`: int64 `(B,)`, each in `1..T`, in any order.
- Per direction (`_f` forward, `_b` backward), in `nn.LSTM`'s layout: `W_ih` `(4H, D)`, `W_hh` `(4H, H)`, `b_ih`, `b_hh` `(4H,)`. The four row blocks are the gates in the order **i, f, g, o**.
- Initial hidden and cell states are zero for both directions.

One step, with $z = x_t W_{ih}^\top + b_{ih} + h_{t-1} W_{hh}^\top + b_{hh}$ split into $z_i, z_f, z_g, z_o$:

$$c_t = \sigma(z_f) \odot c_{t-1} + \sigma(z_i) \odot \tanh(z_g), \qquad h_t = \sigma(z_o) \odot \tanh(c_t)$$

The forward direction reads `t = 0, 1, ..., len - 1`; the backward direction reads `t = len - 1, ..., 0`.

## Return
A tuple `(out, h_n, c_n)`:

- `out`: `(B, T, 2H)`. `out[b, t, :H]` is the forward hidden state after reading position `t`; `out[b, t, H:]` is the backward hidden state after reading position `t`. Both halves are 0 at padded positions.
- `h_n`, `c_n`: `(2, B, H)`. Index 0 holds the forward direction's final state (after position `len - 1`), index 1 the backward direction's final state (after position 0).

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def bilstm_padded(x, lengths, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_b, W_hh_b, b_ih_b, b_hh_b):
    """Return (out (B, T, 2H), h_n (2, B, H), c_n (2, B, H))."""
    B, T, _ = x.shape
    H = W_hh_f.shape[1]

    def run(W_ih, W_hh, b_ih, b_hh, times):
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        outs = x.new_zeros(B, T, H)
        for t in times:
            z = x[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh
            i, f, g, o = z.chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs[:, t] = h
        return outs, h, c

    # TODO: padding is ignored, so short sequences read garbage and the backward pass starts in the padding
    of, hf, cf = run(W_ih_f, W_hh_f, b_ih_f, b_hh_f, range(T))
    ob, hb, cb = run(W_ih_b, W_hh_b, b_ih_b, b_hh_b, range(T - 1, -1, -1))
    return torch.cat([of, ob], dim=-1), torch.stack([hf, hb]), torch.stack([cf, cb])
'''

SOLUTION = '''import torch

def bilstm_padded(x, lengths, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_b, W_hh_b, b_ih_b, b_hh_b):
    B, T, _ = x.shape
    H = W_hh_f.shape[1]
    valid = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)          # (B, T) True on real tokens

    def run(W_ih, W_hh, b_ih, b_hh, times):
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        outs = x.new_zeros(B, T, H)
        for t in times:
            z = x[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh
            i, f, g, o = z.chunk(4, dim=1)                                 # nn.LSTM gate order
            c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h_new = torch.sigmoid(o) * torch.tanh(c_new)
            m = valid[:, t].unsqueeze(1)                                   # (B, 1)
            # Padded steps leave the state untouched. Going backward, padding comes first, so the
            # state is still zero when each sequence reaches its last real token.
            h = torch.where(m, h_new, h)
            c = torch.where(m, c_new, c)
            outs[:, t] = torch.where(m, h_new, torch.zeros_like(h_new))
        return outs, h, c

    of, hf, cf = run(W_ih_f, W_hh_f, b_ih_f, b_hh_f, range(T))
    ob, hb, cb = run(W_ih_b, W_hh_b, b_ih_b, b_hh_b, range(T - 1, -1, -1))
    return torch.cat([of, ob], dim=-1), torch.stack([hf, hb]), torch.stack([cf, cb])
'''

HINTS = [
    'Build a `(B, T)` boolean mask `arange(T) < lengths[:, None]`. Run one batched loop per direction over all `T` positions; at each step compute the new state for every row, then keep it only where the mask is True.',
    'With "keep the old state on padded steps", the forward direction freezes its state after `len - 1`, which is exactly `h_n[0]`. Use `torch.where`, not multiplication by the mask, so garbage padding values (even inf) can\'t leak in.',
    'The backward direction walks `t = T-1 .. 0`. For a short sequence the first steps are padding, so its state stays zero until `t = len - 1`: it starts at the last real token for free. Zero the outputs at padded positions in both halves.',
]

_r = np.random.default_rng(29)

def _w(D, H):
    s = 1 / np.sqrt(H)
    return [_r.uniform(-s, s, size=sh) for sh in ((4 * H, D), (4 * H, H), (4 * H,), (4 * H,))]

def _case(B, T, D, H, lengths, junk=0.0):
    x = _r.normal(size=(B, T, D))
    lengths = np.array(lengths, dtype=np.int64)
    x[np.arange(T)[None, :] >= lengths[:, None]] = junk                  # padding values must not matter
    return (x, lengths, *_w(D, H), *_w(D, H))

def _ones(B, T, D, H, lengths):
    x = np.ones((B, T, D))
    return (x, np.array(lengths, dtype=np.int64), *[0.5 * np.ones_like(a) for a in _w(D, H)],
            *[-0.5 * np.ones_like(a) for a in _w(D, H)])

TESTS = [
    _ones(1, 2, 1, 1, [2]),
    _ones(2, 3, 1, 1, [3, 1]),
    _case(3, 5, 2, 3, [5, 2, 4]),
    _case(2, 4, 3, 2, [1, 4], junk=7.0),
    _case(1, 6, 2, 4, [6]),
    _case(4, 6, 3, 3, [2, 6, 1, 3], junk=-3.0),
    _case(2, 3, 4, 5, [3, 3]),
    _case(5, 7, 2, 2, [7, 1, 4, 2, 5], junk=100.0),
]

EXPLANATION = r"""
## Masked state updates are packing in disguise
`pack_padded_sequence` lets cuDNN skip padded steps entirely. In a plain loop you get the same result by computing every row at every step and keeping the new state only where the token is real. For the forward direction that freezes the state after the last real token, so the value left in `h` at the end is the true final state. For the backward direction the padding comes *first*: the state stays at its zero initial value until the loop reaches `len - 1`, so each sequence starts exactly at its last real token.

## Why the naive version is wrong
Running the backward LSTM from `t = T - 1` on a padded sequence feeds it the padding tokens first. Its "final" state then depends on how much padding the batch had, so the same sentence gets different encodings in different batches. The forward half looks fine at real positions, which is why this bug survives casual testing.

## Complexity
$O(T \cdot B \cdot 4H(D + H))$ per direction, with $T$ sequential steps. Rows of short sequences still do the math on padded steps; sorting by length and shrinking the active batch, which is what packing does, removes that waste.

## Common bugs
- Gate order: PyTorch uses i, f, g, o. Reading the blocks as i, f, o, g gives plausible but wrong numbers.
- Taking `h_n[0]` as `out[:, -1, :H]`, which is zero (or padding state) for short sequences.
- Multiplying by the mask instead of `torch.where`: `0 * inf` is NaN.
- Using one bias: `nn.LSTM` has two, `b_ih + b_hh`. Loading a single-bias checkpoint means putting it in one and zeroing the other.

## Interview follow-ups
- How would you vectorise the input projection $x W_{ih}^\top$ over all time steps before the loop, and why can't the recurrent part be parallelised the same way?
- For sequence classification with a BiLSTM, which states would you concatenate as the sentence representation, and why not `out[:, -1]`?
"""
