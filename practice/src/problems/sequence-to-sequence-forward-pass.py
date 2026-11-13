import numpy as np

META = dict(title='Encoder-decoder RNN with teacher forcing', tags=['NLP', 'Neural Networks', 'Loss Functions'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:04.25', 'air-fanout-v1:04.24'],
            related=['vanilla-rnn-cell', 'gru-cell', 'cross-entropy-logits'])

STATEMENT = r"""
A sequence-to-sequence model compresses the source sentence into the encoder's final state, then a decoder generates the target from it. During training the decoder is fed the **true** previous token (teacher forcing), and padded positions must not count in the loss.

Write `seq2seq_loss(src, src_len, tgt, tgt_len, p, bos)` that returns `(logits, loss)`.

## Inputs
- `src`: int `(N, S)` source token ids, `src_len`: int `(N,)` with $1 \le$ `src_len[n]` $\le S$. Entries past a sequence's length are padding (any id).
- `tgt`: int `(N, T)` target ids, `tgt_len`: int `(N,)` with $1 \le$ `tgt_len[n]` $\le T$.
- `bos`: the id of the start-of-sequence token (a valid row of `E_tgt`).
- `p`: dict of arrays: `E_src` `(V_s, E)`, `E_tgt` `(V_t, E)`, encoder `enc_Wx` `(E, H)`, `enc_Wh` `(H, H)`, `enc_b` `(H,)`, decoder `dec_Wx` `(E, H)`, `dec_Wh` `(H, H)`, `dec_b` `(H,)`, output `W_out` `(H, V_t)`, `b_out` `(V_t,)`.

## Model
1. **Encoder**: $h_0 = 0$, $h_t = \tanh(E_\text{src}[\text{src}_{t}]\,W^\text{enc}_x + h_{t-1} W^\text{enc}_h + b^\text{enc})$ for $t = 1..S$. The context for sample $n$ is its state after step `src_len[n]` (padding steps never update it).
2. **Decoder** (no masking inside, it runs all $T$ steps): $s_0 = $ context. The input token at step $t = 1..T$ is `bos` for $t = 1$ and `tgt[:, t-2]` otherwise (the target shifted right by one). $s_t = \tanh(E_\text{tgt}[\text{in}_t]\,W^\text{dec}_x + s_{t-1} W^\text{dec}_h + b^\text{dec})$ and the logits are $s_t W_\text{out} + b_\text{out}$.
3. **Loss**: cross-entropy of the logits at step $t$ against `tgt[:, t-1]`, averaged over all **valid** positions (step index $<$ `tgt_len[n]`) of all samples together. Use a stable log-softmax.

## Output
`logits`: float `(N, T, V_t)` for every position (including padded ones); `loss`: Python `float`.
"""

STARTER = '''import numpy as np

def seq2seq_loss(src, src_len, tgt, tgt_len, p, bos):
    """Return (logits (N, T, V_t), mean cross-entropy over valid target positions)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def seq2seq_loss(src, src_len, tgt, tgt_len, p, bos):
    P = {k: np.asarray(v, dtype=float) for k, v in p.items()}
    src, tgt = np.asarray(src), np.asarray(tgt)
    N, S = src.shape
    T = tgt.shape[1]
    # encoder: masked tanh RNN, keep each sample's state after its last real token
    h = np.zeros((N, P['enc_Wh'].shape[0]))
    xs = P['E_src'][src] @ P['enc_Wx'] + P['enc_b']             # (N, S, H)
    for t in range(S):
        live = (t < np.asarray(src_len))[:, None]
        h = np.where(live, np.tanh(xs[:, t] + h @ P['enc_Wh']), h)
    # decoder with teacher forcing: inputs are BOS followed by tgt[:, :-1]
    dec_in = np.concatenate([np.full((N, 1), bos), tgt[:, :-1]], axis=1)
    xd = P['E_tgt'][dec_in] @ P['dec_Wx'] + P['dec_b']           # (N, T, H)
    s = h
    states = []
    for t in range(T):
        s = np.tanh(xd[:, t] + s @ P['dec_Wh'])
        states.append(s)
    logits = np.stack(states, axis=1) @ P['W_out'] + P['b_out']   # (N, T, V_t)
    # masked mean cross-entropy
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll = -np.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    valid = np.arange(T)[None, :] < np.asarray(tgt_len)[:, None]
    return logits, float(nll[valid].sum() / valid.sum())
'''

HINTS = [
    'Embedding lookup is fancy indexing: `E_src[src]` has shape `(N, S, E)`. Run the encoder with a per-step mask `t < src_len` that freezes finished sequences.',
    'Build the decoder inputs by shifting: `concatenate([full((N, 1), bos), tgt[:, :-1]], axis=1)`. Start the decoder state from the encoder\'s masked final state and run all $T$ steps.',
    'For the loss, compute a stable log-softmax, pick the target ids with `take_along_axis`, then average only where `arange(T) < tgt_len[:, None]`: divide by the number of valid positions, not by $N T$.',
]

_r = np.random.default_rng(73)
def _params(Vs, Vt, E, H, s=0.6):
    g = lambda *sh: _r.normal(size=sh) * s
    return {'E_src': g(Vs, E), 'E_tgt': g(Vt, E), 'enc_Wx': g(E, H), 'enc_Wh': g(H, H), 'enc_b': g(H),
            'dec_Wx': g(E, H), 'dec_Wh': g(H, H), 'dec_b': g(H), 'W_out': g(H, Vt), 'b_out': g(Vt)}
_tiny = {'E_src': np.array([[0.0], [1.0]]), 'E_tgt': np.array([[0.0], [1.0]]), 'enc_Wx': np.array([[1.0]]), 'enc_Wh': np.array([[0.0]]),
         'enc_b': np.zeros(1), 'dec_Wx': np.array([[1.0]]), 'dec_Wh': np.array([[1.0]]), 'dec_b': np.zeros(1),
         'W_out': np.array([[1.0, -1.0]]), 'b_out': np.zeros(2)}
TESTS = [
    (np.array([[1]]), np.array([1]), np.array([[1, 0]]), np.array([2]), _tiny, 0),
    (np.array([[1, 0]]), np.array([1]), np.array([[0, 1]]), np.array([1]), _tiny, 1),
    (_r.integers(0, 6, size=(3, 5)), np.array([5, 3, 1]), _r.integers(0, 7, size=(3, 4)), np.array([4, 2, 3]), _params(6, 7, 3, 4), 0),
    (_r.integers(0, 6, size=(2, 4)), np.array([4, 4]), _r.integers(0, 5, size=(2, 6)), np.array([6, 6]), _params(6, 5, 2, 3), 4),
    (_r.integers(0, 4, size=(4, 3)), np.array([1, 2, 3, 2]), _r.integers(0, 9, size=(4, 1)), np.array([1, 1, 1, 1]), _params(4, 9, 4, 5), 2),
    (_r.integers(0, 5, size=(2, 6)), np.array([6, 2]), _r.integers(0, 6, size=(2, 5)), np.array([1, 5]), _params(5, 6, 3, 3, s=3.0), 5),
]

EXPLANATION = r"""
## The bottleneck
Everything the decoder knows about the source is the single vector $s_0$. That's fine for short sentences and the main reason quality collapsed on long ones, which motivated **attention**: let every decoder step look at all encoder states instead of only the last.

## Teacher forcing
Feeding the ground-truth previous token makes the $T$ decoder inputs known in advance, so training needs no sampling and the input projections can be computed in one batch. The cost is *exposure bias*: at inference the model consumes its own predictions, a distribution it never saw during training. Scheduled sampling mixes in model predictions to reduce the gap.

## Masking in two places
- Encoder: freeze the state past `src_len`, otherwise the context depends on how much padding a sentence got.
- Loss: average over valid target positions only. Dividing by $N T$ silently down-weights short targets, and averaging per sequence first changes the weighting again; the spec asks for a per-token mean.

## Complexity
$O(N(S + T)H^2)$ for the recurrences plus $O(NTHV_t)$ for the output layer, which dominates for real vocabularies.

## Common bugs
- Feeding `tgt` itself (not shifted) to the decoder: the model then just copies its input.
- Taking `h` after the last time step instead of after `src_len` for padded sources.
- Masking the decoder recurrence: padded target positions come after the valid ones, so they can't affect valid logits anyway.

## Interview follow-ups
- Add Luong (dot-product) attention: which shapes change and where does the context vector enter?
- How does greedy decoding differ from beam search, and why does beam search need length normalisation?
"""
