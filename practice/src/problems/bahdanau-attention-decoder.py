import numpy as np

META = dict(title='Encoder-decoder with additive (Bahdanau) attention', tags=['Attention', 'NLP'], diff='Hard', runtime='numpy',
            lessons=['dlcore:05.03', 'air-fanout-v1:06.01'],
            related=['attention-mechanism', 'sequence-to-sequence-forward-pass', 'gru-cell', 'bigru-text-classifier'])

STATEMENT = r"""
Bahdanau, Cho and Bengio removed the fixed-size bottleneck of encoder-decoder translation: at every output step the decoder computes a weighted average of all encoder states, with weights from a small learned scoring network. Implement the full teacher-forced forward pass for one sentence pair.

Write `bahdanau_forward(src, tgt_in, p)` returning `(logits, attn)`.

## GRU cell (used by all three RNNs)
`gru(x, h, g)` with `g` = dict `W_ih` `(D, 3H)`, `W_hh` `(H, 3H)`, `b_ih`, `b_hh`, blocks $r, z, n$: $r = \sigma(a_r + u_r)$, $z = \sigma(a_z + u_z)$, $n = \tanh(a_n + r\odot u_n)$, $h' = (1-z)\odot n + z\odot h$, where $a = xW_{ih} + b_{ih}$, $u = hW_{hh} + b_{hh}$.

## Encoder
Embed `src` `(S,)` with `E_src`; run `enc_f` left to right and `enc_b` right to left from zero states. Annotation $h_j = [\overrightarrow{h}_j; \overleftarrow{h}_j]$ (width $2H$).

## Decoder (for $i = 0..T-1$)
- Initial state $s_{-1} = \tanh(\overleftarrow{h}_0 W_s)$, the backward state at the first source word.
- Scores $e_{j} = v^\top\tanh(s_{i-1}W_a + h_jU_a)$; $\alpha = \mathrm{softmax}(e)$ over $j$; context $c_i = \sum_j\alpha_jh_j$.
- $s_i = \mathrm{gru}([E_{tgt}[\text{tgt\_in}_i];\ c_i],\ s_{i-1},\ \text{dec})$.
- Logits $= [s_i;\ c_i]\,W_o + b_o$.

`p` keys: `E_src`, `E_tgt`, `enc_f`, `enc_b`, `dec` (GRU dicts; the decoder's input width is $D + 2H$), `W_s` `(H, H)`, `W_a` `(H, A)`, `U_a` `(2H, A)`, `v` `(A,)`, `W_o` `(3H, V)`, `b_o`.

## Output
`logits` `(T, V)`; `attn` `(T, S)` with row $i$ the weights $\alpha$ used at step $i$.
"""

STARTER = '''import numpy as np

def bahdanau_forward(src, tgt_in, p):
    """Return (logits (T, V), attn (T, S))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))

def _gru(x, h, g):
    H = h.shape[-1]
    a = x @ g['W_ih'] + g['b_ih']
    u = h @ g['W_hh'] + g['b_hh']
    r, z = _sig(a[:H] + u[:H]), _sig(a[H:2 * H] + u[H:2 * H])
    n = np.tanh(a[2 * H:] + r * u[2 * H:])
    return (1 - z) * n + z * h

def bahdanau_forward(src, tgt_in, p):
    xs = p['E_src'][np.asarray(src)]
    S, H = len(xs), p['W_s'].shape[0]
    fw, bw = np.zeros((S, H)), np.zeros((S, H))
    h = np.zeros(H)
    for j in range(S):
        h = _gru(xs[j], h, p['enc_f']); fw[j] = h
    h = np.zeros(H)
    for j in range(S - 1, -1, -1):
        h = _gru(xs[j], h, p['enc_b']); bw[j] = h
    ann = np.concatenate([fw, bw], axis=1)                  # (S, 2H)
    keys = ann @ p['U_a']                                   # precomputed once
    s = np.tanh(bw[0] @ p['W_s'])
    logits, attn = [], []
    for y in np.asarray(tgt_in):
        e = np.tanh(s @ p['W_a'] + keys) @ p['v']
        a = np.exp(e - e.max()); a /= a.sum()
        c = a @ ann
        s = _gru(np.concatenate([p['E_tgt'][y], c]), s, p['dec'])
        logits.append(np.concatenate([s, c]) @ p['W_o'] + p['b_o'])
        attn.append(a)
    return np.array(logits), np.array(attn)
'''

HINTS = [
    'Build the annotations first: two GRU passes over the source, stacked to `(S, 2H)`. The term $h_jU_a$ doesn\'t depend on the decoder step, so compute it once.',
    'At each decoder step the scores use the **previous** decoder state $s_{i-1}$; the new state is computed after the context.',
    'The decoder GRU input is the target embedding concatenated with the context, width $D + 2H$.',
]

_r = np.random.default_rng(2014)
def _g(D, H):
    return dict(W_ih=_r.normal(size=(D, 3 * H)) / np.sqrt(D), W_hh=_r.normal(size=(H, 3 * H)) / np.sqrt(H),
                b_ih=0.1 * _r.normal(size=3 * H), b_hh=0.1 * _r.normal(size=3 * H))
def _p(Vs, Vt, D, H, A):
    return dict(E_src=_r.normal(size=(Vs, D)), E_tgt=_r.normal(size=(Vt, D)), enc_f=_g(D, H), enc_b=_g(D, H),
                dec=_g(D + 2 * H, H), W_s=_r.normal(size=(H, H)) / np.sqrt(H), W_a=_r.normal(size=(H, A)) / np.sqrt(H),
                U_a=_r.normal(size=(2 * H, A)) / np.sqrt(2 * H), v=_r.normal(size=A), W_o=_r.normal(size=(3 * H, Vt)) / np.sqrt(3 * H),
                b_o=0.1 * _r.normal(size=Vt))
_p1, _p2 = _p(5, 6, 2, 3, 4), _p(9, 8, 4, 5, 6)
TESTS = [
    (np.array([1, 2]), np.array([0]), _p1),
    (np.array([1, 2, 3, 4]), np.array([0, 5, 2]), _p1),
    (np.array([3]), np.array([0, 1]), _p1),
    (_r.integers(0, 9, size=6), _r.integers(0, 8, size=5), _p2),
    (_r.integers(0, 9, size=9), _r.integers(0, 8, size=7), _p2),
]

EXPLANATION = r"""
## What attention fixed
The earlier encoder-decoder compressed the whole source into one vector, and translation quality fell sharply for long sentences. With attention the decoder looks back at every source annotation each step, and the learned weights $\alpha$ turn out to be soft word alignments (the paper's famous heatmaps). This is the direct ancestor of Transformer attention.

## Additive vs dot-product scoring
Bahdanau scores with a one-hidden-layer network, $v^\top\tanh(sW_a + hU_a)$; Luong et al. later used $s^\top W h$ or $s^\top h$, and the Transformer the scaled dot product, which is cheaper and fits matrix hardware. Additive scoring tends to work better when the query and key widths differ or for small dimensions.

## Complexity
Per decoder step $O(SA)$ for scores plus $O(S\cdot 2H)$ for the context; over the sentence $O(TS(A + H))$. The encoder term $hU_a$ is computed once, $O(S\cdot 2H\cdot A)$.

## Common bugs
- Scoring with the new state $s_i$ instead of $s_{i-1}$ (that is Luong's variant and changes the data flow).
- Initialising the decoder from the forward encoder's last state when the spec says backward-first.
- Softmax over the wrong axis of a batched score matrix.

## Interview follow-ups
- How would you mask padded source positions in a batched version?
- Why does attention make the gradient path from an output word to its aligned source word short?
"""
