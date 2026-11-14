import numpy as np

META = dict(title='Sequence to sequence: reversed-source LSTM encoder and beam search', tags=['NLP', 'Language Models'],
            diff='Hard', runtime='numpy', lessons=['dlcore:05.03'],
            related=['sequence-to-sequence-forward-pass', 'lstm-cell', 'beam-search-decoding', 'lstm-language-model-loss'])

STATEMENT = r"""
Sutskever, Vinyals and Le translated with two LSTMs: one reads the source sentence **in reverse** into its final state, the other starts from that state and generates the target, decoded with a small beam. Implement translation with a trained (here: given) model.

Write `translate(src, p, bos, eos, beam, max_len)` returning `(tokens, score)`.

## LSTM cell
`lstm(x, h, c, g)` with `W_ih` `(D, 4H)`, `W_hh` `(H, 4H)`, `b` `(4H,)`, gate blocks in order $i, f, g, o$: $c' = \sigma(f)\odot c + \sigma(i)\odot\tanh(g)$, $h' = \sigma(o)\odot\tanh(c')$.

## Encoder
Start from zero $(h, c)$ and feed the embeddings `E_src[src[::-1]]` (last source word first) through `enc`. Its final $(h, c)$ initialises the decoder.

## Decoder and beam search
- One decoder step from state $(h, c)$ with input token $y$: $(h', c') = \mathrm{lstm}(E_{tgt}[y], h, c, \text{dec})$; next-token log-probabilities $= \mathrm{log\_softmax}(h'W_o + b_o)$.
- Each hypothesis has tokens (initially empty), a state, its last token (initially `bos`) and a score (sum of log-probabilities, initially 0).
- Each step: expand every live hypothesis by every token; among all candidates keep the `beam` best by score (stable order: earlier hypothesis, then lower token id, first on exact ties). Kept candidates whose new token is `eos` become finished (with `eos` included in their tokens); the others stay live.
- Stop when nothing is live or after `max_len` steps; then live hypotheses also count as finished.
- Return the finished hypothesis with the highest score (the first one on ties), as `(tokens, score)` with `tokens` an int array.
"""

STARTER = '''import numpy as np

def translate(src, p, bos, eos, beam, max_len):
    """Return (tokens, score)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))

def _lstm(x, h, c, g):
    H = h.shape[0]
    a = x @ g['W_ih'] + h @ g['W_hh'] + g['b']
    c = _sig(a[H:2 * H]) * c + _sig(a[:H]) * np.tanh(a[2 * H:3 * H])
    return _sig(a[3 * H:]) * np.tanh(c), c

def translate(src, p, bos, eos, beam, max_len):
    H = p['enc']['W_hh'].shape[0]
    h, c = np.zeros(H), np.zeros(H)
    for w in np.asarray(src)[::-1]:
        h, c = _lstm(p['E_src'][w], h, c, p['enc'])
    live = [([], h, c, bos, 0.0)]
    done = []
    for _ in range(max_len):
        if not live:
            break
        cands = []
        for toks, h, c, last, sc in live:
            h2, c2 = _lstm(p['E_tgt'][last], h, c, p['dec'])
            z = h2 @ p['W_o'] + p['b_o']
            lp = z - z.max() - np.log(np.exp(z - z.max()).sum())
            cands += [(sc + lp[y], toks + [y], h2, c2, y) for y in range(len(lp))]
        order = sorted(range(len(cands)), key=lambda i: -cands[i][0])     # sorted() is stable
        live = []
        for i in order[:beam]:
            sc, toks, h2, c2, y = cands[i]
            if y == eos:
                done.append((toks, sc))
            else:
                live.append((toks, h2, c2, y, sc))
    done += [(toks, sc) for toks, _, _, _, sc in live]
    best = max(range(len(done)), key=lambda i: (done[i][1], -i))
    return np.array(done[best][0], dtype=int), float(done[best][1])
'''

HINTS = [
    'Reverse the source before encoding: `src[::-1]`. The decoder\'s first input is `bos` and its first state is the encoder\'s final $(h, c)$.',
    'Keep each hypothesis as (tokens, h, c, last token, score). Each step, build every candidate from every live hypothesis, sort by score with a stable sort, keep the top `beam`, then split them into finished (ended with `eos`) and live.',
    'Finished hypotheses are compared by raw summed log-probability, so shorter answers have an advantage. The paper did not length-normalise.',
]

_r = np.random.default_rng(2014)
def _l(D, H):
    return dict(W_ih=_r.normal(size=(D, 4 * H)) / np.sqrt(D), W_hh=_r.normal(size=(H, 4 * H)) / np.sqrt(H),
                b=0.1 * _r.normal(size=4 * H))
def _p(Vs, Vt, D, H, s=2.0):
    return dict(E_src=_r.normal(size=(Vs, D)), E_tgt=_r.normal(size=(Vt, D)), enc=_l(D, H), dec=_l(D, H),
                W_o=_r.normal(size=(H, Vt)) * s, b_o=0.3 * _r.normal(size=Vt))
_p1, _p2 = _p(6, 5, 3, 4), _p(10, 8, 4, 6, 3.0)
TESTS = [
    (np.array([1, 2, 3]), _p1, 0, 1, 1, 4),
    (np.array([1, 2, 3]), _p1, 0, 1, 3, 4),
    (np.array([3, 2, 1]), _p1, 0, 1, 3, 6),
    (np.array([5, 7, 2, 9]), _p2, 0, 1, 4, 5),
    (np.array([5, 7, 2, 9]), _p2, 0, 1, 1, 8),
    (np.array([4]), _p2, 0, 7, 2, 3),
]

EXPLANATION = r"""
## Why reverse the source
With the source reversed, the first source words end up close (in time steps) to the first target words, which typically translate them. This creates many short-range dependencies that give the optimiser an easy start, and the paper reports a large BLEU gain from this one change. Attention later removed the need for the trick.

## Beam search
Greedy decoding commits to the single best word each step and can't recover from an early mistake. A beam keeps the $k$ best partial translations; the paper found a beam of 2 already gave most of the benefit, and 12 a little more. Without length normalisation the search prefers short outputs, since every extra token adds a negative log-probability; later systems divide by a length penalty.

## The fixed-size bottleneck
The encoder's final state must hold the entire source sentence. The paper's deep 4-layer LSTMs with 1000 units made this work surprisingly well, but quality still fell for long sentences, which Bahdanau attention addressed.

## Complexity
Encoder $O(S\cdot 4H(D + H))$; each decoder step costs $O(k(4H(D + H) + HV))$ plus a sort over $kV$ candidates.

## Common bugs
- Encoding the source in the original order.
- Letting finished hypotheses keep expanding.
- Comparing finished hypotheses by average log-probability when the spec (and the paper) uses the sum.

## Interview follow-ups
- Why can a larger beam give a worse translation (the "beam search curse")?
- How would you batch beam search for many sentences on a GPU?
"""
