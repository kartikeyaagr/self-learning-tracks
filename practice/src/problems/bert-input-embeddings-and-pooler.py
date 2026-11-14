import numpy as np

META = dict(title='BERT input embeddings and pooler', tags=['Transformers', 'Embeddings', 'Language Models'], diff='Medium', runtime='numpy')

STATEMENT = r"""
Write `bert_embed_and_pool(ids, segs, tok, pos, seg, gamma, beta, Wp, bp, H)`.

1. Input embedding for position $t$: `tok[ids[t]] + pos[t] + seg[segs[t]]`, where `segs` marks sentence A (0) or B (1).
2. Apply LayerNorm over the hidden axis with `eps = 1e-12`, then `gamma` and `beta`.
3. The encoder is outside this problem: you are given its output `H` (shape `(T, d)`).
4. Pooler: `tanh(H[0] @ Wp + bp)`, using the `[CLS]` position only.

Return a tuple `(E, pooled)` where `E` is the normalised input embedding of shape `(T, d)`.
"""

STARTER = '''import numpy as np

def bert_embed_and_pool(ids, segs, tok, pos, seg, gamma, beta, Wp, bp, H):
    """Return (E, pooled)."""
    E = np.asarray(tok)[ids]
    return E, np.asarray(H)[0]
'''

SOLUTION = '''import numpy as np

def bert_embed_and_pool(ids, segs, tok, pos, seg, gamma, beta, Wp, bp, H):
    ids, segs = np.asarray(ids), np.asarray(segs)
    x = np.asarray(tok)[ids] + np.asarray(pos)[: len(ids)] + np.asarray(seg)[segs]
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    E = (x - mu) / np.sqrt(var + 1e-12) * gamma + beta
    pooled = np.tanh(np.asarray(H)[0] @ Wp + bp)
    return E, pooled
'''

HINTS = [
    'The three embeddings are summed, not concatenated, so they share the hidden size d.',
    'LayerNorm uses the population variance over the last axis.',
    'The pooler reads only row 0 of the encoder output, the [CLS] token.',
]

def _case(seed, V, T, d):
    r = np.random.default_rng(seed)
    return (r.integers(0, V, T), np.r_[np.zeros(T // 2, int), np.ones(T - T // 2, int)],
            r.normal(size=(V, d)), r.normal(size=(16, d)), r.normal(size=(2, d)),
            1 + 0.1 * r.normal(size=d), 0.1 * r.normal(size=d),
            r.normal(size=(d, d)) / np.sqrt(d), 0.1 * r.normal(size=d), r.normal(size=(T, d)))

TESTS = [_case(1, 10, 4, 3), _case(2, 20, 6, 4), _case(3, 5, 2, 2), _case(4, 30, 8, 6), _case(5, 12, 5, 8)]

EXPLANATION = r"""
## Three embeddings, one sum
BERT feeds pairs of sentences for tasks such as entailment and next-sentence prediction. Token embeddings say what the word is, learned position embeddings say where it is, and segment embeddings say which sentence it belongs to. Summing them keeps the width at $d$; the network learns to separate the signals.

## Why LayerNorm at the input
The sum of three learned tables can have very different scales per position. Normalising before the first layer keeps the residual stream well conditioned from the start.

## The pooler
The pooler is a dense layer plus tanh on the `[CLS]` vector, trained by next-sentence prediction. Fine-tuned classifiers often put their head on it, but for sentence embeddings mean pooling over tokens usually works better, because NSP is a weak objective.

## Interview follow-ups
- Why did RoBERTa drop the next-sentence task, and what happens to segment embeddings then?
"""
