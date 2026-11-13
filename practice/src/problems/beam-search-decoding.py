import numpy as np

META = dict(title='Beam search with length normalization', tags=['Language Models', 'Inference'], diff='Medium', runtime='numpy',
            lessons=['aieng:03.03', 'ieng:02.05'], related=['greedy-sequence-tagging', 'text-generation-sampling', 'n-gram-language-model'])

STATEMENT = r"""
Beam search keeps the `beam` best partial outputs at every step instead of only the single best one. To keep the problem self-contained, the "model" here is a bigram table: the log-probability of the next token depends only on the previous token.

Write `beam_search(start_logp, trans_logp, beam, max_len, eos, alpha=0.0)` that returns `(tokens, score)`.

## Inputs
- `start_logp`: float array `(V,)`, log-probabilities of the first token.
- `trans_logp`: float array `(V, V)`, `trans_logp[a, b]` is the log-probability of token `b` after token `a`.
- `beam` $\ge 1$, `max_len` $\ge 1$ (maximum tokens per hypothesis), `eos`: the end token id, `alpha` $\ge 0$.

## Algorithm
A hypothesis is a token list with a cumulative log-probability.
1. **Step 1:** the candidates are every one-token hypothesis `[v]` with score `start_logp[v]`, in order of `v`.
2. **Select:** sort the candidates by score, highest first, keeping the candidate order for equal scores (a stable sort), and keep the first `beam`. Of those, hypotheses whose last token is `eos` move to the **finished** list (in that order); the others become the **live** hypotheses (in that order).
3. **Later steps:** while there are live hypotheses and they have fewer than `max_len` tokens, build candidates by extending each live hypothesis (in live order) with every token `v` (in order of `v`), adding `trans_logp[last, v]` to its score, then **select** again.
4. When the loop stops, append any remaining live hypotheses to the finished list, in live order.

## Result
Return the finished hypothesis with the highest normalized score $\text{score} / \text{len}^{\alpha}$, where len counts all its tokens including `eos`. Ties go to the hypothesis that entered the finished list first. Return its tokens as a list of ints and its normalized score as a float.
"""

STARTER = '''import numpy as np

def beam_search(start_logp, trans_logp, beam, max_len, eos, alpha=0.0):
    """Return (best token list, its length-normalized score)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def beam_search(start_logp, trans_logp, beam, max_len, eos, alpha=0.0):
    start_logp, trans_logp = np.asarray(start_logp, dtype=float), np.asarray(trans_logp, dtype=float)
    finished, live = [], []

    def select(cands):
        cands = sorted(cands, key=lambda c: -c[1])[:beam]      # sorted() is stable: ties keep candidate order
        nxt = []
        for c in cands:
            (finished if c[0][-1] == eos else nxt).append(c)
        return nxt

    live = select([([v], float(start_logp[v])) for v in range(len(start_logp))])
    while live and len(live[0][0]) < max_len:                  # all live hypotheses have the same length
        live = select([(toks + [v], sc + float(trans_logp[toks[-1], v]))
                       for toks, sc in live for v in range(trans_logp.shape[1])])
    finished += live

    norm = lambda c: c[1] / len(c[0]) ** alpha
    best = max(finished, key=norm)                             # max() returns the first of equal maxima
    return best[0], float(norm(best))
'''

HINTS = [
    'Represent a hypothesis as `(tokens, logprob)`. One helper `select(candidates)` does the stable sort, keeps `beam`, and splits the survivors into finished and live.',
    'Every step extends **all** live hypotheses by **all** tokens, so there are `len(live) * V` candidates. Finished hypotheses are never extended, so the live set can shrink below `beam`.',
    'Length normalization is applied only at the end, when choosing among finished hypotheses: `score / len(tokens) ** alpha`. With `alpha = 0` it is the raw log-probability, which favours short outputs.',
]

def _logsm(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

_r = np.random.default_rng(47)
_S1, _T1 = _logsm(_r.normal(size=6)), _logsm(_r.normal(size=(6, 6)) * 1.5)
_S2, _T2 = _logsm(_r.normal(size=10)), _logsm(_r.normal(size=(10, 10)) * 2.0)
_ln = np.log
_S0 = _ln(np.array([0.5, 0.45, 0.05]))
_T0 = _ln(np.array([[0.3, 0.3, 0.4], [0.05, 0.05, 0.9], [1 / 3, 1 / 3, 1 / 3]]))
_S3 = _ln(np.array([0.5, 0.05, 0.45]))
_T3 = _ln(np.array([[0.1, 0.85, 0.05], [0.02, 0.03, 0.95], [1 / 3, 1 / 3, 1 / 3]]))
TESTS = [
    (_S0, _T0, 1, 3, 2),
    (_S0, _T0, 2, 3, 2),
    (_S3, _T3, 2, 4, 2),
    (_S3, _T3, 2, 4, 2, 1.0),
    (_ln(np.array([0.7, 0.3])), _ln(np.array([[0.5, 0.5], [0.5, 0.5]])), 3, 1, 1),
    (_ln(np.array([0.2, 0.8])), _ln(np.array([[0.9, 0.1], [0.5, 0.5]])), 2, 4, 1),
    (_S1, _T1, 3, 6, 0, 0.7),
    (_S2, _T2, 4, 8, 3, 1.0),
    (_S2, _T2, 1, 8, 3),
]

EXPLANATION = r"""
## Why it works
Greedy decoding commits to the single best token at each step, so a slightly worse first token that leads to a much better continuation is lost forever. Keeping `beam` hypotheses explores those alternatives at a cost of `beam` times the work. In test 1 (beam 1, which is greedy) the first token is 0 because it is the most likely start, and the output is `[0, 2]` with probability $0.5 \cdot 0.4 = 0.2$. With beam 2 (test 2) the start `[1]` survives the first step and leads to `[1, 2]` with probability $0.45 \cdot 0.9 = 0.405$, twice as likely.

## Length normalization
Every extra token multiplies the probability by a number below 1, so raw log-probabilities favour stopping early. In test 3 the winner is `[2]`, an immediate end of sequence with probability 0.45, beating `[0, 1, 2]` at $0.5 \cdot 0.85 \cdot 0.95 \approx 0.40$. With $\alpha = 1$ (test 4) the longer output wins. Dividing by $\text{len}^{\alpha}$ removes part of that bias: $\alpha = 0$ is no correction and $\alpha = 1$ is the average log-probability per token. Production systems use $\alpha$ between 0.6 and 1 or the GNMT penalty $\big((5 + \text{len}) / 6\big)^{\alpha}$.

## Complexity
Each step scores `beam` $\times V$ candidates and sorts them: $O(\text{max\_len} \cdot bV \log(bV))$. `np.argpartition` on the flattened score matrix brings the selection down to $O(bV)$. With a real model the forward pass dominates, and the `beam` hypotheses run as one batch.

## Common bugs
- Extending finished hypotheses, so `eos` is followed by more tokens.
- Normalizing scores during selection rather than at the end (that's a different, also common, variant: say which one you implement).
- Forgetting the live hypotheses that hit `max_len` without `eos`.
- Unstable sorting, which makes results depend on the sort implementation.

## Interview follow-ups
- Why does beam search with a large beam often produce *worse* text for open-ended generation (repetition, blandness), while it helps translation? (High-probability text is not human-like text; sampling methods fix this for open-ended tasks.)
- How do you implement beam search efficiently with a KV cache? (Reorder the cache rows by the parent index of each surviving hypothesis.)
"""
