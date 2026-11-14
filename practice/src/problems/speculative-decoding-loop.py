import numpy as np

META = dict(title='Speculative decoding end to end with a draft and a target bigram model', tags=['Inference', 'Sampling'],
            diff='Hard', runtime='numpy', lessons=['aieng:07.04', 'ieng:16.02', 'ieng:16.03'],
            related=['perform-exact-speculative-sampling', 'speculative-decoding-and-token-verification', 'speculative-decoding-acceptance'])

STATEMENT = r"""
Speculative decoding lets a cheap draft model propose several tokens and has the expensive target model check them all in one pass, while producing exactly the target model's distribution. Run the whole loop with two bigram models and supplied uniforms.

Write `speculative_generate(P_target, P_draft, start, n_new, k, u_draft, u_accept, u_extra)` returning `(tokens, accepted)`.

## Models and sampling
`P_target`, `P_draft` `(V, V)`: row $a$ is the next-token distribution after token $a$. To **sample** from a distribution $p$ with a uniform $u$, return the smallest $i$ with $\sum_{j\le i}p_j > u$ (inverse CDF).

## One round $r$ (starting from the last token so far, initially `start`)
1. **Draft**: sample $k$ tokens $x_1..x_k$ autoregressively from `P_draft`, token $j$ using `u_draft[r, j]`.
2. **Verify** in order: with $q = $ the draft distribution and $p = $ the target distribution at that position (both conditioned on the previous token), accept $x_j$ if `u_accept[r, j]` $< \min(1, p(x_j)/q(x_j))$.
3. At the first rejection, discard the remaining drafts and append one token sampled with `u_extra[r]` from the residual $\max(0, p - q)$, normalised. If all $k$ are accepted, append a bonus token sampled from the target's next distribution after $x_k$ with `u_extra[r]`.

Stop once at least `n_new` tokens have been generated and return the first `n_new` as `tokens`, plus `accepted` `(rounds,)`: the number of drafts accepted in each round.
"""

STARTER = '''import numpy as np

def speculative_generate(P_target, P_draft, start, n_new, k, u_draft, u_accept, u_extra):
    """Return (tokens (n_new,), accepted (rounds,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sample(p, u):
    return int(min(np.searchsorted(np.cumsum(p), u, side='right'), len(p) - 1))

def speculative_generate(P_target, P_draft, start, n_new, k, u_draft, u_accept, u_extra):
    Pt, Pd = np.asarray(P_target, dtype=float), np.asarray(P_draft, dtype=float)
    out, acc, last, r = [], [], int(start), 0
    while len(out) < n_new:
        prev, draft = last, []
        for j in range(k):
            x = _sample(Pd[prev], u_draft[r][j])
            draft.append(x)
            prev = x
        prev, n_ok = last, 0
        for j, x in enumerate(draft):
            p, q = Pt[prev], Pd[prev]
            if u_accept[r][j] < min(1.0, p[x] / q[x]):
                out.append(x)
                prev, n_ok = x, n_ok + 1
            else:
                res = np.maximum(p - q, 0)
                out.append(_sample(res / res.sum(), u_extra[r]))
                break
        else:
            out.append(_sample(Pt[prev], u_extra[r]))
        acc.append(n_ok)
        last = out[-1]
        r += 1
    return np.array(out[:n_new]), np.array(acc)
'''

HINTS = [
    'Each round produces between 1 and $k + 1$ tokens: the accepted drafts plus one token from the target (a correction or a bonus).',
    'Both the draft and the target distributions at position $j$ are conditioned on the previous **draft** token, which is what the target scores in its single parallel pass.',
    'The residual $\\max(0, p - q)$ is only used after a rejection, which can only happen where $p(x) < q(x)$, so its sum is positive.',
]

_r = np.random.default_rng(2301)
def _models(V, closeness):
    Pt = _r.dirichlet(np.ones(V) * 0.7, size=V)
    noise = _r.dirichlet(np.ones(V), size=V)
    return Pt, closeness * Pt + (1 - closeness) * noise
def _c(V, n_new, k, closeness, R=None):
    Pt, Pd = _models(V, closeness)
    R = R or n_new
    return Pt, Pd, int(_r.integers(V)), n_new, k, _r.random((R, k)), _r.random((R, k)), _r.random(R)
TESTS = [
    _c(3, 4, 2, 0.5),
    _c(4, 10, 3, 0.9),
    _c(6, 20, 4, 0.2),
    _c(5, 15, 1, 0.7),
    _c(8, 30, 5, 0.95),
    _c(4, 12, 3, 1.0),
]

EXPLANATION = r"""
## Why it is exact
A draft token is kept with probability $\min(1, p/q)$; when it isn't, the replacement comes from the normalised $\max(0, p - q)$. Adding the two cases gives exactly $p$ for every token, so the output is distributed as if sampled from the target alone, whatever the draft model is. A poor draft only costs speed.

## Why it is faster
Decoding a large model is memory-bandwidth bound: one forward pass on $k + 1$ positions costs nearly the same as on one. Each round costs $k$ cheap draft steps plus one target pass and yields $1 + (\text{accepted})$ tokens. With acceptance rate $\alpha$ per token, the expected yield per round is $(1 - \alpha^{k+1})/(1 - \alpha)$. Test 6 uses a draft identical to the target, so every draft is accepted and each round yields $k + 1$ tokens.

## Choosing $k$
Larger $k$ helps only while the draft keeps agreeing with the target; the chance of reaching position $j$ decays like $\alpha^j$. Under heavy batching the target is no longer bandwidth-bound and the speedup shrinks.

## Common bugs
- Sampling the correction from $p$ instead of the residual (biases the output).
- Conditioning the target's distributions on the accepted prefix only, instead of on each draft token in turn.
- Forgetting the bonus token when all drafts are accepted.

## Interview follow-ups
- How do Medusa or EAGLE get drafts without a separate model?
- How does greedy (temperature 0) speculative decoding simplify the acceptance rule?
"""
