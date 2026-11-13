import numpy as np

META = dict(title='Sequence-mean vs token-mean loss aggregation', tags=['Reinforcement Learning', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.05'],
            related=['normalize-grpo-group-advantages', 'grpo-group-relative-advantage', 'gradient-accumulation'])

STATEMENT = r"""
Policy-gradient losses for language models are computed per token and then reduced to one number. The reduction looks like a detail but decides how much each token's gradient counts, and in RL on reasoning traces the responses have wildly different lengths.

Write `aggregate_losses(token_loss, mask, max_len)` and return a dict with one entry per method.

## Inputs
- `token_loss`: float64 `(B, T)`; `mask`: 0/1 `(B, T)` with at least one 1 per row. Masked-out entries of `token_loss` are arbitrary.
- `max_len`: int, a fixed normaliser (the generation budget), $\ge T$.

## Methods
With $n_b$ = valid tokens in row $b$ and $\ell_{b,t}$ the token losses:

| method | loss |
|---|---|
| `"seq_mean"` | $\frac1B \sum_b \frac{1}{n_b}\sum_t m_{b,t}\,\ell_{b,t}$ (average each response, then the batch) |
| `"token_mean"` | $\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}{\sum_{b,t} m_{b,t}}$ (pool all tokens) |
| `"constant"` | $\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}{B \cdot \text{max\_len}}$ (a length-independent constant) |

## Output
A dict mapping each method to a tuple `(loss, weights)`:
- `loss`: float.
- `weights`: float64 `(B, T)`, the gradient $\partial\,\text{loss} / \partial \ell_{b,t}$: the weight each token receives. It is 0 on masked positions.

Masked-out losses may be huge or non-finite-looking garbage such as `1e30`; they must not affect anything.
"""

STARTER = '''import numpy as np

def aggregate_losses(token_loss, mask, max_len):
    """Return {method: (loss, weights (B, T))} for seq_mean, token_mean, constant."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def aggregate_losses(token_loss, mask, max_len):
    m = np.asarray(mask, dtype=float)
    l = np.where(m > 0, np.asarray(token_loss, dtype=float), 0.0)    # drop garbage before any arithmetic
    B = m.shape[0]
    n = m.sum(axis=1, keepdims=True)                                 # (B, 1) valid tokens per row
    # Every method is a weighted sum of token losses, so build the weights and reuse them for the loss.
    weights = {
        "seq_mean": m / (B * n),
        "token_mean": m / m.sum(),
        "constant": m / (B * max_len),
    }
    return {name: (float((w * l).sum()), w) for name, w in weights.items()}
'''

HINTS = [
    'All three losses are linear in the token losses, so each is $\\sum_{b,t} w_{b,t}\\,\\ell_{b,t}$ for some weight matrix. Compute the weights first; the loss and the gradient both follow.',
    'For `seq_mean` a token in row $b$ has weight $1/(B\\,n_b)$; for `token_mean` it is $1/\\sum m$; for `constant` it is $1/(B \\cdot \\text{max\\_len})$, all times the mask.',
    'Replace masked losses with 0 using `np.where` before multiplying: `1e30 * 0` is fine, but `inf * 0` would be `nan`.',
]

_r = np.random.default_rng(41)
_len = np.array([3, 12, 1, 7, 20])
_m = (np.arange(20)[None, :] < _len[:, None]).astype(np.int64)
TESTS = [
    (np.array([[1.0, 1.0, 1.0, 1.0], [4.0, 9.0, 9.0, 9.0]]), np.array([[1, 1, 1, 1], [1, 0, 0, 0]]), 4),
    (np.array([[2.0, 0.0], [1.0, 3.0]]), np.array([[1, 0], [1, 1]]), 8),
    (np.where(_m == 1, _r.normal(size=(5, 20)), 1e30), _m, 32),
    (np.where(_m == 1, np.abs(_r.normal(size=(5, 20))), -1e30), _m, 20),
    (np.ones((3, 5)), np.ones((3, 5), dtype=np.int64), 5),
    (_r.normal(size=(1, 6)), np.array([[0, 1, 1, 0, 1, 0]]), 6),
    (np.where(_m[:3] == 1, 1.0, 7.0), _m[:3], 20),
]

EXPLANATION = r"""
## Where the length bias comes from
Under `seq_mean`, every response gets total weight $1/B$, shared among its tokens. A 1,000-token response gives each token a hundredth of the weight a 10-token response does. With a negative advantage (a wrong answer), the per-token penalty on a long wrong answer is therefore tiny, so the policy is pushed less hard away from rambling, and responses tend to grow longer over training. That's one of the biases Dr. GRPO identified.

`token_mean` gives every token in the batch the same weight, which fixes the per-response bias but makes the weight depend on the batch's total length, so one very long response dilutes every other token's gradient. `constant` uses a fixed normaliser: weights are independent of all lengths, and the gradient is an unbiased (up to a constant) sum over tokens. Test 1 makes the difference visible with one 4-token and one 1-token response.

## When they agree
If every response has the same length $n$, then `seq_mean` = `token_mean` = $\frac{\text{max\_len}}{n}\cdot$`constant` (test 5, where they coincide exactly with `max_len` $= n$).

## Complexity
$O(BT)$.

## Common bugs
- `token_loss.mean()` over the padded array: counts padding as zero-loss tokens.
- Multiplying garbage by the mask after it could have become `inf` or `nan`.
- In gradient accumulation across micro-batches, applying `token_mean` per micro-batch and averaging: that is neither method, because the token counts differ between micro-batches.

## Interview follow-ups
- With gradient accumulation over 8 micro-batches, how do you implement exact `token_mean` without knowing the total token count in advance?
- Which aggregation would you choose for SFT on chat data with mixed short and long answers, and why?
"""
