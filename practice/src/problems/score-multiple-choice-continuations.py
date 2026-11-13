import numpy as np

META = dict(title='Score multiple-choice answers four ways', tags=['Language Models', 'Evaluation Metrics'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:10.05'],
            related=['conditional-perplexity', 'beam-search-decoding', 'fit-elo-ratings-from-pairwise-votes'])

STATEMENT = r"""
Base models are evaluated on multiple-choice benchmarks without generating anything: each answer option is appended to the question and the model's log-likelihood of the option's tokens decides the prediction. How that likelihood is normalised changes the accuracy noticeably, so evaluation harnesses report several variants.

Write `score_choices(logprobs, mask, n_bytes, uncond_logprobs, answer)` and return a dict.

## Inputs
For $Q$ questions with $C$ options each, padded to $T$ tokens:
- `logprobs`: float64 `(Q, C, T)`, `logprobs[i, c, t]` is $\log p(\text{token}_t \mid \text{question}, \text{earlier option tokens})$.
- `mask`: 0/1 `(Q, C, T)`, 1 on real option tokens (left-aligned, at least one per option). Padded log-probs are arbitrary.
- `n_bytes`: int `(Q, C)`, UTF-8 length of each option's text.
- `uncond_logprobs`: float64 `(Q, C, T)`, the same option tokens scored after a neutral prompt such as `"Answer:"` (same mask).
- `answer`: int `(Q,)`, index of the correct option.

## Methods
With $S = \sum_t \text{mask} \cdot \text{logprobs}$ per option:

| method | score |
|---|---|
| `"sum"` | $S$ |
| `"per_token"` | $S / n_\text{tokens}$ |
| `"per_byte"` | $S / \text{n\_bytes}$ |
| `"calibrated"` | $S - S_\text{uncond}$, a pointwise mutual information |

Each method predicts the option with the **highest** score; on an exact tie, the lowest option index.

## Output
A dict mapping each method name to a dict `{"pred": int array (Q,), "acc": float}`, where `acc` is the fraction of questions with `pred == answer`.
"""

STARTER = '''import numpy as np

def score_choices(logprobs, mask, n_bytes, uncond_logprobs, answer):
    """Return {method: {"pred": (Q,) ints, "acc": float}} for sum, per_token, per_byte, calibrated."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def score_choices(logprobs, mask, n_bytes, uncond_logprobs, answer):
    m = np.asarray(mask, dtype=float)
    s = (np.asarray(logprobs, dtype=float) * m).sum(axis=-1)           # (Q, C); padding zeroed by the mask
    s_uncond = (np.asarray(uncond_logprobs, dtype=float) * m).sum(axis=-1)
    scores = {
        "sum": s,
        "per_token": s / m.sum(axis=-1),
        "per_byte": s / np.asarray(n_bytes, dtype=float),
        "calibrated": s - s_uncond,
    }
    out = {}
    for name, sc in scores.items():
        pred = np.argmax(sc, axis=1)                  # first maximum = lowest index on ties
        out[name] = {"pred": pred, "acc": float(np.mean(pred == answer))}
    return out
'''

HINTS = [
    'Mask first: `(logprobs * mask).sum(-1)` gives the `(Q, C)` sequence log-likelihoods. Padding values can be anything, including huge numbers, so never sum them unmasked.',
    'The four scores are elementwise transforms of that `(Q, C)` matrix: divide by the token count, divide by the byte count, or subtract the unconditional sum.',
    '`np.argmax(scores, axis=1)` already picks the lowest index among ties. Accuracy is `np.mean(pred == answer)` as a float.',
]


def _bank(seed, Q, C, T):
    r = np.random.default_rng(seed)
    lens = r.integers(1, T + 1, size=(Q, C))
    mask = (np.arange(T)[None, None, :] < lens[..., None]).astype(np.int64)
    lp = -r.gamma(2.0, 1.0, size=(Q, C, T))
    ulp = lp - r.normal(0.5, 1.0, size=(Q, C, T))
    lp = np.where(mask == 1, lp, 99.0)                 # loud padding
    return lp, mask, lens * r.integers(3, 6, size=(Q, C)), np.where(mask == 1, ulp, -99.0), r.integers(0, C, size=Q)


TESTS = [
    (np.array([[[-1.0, -1.0], [-1.5, 0.0]]]), np.array([[[1, 1], [1, 0]]]), np.array([[6, 3]]),
     np.array([[[-2.0, -2.0], [-1.0, 0.0]]]), np.array([0])),
    (np.array([[[-0.5, -0.5, -0.5, -0.5], [-1.0, 0.0, 0.0, 0.0], [-0.9, -0.9, 0.0, 0.0]]]),
     np.array([[[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]]]), np.array([[8, 2, 9]]),
     np.array([[[-0.55, -0.55, -0.55, -0.55], [-1.1, 0.0, 0.0, 0.0], [-1.5, -1.5, 0.0, 0.0]]]), np.array([0])),
    _bank(0, 6, 4, 5),
    _bank(1, 20, 5, 8),
    (np.array([[[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]]), np.ones((1, 3, 2), dtype=np.int64) * np.array([1, 0]),
     np.array([[4, 4, 4]]), np.array([[[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]]), np.array([1])),
    _bank(2, 50, 4, 12),
]

EXPLANATION = r"""
## Why the normalisations differ
- **sum** is the model's probability of the exact option string. It penalises long options: every extra token multiplies in a probability below 1.
- **per_token** removes most of that length penalty, but depends on the tokenizer.
- **per_byte** normalises by the text's length instead, so it's comparable across tokenizers; harnesses often call this "acc_norm".
- **calibrated** divides out how likely the option is *a priori* (after a content-free prompt). An option like "None of the above" may be likely regardless of the question; subtracting $S_\text{uncond}$ asks how much the question **raised** its probability.

Test 2 is built so that `sum`, `per_token` and `per_byte` each pick a different option.

## Ties
Exact ties do happen in practice (identical options, or symmetric scores). Test 5 has three identical options and expects index 0 everywhere. Breaking ties randomly would make the benchmark score nondeterministic.

## Complexity
$O(QCT)$, two masked reductions.

## Common bugs
- Summing padded positions (the tests fill them with $\pm 99$).
- Taking `argmin` of negative log-likelihoods but then normalising the wrong sign.
- Using the number of options or the max length $T$ instead of each option's own token count.

## Interview follow-ups
- Why do multiple-choice scores for small models sit near chance while generation-based scores are 0, and which one shows "emergence" more sharply?
- How would you evaluate an instruction-tuned model that answers in free text ("The answer is (B) because ...") instead?
"""
