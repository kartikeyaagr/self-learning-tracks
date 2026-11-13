import numpy as np

META = dict(title='Conditional perplexity and bits per byte', tags=['Language Models', 'Evaluation Metrics'],
            diff='Medium', runtime='numpy', lessons=['aieng:03.02', 'ieng:02.04'],
            related=['perplexity-computation', 'score-multiple-choice-continuations', 'softmax-numerical-stability-and-log-sum-exp'])

STATEMENT = r"""
Many evaluations ask how well a model predicts a **continuation given a prompt**: the prompt tokens are context and must not be scored. And because models with different tokenizers split the same text into different numbers of tokens, per-token perplexity can't compare them. **Bits per byte** normalises by the length of the text in UTF-8 bytes instead.

Write `conditional_ppl(logits, ids, prompt_len, total_len, n_bytes)` and return a dict.

## Inputs
- `logits`: float64 `(B, T, V)`. `logits[b, t]` is the prediction for the token at position `t + 1`.
- `ids`: int `(B, T)` token ids.
- `prompt_len`, `total_len`: int `(B,)` with `1 <= prompt_len[b] < total_len[b] <= T`. Positions `0 .. prompt_len-1` are the prompt, `prompt_len .. total_len-1` the continuation, and anything after is padding.
- `n_bytes`: int `(B,)`, the UTF-8 length of each continuation's text (positive).

## Scoring
Each continuation position $j$ contributes $\ell_{b,j} = -\ln \operatorname{softmax}(\text{logits}[b, j-1])_{\text{ids}[b, j]}$, computed stably (log-sum-exp). Let $n_b$ be the number of continuation tokens.

## Output
| key | value |
|---|---|
| `"nll"` | float64 `(B,)`, $\sum_j \ell_{b,j}$ in nats |
| `"ppl"` | float64 `(B,)`, $\exp(\text{nll}_b / n_b)$ |
| `"corpus_ppl"` | float, $\exp\big(\sum_b \text{nll}_b / \sum_b n_b\big)$ (token-weighted, **not** the mean of `ppl`) |
| `"bits_per_byte"` | float, $\sum_b \text{nll}_b \,/\, (\ln 2 \cdot \sum_b \text{n\_bytes}_b)$ |

No Python loops over the batch or time axes.
"""

STARTER = '''import numpy as np

def conditional_ppl(logits, ids, prompt_len, total_len, n_bytes):
    """Return a dict: nll (B,), ppl (B,), corpus_ppl, bits_per_byte."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def conditional_ppl(logits, ids, prompt_len, total_len, n_bytes):
    logits = np.asarray(logits, dtype=float)
    B, T, V = logits.shape
    # Stable log-softmax over the vocabulary.
    m = logits.max(axis=-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
    # Position j is predicted by logits[:, j-1]: align predictions [0, T-1) with targets [1, T).
    tok_lp = np.take_along_axis(logp[:, :-1], ids[:, 1:, None], axis=-1)[..., 0]   # (B, T-1)
    j = np.arange(1, T)[None, :]
    scored = (j >= prompt_len[:, None]) & (j < total_len[:, None])                  # continuation only
    nll = -(tok_lp * scored).sum(axis=1)
    n = scored.sum(axis=1)
    return {
        "nll": nll,
        "ppl": np.exp(nll / n),
        "corpus_ppl": float(np.exp(nll.sum() / n.sum())),
        "bits_per_byte": float(nll.sum() / (np.log(2) * np.sum(n_bytes))),
    }
'''

HINTS = [
    'Compute a stable log-softmax once for the whole `(B, T, V)` array, then gather the target token\'s log-prob with `np.take_along_axis` using `ids[:, 1:]` against `logp[:, :-1]`.',
    'Build the mask over target positions $j = 1 \\dots T-1$: scored iff `prompt_len[b] <= j < total_len[b]`. Broadcasting `np.arange(1, T)[None, :]` against `prompt_len[:, None]` does it in one line.',
    'Corpus perplexity pools nats and tokens before exponentiating. Bits per byte divides the pooled nats by $\\ln 2$ times the pooled bytes.',
]

_r = np.random.default_rng(17)
_L = _r.normal(size=(3, 7, 11)) * 2.0
_ids = _r.integers(0, 11, size=(3, 7))
TESTS = [
    (np.zeros((1, 3, 4)), np.array([[0, 1, 2]]), np.array([1]), np.array([3]), np.array([4])),
    (np.array([[[2.0, 0.0], [0.0, 2.0], [5.0, 5.0]]]), np.array([[0, 1, 0]]), np.array([2]), np.array([3]), np.array([1])),
    (_L, _ids, np.array([1, 3, 6]), np.array([7, 5, 7]), np.array([20, 6, 3])),
    (_L, _ids, np.array([1, 1, 1]), np.array([7, 7, 7]), np.array([30, 30, 30])),
    (np.array([[[1000.0, 0.0, -1000.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 0.0]]]), np.array([[0, 0, 1]]), np.array([1]),
     np.array([3]), np.array([2])),
    (_r.normal(size=(4, 9, 50)) * 3.0, _r.integers(0, 50, size=(4, 9)), np.array([2, 4, 8, 1]), np.array([9, 6, 9, 2]),
     np.array([25, 7, 4, 1])),
    (np.tile(np.log(np.array([0.5, 0.25, 0.25])), (2, 4, 1)), np.array([[0, 0, 1, 2], [2, 2, 2, 2]]), np.array([2, 1]),
     np.array([4, 4]), np.array([2, 3])),
]

EXPLANATION = r"""
## Why condition on the prompt
A benchmark item like "Translate: ... Answer: ..." contains a long prompt whose tokens are not what is being evaluated. Scoring them would mix the model's skill at the task with how predictable the prompt template is, and would reward models for memorising boilerplate. Masking to the continuation measures $p(\text{answer} \mid \text{prompt})$ only.

## Why bits per byte
Token-level perplexity depends on the tokenizer: a tokenizer with longer tokens has fewer, harder predictions, so its per-token perplexity is higher even if the model assigns the text the same total probability. The total log-likelihood of a string is tokenizer-independent, so dividing it by the number of bytes gives a comparable number. Converting nats to bits ($\div \ln 2$) makes it read as "how many bits a compressor based on this model would need per byte of text".

## Corpus vs mean perplexity
`corpus_ppl` weights every token equally. The mean of per-sequence perplexities over-weights short continuations and, because $\exp$ is convex, is always at least as large as the token-weighted value. Tests 3 and 4 use the same logits with different prompt splits.

## Complexity
$O(BTV)$ for the log-softmax; everything else is $O(BT)$.

## Common bugs
- Off-by-one alignment: scoring `logits[b, j]` against `ids[b, j]` (predicting a token from itself).
- Scoring the first continuation token with the wrong logits, or dropping it: it is predicted by the **last prompt position**, `logits[b, prompt_len - 1]`.
- Computing `log(softmax(x))` naively: test 5 has logits of $\pm 1000$, which overflow `exp`.

## Interview follow-ups
- Why is perplexity on a held-out set a poor proxy for downstream accuracy after instruction tuning?
- How would you detect benchmark contamination using per-token losses on the continuation?
"""
