import numpy as np

META = dict(title='Perplexity from logits with a padding mask', tags=['Language Models', 'Evaluation Metrics'], diff='Medium', runtime='numpy',
            lessons=['ieng:02.04', 'aieng:03.02'], related=['language-modeling', 'n-gram-language-model', 'cross-entropy-logits'])

STATEMENT = r"""
Perplexity is the exponential of the average negative log-likelihood per predicted token. A perplexity of 20 means the model is, on average, as uncertain as a uniform choice among 20 tokens.

Write `perplexity(logits, ids, mask)` that returns `(corpus_ppl, per_seq_ppl)`.

## Inputs
- `logits`: float array `(B, T, V)`. `logits[b, t]` is the model's prediction for the token at position `t + 1`.
- `ids`: int array `(B, T)` of token ids.
- `mask`: array `(B, T)` of 0/1 (or bools), 1 at real tokens. Padding can be on either side.

## Targets
Position `t + 1` of sequence `b` (for `t = 0 .. T-2`) is a target if **both** `mask[b, t]` and `mask[b, t+1]` are 1: the token is real and it is predicted from a real token. Its loss is
$$\ell_{b,t+1} = -\ln \mathrm{softmax}(\text{logits}[b, t])_{\,\text{ids}[b, t+1]}$$
computed in a numerically stable way.

## Output
- `corpus_ppl`: $\exp\big(\sum \ell / \text{number of targets}\big)$ over **all** targets in the batch, as a float. The tests always have at least one target.
- `per_seq_ppl`: float array `(B,)`, the same formula restricted to each sequence, or `nan` for a sequence with no targets.
"""

STARTER = '''import numpy as np

def perplexity(logits, ids, mask):
    """Return (corpus perplexity float, per-sequence perplexities (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def perplexity(logits, ids, mask):
    logits, ids = np.asarray(logits, dtype=float), np.asarray(ids)
    mask = np.asarray(mask).astype(bool)
    z = logits[:, :-1]                                      # predictions for positions 1..T-1
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    tgt = ids[:, 1:]
    nll = -np.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]   # (B, T-1)
    valid = mask[:, :-1] & mask[:, 1:]
    nll = np.where(valid, nll, 0.0)
    count = valid.sum(axis=1)
    per_seq = np.full(len(ids), np.nan)
    has = count > 0
    per_seq[has] = np.exp(nll.sum(axis=1)[has] / count[has])
    return float(np.exp(nll.sum() / count.sum())), per_seq
'''

HINTS = [
    'Shift once: `logits[:, :-1]` lines up with targets `ids[:, 1:]`. Everything after that is elementwise over a `(B, T-1)` grid.',
    'Use a stable log-softmax and gather the target column with `np.take_along_axis`. The valid-target mask is `mask[:, :-1] & mask[:, 1:]`.',
    'Pool the corpus number over tokens, not over sequences: total NLL divided by total target count, then `exp`. Per-sequence values divide by each row\'s own count, with `nan` where it is 0.',
]

_r = np.random.default_rng(59)
_B, _T, _V = 4, 7, 9
_lg = _r.normal(size=(_B, _T, _V)) * 2
_ids = _r.integers(0, _V, size=(_B, _T))
_m = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0]])
TESTS = [
    (np.zeros((1, 3, 4)), np.array([[0, 1, 2]]), np.array([[1, 1, 1]])),
    (np.log(np.array([[[0.5, 0.5], [0.9, 0.1], [0.5, 0.5]]])), np.array([[0, 0, 1]]), np.array([[1, 1, 1]])),
    (np.zeros((2, 3, 5)), np.array([[0, 1, 2], [3, 3, 3]]), np.array([[1, 1, 0], [1, 0, 0]])),
    (np.array([[[1000.0, 0.0], [0.0, 1000.0]]]), np.array([[1, 0]]), np.array([[True, True]])),
    (_lg, _ids, _m),
    (_lg, _ids, np.ones((_B, _T), dtype=int)),
    (_lg * 0.1, _ids, _m),
]

EXPLANATION = r"""
## Why it works
Perplexity is $\exp(H)$ where $H$ is the cross-entropy in nats per token. Shifting by one aligns each prediction with the token it predicts, and the two-sided mask removes padding targets as well as the first real token of a left-padded sequence (it has no real context). A model that outputs uniform logits over $V$ tokens has perplexity exactly $V$ (test 1 gives 4).

## Corpus vs average of sequences
The corpus value pools all tokens, so long sequences weigh more. Averaging per-sequence perplexities gives a different, usually larger, number because $\exp$ is convex (Jensen's inequality). Papers report the pooled version; always say which one you compute.

## Numerical stability
Test 4 has logits of 1000 where the correct token is the *unlikely* one. A naive `np.log(np.exp(z) / sum)` overflows to `nan`; the stable log-softmax gives a loss of exactly 1000 per token. Perplexity itself then overflows to `inf`, which is the honest answer.

## Complexity
$O(BTV)$ for the log-softmax, the same as one forward pass of the output layer.

## Common bugs
- Not shifting, so each position "predicts" itself.
- Dividing by $B \cdot (T - 1)$ instead of the number of real targets.
- Averaging per-sequence perplexities and calling it corpus perplexity.
- Comparing perplexities of models with different tokenizers: per-token numbers are not comparable, bits per byte or per character are.

## Interview follow-ups
- How do you evaluate a document longer than the context window? (A sliding window with a stride: each token is scored once, with as much preceding context as fits.)
- Why does lower perplexity not always mean a better chatbot? (It measures fit to the evaluation text, not helpfulness or factuality.)
"""
