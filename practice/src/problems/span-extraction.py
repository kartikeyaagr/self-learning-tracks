import numpy as np

META = dict(title='Best answer span from start and end logits', tags=['Transformers', 'NLP'], diff='Medium', runtime='numpy',
            related=['next-sentence-prediction', 'named-entity-recognition', 'token-classification'])

STATEMENT = r"""
Extractive question answering models output two scores per token: how likely the answer **starts** there and how likely it **ends** there. The answer is the span with the best combined score, subject to a few constraints. Models trained on data with unanswerable questions also use the `[CLS]` token at position 0 to say "no answer".

Write `best_span(start_logits, end_logits, context_mask, max_answer_len, null_threshold=0.0)` that returns `(start, end, score)`.

## Inputs
- `start_logits`, `end_logits`: float arrays `(T,)`.
- `context_mask`: bool array `(T,)`, `True` at tokens of the context (the answer must lie there). Position 0 is never context.
- `max_answer_len`: int $\ge 1$, the maximum number of tokens in an answer.

## Rules
1. A span `(i, j)` is valid if `context_mask[i]` and `context_mask[j]` are true and $i \le j \le i + \text{max\_answer\_len} - 1$. Its score is `start_logits[i] + end_logits[j]`.
2. Take the valid span with the highest score. Ties go to the smallest `i`, then the smallest `j`.
3. The null score is `start_logits[0] + end_logits[0]`. Return the null answer `(0, 0, null_score)` if there is no valid span, or if `null_score > best_score + null_threshold`. Otherwise return the best span.

`start` and `end` are Python ints (inclusive end), `score` a Python float.
"""

STARTER = '''import numpy as np

def best_span(start_logits, end_logits, context_mask, max_answer_len, null_threshold=0.0):
    """Return (start, end_inclusive, score), or (0, 0, null_score) for no answer."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def best_span(start_logits, end_logits, context_mask, max_answer_len, null_threshold=0.0):
    s = np.asarray(start_logits, dtype=float)
    e = np.asarray(end_logits, dtype=float)
    ctx = np.asarray(context_mask, dtype=bool)
    T = len(s)
    best = None                                   # (score, i, j)
    for i in np.flatnonzero(ctx):                 # increasing i
        for j in range(i, min(T, i + max_answer_len)):   # increasing j
            if ctx[j]:
                sc = s[i] + e[j]
                if best is None or sc > best[0]:  # strict >: earlier (i, j) wins ties
                    best = (sc, int(i), j)
    null = float(s[0] + e[0])
    if best is None or null > best[0] + null_threshold:
        return 0, 0, null
    return best[1], best[2], float(best[0])
'''

HINTS = [
    'Only pairs with `i <= j < i + max_answer_len` matter, so loop over context starts and, for each, over at most `max_answer_len` ends: $O(T \\cdot L)$ instead of $O(T^2)$.',
    'Visit `i` in increasing order and `j` in increasing order, and replace the best only on a strictly larger score. That implements the tie rule without extra work.',
    'Compare against the null score last, using the threshold: `null > best + null_threshold` means "no answer". Return plain ints and a float.',
]

_r = np.random.default_rng(43)
_T = 24
_ctx = np.zeros(_T, dtype=bool)
_ctx[8:23] = True
_s, _e = _r.normal(size=_T) * 2, _r.normal(size=_T) * 2
TESTS = [
    (np.array([0.0, 0.0, 5.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 4.0]), np.array([False, False, True, True, True]), 5),
    (np.array([0.0, 0.0, 5.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 4.0]), np.array([False, False, True, True, True]), 2),
    (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([False, False]), 3),
    (np.array([9.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([False, True, True]), 2),
    (np.array([9.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([False, True, True]), 2, 10.0),
    (np.array([0.0, 3.0, 0.0, 3.0]), np.array([0.0, 2.0, 0.0, 2.0]), np.array([False, True, False, True]), 4),
    (np.array([0.0, 1.0, 5.0, 1.0]), np.array([0.0, 5.0, 0.0, 1.0]), np.array([False, True, True, True]), 3),
    (_s, _e, _ctx, 6),
    (_s, _e, _ctx, 30, -100.0),
]

EXPLANATION = r"""
## Why it works
The model treats start and end as independent softmaxes, so the log-probability of a span is (up to constants) `start_logits[i] + end_logits[j]`, and maximizing the sum maximizes the span's probability. The constraints remove spans that cannot be answers: ends before starts, spans in the question or padding, and spans longer than any real answer. In test 7 the best start (2) and best end (1) are incompatible, which is exactly why you cannot take the two argmaxes separately.

## The null answer
SQuAD 2.0-style models point both start and end at `[CLS]` for unanswerable questions. Comparing the null score with the best real span, shifted by a threshold tuned on validation data, trades off "abstaining" against "guessing". A large threshold (test 5) makes the model answer more often; a very negative one (test 9) always abstains.

## Complexity
$O(T \cdot L)$ with $L$ = `max_answer_len`. Libraries usually restrict to the top 20 starts and ends first, which is $O(400)$ per example.

## Common bugs
- Taking `argmax(start)` and `argmax(end)` independently, which can give `end < start` or a span outside the context.
- Forgetting that `max_answer_len` counts tokens, so `j - i + 1 <= L`.
- Letting the `[CLS]` position compete as a normal span start.

## Interview follow-ups
- With a long document split into overlapping windows, how do you pick the final answer? (Score spans in every window and compare their scores, mapping tokens back to characters with the offsets.)
- How would you calibrate `null_threshold`? (Sweep it on a validation set and maximise F1 or exact match.)
"""
