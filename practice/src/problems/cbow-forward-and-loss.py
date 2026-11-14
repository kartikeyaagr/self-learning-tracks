import numpy as np

META = dict(title='CBOW forward pass and full-softmax loss', tags=['NLP', 'Embeddings'], diff='Medium', runtime='numpy',
            related=['word2vec', 'word2vec-sgns-epoch'])

STATEMENT = r"""
Continuous bag-of-words predicts a centre word from its context. Write `cbow_loss(tokens, window, W_in, W_out)`.

- For every position $t$ that has at least one context word, the context is the tokens at offsets $-$`window`..$+$`window` excluding $t$ itself, clipped to the sequence.
- Hidden vector $h_t$ = mean of `W_in` rows for the context tokens.
- Scores $z = h_t$ `@ W_out` (shape `(V,)`), probabilities by softmax, loss $-\log p(\text{tokens}[t])$.

Return the mean loss over those positions as a float. Use a numerically stable log-softmax.
"""

STARTER = '''import numpy as np

def cbow_loss(tokens, window, W_in, W_out):
    """Mean CBOW cross-entropy over positions."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cbow_loss(tokens, window, W_in, W_out):
    W_in, W_out = np.asarray(W_in, float), np.asarray(W_out, float)
    n, losses = len(tokens), []
    for t in range(n):
        ctx = [tokens[j] for j in range(max(0, t - window), min(n, t + window + 1)) if j != t]
        if not ctx:
            continue
        z = W_in[ctx].mean(0) @ W_out
        m = z.max()
        logp = z - m - np.log(np.exp(z - m).sum())
        losses.append(-logp[tokens[t]])
    return float(np.mean(losses)) if losses else 0.0
'''

HINTS = [
    'Average the input vectors of the context words; that average is the whole hidden layer.',
    'Subtract the max score before exponentiating to keep the log-softmax stable.',
]

def _case(seed, V, d, n, w):
    r = np.random.default_rng(seed)
    return (list(r.integers(0, V, n)), w, r.normal(size=(V, d)), r.normal(size=(d, V)))

TESTS = [([0, 1, 2], 1, np.eye(3), np.eye(3)), _case(1, 6, 3, 8, 2), _case(2, 10, 4, 12, 1),
         _case(3, 5, 2, 5, 3), ([4], 2, np.ones((5, 2)), np.ones((2, 5))), _case(4, 8, 5, 20, 2)]

EXPLANATION = r"""
## CBOW versus skip-gram
Skip-gram predicts each context word from the centre; CBOW predicts the centre from the averaged context. Averaging smooths over the context, so CBOW trains faster and does well on frequent words, while skip-gram gives better vectors for rare words because each occurrence produces several updates.

## Why the full softmax is only a teaching tool
Every step costs $O(Vd)$ for the output scores. Real word2vec replaces it with negative sampling or hierarchical softmax, which cost $O(kd)$ or $O(d \log V)$ per step.

## Interview follow-ups
- What does averaging lose about word order in the window?
- Which matrix do you keep as the embedding table, and why do some implementations average the two?
"""
