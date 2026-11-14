import numpy as np

META = dict(title='Top activating examples for one feature', tags=['Neural Networks', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['feature-activation-frequency'])

STATEMENT = r"""
To guess what an SAE feature means, look at the tokens where it fires hardest, with some context.

Write `top_examples(acts, tokens, k, window)`:

- `acts (n_docs, seq_len)` holds one feature's activation at every token; `tokens` is the matching list of token-string lists.
- Keep only positions with activation strictly above 0.
- Sort by activation, highest first; break ties by document index, then position, both ascending.
- Return up to `k` dicts `{"doc", "pos", "act", "context"}`, where `context` is the tokens from `pos - window` to `pos` inclusive (clipped at the start of the document).
"""

STARTER = '''import numpy as np

def top_examples(acts, tokens, k, window):
    """The k strongest activations of one feature, with left context."""
    acts = np.asarray(acts, float)
    flat = np.argsort(acts.ravel())[-k:]
    return [{"doc": int(i // acts.shape[1]), "pos": int(i % acts.shape[1]), "act": float(acts.ravel()[i]), "context": []} for i in flat]
'''

SOLUTION = '''import numpy as np

def top_examples(acts, tokens, k, window):
    acts = np.asarray(acts, float)
    hits = [(-acts[d, p], d, p) for d in range(acts.shape[0]) for p in range(acts.shape[1]) if acts[d, p] > 0]
    hits.sort()
    out = []
    for neg, d, p in hits[:k]:
        out.append({"doc": d, "pos": p, "act": float(-neg), "context": list(tokens[d][max(0, p - window):p + 1])})
    return out
'''

HINTS = [
    'Sort tuples of (-activation, doc, pos) to get descending activation with ascending tie-breaks in one go.',
    'The context window ends at the firing token, because the model at that position has only seen tokens to its left.',
]

_docs = [["the", "cat", "sat", "on", "the", "mat"], ["a", "dog", "ran", "to", "the", "park"], ["the", "end", "of", "the", "day", "."]]
_acts = [[0.0, 0.0, 0.0, 0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0, 0.0], [3.0, 0.0, 0.0, 1.5, 0.0, 0.0]]

TESTS = [
    (_acts, _docs, 2, 2),
    (_acts, _docs, 10, 1),
    (_acts, _docs, 3, 0),
    ([[0.0, 0.0], [0.0, 0.0]], [["x", "y"], ["z", "w"]], 3, 1),
    ([[0.5, 0.2, 0.9]], [["p", "q", "r"]], 2, 5),
]

EXPLANATION = r"""
## Reading a feature
The top examples are the fastest way to form a hypothesis: if every top context ends in "the" right before a noun about animals, that is a guess to test.

## Why the top is not enough
The strongest activations show what the feature means when it is most confident, which can be much narrower than what it does at moderate strength. Good practice samples examples from several activation quantiles, checks random tokens where the feature is off, and tests the hypothesis by steering: add the decoder direction to the residual stream and see whether the model's output changes the way the label predicts.

## Interview follow-ups
- How would you automate labelling thousands of features, and how would you score whether a label is right?
"""
