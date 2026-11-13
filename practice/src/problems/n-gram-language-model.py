import math

META = dict(title='Add-k smoothed n-gram language model', tags=['Language Models', 'NLP'], diff='Medium', runtime='python',
            lessons=['aieng:03.02'], related=['perplexity-computation', 'language-modeling', 'vocabulary-building'])

STATEMENT = r"""
An n-gram language model predicts each word from the previous $n-1$ words using counts. Write `ngram_logprob(train, test, n, k=1.0)` that trains on `train` and returns the natural-log probability of each sentence in `test`.

## Preparing a sentence
Prepend $n-1$ copies of `"<s>"` and append one `"</s>"`. For `n = 2`, `["a", "b"]` becomes `["<s>", "a", "b", "</s>"]`. Each position from the first real word through `"</s>"` is predicted from the $n-1$ tokens before it.

## Vocabulary
$V$ is the set of distinct training words, plus `"</s>"` and `"<unk>"` (`"<s>"` is never predicted, so it is not in $V$). Before scoring, replace every test word that never appeared in training with `"<unk>"`.

## Probabilities
With $c(h, w)$ the number of times word $w$ follows history $h$ in the prepared training sentences, and $c(h) = \sum_w c(h, w)$:
$$P(w \mid h) = \frac{c(h, w) + k}{c(h) + k\,|V|}$$
Histories never seen in training get $c(h) = 0$, so their distribution is uniform.

## Output
A list of floats: for each test sentence, the sum of $\ln P(w \mid h)$ over its predicted positions (including `"</s>"`).

$n \ge 1$ and $k > 0$. For $n = 1$ the history is empty and every word shares the same counts.
"""

STARTER = '''import math

def ngram_logprob(train, test, n, k=1.0):
    """Return [log P(sentence) for sentence in test] under an add-k n-gram model."""
    raise NotImplementedError
'''

SOLUTION = '''import math
from collections import Counter

def ngram_logprob(train, test, n, k=1.0):
    pad = lambda s: ["<s>"] * (n - 1) + list(s) + ["</s>"]
    vocab = {w for s in train for w in s} | {"</s>", "<unk>"}
    pair, hist = Counter(), Counter()
    for s in train:
        p = pad(s)
        for i in range(n - 1, len(p)):
            h = tuple(p[i - n + 1:i])            # empty tuple when n == 1
            pair[h, p[i]] += 1
            hist[h] += 1
    out = []
    for s in test:
        p = pad(w if w in vocab else "<unk>" for w in s)
        lp = 0.0
        for i in range(n - 1, len(p)):
            h = tuple(p[i - n + 1:i])
            lp += math.log((pair[h, p[i]] + k) / (hist[h] + k * len(vocab)))
        out.append(lp)
    return out
'''

HINTS = [
    'Pad every sentence the same way, then slide a window: the history is the `n-1` tokens before position `i`, and the target is `p[i]`.',
    'Two counters are enough: one for `(history, word)` pairs and one for histories. Use tuples as history keys so they are hashable.',
    "Map unseen test words to `\"<unk>\"` before padding. The denominator always adds `k * |V|`, where V includes `</s>` and `<unk>` but not `<s>`. A `Counter` returns 0 for unseen keys, which handles unseen histories for free.",
]

_train = [["the", "cat", "sat"], ["the", "dog", "sat"], ["a", "cat", "ran"], ["the", "cat", "ran", "fast"]]
TESTS = [
    ([["a", "b"], ["a", "c"]], [["a", "b"]], 2),
    (_train, [["the", "cat", "sat"], ["the", "bird", "sat"]], 2, 0.5),
    (_train, [[]], 2),
    (_train, [["cat", "the"], ["sat"]], 1, 1.0),
    (_train, [["the", "cat", "ran", "fast"], ["a", "dog", "sat"]], 3, 0.1),
    ([[]], [["x"], []], 2),
    (_train, [["zebra", "zebra"]], 3, 2.0),
]

EXPLANATION = r"""
## Why it works
The maximum-likelihood estimate $c(h, w)/c(h)$ gives probability 0 to anything unseen, and one zero makes a whole sentence impossible ($\ln 0 = -\infty$). Add-$k$ pretends every word followed every history $k$ extra times. The denominator grows by $k|V|$ so each conditional distribution still sums to 1 over $V$. `"<unk>"` gives unseen words a share of probability mass, and `"</s>"` makes the model a distribution over sentences of every length, not just over words.

## Complexity
Training is $O(n \cdot T)$ for $T$ training tokens (building each history tuple costs $n$). Scoring is the same per test token. Memory grows with the number of distinct n-grams, which explodes as $n$ increases.

## Common bugs
- Leaving `"<s>"` in $V$ or `"</s>"` out of it, which changes $|V|$.
- Forgetting to predict `"</s>"`.
- Mapping unknown words to `"<unk>"` in the test set but never giving `"<unk>"` a place in $V$.
- Using string concatenation for history keys (`"a b"`), which collides when tokens contain spaces.

## Interview follow-ups
- Add-one smoothing moves too much mass to unseen events. What do Kneser-Ney and interpolation do better? (They back off to lower orders, and KN uses how many contexts a word appears in rather than its raw count.)
- How do you turn these log-probabilities into perplexity? (Divide the negated total by the number of predicted tokens and exponentiate.)
"""
