META = dict(title='Remove stopwords while keeping negations', tags=['NLP', 'Data Processing'], diff='Easy', runtime='python',
            related=['tokenization'])

STATEMENT = r"""
Write `remove_stopwords(text, stopwords, keep)`.

- Lowercase the text and split it into tokens with the regex `[a-z0-9']+` (apostrophes stay inside a token, so `don't` is one token).
- Drop every token in `stopwords` unless it is also in `keep`.
- Return the remaining tokens as a list, in order.

`keep` exists because standard stopword lists contain negations such as `not` and `no`, and removing them flips the meaning of a sentiment example.
"""

STARTER = '''import re

def remove_stopwords(text, stopwords, keep):
    """Return the tokens of text with stopwords removed, except those in keep."""
    return text.split()
'''

SOLUTION = '''import re

def remove_stopwords(text, stopwords, keep):
    stop = set(stopwords) - set(keep)
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in stop]
'''

HINTS = [
    'Lowercase before matching; the regex only matches lowercase letters.',
    'Build one set, stopwords minus keep, then filter once.',
]

_S = ["the", "a", "is", "not", "no", "and", "of", "to", "it", "this", "don't"]
TESTS = [
    ("This movie is NOT good.", _S, ["not", "no"]),
    ("This movie is NOT good.", _S, []),
    ("I don't like it, and no one should.", _S, ["no", "don't"]),
    ("", _S, []),
    ("The cost of 3 items is 12 dollars", _S, []),
    ("A a A the THE", _S, ["a"]),
]

EXPLANATION = r"""
## When stopword removal helps
For bag-of-words models (TF-IDF plus a linear classifier, BM25 retrieval) common function words carry little signal and dominate counts, so removing them shrinks the vocabulary and sharpens weights.

## When it hurts
Negations and modal words change meaning. "not good" with `not` removed becomes "good". Transformer models learn from word order and context, so modern pipelines usually skip stopword removal entirely.

## Practical notes
- Tokenise the same way at training and serving time, or the lists will not match.
- IDF weighting already down-weights frequent words, which is one reason BM25 works without explicit stopword lists.

## Interview follow-ups
- How would you build a domain-specific stopword list from corpus statistics?
"""
