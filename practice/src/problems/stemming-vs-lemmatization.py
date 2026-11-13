META = dict(title='A suffix stemmer next to a lemma lookup', tags=['NLP'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:04.24'], related=['text-normalization', 'vocabulary-building'])

STATEMENT = r"""
Stemming and lemmatization both map inflected words onto a shared base so that *connects*, *connected* and *connecting* count as one feature. A **stemmer** chops suffixes by rule and may produce non-words (`running` becomes `runn`). A **lemmatizer** looks the word up and returns a real dictionary form (`ran` becomes `run`), but only for words it knows.

Write `stem_and_lemma(words, lemmas)` that returns a list of `(stem, lemma)` tuples, one per word.

## Stemmer
Lower-case the word, then scan this table from top to bottom. Apply the **first** rule whose suffix the word ends with **and** whose remaining stem (the word with the suffix removed) has at least 3 characters: replace the suffix and stop. If no rule applies, the stem is the lower-cased word.

| suffix | replacement |
| --- | --- |
| `ational` | `ate` |
| `ization` | `ize` |
| `fulness` | `ful` |
| `ousness` | `ous` |
| `iveness` | `ive` |
| `ness` | (nothing) |
| `ment` | (nothing) |
| `ing` | (nothing) |
| `ies` | `i` |
| `ied` | `i` |
| `ed` | (nothing) |
| `ly` | (nothing) |
| `sses` | `ss` |
| `ss` | `ss` |
| `s` | (nothing) |

A rule whose stem would be too short is skipped and the scan continues with the next row, so `ties` skips `ies` (stem `t`) and then matches `s`, giving `tie`.

## Lemmatizer
`lemmas` is a dict from lower-case word forms to lemmas. The lemma is `lemmas[w]` for the lower-cased word `w`, or `w` itself if it is not in the dict.
"""

STARTER = '''RULES = [("ational", "ate"), ("ization", "ize"), ("fulness", "ful"), ("ousness", "ous"), ("iveness", "ive"),
         ("ness", ""), ("ment", ""), ("ing", ""), ("ies", "i"), ("ied", "i"), ("ed", ""), ("ly", ""),
         ("sses", "ss"), ("ss", "ss"), ("s", "")]

def stem_and_lemma(words, lemmas):
    """Return [(stem, lemma), ...] for each word."""
    raise NotImplementedError
'''

SOLUTION = '''RULES = [("ational", "ate"), ("ization", "ize"), ("fulness", "ful"), ("ousness", "ous"), ("iveness", "ive"),
         ("ness", ""), ("ment", ""), ("ing", ""), ("ies", "i"), ("ied", "i"), ("ed", ""), ("ly", ""),
         ("sses", "ss"), ("ss", "ss"), ("s", "")]

def stem(w):
    for suf, rep in RULES:                       # order matters: longer, more specific suffixes first
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:len(w) - len(suf)] + rep
    return w

def stem_and_lemma(words, lemmas):
    out = []
    for word in words:
        w = word.lower()
        out.append((stem(w), lemmas.get(w, w)))
    return out
'''

HINTS = [
    'Keep the rules as an ordered list of `(suffix, replacement)` pairs and loop over it. The first rule that applies wins.',
    'A rule applies only if `w.endswith(suffix)` and `len(w) - len(suffix) >= 3`. If the length check fails, keep scanning rather than stopping.',
    'Why is `("ss", "ss")` in the table? It catches `glass` before the final `s` rule can strip it. The lemma is just `lemmas.get(w, w)` on the lower-cased word.',
]

_LEM = {"ran": "run", "running": "run", "better": "good", "mice": "mouse", "flies": "fly", "studies": "study",
        "was": "be", "geese": "goose", "happily": "happy", "connected": "connect"}
TESTS = [
    (["running", "flies", "Happily"], _LEM),
    (["glass", "classes", "ties", "sing"], {}),
    ([], _LEM),
    (["Relational", "organization", "hopefulness", "callousness", "decisiveness"], _LEM),
    (["ran", "mice", "geese", "better", "was"], _LEM),
    (["kindness", "payment", "studied", "studies", "connected", "fly"], _LEM),
    (["is", "bus", "red", "bed", "news", "ing"], {}),
    (["CATS", "Dogs", "boss", "Mess", "cried", "fed"], {"cats": "cat", "dogs": "dog"}),
]

EXPLANATION = r"""
## Why the order of the rules matters
Several suffixes overlap: every word ending in `ies` or `sses` also ends in `s`. Putting the longer suffix first makes the most specific rule win. `("ss", "ss")` changes nothing, but by matching first it protects `glass` and `boss` from the plain `s` rule. The minimum-stem check stops `is`, `red` and `sing` from being cut down to nothing.

## Stems vs lemmas
| word | stem | lemma |
| --- | --- | --- |
| running | runn | run |
| flies | flie | fly |
| ran | ran | run |
| better | better | good |

The stemmer never needs a dictionary and handles any word, but it cannot do irregular forms (`ran`, `mice`) and it produces non-words. The lemmatizer gets irregulars right but returns unknown words unchanged. Real lemmatizers also use the part of speech: *meeting* is `meet` as a verb and `meeting` as a noun.

## Complexity
$O(R)$ suffix checks per word for $R$ rules, plus one dict lookup.

## Common bugs
- Stopping the scan as soon as a suffix matches, even when the stem is too short.
- Forgetting to lower-case before both the stemmer and the dictionary lookup.

## Interview follow-ups
- When does stemming help a model and when does it hurt? (It helps sparse bag-of-words models on small data by merging features. It mostly hurts subword-based neural models, which learn morphology themselves.)
- How would you evaluate a stemmer? (Under-stemming vs over-stemming errors against groups of words that should or should not share a base.)
"""
