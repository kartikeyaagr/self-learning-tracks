META = dict(title='Most-frequent-tag POS baseline', tags=['NLP', 'Classification'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:04.24'], related=['greedy-sequence-tagging', 'named-entity-recognition', 'text-classification'])

STATEMENT = r"""
A surprisingly strong part-of-speech tagger assigns every word the tag it had most often in training. Every real tagger is compared against this baseline.

Write `baseline_tagger(train, test)` that returns `(predictions, accuracy)`.

## Inputs
- `train`: a list of sentences, each a list of `(word, tag)` tuples.
- `test`: a list of sentences in the same format; the tags are the gold answers.

## Rules
- A word seen in training gets its most frequent training tag. Ties go to the tag that is smallest in string order.
- An unseen word gets the most frequent tag over the whole training set, ties again to the smallest string.
- Words are matched exactly (case-sensitive).
- The training set has at least one token.

## Output
- `predictions`: a list of tag lists, one per test sentence.
- `accuracy`: correct tags divided by the number of test tokens, as a float. If the test set has no tokens, return `0.0`.
"""

STARTER = '''def baseline_tagger(train, test):
    """Return (list of predicted tag lists, accuracy float)."""
    raise NotImplementedError
'''

SOLUTION = '''from collections import Counter, defaultdict

def best(counter):
    return min(counter, key=lambda t: (-counter[t], t))       # most frequent, then alphabetical

def baseline_tagger(train, test):
    per_word, overall = defaultdict(Counter), Counter()
    for sent in train:
        for w, t in sent:
            per_word[w][t] += 1
            overall[t] += 1
    choice = {w: best(c) for w, c in per_word.items()}
    fallback = best(overall)
    preds, right, total = [], 0, 0
    for sent in test:
        tags = [choice.get(w, fallback) for w, _ in sent]
        preds.append(tags)
        right += sum(p == g for p, (_, g) in zip(tags, sent))
        total += len(sent)
    return preds, right / total if total else 0.0
'''

HINTS = [
    'You need a tag counter per word and one counter over all tags. `defaultdict(Counter)` builds the first in one loop.',
    'Pick the winner of a counter with `min(c, key=lambda t: (-c[t], t))`: highest count first, then the smallest tag string.',
    'Precompute the choice for every training word and the fallback tag once, then tagging is a dict lookup with a default.',
]

_train = [[("the", "DT"), ("dog", "NN"), ("runs", "VBZ")], [("the", "DT"), ("run", "NN"), ("ended", "VBD")],
          [("dogs", "NNS"), ("run", "VBP"), ("fast", "RB")], [("they", "PRP"), ("run", "VBP")],
          [("a", "DT"), ("fast", "JJ"), ("dog", "NN")], [("The", "DT"), ("park", "NN")]]
TESTS = [
    ([[("a", "DT"), ("cat", "NN")], [("cat", "NN"), ("a", "DT")]], [[("a", "DT"), ("cat", "NN"), ("sat", "VBD")]]),
    (_train, [[("the", "DT"), ("dog", "NN"), ("run", "VBP"), ("fast", "RB")]]),
    (_train, []),
    (_train, [[]]),
    (_train, [[("The", "DT"), ("cats", "NNS"), ("run", "NN")], [("They", "PRP"), ("ran", "VBD")]]),
    ([[("x", "B"), ("x", "A"), ("y", "C")]], [[("x", "A"), ("z", "A")]]),
    (_train * 3 + [[("fast", "RB")]], [[("fast", "RB"), ("park", "NN"), ("dogs", "NNS"), ("ended", "VBD")]]),
]

EXPLANATION = r"""
## Why it works so well
Most words are unambiguous, and ambiguous words usually have one dominant tag (*run* is a verb more often than a noun). On standard English treebanks this baseline tags roughly nine out of ten tokens correctly. Good taggers reach about 97%, so all the hard work is in that last gap: ambiguous words in context and unknown words.

## Tie-breaking
Without a rule, a word seen once as `A` and once as `B` would get whichever tag the dict met first, so the result would depend on the order of the training data. The explicit rule makes it a pure function of the counts.

## Complexity
$O(N)$ to count training tokens, $O(\text{tags})$ per word to pick a winner, and $O(1)$ per test token.

## Common bugs
- Returning integer division or forgetting the empty-test case.
- Lower-casing silently (here `The` and `the` are different words, test 5).
- Falling back to a fixed tag like `NN` instead of the training set's most frequent tag.

## Interview follow-ups
- How would you improve unknown-word handling? (Suffix rules or features: `-ly` to `RB`, `-ed` to `VBD`, capitalised to `NNP`, digits to `CD`.)
- How does an HMM tagger with Viterbi use context that this baseline ignores? (Transition probabilities between neighbouring tags.)
"""
