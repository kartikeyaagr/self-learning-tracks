META = dict(title='Align word labels to subword tokens', tags=['NLP', 'Transformers'], diff='Easy', runtime='python',
            lessons=['ieng:01.02'], related=['named-entity-recognition', 'subword-tokenization', 'masked-language-modeling'])

STATEMENT = r"""
NER datasets label **words**, but a Transformer sees **subword tokens**: `Washington` might become `Wash`, `##ington`, and `[CLS]`/`[SEP]` belong to no word at all. Before training, the word labels have to be spread over the tokens.

Write `align_labels(word_ids, word_labels, label2id, label_all=False, ignore=-100)` that returns a list of ints, one per token.

## Inputs
- `word_ids`: for each token, the index of the word it came from, or `None` for special tokens. Tokens of the same word are consecutive.
- `word_labels`: one string label per word, such as `"B-LOC"`, `"I-LOC"`, `"O"`.
- `label2id`: dict from label string to int id.

## Rules
- A token with word id `None` gets `ignore`.
- The **first** token of a word (its word id differs from the previous token's) gets `label2id[word_labels[w]]`.
- Any **later** token of the same word gets `ignore` when `label_all` is false. When `label_all` is true it gets the word's label, except that a label starting with `"B-"` is changed to `"I-"` with the same type (the entity has already begun). That label is guaranteed to be in `label2id`.
"""

STARTER = '''def align_labels(word_ids, word_labels, label2id, label_all=False, ignore=-100):
    """Return one label id per token."""
    raise NotImplementedError
'''

SOLUTION = '''def align_labels(word_ids, word_labels, label2id, label_all=False, ignore=-100):
    out, prev = [], None
    for w in word_ids:
        if w is None:
            out.append(ignore)
        elif w != prev:
            out.append(label2id[word_labels[w]])          # first piece carries the word's label
        elif label_all:
            lab = word_labels[w]
            if lab.startswith("B-"):
                lab = "I-" + lab[2:]                       # continuation of an entity that already began
            out.append(label2id[lab])
        else:
            out.append(ignore)
        prev = w
    return out
'''

HINTS = [
    'Walk the tokens remembering the previous word id. Three cases: special token, first piece of a word, later piece of the same word.',
    'A token is the first piece of its word exactly when `w != prev`. Update `prev` after every token, including special tokens.',
    'For later pieces with `label_all=True`, rewrite `"B-X"` as `"I-" + label[2:]`; `"I-X"` and `"O"` stay as they are.',
]

_L = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
TESTS = [
    ([None, 0, 1, 1, 2, None], ["B-PER", "B-LOC", "O"], _L),
    ([None, 0, 1, 1, 2, None], ["B-PER", "B-LOC", "O"], _L, True),
    ([None, None], [], _L),
    ([0, 0, 0], ["I-ORG"], _L, True, -1),
    ([None, 0, 1, 2, 2, 2, 3, None, None], ["O", "B-ORG", "I-ORG", "O"], _L, True),
    ([None, 0, 0, 1, 2, 2, None], ["B-PER", "I-PER", "B-LOC"], _L),
    ([0, 1, 1, 1, 2, 3, 3], ["O", "B-LOC", "O", "B-PER"], {"O": 7, "B-LOC": 3, "I-LOC": 9, "B-PER": 1, "I-PER": 2}, True),
]

EXPLANATION = r"""
## Why it works
Labelling only the first piece keeps exactly one prediction per word, so token-level training matches word-level evaluation, and `ignore` (the default `ignore_index` of PyTorch's cross-entropy) removes the other pieces from the loss. `label_all` instead gives every piece a target, which adds training signal; the B-to-I rewrite keeps the tag sequence valid, since only the first piece of an entity can be its beginning.

## Detecting the first piece
Comparing with the previous token's word id works because the tokens of a word are always contiguous. Special tokens break runs naturally because their id is `None`.

## Complexity
$O(n)$ for $n$ tokens.

## Common bugs
- Giving `[CLS]`/`[SEP]` the `O` label, which teaches the model something meaningless and skews accuracy.
- Copying `B-PER` onto every piece with `label_all`, so one word looks like several entities.
- Forgetting to update `prev` on special tokens.

## Interview follow-ups
- At inference time, how do you turn token predictions back into word labels? (Take the prediction of each word's first piece, or aggregate the pieces' probabilities.)
- With a sliding window over long documents, a word appears in two windows. Which prediction do you keep? (The one with the most context on both sides, i.e. furthest from the window edge.)
"""
