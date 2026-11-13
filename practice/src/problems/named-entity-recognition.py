META = dict(title='Decode BIO tags into entity spans', tags=['NLP'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:04.24'], related=['token-classification', 'greedy-sequence-tagging', 'span-extraction'])

STATEMENT = r"""
Named entity recognition is usually trained as tagging: each token gets `B-TYPE` (begins an entity), `I-TYPE` (inside one) or `O` (outside). To report or score entities you have to turn the tag sequence back into spans.

Write `bio_to_spans(tags)` that returns a list of `(type, start, end)` tuples, with `end` exclusive, in order of `start`.

## Rules
- `B-X` always starts a new entity of type `X` (closing any open entity).
- `I-X` continues the open entity if it has type `X`. Otherwise (after `O`, at the start, or after a different type) it **starts** a new entity of type `X`. This lenient rule is what most evaluation scripts use for malformed sequences.
- `O` closes any open entity.
- An entity still open at the end of the list ends there.

The type is everything after the first `-`, so `B-WORK-OF-ART` has type `WORK-OF-ART`.

## Example
`["B-PER", "I-PER", "O", "B-LOC"]` gives `[("PER", 0, 2), ("LOC", 3, 4)]`.
"""

STARTER = '''def bio_to_spans(tags):
    """Return [(type, start, end_exclusive), ...]."""
    raise NotImplementedError
'''

SOLUTION = '''def bio_to_spans(tags):
    spans, cur, start = [], None, 0          # cur = type of the open entity, or None
    for i, tag in enumerate(tags):
        prefix, _, typ = tag.partition("-")
        if prefix == "I" and cur == typ:
            continue                          # entity keeps growing
        if cur is not None:
            spans.append((cur, start, i))     # anything else closes the open entity
            cur = None
        if prefix in ("B", "I"):
            cur, start = typ, i               # B-X, or an I-X that cannot continue
    if cur is not None:
        spans.append((cur, start, len(tags)))
    return spans
'''

HINTS = [
    'Track one open entity: its type and its start index. Every tag either extends it, closes it, or closes it and opens a new one.',
    'Only one case extends: an `I-` tag whose type equals the open type. In every other case, first close the open entity (if any).',
    "After closing, open a new entity if the tag starts with `B` or `I`. `tag.partition(\"-\")` splits at the first dash, which keeps types like `WORK-OF-ART` whole. Don't forget the entity still open after the loop.",
]

TESTS = [
    (["B-PER", "I-PER", "O", "B-LOC"],),
    (["I-ORG", "I-ORG", "B-ORG", "I-PER", "O"],),
    ([],),
    (["O", "O", "O"],),
    (["B-LOC", "B-LOC", "I-LOC", "I-LOC"],),
    (["B-WORK-OF-ART", "I-WORK-OF-ART", "I-DATE", "O", "I-DATE"],),
    (["B-PER"],),
    (["O", "B-MISC", "I-MISC", "I-PER", "I-PER", "B-ORG", "O", "B-LOC", "I-LOC"],),
]

EXPLANATION = r"""
## Why it works
At each token there is at most one open entity, so a single pass with two variables is enough. The decision depends only on the current tag and the open type, which is why the code reduces to "extend if `I-` of the same type, else close and maybe open".

## Malformed sequences
Models produce invalid sequences such as `O I-PER` or `B-LOC I-PER`. The lenient rule treats such an `I-` as the start of a new entity. A strict decoder would drop it instead; scores can differ noticeably between the two, so evaluation code must state which one it uses. Constrained decoding (a CRF or masking invalid transitions) avoids the problem by never producing them.

## Complexity
$O(n)$ time, $O(\text{entities})$ output.

## Common bugs
- Merging `B-LOC B-LOC` into one entity: two consecutive `B-` tags are two entities.
- Losing the last entity because it is only appended when an `O` arrives.
- `tag.split("-")[1]`, which cuts `WORK-OF-ART` down to `WORK`.

## Interview follow-ups
- How do you compute entity-level precision, recall and F1 from predicted and gold spans? (Exact match on `(type, start, end)`, as sets.)
- What do BIOES (or BILOU) tags add, and why do some systems train better with them? (Explicit end and single-token tags make the boundaries local decisions.)
"""
