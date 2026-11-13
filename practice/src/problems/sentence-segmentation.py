META = dict(title='Rule-based sentence splitter', tags=['NLP'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:04.24'], related=['tokenization', 'regex-patterns-for-nlp'])

STATEMENT = r"""
Splitting on every `.` breaks on `Dr. Smith`, `3.14` and `e.g. this`. Write `split_sentences(text, abbreviations)` that follows these rules.

## Candidate boundaries
A **terminator** is a run of one or more characters from `.!?`, followed by zero or more closing characters from `"')]` (the regex `[.!?]+["')\]]*`). Scan terminators left to right, non-overlapping. Let the terminator span `text[s:e]`. It is a sentence boundary unless one of these holds:

1. `e < len(text)` and `text[e]` is not whitespace (`3.14`, `e.g.,`, `Yahoo!'s`).
2. The first non-whitespace character after `e` exists and is lower-case (`str.islower()`).
3. The `[.!?]` part of the terminator is exactly one `.`, and the **word before it** is an abbreviation or a single letter. The word before it is the maximal run of non-whitespace characters ending at `s`, with leading characters from `"'([` removed, then lower-cased. It blocks the boundary if it is in `abbreviations` (a list of lower-case strings without the final period, like `"dr"` or `"e.g"`) or if it is one alphabetic character (an initial, as in `J. Smith`).

At the end of the text (only whitespace or nothing after `e`) rules 1 and 2 never block, but rule 3 still does.

## Output
Cut the text after every boundary. Return the pieces with `str.strip()` applied, dropping pieces that are empty after stripping. The text after the last boundary is the final sentence if it is non-empty.
"""

STARTER = '''import re

def split_sentences(text, abbreviations):
    """Return a list of sentence strings."""
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
'''

SOLUTION = '''import re

TERM = re.compile(r"[.!?]+[\\"')\\]]*")

def split_sentences(text, abbreviations):
    abbreviations = set(abbreviations)               # O(1) lookups
    out, start = [], 0
    for m in TERM.finditer(text):
        s, e = m.span()
        if e < len(text) and not text[e].isspace():
            continue                                    # glued to the next token: 3.14, e.g.,
        rest = text[e:].lstrip()
        if rest and rest[0].islower():
            continue                                    # sentences don't start lower-case
        if m.group().rstrip("\\"')]") == ".":
            k = s
            while k > 0 and not text[k - 1].isspace():
                k -= 1
            word = text[k:s].lstrip("\\"'([").lower()
            if word in abbreviations or (len(word) == 1 and word.isalpha()):
                continue                                # Dr. / e.g. / J.
        piece = text[start:e].strip()
        if piece:
            out.append(piece)
        start = e
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
'''

HINTS = [
    'Find every candidate with one regex, `[.!?]+["\')\\]]*`, and then decide per match whether to cut. Keep the index where the current sentence started.',
    'Check in order: is the next character non-space (no cut)? Is the next non-space character lower-case (no cut)? Is it a lone `.` after an abbreviation or an initial (no cut)?',
    'For the abbreviation test walk left from the match start until whitespace to get the word, strip leading `"\'([`, lower-case it, and look it up. `...` and `?!` are not a lone period, so the abbreviation rule does not apply to them.',
]

_AB = ["dr", "e.g", "etc", "i.e", "mr", "mrs", "no", "prof", "st", "u.s", "vs"]
TESTS = [
    ("Dr. Smith arrived. He was late!", _AB),
    ("Pi is 3.14. Is it? Yes!! \"Really.\" Then we left.", _AB),
    ("", _AB),
    ("   ", _AB),
    ("No terminator at all", []),
    ("Use tools, e.g. a hammer. J. R. R. Tolkien wrote it. The U.S. Army won.", _AB),
    ("Wait... What? (He left.) she said. \"Go!\" OK", _AB),
    ("Visit St. Paul's. it's lower. Prof. X vs. Y! End. ", _AB),
    ("Line one.\nLine two!\n\n(Line three?) [Four.]", []),
]

EXPLANATION = r"""
## Why it works
The regex proposes every place a sentence could end, and three cheap tests veto the false alarms. A real boundary needs whitespace after it, a next sentence that does not start lower-case, and, for a lone period, a word before it that is not a known abbreviation or an initial. Including closers in the terminator keeps `"Really."` and `(He left.)` together with their quotes and brackets.

## Where rules fail
- `The U.S. Army won.` stays one sentence, but `I live in the U.S. It is big.` should split and won't: the abbreviation rule cannot tell. Statistical splitters (Punkt) learn which abbreviations often end sentences from collocation counts.
- `Plan B. Then...` is treated as an initial and not split.
- Lower-case starts are common in chat and lyrics; rule 2 will merge those sentences.

## Complexity
The regex scan is $O(n)$. Walking back for the previous word costs at most its length, so the total is $O(n)$.

## Common bugs
- `re.split(r"[.!?]")`, which splits `3.14` and loses the punctuation.
- Leaving a closing quote at the start of the next sentence because closers were not part of the terminator.
- Applying the abbreviation rule to `!` or `...`.

## Interview follow-ups
- How would you learn the abbreviation list from raw text? (Count how often a token appears followed by a period, and how often that period is followed by a lower-case word.)
- Why do many LLM data pipelines not split into sentences at all? (Models train on long documents, and sentence boundaries only matter for tasks like alignment, summarization labels or TTS.)
"""
