META = dict(title='Unicode-aware text normalization', tags=['NLP', 'Data Processing'], diff='Easy', runtime='python',
            lessons=['ieng:01.02'], related=['tokenization', 'regex-patterns-for-nlp', 'text-deduplication-minhash'])

STATEMENT = r"""
Two strings that look the same can differ byte for byte: `"café"` may store `é` as one code point or as `e` plus a combining accent, a ligature `"ﬁ"` is one character, and `"ß"` should match `"SS"` in a case-insensitive search. Normalizing first makes counting, deduplication and lookup reliable.

Write `normalize(text, strip_accents=True)` using the standard library module `unicodedata`. Apply these steps in order:

1. `unicodedata.normalize("NFKD", text)`: decompose characters and fold compatibility forms (ligatures, full-width letters, superscripts).
2. If `strip_accents` is true, delete every character `c` with `unicodedata.combining(c) != 0`.
3. `str.casefold()`: aggressive lower-casing (`"ß"` becomes `"ss"`).
4. Collapse whitespace: split on any run of whitespace and re-join the pieces with a single space, which also trims both ends.
5. `unicodedata.normalize("NFC", ...)`: recompose what is left, so kept accents are stored as single code points.

Return the resulting string.
"""

STARTER = '''import unicodedata

def normalize(text, strip_accents=True):
    """NFKD -> (strip combining marks) -> casefold -> collapse whitespace -> NFC."""
    return text.lower().strip()
'''

SOLUTION = '''import unicodedata

def normalize(text, strip_accents=True):
    s = unicodedata.normalize("NFKD", text)                     # é -> e + U+0301, ﬁ -> f i
    if strip_accents:
        s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()                                            # stronger than lower(): ß -> ss
    s = " ".join(s.split())                                     # any whitespace run -> one space, trimmed
    return unicodedata.normalize("NFC", s)                      # recompose kept accents
'''

HINTS = [
    'Accents become separate "combining" code points only after decomposition. Which normalization form decomposes and also folds ligatures?',
    'After NFKD, filter with `unicodedata.combining(c)`: it is non-zero exactly for the accent marks. Then `casefold()`, not `lower()`.',
    '`" ".join(s.split())` collapses tabs, newlines and non-breaking spaces into single spaces and trims the ends. Finish with NFC so the output is canonical when accents are kept.',
]

TESTS = [
    ("  Café\tCRÈME  ",),
    ("Straße ﬁnal",),
    ("Crème brûlée", False),
    ("",),
    ("ＦＵＬＬ　ｗｉｄｔｈ ２０２４",),
    ("x² + H₂O\n\n done",),
    ("Ångström é vs é", False),
    ("ÉCOLE Ölçek naïve",),
]

EXPLANATION = r"""
## Why the order matters
Accents can only be removed once they are separate code points, so decomposition (NFKD) comes first. The **K** ("compatibility") part also maps look-alikes onto plain characters: full-width `Ｆ` to `F`, `²` to `2`, the ligature `ﬁ` to `fi`. Case folding comes after stripping so that every decomposed letter gets folded. NFC at the end matters only when accents are kept: `e` + U+0301 and the single code point `é` both come out as `é`.

## casefold vs lower
`lower()` maps `"ß"` to itself, so `"STRASSE".lower() != "straße".lower()`. `casefold()` is designed for caseless matching and maps `"ß"` to `"ss"`.

## Complexity
Linear in the length of the text: each step is one pass.

## Common bugs
- Stripping combining marks after NFC: there are none left to strip.
- `text.replace("  ", " ")` for whitespace: misses tabs, newlines, ` ` and runs of three.
- Applying NFKD where the distinction matters: it turns `x²` into `x2`, which is wrong for maths or chemistry text.

## Interview follow-ups
- Why do BERT's uncased models strip accents but cased multilingual models usually don't? (Accents carry meaning in many languages, e.g. Spanish *año* vs *ano*.)
- Normalization changes string lengths. How do you keep character offsets into the original text? (Record, for each output character, the index of the input character it came from.)
"""
