META = dict(title='Extract URLs, emails, mentions and hashtags', tags=['NLP', 'Data Processing'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:04.24'], related=['tokenization', 'text-normalization'])

STATEMENT = r"""
Social-media text mixes words with URLs, email addresses, `@mentions` and `#hashtags`. A common preprocessing step pulls these out (or replaces them with placeholder tokens) before tokenizing.

Write `extract_entities(text)` with Python's `re` module. Return a dict with four keys, each a list of matched strings in order of appearance:

| key | pattern |
| --- | --- |
| `"urls"` | `http://` or `https://` followed by one or more non-whitespace characters. Then strip any trailing characters from the set `.,!?;:)` |
| `"emails"` | one or more of `[A-Za-z0-9._%+-]`, then `@`, then one or more of `[A-Za-z0-9-]`, then one or more groups of `.` followed by one or more of `[A-Za-z0-9-]` |
| `"mentions"` | `@` followed by one or more of `[A-Za-z0-9_]`, where the `@` is **not** immediately preceded by a character in `[A-Za-z0-9_]` |
| `"hashtags"` | `#` followed by one or more of `[A-Za-z0-9_]`, with the same rule for the character before `#` |

## Precedence
Find URLs first, then emails, then mentions and hashtags. After each stage, overwrite every reported match with spaces of the same length, and run the next pattern on that blanked text. So text inside a URL is never an email, mention or hashtag, text inside an email is never a mention, and the "preceding character" check sees a space where an earlier match used to be (`a@b.co#x` reports the email and the hashtag `#x`).

Mentions and hashtags keep their `@` / `#`; matching is case-sensitive and returns the text exactly as written.
"""

STARTER = '''import re

def extract_entities(text):
    """Return {"urls": [...], "emails": [...], "mentions": [...], "hashtags": [...]}."""
    return {"urls": [], "emails": [], "mentions": re.findall(r"@\\w+", text), "hashtags": re.findall(r"#\\w+", text)}
'''

SOLUTION = '''import re

URL = re.compile(r"https?://\\S+")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)+")
MENTION = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]+")
HASHTAG = re.compile(r"(?<![A-Za-z0-9_])#[A-Za-z0-9_]+")

def _take(pattern, text, strip=""):
    """Matches of pattern (trailing `strip` chars removed), and text with those spans blanked out."""
    found, chars = [], list(text)
    for m in pattern.finditer(text):
        s = m.group().rstrip(strip)
        if s:
            found.append(s)
            chars[m.start():m.start() + len(s)] = " " * len(s)   # blank only what we kept
    return found, "".join(chars)

def extract_entities(text):
    urls, text = _take(URL, text, ".,!?;:)")
    emails, text = _take(EMAIL, text)
    return {"urls": urls, "emails": emails,
            "mentions": MENTION.findall(text), "hashtags": HASHTAG.findall(text)}
'''

HINTS = [
    'Write one compiled pattern per entity type, and run them in precedence order: URLs, then emails, then mentions and hashtags.',
    'A negative look-behind `(?<![A-Za-z0-9_])@` stops the `@` inside `bob@mail.com` from looking like a mention. Blanking earlier matches with spaces stops `#frag` inside a URL from becoming a hashtag.',
    'For URLs, match `https?://\\S+` and then `.rstrip(".,!?;:)")`. Blank only the kept part, so a closing `)` after the URL is still ordinary text.',
]

TESTS = [
    ("Loved the talk by @ana_k! Slides: https://ex.com/talk?id=7. #NLP #ml",),
    ("Mail bob.smith+nlp@uni-x.ac.uk or ping @bob, not bob@ (and not a@b).",),
    ("",),
    ("no entities here, just words",),
    ("See (https://site.org/a#frag) and http://x.io/@user!!",),
    ("email@domain.com#tag x#notatag #ok_2 @@double end.",),
    ("Multi: https://a.com, https://b.com/path;q=1; done #one#two",),
    ("UPPER@CASE.COM @Mixed_Case #CamelCase http://h.co:8080/x)",),
]

EXPLANATION = r"""
## Why precedence matters
The patterns overlap. `https://x.io/@user` contains something mention-shaped and `bob@mail.com` contains `@mail`. Matching the most specific, longest structures first and erasing them leaves only genuine mentions and hashtags for the last two patterns. Replacing with spaces (not deleting) keeps the remaining text's positions and keeps neighbouring words from being glued together.

## Look-behind
`(?<![A-Za-z0-9_])` is a zero-width assertion: it checks the previous character without consuming it, and it also succeeds at the start of the string. That is how `x#notatag` is rejected while `#ok_2` is accepted. In `#one#two` the second `#` follows `e`, so only `#one` is a hashtag.

## Complexity
Each pattern scans the text once, so $O(n)$ per pattern for these regexes (none of them backtracks badly).

## Common bugs
- `\w` instead of `[A-Za-z0-9_]`. In Python 3 `\w` is Unicode-aware and matches accented letters, which changes the results.
- Keeping sentence punctuation on URLs (`...?id=7.`). Real-world URL extractors all have some trailing-punctuation rule.
- Running the mention pattern before the email pattern.

## Interview follow-ups
- Why replace URLs and mentions with placeholder tokens like `<url>` and `<user>` before training a classifier? (They are high-cardinality and mostly noise, but their presence can still be a useful signal.)
- What is catastrophic backtracking, and how would you spot a regex that is vulnerable to it?
"""
