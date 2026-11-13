META = dict(title='Word tokenizer with character offsets', tags=['NLP', 'Data Processing'], diff='Easy', runtime='python',
            lessons=['ieng:01.02', 'air-fanout-v1:04.24'], related=['text-normalization', 'regex-patterns-for-nlp', 'subword-tokenization'])

STATEMENT = r"""
Before any model sees text, it is cut into tokens. Many pipelines also keep each token's **character span** so predictions (an entity, an answer span) can be mapped back onto the original string.

Write `tokenize(text)` that returns a list of `(token, start, end)` tuples, where `text[start:end] == token`, in left-to-right order.

## Rules
1. A **word** is a maximal run of characters for which `str.isalnum()` is true.
2. An apostrophe `'` that has an alphanumeric character immediately on **both** sides belongs to the word, so `don't` and `O'Neil` are single tokens.
3. Every other non-whitespace character is a token on its own: `"?!"` gives two tokens, `"..."` gives three.
4. Whitespace (`str.isspace()`) separates tokens and never appears in one.

An empty or all-whitespace string gives `[]`.

## Example
`tokenize("Don't stop!")` returns `[("Don't", 0, 5), ("stop", 6, 10), ("!", 10, 11)]`.
"""

STARTER = '''def tokenize(text):
    """Return a list of (token, start, end) with text[start:end] == token."""
    raise NotImplementedError
'''

SOLUTION = '''def tokenize(text):
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c.isalnum():
            j = i + 1
            while j < n and (text[j].isalnum()
                             or (text[j] == "'" and j + 1 < n and text[j + 1].isalnum())):
                j += 1       # an apostrophe is kept only when a letter/digit follows (one already precedes)
            out.append((text[i:j], i, j))
            i = j
        else:
            out.append((c, i, i + 1))   # any other symbol is its own token
            i += 1
    return out
'''

HINTS = [
    'Walk the string with an index. At each position decide: skip whitespace, start a word, or emit a single symbol.',
    'Inside a word, keep extending while the next character is alphanumeric. The apostrophe needs a look-ahead: extend over it only if the character after it is alphanumeric too.',
    'You are already inside a word when you see the apostrophe, so its left neighbour is alphanumeric for free. Check `text[j + 1].isalnum()` (and that `j + 1 < n`) before consuming it.',
]

TESTS = [
    ("Don't stop!",),
    ("Wait... what?!",),
    ("",),
    ("   \t\n ",),
    ("rock 'n' roll, O'Neil's 3rd try",),
    ("x=42;y='a'",),
    ("Café naïve 東京 2024-01-05",),
    ("end'  'start  mid'dle''",),
]

EXPLANATION = r"""
## Why it works
The scanner is a tiny state machine with three cases. Words are extended greedily, so each word is maximal. The apostrophe rule needs only one character of look-ahead, because the character before the apostrophe is the one the word just consumed. Offsets come for free: the token is always `text[i:j]`.

## Complexity
Each character is examined a constant number of times: $O(n)$ time, and $O(n)$ for the output.

## Common bugs
- Using `str.split()` first. It loses the offsets and glues punctuation onto words (`"stop!"`).
- Using the regex class `\w`, which also matches `_`. That's a different tokenizer from the one specified here.
- Keeping a trailing or leading apostrophe (`end'`, `'start`) inside the word.
- Forgetting non-ASCII letters: `str.isalnum()` is Unicode-aware, so `Café` and `東京` stay whole.

## Interview follow-ups
- How would you map a model's subword predictions back to these words? (Keep the offsets of every subword and group by the word span that contains them.)
- Why do modern LLM tokenizers skip word splitting entirely and run byte-level BPE on raw text? (No language-specific rules, no unknown characters, and the round trip back to text is exact.)
"""
