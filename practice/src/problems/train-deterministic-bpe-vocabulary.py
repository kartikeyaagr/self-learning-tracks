META = dict(title='Train a byte-level BPE vocabulary', tags=['NLP', 'Language Models'], diff='Medium', runtime='python',
            lessons=['ieng:01.02'], related=['apply-ranked-bpe-merges', 'byte-pair-encoding', 'bpe-merge-step'])

STATEMENT = r"""
Modern LLM tokenizers run BPE on **bytes**, not characters: every string is UTF-8 encoded first, so any text (emoji, accents, code) can be represented with no unknown token. Training has to be fully deterministic, because two machines that train on the same text must produce the same vocabulary.

Write `train_bpe(text, vocab_size, special_tokens)` that returns `(merges, new_tokens)`.

## Initial vocabulary
- Ids `0..255` are the 256 single bytes: id `b` is the byte with value `b`.
- Then the special tokens, in the order given: `special_tokens[i]` gets id `256 + i`.
- Each merge then adds one id, starting at `256 + len(special_tokens)`.

## Preparing the training data
1. **Cut out special tokens.** Split `text` at every occurrence of a special token and drop the special tokens themselves, so no merge can span one. When several special tokens could match at the same place, the longest wins.
2. **Pre-tokenize** each remaining piece with `re.findall(PAT, piece)` where
   ```python
   PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
   ```
   Merges never cross the boundary between two chunks.
3. Encode each chunk with UTF-8; each byte is one starting symbol (its id is the byte value). Identical chunks can be counted once with a frequency.

## Training loop
While the vocabulary has fewer than `vocab_size` ids:
1. Count every adjacent pair of ids across all chunks, weighting each chunk by how often it occurs.
2. If there are no pairs, stop.
3. Pick the pair with the highest count. **Ties** go to the pair whose byte strings are lexicographically greatest: compare `(bytes_of(a), bytes_of(b))` as a tuple of Python `bytes` objects.
4. Give it the next id; its byte string is `bytes_of(a) + bytes_of(b)`. Replace every occurrence in every chunk, scanning left to right without overlaps.

`vocab_size` is at least `256 + len(special_tokens)`.

## Output
- `merges`: list of `(a, b)` tuples of ids, in the order they were learned.
- `new_tokens`: list with one entry per merge, the byte values of that new token as a list of ints (e.g. `[108, 111]` for `b"lo"`).
"""

STARTER = '''import re

PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+"

def train_bpe(text, vocab_size, special_tokens):
    """Return (merges: list of (id, id), new_tokens: list of byte-value lists)."""
    merges, new_tokens = [], []
    # TODO: split out special tokens, pre-tokenize, then merge until vocab_size
    return merges, new_tokens
'''

SOLUTION = '''import re
from collections import Counter

PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+"

def merge(word, pair, new_id):
    """Replace non-overlapping occurrences of `pair` in the tuple `word`, left to right."""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

def train_bpe(text, vocab_size, special_tokens):
    vocab = [bytes([b]) for b in range(256)] + [s.encode('utf-8') for s in special_tokens]
    first_new = len(vocab)

    pieces = [text]
    if special_tokens:                                   # longest first, so overlapping specials resolve correctly
        alt = '|'.join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
        pieces = re.split(alt, text)
    words = Counter(tuple(chunk.encode('utf-8')) for p in pieces for chunk in re.findall(PAT, p))

    merges = []
    while len(vocab) < vocab_size:
        pairs = Counter()
        for w, n in words.items():
            for pair in zip(w, w[1:]):
                pairs[pair] += n
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], vocab[p[0]], vocab[p[1]]))   # count, then greatest bytes
        new_id = len(vocab)
        vocab.append(vocab[best[0]] + vocab[best[1]])
        merges.append(best)
        nxt = Counter()
        for w, n in words.items():
            nxt[merge(w, best, new_id)] += n
        words = nxt
    return merges, [list(t) for t in vocab[first_new:]]
'''

HINTS = [
    'Store each distinct chunk once as a tuple of byte ids with its frequency (`Counter(tuple(chunk.encode("utf-8")) ...)`). Pair counts are then frequency-weighted sums.',
    'Keep a list `vocab` mapping id to `bytes`. The tie-break is then just `max(pairs, key=lambda p: (count[p], vocab[p[0]], vocab[p[1]]))`.',
    'Remove special tokens with `re.split` on an alternation of `re.escape`d specials sorted by length, longest first. The specials still occupy ids 256.. even though they never appear in the training chunks.',
]

_words = ['the', 'cat', 'sat', 'on', 'mat', 'hat', 'that', '42', 'at', 'a', '!', 'café']
_rng = __import__('random').Random(7)
_long = ' '.join(_rng.choice(_words) for _ in range(300))

TESTS = [
    ('low low low lower', 259, []),
    ('aaab', 260, []),
    ('ab<|end|>ab<|end|>ba', 262, ['<|end|>']),
    ('hi<|a|>bhi hi<|a|>b', 262, ['<|a|>', '<|a|>b']),
    ('naïve café café', 262, []),
    ('x y', 300, []),
    ('ab ba', 258, []),
    ('12 12 123 1234', 261, ['<s>']),
    (_long, 290, ['<|endoftext|>']),
]

EXPLANATION = r"""
## Why bytes and pre-tokenization
Starting from 256 byte ids means the vocabulary covers every possible string, so there is never an unknown token. Pre-tokenization (the regex) stops merges from crossing word boundaries and from fusing letters with digits or punctuation. Without it, frequent sequences such as `"e t"` across a space become tokens and the vocabulary is wasted on junk.

## Why the tie-break matters
In small corpora many pairs share the top count: in `"ab ba"` the pairs `(a, b)`, `(b, a)` and `( , b)` all occur once, and the rule picks `(b, a)`. Any rule works as long as everyone uses the same one. Without a fixed rule, two runs could disagree because of dictionary iteration order, and token ids would differ between the machine that trained the tokenizer and the one that serves it.

## Special tokens
They are cut out before counting, so `"<|end|>"` never becomes part of a merge and a document boundary can never be fused into a neighbouring word. They still take ids `256..` so their ids stay fixed however many merges you learn.

## Complexity
Each step recounts all pairs over the distinct chunks: $O(\text{merges} \times \text{total chunk length})$. Production trainers keep a pair-count table and an index from pairs to the chunks containing them, updating only the chunks a merge touches. That turns hours into minutes on large corpora.

## Common bugs
- Merging across chunks or across special tokens.
- Overlapping replacement: `aaa` merged on `(a, a)` is `[aa, a]`, not `[aa, aa]`.
- Comparing ids instead of byte strings in the tie-break.
- Forgetting that `é` is two UTF-8 bytes, so it starts as two symbols.

## Interview follow-ups
- How would you make one merge step $O(\text{affected chunks})$ instead of a full recount?
- Why do GPT-style regexes keep the leading space with the word (`" cat"`) instead of with the previous word?
"""
