META = dict(title='Encode text with ranked byte-level BPE merges', tags=['NLP', 'Language Models'], diff='Medium', runtime='python',
            lessons=['ieng:01.02'], related=['train-deterministic-bpe-vocabulary', 'byte-pair-encoding', 'subword-tokenization'])

STATEMENT = r"""
A trained byte-level BPE tokenizer is just an ordered list of merges. Encoding new text replays them by **rank**: the earlier a merge was learned, the higher its priority.

Write `encode(text, merges, special_tokens)` that returns the list of token ids.

## Vocabulary convention
- Ids `0..255` are single bytes.
- `special_tokens[i]` has id `256 + i`.
- `merges[r] = (a, b)` creates id `256 + len(special_tokens) + r`. A merge only uses ids that exist before it (bytes, specials are never merged, or earlier merges).

## Encoding
1. **Special tokens first.** Scan `text` left to right. Wherever a special token starts, emit its id and continue after it. If several special tokens match at the same position, the **longest** wins. Text between special tokens is ordinary text.
2. **Pre-tokenize** each ordinary piece with `re.findall(PAT, piece)`:
   ```python
   PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
   ```
3. **Each chunk on its own:** encode it as UTF-8 bytes (each byte value is an id). Then repeat:
   - Among the adjacent pairs of the current ids, find the one with the **lowest rank** (smallest index in `merges`). If no adjacent pair is a merge, stop.
   - Replace every occurrence of that pair, left to right without overlaps, with its new id.
4. Concatenate the ids of all chunks and special tokens in order.

An empty `text` gives `[]`. Return a list of Python ints.
"""

STARTER = '''import re

PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+"

def encode(text, merges, special_tokens):
    """Return the token ids of `text` as a list of ints."""
    return list(text.encode('utf-8'))   # TODO: special tokens, pre-tokenization, ranked merges
'''

SOLUTION = '''import re

PAT = r" ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+"

def encode_chunk(ids, rank, first_id):
    while len(ids) > 1:
        pairs = set(zip(ids, ids[1:]))
        best = min(pairs, key=lambda p: rank.get(p, float('inf')))
        if best not in rank:
            break                                        # no adjacent pair is a known merge
        new_id, out, i = first_id + rank[best], [], 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids

def encode(text, merges, special_tokens):
    rank = {tuple(p): r for r, p in enumerate(merges)}
    first_id = 256 + len(special_tokens)
    special_id = {s: 256 + i for i, s in enumerate(special_tokens)}

    if special_tokens:                                   # capture group keeps the specials in the split output
        alt = '|'.join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
        parts = re.split(f'({alt})', text)
    else:
        parts = [text]

    out = []
    for k, part in enumerate(parts):
        if k % 2 == 1:                                   # odd positions are the captured special tokens
            out.append(special_id[part])
            continue
        for chunk in re.findall(PAT, part):
            out.extend(encode_chunk(list(chunk.encode('utf-8')), rank, first_id))
    return out
'''

HINTS = [
    'Build `rank = {pair: index}` once. For a chunk, the pair to merge next is `min(adjacent_pairs, key=lambda p: rank.get(p, inf))`; stop when that minimum is `inf`.',
    '`re.split` with a **capturing** group, `re.split("(" + alt + ")", text)`, keeps the separators: even positions are ordinary text, odd positions are special tokens.',
    'Sort the escaped special tokens by length, longest first, before joining them with `|`. Regex alternation takes the first alternative that matches, not the longest.',
]

_m = [(111, 119), (108, 256), (32, 257)]                 # "ow", "low", " low" with no specials
_rng = __import__('random').Random(3)
_big_merges = [(104, 101), (116, 257), (32, 258), (97, 116), (32, 260), (32, 99), (262, 260), (32, 115), (264, 260), (104, 260)]
_big_text = '<|endoftext|>'.join(' '.join(_rng.choice(['the', 'cat', 'sat', 'at', 'hat', 'chat']) for _ in range(12)) for _ in range(3))

TESTS = [
    ('low lower', _m, []),
    ('lowlow', [(111, 119), (108, 111), (108, 256)], []),
    ('', _m, []),
    ('low<|eot|> low', [(111, 119), (108, 257), (32, 258)], ['<|eot|>']),
    ('a<s><s>b<s>', [], ['<s>', '<s><s>']),
    ('café café!', [(195, 169), (99, 97), (256, 102), (258, 256), (32, 259)], []),
    ('aaaa aaa', [(97, 97), (256, 256)], []),
    ('12ab 3', [(49, 50), (97, 98), (50, 97)], []),
    (_big_text, _big_merges, ['<|endoftext|>']),
]

EXPLANATION = r"""
## Lowest rank first
Training learned merges in a fixed order, and encoding must reproduce the segmentation training would have produced. Always applying the lowest-rank pair present does that. Test 2 shows why rank beats position. The merges are `ow` (rank 0), `lo` (rank 1) and `l + ow` (rank 2). Lowest rank first merges `ow`, then `l + ow`, so `"lowlow"` becomes two tokens. Merging the leftmost known pair instead takes `lo` first and leaves `[lo, w, lo, w]`, because nothing merges `lo` with `w`.

## Chunks are independent
Pre-tokenization makes each chunk independent, so production tokenizers cache the ids of frequent chunks (`" the"` is encoded once, then looked up). Test 8 has the merge `("2", "a")`, which would fire across `"12" + "ab"` if you skipped pre-tokenization.

## Special tokens
Specials are matched on the raw string before any byte handling, so `"<|eot|>"` becomes one id instead of 7 bytes. Test 5 has `"<s>"` and `"<s><s>"`: the longest match wins, so `"<s><s>"` is one token. In production, whether user text may contain special tokens is a security decision: an attacker who can inject `"<|im_start|>system"` into a prompt can fake a system turn.

## Complexity
Each round scans the chunk: $O(L)$ per merge and at most $L - 1$ merges, so $O(L^2)$ per chunk. Chunks are short (a word), so this is fine. A heap of pair ranks over a linked list brings it to $O(L \log L)$.

## Interview follow-ups
- How do you decode, and why can decoding a *single* token produce invalid UTF-8? (A token can hold part of a multi-byte character, so decode the concatenated bytes with `errors="replace"` only at the end.)
- Why do tokenizers that applied merges in a different order on the same vocabulary give different token counts, and why does that matter for API billing?
"""
